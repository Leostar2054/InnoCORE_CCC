#!/usr/bin/env python3
"""
LDV pilot-dataset analysis suite  (Hann + Chirp in one file)
============================================================
First (pilot) dataset, 2026-05-26, single PZT source, sub-region scan, SUS304.

Runs the FULL pipeline on BOTH excitation files:
    Data260526_144631_Innocore_pilot1_5Hann.npy   # 5-cycle Hann tone-burst
    Data260526_145441_Innocore_pilot2_Chirp.npy   # broadband chirp

Array layout (both files)
    shape  : (T, Y, X) = (2048, 901, 521)   int16, raw LDV ADC counts
    time   : dt = 100 ns (10 MHz) -> 2048 samples = 204.8 us    [CONFIRMED]
    space  : 0.25 mm step -> Y=0..225 mm (901), X=0..130 mm (521), origin lower-left
    note   : un-filtered/un-normalised; +/-32767 = LDV signal-loss (masked below)

Analyses (per file, saved under ./pilot_analysis/)
    1. summary + single-point A-scan + amplitude spectrum  (recovers excitation band)
    2. wavefield snapshots                                 (wavefront + scattering)
    3. RMS energy map                                      (label-free damage indicator)
    4. B-scan (t-x)                                        (wave speed, reflections)
    5. F-K (frequency-wavenumber) spectrum + analytical A0/S0 overlay  (mode ID / dispersion)
    6. local-wavenumber (UWM) map                          (images the wall-thinning)
    7. thickness map  (A0 Rayleigh-Lamb inversion)         (quantitative depth proxy)
    + optional wavefield GIF
Then a Hann-vs-Chirp comparison (spectra, A0 dispersion, defect-region statistics).

Dependencies: numpy, scipy, matplotlib (Pillow only for the optional GIF).
Run:  python ldv_pilot_analysis.py        (edit FILES paths first)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter, zoom

# =============================== CONFIG =================================
FILES = {
    "Hann":  "/home/awais/Desktop/Awais Work 2026/KAIST Work/Team 1/KAIST Data/20260526_Data1/Data260526_144631_Innocore_pilot1_5Hann.npy",   # <-- your paths
    "Chirp": "/home/awais/Desktop/Awais Work 2026/KAIST Work/Team 1/KAIST Data/20260526_Data1/Data260526_145441_Innocore_pilot2_Chirp.npy",
}
DT          = 100e-9        # 100 ns (10 MHz)  [CONFIRMED]
STEP_MM     = 0.25          # scan step (Y and X)
SAT         = 32767         # int16 saturation (LDV signal-loss)

# material (SUS304) for the Lamb-wave dispersion / thickness inversion
CL, CT      = 5790.0, 3100.0    # m/s (longitudinal, shear)
D_NOMINAL_MM = 2.0              # nominal plate thickness

# analysis parameters
PROBE_MM    = (112.0, 65.0)         # (y, x) point for A-scan / spectrum
SNAP_US     = [10, 20, 30, 50, 80, 120]
BSCAN_Y_MM  = 112.0
FK_Y_BAND_MM = (90.0, 135.0)        # y-line band averaged for the F-K spectrum
UWM_FREQS_KHZ = {                   # frequencies used for local-wavenumber / thickness
    "Hann":  [100.0],
    "Chirp": [100.0, 150.0, 200.0],
}
UWM_WIN     = 48           # window size (px) for local-FFT wavenumber (48 px = 12 mm)
UWM_STRIDE  = 6            # window stride (px)

OUTDIR      = "pilot_analysis"
SAVE_PNG    = True
MAKE_GIF    = False
CMAP_FIELD  = "RdBu_r"
# =======================================================================


# =========================== Lamb dispersion ===========================
def _pq(w, k):
    p = np.sqrt((w / CL) ** 2 - k ** 2 + 0j)
    q = np.sqrt((w / CT) ** 2 - k ** 2 + 0j)
    return p, q


def lamb_det(k, w, h, mode):
    """Rayleigh-Lamb characteristic function (complex). mode 'A' or 'S'. h = half-thickness."""
    p, q = _pq(w, k)
    sp, cp = np.sin(p * h), np.cos(p * h)
    sq, cq = np.sin(q * h), np.cos(q * h)
    if mode == "A":
        return ((k ** 2 - q ** 2) ** 2) * sp * cq + 4 * k ** 2 * p * q * cp * sq
    return ((k ** 2 - q ** 2) ** 2) * cp * sq + 4 * k ** 2 * p * q * sp * cq


def lamb_k(f, d_mm, mode="A", kmax=2500.0):
    """Fundamental-mode wavenumber (rad/m) at frequency f for plate thickness d_mm."""
    w = 2 * np.pi * f
    h = d_mm * 1e-3 / 2.0
    kg = np.linspace(1.0, kmax, 6000)
    val = np.abs(lamb_det(kg, w, h, mode))
    loc = np.where((val[1:-1] < val[:-2]) & (val[1:-1] < val[2:]))[0] + 1
    if len(loc) == 0:
        return np.nan
    k0 = kg[loc[np.argmax(kg[loc])]]      # fundamental = largest-k minimum at low fd
    r = minimize_scalar(lambda k: np.abs(lamb_det(k, w, h, mode)),
                        bounds=(max(k0 - 3, 1.0), k0 + 3), method="bounded")
    return r.x


def lamb_curve(freqs_hz, d_mm, mode="A"):
    return np.array([lamb_k(f, d_mm, mode) for f in freqs_hz])


def group_velocity(f, d_mm, mode="A", df=1000.0):
    kp, km = lamb_k(f + df, d_mm, mode), lamb_k(f - df, d_mm, mode)
    return 2 * np.pi * (2 * df) / (kp - km)


def thickness_from_k(kmap_rad_m, f0, d_grid_mm=None):
    """Invert A0 k(d) at frequency f0 -> thickness map (mm)."""
    if d_grid_mm is None:
        d_grid_mm = np.linspace(0.4, 2.6, 130)
    ks = np.array([lamb_k(f0, dd, "A") for dd in d_grid_mm])   # rad/m
    order = np.argsort(ks)
    return np.interp(kmap_rad_m, ks[order], d_grid_mm[order])


# ================================ I/O ==================================
def load_scan(path):
    return np.load(path, mmap_mode="r")


def get_axes(vol):
    T, Y, X = vol.shape
    return (np.arange(T) * DT * 1e6, np.arange(Y) * STEP_MM, np.arange(X) * STEP_MM)


def mm_to_idx(mm, n):
    return int(np.clip(round(mm / STEP_MM), 0, n - 1))


def us_to_idx(t_us, n):
    return int(np.clip(round(t_us * 1e-6 / DT), 0, n - 1))


def print_summary(label, vol):
    T, Y, X = vol.shape
    print("-" * 64)
    print(f"[{label}]  shape={vol.shape}  dtype={vol.dtype}")
    print(f"   time : dt={DT*1e9:.0f} ns  fs={1/DT/1e6:.1f} MHz  dur={T*DT*1e6:.1f} us")
    print(f"   space: {Y}x{X} @ {STEP_MM} mm -> {(Y-1)*STEP_MM:.0f}(Y) x {(X-1)*STEP_MM:.0f}(X) mm")
    print(f"   range: [{int(vol.min())}, {int(vol.max())}]")


# ======================= streaming field pass ==========================
def field_pass(vol, freqs_hz, chunk=128):
    """
    ONE streaming pass over the cube -> per-pixel:
        mean, rms (DC-free energy), sat_mask, and the complex monochromatic
        field W[f] at each requested frequency (single-frequency DFT).
    """
    T, Y, X = vol.shape
    s1 = np.zeros((Y, X), np.float64)
    s2 = np.zeros((Y, X), np.float64)
    sat = np.zeros((Y, X), np.int64)
    W = {f: np.zeros((Y, X), np.complex128) for f in freqs_hz}
    tt = np.arange(T) * DT
    phasor = {f: np.exp(-2j * np.pi * f * tt).astype(np.complex64) for f in freqs_hz}
    for t0 in range(0, T, chunk):
        blk = vol[t0:t0 + chunk].astype(np.float32)
        s1 += blk.sum(axis=0)
        s2 += np.square(blk).sum(axis=0)
        sat += (np.abs(blk) >= SAT).sum(axis=0)
        for f in freqs_hz:
            W[f] += np.einsum("t,tyx->yx", phasor[f][t0:t0 + chunk], blk)
    mean = s1 / T
    rms = np.sqrt(np.maximum(s2 / T - mean ** 2, 0.0))
    return {"mean": mean, "rms": rms, "sat_mask": sat > 0, "W": W}


# ============================= extractors ==============================
def waveform_at(vol, y_mm, x_mm):
    iy, ix = mm_to_idx(y_mm, vol.shape[1]), mm_to_idx(x_mm, vol.shape[2])
    w = vol[:, iy, ix].astype(np.float32)
    w -= w.mean()
    return w, (iy, ix)


def amplitude_spectrum(w):
    f = np.fft.rfftfreq(len(w), d=DT)
    W = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    return f, W


def snapshot(vol, t_us, mean_map=None):
    it = us_to_idx(t_us, vol.shape[0])
    raw = vol[it]
    fr = raw.astype(np.float32) - (mean_map if mean_map is not None else raw.mean())
    fr[np.abs(raw) >= SAT] = np.nan
    return fr, it


def bscan_line(vol, y_mm):
    iy = mm_to_idx(y_mm, vol.shape[1])
    img = vol[:, iy, :].astype(np.float32)
    img -= img.mean(axis=0, keepdims=True)
    return img, iy


def fk_transform(vol, y_band_mm):
    iy0, iy1 = mm_to_idx(y_band_mm[0], vol.shape[1]), mm_to_idx(y_band_mm[1], vol.shape[1])
    rows = range(min(iy0, iy1), max(iy0, iy1) + 1)
    acc = None
    for iy in rows:
        line = vol[:, iy, :].astype(np.float32)
        line -= line.mean(axis=0, keepdims=True)
        F = np.abs(np.fft.fftshift(np.fft.fft2(line)))
        acc = F if acc is None else acc + F
    acc /= len(list(rows))
    T, X = acc.shape
    f = np.fft.fftshift(np.fft.fftfreq(T, d=DT))
    kx = np.fft.fftshift(np.fft.fftfreq(X, d=STEP_MM * 1e-3)) * 2 * np.pi   # rad/m
    return f, kx, acc


def local_wavenumber_map(Wc, win=UWM_WIN, stride=UWM_STRIDE):
    """
    Local dominant wavenumber via a sliding-window spatial FFT of the complex
    monochromatic field. Robust to the multi-directional (reverberant) content
    that defeats simple phase-gradient estimators. Returns |k| (rad/m) and the
    peak amplitude, both upsampled to the full grid.
    """
    step_m = STEP_MM * 1e-3
    Y, X = Wc.shape
    kfreq = 2 * np.pi * np.fft.fftfreq(win, d=step_m)          # rad/m
    KX, KY = np.meshgrid(kfreq, kfreq)
    KMAG = np.sqrt(KX ** 2 + KY ** 2)
    kmin = 1.5 * (2 * np.pi / (win * step_m))                  # ignore near-DC bins
    valid = KMAG > kmin
    han = np.outer(np.hanning(win), np.hanning(win))

    ylist = list(range(0, Y - win, stride))
    xlist = list(range(0, X - win, stride))
    kmap = np.zeros((len(ylist), len(xlist)))
    amp = np.zeros_like(kmap)
    for i, y0 in enumerate(ylist):
        for j, x0 in enumerate(xlist):
            patch = Wc[y0:y0 + win, x0:x0 + win] * han
            P = np.abs(np.fft.fft2(patch))
            P[~valid] = 0.0
            idx = np.argmax(P)
            kmap[i, j] = KMAG.ravel()[idx]
            amp[i, j] = P.ravel()[idx]
    kfull = zoom(kmap, (Y / kmap.shape[0], X / kmap.shape[1]), order=1)[:Y, :X]
    afull = zoom(amp, (Y / amp.shape[0], X / amp.shape[1]), order=1)[:Y, :X]
    return gaussian_filter(kfull, 1.0), afull


# ============================== plotting ===============================
def _extent(vol):
    T, Y, X = vol.shape
    return [0, (X - 1) * STEP_MM, 0, (Y - 1) * STEP_MM]


def _finish(fig, label, name):
    if SAVE_PNG:
        os.makedirs(OUTDIR, exist_ok=True)
        path = os.path.join(OUTDIR, f"{label}_{name}.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"   saved {path}")
    plt.close(fig)


def fig_waveform_spectrum(label, vol):
    t_us, _, _ = get_axes(vol)
    w, (iy, ix) = waveform_at(vol, *PROBE_MM)
    f, W = amplitude_spectrum(w)
    fp = f[np.argmax(W)] / 1e3
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t_us, w, lw=0.7)
    ax[0].set(title=f"[{label}] A-scan @ (y={iy*STEP_MM:.0f}, x={ix*STEP_MM:.0f}) mm",
              xlabel="time (us)", ylabel="amplitude (counts)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(f / 1e3, W, lw=0.9)
    ax[1].axvline(fp, color="r", ls="--", lw=0.8, label=f"peak ~ {fp:.0f} kHz")
    ax[1].set(title="amplitude spectrum", xlabel="frequency (kHz)",
              ylabel="|FFT|", xlim=(0, 400))
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    _finish(fig, label, "01_waveform_spectrum")
    return fp


def fig_snapshots(label, vol, mean_map):
    ext = _extent(vol)
    fmid, _ = snapshot(vol, SNAP_US[len(SNAP_US) // 2], mean_map)
    vlim = np.nanpercentile(np.abs(fmid), 99) or 1.0
    cols = 3
    rows = int(np.ceil(len(SNAP_US) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5.2 * rows))
    axflat = np.atleast_1d(axes).ravel()
    im = None
    for k, ax in enumerate(axflat):
        if k >= len(SNAP_US):
            ax.axis("off"); continue
        fr, it = snapshot(vol, SNAP_US[k], mean_map)
        im = ax.imshow(fr, origin="lower", extent=ext, aspect="equal",
                       cmap=CMAP_FIELD, vmin=-vlim, vmax=vlim)
        ax.set(title=f"t = {it*DT*1e6:.0f} us", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(im, ax=axflat.tolist(), shrink=0.6, label="displacement (a.u.)")
    fig.suptitle(f"[{label}] LDV wavefield snapshots")
    _finish(fig, label, "02_snapshots")


def fig_rms(label, fp, vol):
    ext = _extent(vol)
    rms = fp["rms"].copy()
    rms[fp["sat_mask"]] = np.nan
    fig, ax = plt.subplots(1, 2, figsize=(11, 6))
    im0 = ax[0].imshow(rms, origin="lower", extent=ext, aspect="equal", cmap="inferno")
    ax[0].set(title=f"[{label}] RMS energy map", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(im0, ax=ax[0], shrink=0.7, label="RMS amplitude")
    ax[1].imshow(fp["sat_mask"], origin="lower", extent=ext, aspect="equal", cmap="gray_r")
    ax[1].set(title="saturated / signal-loss pixels", xlabel="x (mm)", ylabel="y (mm)")
    fig.tight_layout()
    _finish(fig, label, "03_rms_map")


def fig_bscan(label, vol):
    img, iy = bscan_line(vol, BSCAN_Y_MM)
    T, X = img.shape
    vlim = np.nanpercentile(np.abs(img), 99) or 1.0
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img, origin="lower", aspect="auto", cmap=CMAP_FIELD,
                   vmin=-vlim, vmax=vlim,
                   extent=[0, (X - 1) * STEP_MM, 0, T * DT * 1e6])
    ax.set(title=f"[{label}] B-scan (t-x) at y={iy*STEP_MM:.0f} mm",
           xlabel="x (mm)", ylabel="time (us)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="displacement (a.u.)")
    fig.tight_layout()
    _finish(fig, label, "04_bscan")


def fig_fk(label, vol):
    f, kx, P = fk_transform(vol, FK_Y_BAND_MM)
    pos = f >= 0
    fpos = f[pos] / 1e3
    Pp = P[pos]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.log1p(Pp), origin="lower", aspect="auto", cmap="magma",
              extent=[kx.min(), kx.max(), fpos.min(), fpos.max()])
    # analytical A0 / S0 overlay (validation)
    fov = np.linspace(20e3, min(400e3, f.max()), 60)
    for mode, col in [("A", "cyan"), ("S", "lime")]:
        kk = lamb_curve(fov, D_NOMINAL_MM, mode)
        ax.plot(kk, fov / 1e3, col, lw=1.4, label=f"{mode}0 (2 mm)")
        ax.plot(-kk, fov / 1e3, col, lw=1.4)
    ax.set(title=f"[{label}] F-K spectrum + analytical dispersion",
           xlabel="wavenumber k (rad/m)", ylabel="frequency (kHz)",
           xlim=(kx.min(), kx.max()), ylim=(0, min(400, f.max() / 1e3)))
    ax.legend(loc="upper right")
    _finish(fig, label, "05_fk_spectrum")


def fig_uwm_thickness(label, fp, vol, freqs_hz):
    ext = _extent(vol)
    # local-wavenumber map at the primary frequency
    f0 = freqs_hz[0]
    kmap, amp = local_wavenumber_map(fp["W"][f0])
    lowamp = amp < np.nanpercentile(amp, 25)
    kshow = kmap.copy()
    kshow[fp["sat_mask"] | lowamp] = np.nan

    # thickness: invert A0 dispersion; for chirp average over the freq set
    thick_stack = []
    for f in freqs_hz:
        km, _ = local_wavenumber_map(fp["W"][f]) if f != f0 else (kmap, amp)
        thick_stack.append(thickness_from_k(km, f))
    thick = np.nanmean(thick_stack, axis=0)
    thick[fp["sat_mask"] | lowamp] = np.nan
    thick = np.clip(thick, 0.4, 2.6)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    im0 = ax[0].imshow(kshow, origin="lower", extent=ext, aspect="equal", cmap="viridis")
    ax[0].set(title=f"[{label}] local wavenumber @ {f0/1e3:.0f} kHz",
              xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(im0, ax=ax[0], shrink=0.7, label="|k| (rad/m)  [higher = thinner]")
    im1 = ax[1].imshow(thick, origin="lower", extent=ext, aspect="equal",
                       cmap="inferno_r", vmin=0.6, vmax=2.1)
    ax[1].set(title="thickness map (A0 Lamb inversion)", xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(im1, ax=ax[1], shrink=0.7, label="thickness (mm)")
    fig.tight_layout()
    _finish(fig, label, "06_wavenumber_thickness")

    healthy = np.nanmedian(thick)
    thin = np.nanpercentile(thick, 2)
    return {"k_at_f0_median": np.nanmedian(kshow), "thick_median": healthy, "thick_min": thin}


def make_gif(label, vol, mean_map, out=None, t0_us=5, t1_us=120, n=60, fps=15):
    from matplotlib.animation import FuncAnimation, PillowWriter
    os.makedirs(OUTDIR, exist_ok=True)
    out = out or os.path.join(OUTDIR, f"{label}_wavefield.gif")
    ext = _extent(vol)
    times = np.linspace(t0_us, t1_us, n)
    f0, _ = snapshot(vol, times[n // 2], mean_map)
    vlim = np.nanpercentile(np.abs(f0), 99) or 1.0
    fig, ax = plt.subplots(figsize=(4.5, 6.5))
    im = ax.imshow(f0, origin="lower", extent=ext, aspect="equal",
                   cmap=CMAP_FIELD, vmin=-vlim, vmax=vlim)
    ttl = ax.set_title("")
    ax.set(xlabel="x (mm)", ylabel="y (mm)")

    def upd(k):
        fr, it = snapshot(vol, times[k], mean_map)
        im.set_data(fr); ttl.set_text(f"[{label}] t = {it*DT*1e6:.0f} us")
        return [im, ttl]

    FuncAnimation(fig, upd, frames=n, blit=False).save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"   saved {out}")


# ============================== drivers ================================
def run_all(label, path):
    print("=" * 64)
    if not os.path.exists(path):
        print(f"[{label}] FILE NOT FOUND: {path}  (edit FILES in CONFIG)"); return None
    vol = load_scan(path)
    print_summary(label, vol)

    freqs_hz = [f * 1e3 for f in UWM_FREQS_KHZ[label]]
    print(f"[{label}] streaming pass (mean, rms, sat, monochromatic fields)...")
    fp = field_pass(vol, freqs_hz)

    peak = fig_waveform_spectrum(label, vol)
    fig_snapshots(label, vol, fp["mean"])
    fig_rms(label, fp, vol)
    fig_bscan(label, vol)
    fig_fk(label, vol)
    stats = fig_uwm_thickness(label, fp, vol, freqs_hz)
    if MAKE_GIF:
        make_gif(label, vol, fp["mean"])

    cp = 2 * np.pi * freqs_hz[0] / lamb_k(freqs_hz[0], D_NOMINAL_MM)
    cg = group_velocity(freqs_hz[0], D_NOMINAL_MM)
    out = {"label": label, "spectrum_peak_kHz": peak,
           "A0_cp_mps": cp, "A0_cg_mps": cg, **stats}
    print(f"[{label}] spectrum peak ~{peak:.0f} kHz | A0 @{freqs_hz[0]/1e3:.0f}kHz: "
          f"cp={cp:.0f} m/s cg={cg:.0f} m/s | thickness median={stats['thick_median']:.2f} mm "
          f"min={stats['thick_min']:.2f} mm")
    return out


def compare(results, vols):
    results = [r for r in results if r]
    if len(results) < 2:
        return
    # overlaid spectra
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for label, vol in vols.items():
        w, _ = waveform_at(vol, *PROBE_MM)
        f, W = amplitude_spectrum(w)
        ax[0].plot(f / 1e3, W / W.max(), lw=1.0, label=label)
    ax[0].set(title="Hann vs Chirp - normalised spectrum (same probe point)",
              xlabel="frequency (kHz)", ylabel="|FFT| (norm)", xlim=(0, 400))
    ax[0].legend(); ax[0].grid(alpha=0.3)
    # analytical dispersion (shared physics)
    fov = np.linspace(20e3, 300e3, 80)
    for mode, col in [("A", "tab:blue"), ("S", "tab:orange")]:
        cp = 2 * np.pi * fov / lamb_curve(fov, D_NOMINAL_MM, mode)
        ax[1].plot(fov / 1e3, cp, col, label=f"{mode}0 phase vel (2 mm)")
    ax[1].set(title="A0/S0 phase velocity (SUS304, 2 mm)",
              xlabel="frequency (kHz)", ylabel="phase velocity (m/s)", ylim=(0, 6000))
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    _finish(fig, "COMPARE", "spectra_dispersion")

    print("=" * 64)
    print(f"{'file':6s} {'peak kHz':>9s} {'A0 cp':>8s} {'A0 cg':>8s} "
          f"{'thk_med':>8s} {'thk_min':>8s}")
    for r in results:
        print(f"{r['label']:6s} {r['spectrum_peak_kHz']:9.0f} {r['A0_cp_mps']:8.0f} "
              f"{r['A0_cg_mps']:8.0f} {r['thick_median']:8.2f} {r['thick_min']:8.2f}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    results, vols = [], {}
    for label, path in FILES.items():
        r = run_all(label, path)
        results.append(r)
        if r is not None:
            vols[label] = load_scan(path)
    compare(results, vols)
    print("=" * 64)
    print(f"done. figures in ./{OUTDIR}/")


if __name__ == "__main__":
    main()
