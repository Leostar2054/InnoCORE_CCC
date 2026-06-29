"""
paper5_utils.py
===============
Shared functions for the Paper 5 cross-modal consistency study
(full-field LDV reference  vs  sparse 6-PZT network), SUS304 plate.

Two data sources:
  Folder 1  "260609_pzt_network"        -> PZT pitch-catch tensors (6x6x1000), 2.2 MS/s
  Folder 2  "2606010_Innocore_STS304_6PZT" -> 12 LDV scans (2048x1201x1201), 10 MS/s

Coordinate convention: (y, x) in mm, plate centre (150, 150), origin lower-left.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter, zoom

# ============================ constants ============================
PLATE_MM = 300.0
CENTER = (150.0, 150.0)

# 6-PZT hexagonal layout, (y, x) mm  (README)
PZT = {
    1: (150.00, 275.00), 2: (258.25, 212.50), 3: (258.25, 87.50),
    4: (150.00, 25.00),  5: (41.75, 87.50),   6: (41.75, 212.50),
}

# ground truth (mm)
WALL_THINNING_GT = (148.39, 149.88)   # LDV multi-source consensus (README)
MAGNET_GT        = (215.00, 150.00)   # added mass

# PZT network timing
FS_PZT = 2.2e6
DT_PZT = 1.0 / FS_PZT
CROSSTALK_US = 60.0                    # skip 0-60 us electrical crosstalk

# LDV timing / geometry
FS_LDV = 10e6
DT_LDV = 1.0 / FS_LDV
STEP_MM = 0.25
SAT = 32767                           # int16 saturation (LDV signal loss)

# SUS304 material + A0 reference
CL, CT = 5790.0, 3100.0
D_NOMINAL_MM = 2.0
VG_A0_MM_US = 2.52                    # A0 group velocity @100 kHz (README opposite path)
LAMBDA_A0_MM = 24.0                   # ~A0 wavelength @100 kHz on 2 mm steel


# ============================ PZT branch ============================
def pzt_time_us(n=1000):
    return np.arange(n) * DT_PZT * 1e6


def _gate_index(gate_us):
    return int(round(gate_us * 1e-6 * FS_PZT))


def prep_path(sig, gate_us=CROSSTALK_US):
    """DC-remove a single pitch-catch trace and zero the crosstalk window."""
    s = np.asarray(sig, dtype=float).copy()
    s = s - np.nanmean(s)
    s[:_gate_index(gate_us)] = 0.0
    return s


def damage_index_matrix(baseline_tensor, damage_tensor, gate_us=CROSSTALK_US):
    """
    Per-path damage index = 1 - Pearson correlation between baseline and damage
    traces over the post-crosstalk window. ~0 = unchanged, larger = more change.
    Returns a 6x6 matrix (diagonal / self-paths = NaN).
    """
    B = np.asarray(baseline_tensor, float)
    D = np.asarray(damage_tensor, float)
    n = B.shape[0]
    n0 = _gate_index(gate_us)
    DI = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x = prep_path(B[i, j], gate_us)[n0:]
            y = prep_path(D[i, j], gate_us)[n0:]
            if x.size == 0 or x.std() < 1e-9 or y.std() < 1e-9:
                continue
            DI[i, j] = 1.0 - np.corrcoef(x, y)[0, 1]
    return DI


def rapid_image(DI, sigma_mm=15.0, step_mm=2.0):
    """
    RAPID damage-probability map (Gaussian elliptical form, matches the deck):
        P(p) = sum_paths DI * exp( -(R_ap + R_sp - R_as)^2 / (2 sigma^2) )
    Returns (gy, gx, P).
    """
    gy = np.arange(0, PLATE_MM + step_mm, step_mm)
    gx = np.arange(0, PLATE_MM + step_mm, step_mm)
    YY, XX = np.meshgrid(gy, gx, indexing="ij")
    P = np.zeros_like(YY, float)
    n = DI.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(DI[i, j]):
                continue
            ay, ax = PZT[i + 1]
            sy, sx = PZT[j + 1]
            R_as = np.hypot(ay - sy, ax - sx)
            R_ap = np.hypot(YY - ay, XX - ax)
            R_sp = np.hypot(YY - sy, XX - sx)
            excess = R_ap + R_sp - R_as
            P += max(DI[i, j], 0.0) * np.exp(-(excess ** 2) / (2 * sigma_mm ** 2))
    return gy, gx, P


def rapid_image_beta(DI, beta=1.03, step_mm=2.0):
    """Classical elliptical RAPID (linear-decay distribution, foci = the two PZTs).
    Often sharper than the Gaussian form for a discrete scatterer. Returns (gy,gx,P)."""
    gy = np.arange(0, PLATE_MM + step_mm, step_mm)
    gx = np.arange(0, PLATE_MM + step_mm, step_mm)
    YY, XX = np.meshgrid(gy, gx, indexing="ij")
    P = np.zeros_like(YY, float)
    n = DI.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j or not np.isfinite(DI[i, j]):
                continue
            ay, ax = PZT[i + 1]; sy, sx = PZT[j + 1]
            R_as = np.hypot(ay - sy, ax - sx)
            ratio = (np.hypot(YY - ay, XX - ax) + np.hypot(YY - sy, XX - sx)) / R_as
            P += max(DI[i, j], 0.0) * np.clip((beta - ratio) / (beta - 1.0), 0, None)
    return gy, gx, P


def das_scatter_image(baseline_tensor, damage_tensor, vg_mm_us=VG_A0_MM_US,
                      gate_us=CROSSTALK_US, step_mm=2.0):
    """
    Scattered-field delay-and-sum: image the magnet by back-projecting the
    residual (damage - baseline) envelopes to their scatter-arrival time
    t = (R_ap + R_sp) / vg for each grid point. Good for a discrete scatterer.
    Returns (gy, gx, P).
    """
    from scipy.signal import hilbert
    B = np.asarray(baseline_tensor, float)
    D = np.asarray(damage_tensor, float)
    n = B.shape[0]
    gy = np.arange(0, PLATE_MM + step_mm, step_mm)
    gx = np.arange(0, PLATE_MM + step_mm, step_mm)
    YY, XX = np.meshgrid(gy, gx, indexing="ij")
    P = np.zeros_like(YY, float)
    tt = pzt_time_us(B.shape[2])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            res = prep_path(D[i, j], gate_us) - prep_path(B[i, j], gate_us)
            env = np.abs(hilbert(res))
            ay, ax = PZT[i + 1]; sy, sx = PZT[j + 1]
            tof = (np.hypot(YY - ay, XX - ax) + np.hypot(YY - sy, XX - sx)) / vg_mm_us
            idx = np.clip(np.round(tof / (DT_PZT * 1e6)).astype(int), 0, len(env) - 1)
            P += env[idx]
    return gy, gx, P


def localize_peak(gy, gx, P):
    iy, ix = np.unravel_index(np.argmax(P), P.shape)
    return (float(gy[iy]), float(gx[ix]))


def loc_error(est, gt):
    return float(np.hypot(est[0] - gt[0], est[1] - gt[1]))


def tof_envelope(sig, gate_us=CROSSTALK_US, frac=0.5):
    """First-arrival time (us) via envelope leading edge after the crosstalk gate."""
    from scipy.signal import hilbert
    s = prep_path(sig, gate_us)
    env = np.abs(hilbert(s))
    n0 = _gate_index(gate_us)
    seg = env[n0:]
    if seg.max() < 1e-9:
        return np.nan
    thr = frac * seg.max()
    idx = np.argmax(seg > thr)
    return (n0 + idx) * DT_PZT * 1e6


def cross_corr_paths(ldv_virtual_tensor, pzt_tensor, gate_us=CROSSTALK_US):
    """Signal-level cross-modal agreement: per-path Pearson r between the
    LDV-virtual-PZT tensor and the actual PZT tensor. Returns a 6x6 matrix."""
    A = np.asarray(ldv_virtual_tensor, float)
    Bt = np.asarray(pzt_tensor, float)
    n = Bt.shape[0]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = A[i, j]
            b = Bt[i, j]
            m = min(len(a), len(b))
            a = np.asarray(a[:m], float) - np.nanmean(a[:m])
            b = np.asarray(b[:m], float) - np.nanmean(b[:m])
            if np.nanstd(a) < 1e-9 or np.nanstd(b) < 1e-9:
                continue
            out[i, j] = np.corrcoef(np.nan_to_num(a), np.nan_to_num(b))[0, 1]
    return out


# ============================ LDV branch ============================
def _pq(w, k):
    p = np.sqrt((w / CL) ** 2 - k ** 2 + 0j)
    q = np.sqrt((w / CT) ** 2 - k ** 2 + 0j)
    return p, q


def lamb_det(k, w, h, mode):
    p, q = _pq(w, k)
    sp, cp = np.sin(p * h), np.cos(p * h)
    sq, cq = np.sin(q * h), np.cos(q * h)
    if mode == "A":
        return ((k ** 2 - q ** 2) ** 2) * sp * cq + 4 * k ** 2 * p * q * cp * sq
    return ((k ** 2 - q ** 2) ** 2) * cp * sq + 4 * k ** 2 * p * q * sp * cq


def lamb_k(f, d_mm, mode="A", kmax=2500.0):
    """Fundamental-mode wavenumber (rad/m) at frequency f for thickness d_mm."""
    w = 2 * np.pi * f
    h = d_mm * 1e-3 / 2.0
    kg = np.linspace(1.0, kmax, 6000)
    val = np.abs(lamb_det(kg, w, h, mode))
    loc = np.where((val[1:-1] < val[:-2]) & (val[1:-1] < val[2:]))[0] + 1
    if len(loc) == 0:
        return np.nan
    k0 = kg[loc[np.argmax(kg[loc])]]
    r = minimize_scalar(lambda k: np.abs(lamb_det(k, w, h, mode)),
                        bounds=(max(k0 - 3, 1.0), k0 + 3), method="bounded")
    return r.x


def thickness_from_k(kmap_rad_m, f0, d_grid_mm=None):
    if d_grid_mm is None:
        d_grid_mm = np.linspace(0.4, 2.6, 130)
    ks = np.array([lamb_k(f0, dd) for dd in d_grid_mm])
    order = np.argsort(ks)
    return np.interp(kmap_rad_m, ks[order], d_grid_mm[order])


def ldv_field_pass(vol, f0_hz, gate_us=(0.0, 120.0), chunk=128):
    """
    ONE streaming pass over a TIME-GATED window of the LDV cube ->
        rms        : DC-free energy map (label-free damage indicator)
        sat_mask   : saturated / signal-loss pixels
        W          : complex monochromatic field at f0 (single-frequency DFT)
    Time-gating to the first-arrival-dominated window is essential: UWM on the
    full reverberant record is biased (the pilot lesson).
    """
    T, Y, X = vol.shape
    n0 = max(0, int(round(gate_us[0] * 1e-6 * FS_LDV)))
    n1 = min(T, int(round(gate_us[1] * 1e-6 * FS_LDV)))
    s1 = np.zeros((Y, X)); s2 = np.zeros((Y, X)); sat = np.zeros((Y, X), np.int64)
    Wc = np.zeros((Y, X), np.complex128)
    tt = np.arange(T) * DT_LDV
    for t0 in range(n0, n1, chunk):
        t1 = min(t0 + chunk, n1)
        blk = vol[t0:t1].astype(np.float32)
        s1 += blk.sum(0); s2 += np.square(blk).sum(0)
        sat += (np.abs(blk) >= SAT).sum(0)
        e = np.exp(-2j * np.pi * f0_hz * tt[t0:t1]).astype(np.complex64)
        Wc += np.einsum("t,tyx->yx", e, blk)
    ng = max(n1 - n0, 1)
    mean = s1 / ng
    rms = np.sqrt(np.maximum(s2 / ng - mean ** 2, 0.0))
    return {"rms": rms, "sat_mask": sat > 0, "W": Wc}


def local_wavenumber_map(Wc, win=48, stride=6):
    """Local dominant wavenumber (rad/m) via sliding-window spatial FFT of the
    monochromatic field. Returns (k_map, amplitude_map) on the full grid."""
    step_m = STEP_MM * 1e-3
    Y, X = Wc.shape
    kf = 2 * np.pi * np.fft.fftfreq(win, d=step_m)
    KX, KY = np.meshgrid(kf, kf)
    KMAG = np.sqrt(KX ** 2 + KY ** 2)
    kmin = 1.5 * (2 * np.pi / (win * step_m))
    valid = KMAG > kmin
    han = np.outer(np.hanning(win), np.hanning(win))
    ys = list(range(0, Y - win, stride))
    xs = list(range(0, X - win, stride))
    kmap = np.zeros((len(ys), len(xs)))
    amp = np.zeros_like(kmap)
    for a, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            patch = Wc[y0:y0 + win, x0:x0 + win] * han
            Pw = np.abs(np.fft.fft2(patch))
            Pw[~valid] = 0.0
            idx = np.argmax(Pw)
            kmap[a, c] = KMAG.ravel()[idx]
            amp[a, c] = Pw.ravel()[idx]
    kfull = zoom(kmap, (Y / kmap.shape[0], X / kmap.shape[1]), order=1)[:Y, :X]
    afull = zoom(amp, (Y / amp.shape[0], X / amp.shape[1]), order=1)[:Y, :X]
    return gaussian_filter(kfull, 1.0), afull


def anomaly_centroid(field, amp, sat_mask, roi_mm=((100, 200), (100, 200)),
                     pct=92, high=True):
    """
    Weighted centroid (y, x mm) of an anomaly within a central ROI.
    high=True selects high values (e.g. high wavenumber = thinner = wall-thinning).
    Works on a wavenumber map or an RMS map.
    """
    Y, X = field.shape
    yy = np.arange(Y) * STEP_MM
    xx = np.arange(X) * STEP_MM
    roi = np.zeros((Y, X), bool)
    iy = (yy >= roi_mm[0][0]) & (yy <= roi_mm[0][1])
    ix = (xx >= roi_mm[1][0]) & (xx <= roi_mm[1][1])
    roi[np.ix_(iy, ix)] = True
    amp_ok = amp >= np.nanpercentile(amp[roi], 40) if amp is not None else np.ones_like(roi, bool)
    valid = roi & (~sat_mask) & amp_ok
    f = np.where(valid, field, np.nan)
    if not np.isfinite(f).any():
        return (np.nan, np.nan)
    thr = np.nanpercentile(f, pct if high else 100 - pct)
    sel = valid & ((field >= thr) if high else (field <= thr))
    if sel.sum() == 0:
        return (np.nan, np.nan)
    Yg, Xg = np.meshgrid(yy, xx, indexing="ij")
    w = np.abs(field[sel] - np.nanmedian(f))
    cy = np.average(Yg[sel], weights=w)
    cx = np.average(Xg[sel], weights=w)
    return (float(cy), float(cx))


# ============================ cross-modal ============================
def agreement_report(c_ldv, c_pzt, framing="A"):
    """
    Cross-modal agreement summary.
    Framing A (current data): different targets -> report each modality's error
                              against its own ground truth.
    Framing B (needs magnet LDV scan): same target -> Delta is the headline.
    """
    e_ldv = loc_error(c_ldv, WALL_THINNING_GT)
    e_pzt = loc_error(c_pzt, MAGNET_GT)
    out = {
        "c_ldv_mm": c_ldv, "c_pzt_mm": c_pzt,
        "e_ldv_vs_wallthinning_mm": e_ldv,
        "e_pzt_vs_magnet_mm": e_pzt,
        "framing": framing,
    }
    if framing == "B":
        delta = loc_error(c_ldv, c_pzt)
        out["delta_same_target_mm"] = delta
        out["delta_over_lambda"] = delta / LAMBDA_A0_MM
    return out


# ============================ plotting helpers ============================
def plate_extent():
    return [0, PLATE_MM, 0, PLATE_MM]


def draw_pzt(ax):
    for k, (y, x) in PZT.items():
        ax.plot(x, y, "o", ms=9, mfc="none", mec="k", mew=1.5)
        ax.annotate(f"P{k}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)


def draw_truth(ax):
    wy, wx = WALL_THINNING_GT
    my, mx = MAGNET_GT
    ax.plot(wx, wy, "x", color="lime", ms=12, mew=2.5, label="wall-thinning (GT)")
    ax.plot(mx, my, "+", color="red", ms=14, mew=2.5, label="magnet (GT)")
