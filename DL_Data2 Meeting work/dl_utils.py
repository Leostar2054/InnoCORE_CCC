from pathlib import Path
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def choose_device(mode="auto"):
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def find_feature_file(pzt_dir, candidates):
    for name in candidates:
        path = Path(pzt_dir) / name
        if path.exists():
            return path
    raise FileNotFoundError("No feature file found")

def to_sample_first(arr, n_samples):
    x = np.asarray(arr)
    axes = [i for i, s in enumerate(x.shape) if s == n_samples]
    if not axes:
        raise ValueError(f"No axis of length {n_samples} in shape {x.shape}")
    if axes[0] != 0:
        x = np.moveaxis(x, axes[0], 0)
    return x

def flatten_samples(arr):
    x = np.asarray(arr, dtype=np.float32)
    return x.reshape(x.shape[0], -1)

def normalize_pair_idx(pair_idx):
    p = np.asarray(pair_idx).astype(int)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError(f"pair_idx must be (N,2), got {p.shape}")
    if p.min() == 0:
        p = p + 1
    return p

def reciprocal_group_ids(pair_idx):
    p = normalize_pair_idx(pair_idx)
    keys = [tuple(sorted((int(a), int(b)))) for a, b in p]
    unique = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([unique[k] for k in keys], dtype=int)

def rapid_sensitivity(xx, yy, src_yx, rx_yx, beta=1.08):
    sy, sx = src_yx
    ry, rx = rx_yx
    direct = np.hypot(sx-rx, sy-ry)
    ratio = (np.hypot(xx-sx, yy-sy) + np.hypot(xx-rx, yy-ry)) / direct
    sens = (beta-ratio)/(beta-1.0)
    sens[(ratio < 1.0) | (ratio >= beta)] = 0.0
    return np.clip(sens, 0.0, 1.0)

def minmax(v):
    v = np.asarray(v, dtype=float)
    mn, mx = np.nanmin(v), np.nanmax(v)
    return np.zeros_like(v) if mx == mn else (v-mn)/(mx-mn)
