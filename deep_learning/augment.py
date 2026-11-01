from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .config import AugmentConfig
from .data import InputLayout

GYRO_JITTER_DPS = 1.5
BARO_JITTER_M = 0.02

def _rotation_matrices(rng: np.random.Generator, n: int, max_deg: float) -> np.ndarray:
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    theta = np.deg2rad(rng.uniform(-max_deg, max_deg, size=n))
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s, C = np.cos(theta), np.sin(theta), 1.0 - np.cos(theta)
    return np.stack([
        np.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], axis=-1),
        np.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], axis=-1),
        np.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], axis=-1),
    ], axis=1)

def _time_warp(window: np.ndarray, rng: np.random.Generator, sigma: float, knots: int) -> np.ndarray:
    T = window.shape[0]
    positions = np.linspace(0, T - 1, knots + 2)
    speed = np.clip(rng.normal(1.0, sigma, size=knots + 2), 0.5, 1.5)
    cumulative = np.cumsum(np.interp(np.arange(T), positions, speed))
    warped = (cumulative - cumulative[0]) / (cumulative[-1] - cumulative[0]) * (T - 1)
    grid = np.arange(T)
    return np.column_stack([np.interp(warped, grid, window[:, c]) for c in range(window.shape[1])])

def augment_batch(raw: np.ndarray, rng: np.random.Generator, cfg: AugmentConfig,
                  layout: InputLayout) -> np.ndarray:
    if not cfg.enabled:
        return raw
    x = raw.astype(np.float32, copy=True)
    B = x.shape[0]
    acc = slice(0, 3)
    gyr = slice(3, 6) if layout.use_gyro else None
    baro = x.shape[2] - 1 if layout.use_baro else None

    mask = rng.random(B) < cfg.p_rotate
    if mask.any():
        R = _rotation_matrices(rng, int(mask.sum()), cfg.rotate_max_deg).astype(np.float32)
        x[mask, :, acc] = np.einsum("btj,bkj->btk", x[mask, :, acc], R)
        if gyr is not None:
            x[mask, :, gyr] = np.einsum("btj,bkj->btk", x[mask, :, gyr], R)

    mask = rng.random(B) < cfg.p_scale
    if mask.any():
        s = rng.uniform(cfg.scale_low, cfg.scale_high, size=int(mask.sum())).astype(np.float32)
        a = x[mask, :, acc]
        g = uniform_filter1d(a, size=cfg.gravity_window, axis=1, mode="nearest")
        x[mask, :, acc] = g + s[:, None, None] * (a - g)
        if gyr is not None:
            x[mask, :, gyr] *= s[:, None, None]

    for b in np.flatnonzero(rng.random(B) < cfg.p_warp):
        x[b] = _time_warp(x[b], rng, cfg.warp_sigma, cfg.warp_knots)

    mask = rng.random(B) < cfg.p_jitter
    if mask.any():
        n = int(mask.sum())
        x[mask, :, acc] += rng.normal(0.0, cfg.jitter_sigma_g, size=(n, x.shape[1], 3)).astype(np.float32)
        if gyr is not None:
            x[mask, :, gyr] += rng.normal(0.0, GYRO_JITTER_DPS, size=(n, x.shape[1], 3)).astype(np.float32)
        if baro is not None:
            x[mask, :, baro] += rng.normal(0.0, BARO_JITTER_M, size=(n, x.shape[1])).astype(np.float32)
    return x
