from __future__ import annotations

from pathlib import Path

import numpy as np


def load_profiles(path: Path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)


def normalize_and_resample(profile: np.ndarray, lo: float, hi: float, num_points: int = 200) -> np.ndarray:
    y = np.zeros_like(profile) if hi - lo < 1e-12 else (profile - lo) / (hi - lo)
    x_src = np.linspace(0.0, 1.0, num=len(profile))
    x_dst = np.linspace(0.0, 1.0, num=num_points)
    return np.interp(x_dst, x_src, y)


def _trapz_uniform(y: np.ndarray, dx: float) -> float:
    if len(y) < 2:
        return 0.0
    return float(dx * (np.sum(y) - 0.5 * (y[0] + y[-1])))


def pairwise_area(profiles: np.ndarray) -> float:
    n = len(profiles)
    if n < 2:
        return 0.0
    dx = 1.0 / (profiles.shape[1] - 1) if profiles.shape[1] > 1 else 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += _trapz_uniform(np.abs(profiles[i] - profiles[j]), dx)
    return total


def score_segment_length(profiles_path: Path, num_points: int = 200) -> float:
    raw = load_profiles(profiles_path)
    lo, hi = float(raw.min()), float(raw.max())
    normalized = np.array([normalize_and_resample(row, lo, hi, num_points) for row in raw])
    return pairwise_area(normalized)


def pick_best_length(scores: dict[int, float]) -> int:
    if not scores:
        raise ValueError("no scores to pick from")
    return max(sorted(scores), key=scores.get)
