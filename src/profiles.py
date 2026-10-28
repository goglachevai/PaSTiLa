from __future__ import annotations

from pathlib import Path

import numpy as np

def load_profiles(path: Path) -> np.ndarray:
    return np.loadtxt(path, ndmin=2)


def area_norm(raw: np.ndarray) -> float:
    raw = np.atleast_2d(raw)
    a_max = np.nanmax(raw) if np.isfinite(raw).any() else 0.0
    if a_max <= 0:
        return 0.0
    total = 0.0
    for p in range(len(raw)):
        for q in range(p + 1, len(raw)):
            total += float(np.nansum(np.abs(raw[p] - raw[q])))
    return total / float(a_max)


def score_segment_length(profiles_path: Path, num_patterns: int | None = None) -> float:
    raw = load_profiles(profiles_path)
    if num_patterns is not None:
        raw = raw[:num_patterns]
    return area_norm(raw)


def pick_best_length(scores: dict[int, float]) -> int:
    if not scores:
        raise ValueError("no scores to pick from")
    return max(sorted(scores), key=scores.get)
