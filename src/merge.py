from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PatternSet:
    segment_length: int
    starts: np.ndarray
    snippets: np.ndarray
    profiles: np.ndarray
    fracs: np.ndarray
    labels: np.ndarray
    runtime_ms: float = float("nan")
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, outdir: Path, segment_length: int, runtime_ms: float = float("nan")) -> "PatternSet":
        outdir = Path(outdir)
        return cls(
            segment_length=segment_length,
            starts=np.loadtxt(outdir / "indicies.txt", dtype=np.int64, ndmin=1),
            snippets=np.loadtxt(outdir / "snippets.txt", ndmin=2),
            profiles=np.loadtxt(outdir / "profiles.txt", ndmin=2),
            fracs=np.loadtxt(outdir / "fracs.txt", ndmin=1),
            labels=np.loadtxt(outdir / "labels.txt", dtype=np.int64, ndmin=1),
            runtime_ms=runtime_ms,
        )

    def write(self, outdir: Path) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        _write_row(outdir / "indicies.txt", self.starts, "%d")
        _write_row(outdir / "fracs.txt", self.fracs, "%g")
        _write_row(outdir / "labels.txt", self.labels, "%d")
        _write_rows(outdir / "snippets.txt", self.snippets)
        _write_rows(outdir / "profiles.txt", self.profiles)


def _write_row(path: Path, values, fmt: str) -> None:
    path.write_text(" ".join(fmt % v for v in np.asarray(values).ravel()) + " \n")


def _write_rows(path: Path, rows: np.ndarray) -> None:
    with path.open("w") as f:
        for row in np.atleast_2d(rows):
            f.write(" ".join("%g" % v for v in row) + " \n")


def merge_patterns(ps: PatternSet, k: int) -> tuple[PatternSet, list[list[int]]]:
    p_count = len(ps.starts)
    if k >= p_count:
        return ps, [[p] for p in range(p_count)]
    raw = ps.profiles.astype(float)
    finite = np.isfinite(raw)
    fill = raw[finite].max() if finite.any() else 1.0
    prof = {g: np.where(finite[g], raw[g], fill) for g in range(p_count)}
    frac = {g: float(ps.fracs[g]) for g in range(p_count)}
    groups = {g: [g] for g in range(p_count)}

    dist = np.full((p_count, p_count), np.inf)
    for g in range(p_count):
        for h in range(g + 1, p_count):
            dist[g, h] = float(np.mean(np.abs(prof[g] - prof[h])))

    def pair(a: int, b: int) -> float:
        return dist[min(a, b), max(a, b)]

    while len(groups) > k:
        keys = sorted(groups)
        best, best_pair = np.inf, None
        for i, g in enumerate(keys):
            for h in keys[i + 1:]:
                if best_pair is None or pair(g, h) < best:
                    best, best_pair = pair(g, h), (g, h)
        g, h = best_pair
        if frac[h] > frac[g]:
            g, h = h, g
        groups[g] += groups.pop(h)
        prof[g] = np.minimum(prof[g], prof.pop(h))
        frac[g] += frac.pop(h)
        for other in groups:
            if other != g:
                dist[min(g, other), max(g, other)] = float(np.mean(np.abs(prof[g] - prof[other])))

    keep = sorted(groups)
    new_label = np.empty(p_count, dtype=np.int64)
    for new, g in enumerate(keep):
        for member in groups[g]:
            new_label[member] = new
    merged = PatternSet(
        segment_length=ps.segment_length,
        starts=ps.starts[keep],
        snippets=ps.snippets[keep],
        profiles=np.array([np.fmin.reduce(raw[groups[g]], axis=0) for g in keep]),
        fracs=np.array([frac[g] for g in keep]),
        labels=new_label[ps.labels],
        runtime_ms=ps.runtime_ms,
        extra=dict(ps.extra),
    )
    return merged, [sorted(groups[g]) for g in keep]
