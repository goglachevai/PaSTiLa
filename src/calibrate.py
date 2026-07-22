from __future__ import annotations

import csv
import tempfile
import warnings
from pathlib import Path

import numpy as np

from model import RuntimeModel
from paths import DEFAULT_MODEL_DIR, DEFAULT_PDSS_EXE, DEFAULT_PSF_EXE
from runner import run_task

DEFAULT_TS_LENGTHS = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
DEFAULT_SEGMENT_LENGTHS = [200, 300, 500, 750, 1_000, 1_500]
DEFAULT_NUM_SNIPPETS = 3


def make_dummy_series(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=length)).astype(np.float32)


def calibrate(
    algo: str,
    exe: Path | None = None,
    ts_lengths: list[int] | None = None,
    segment_lengths: list[int] | None = None,
    num_snippets: int = DEFAULT_NUM_SNIPPETS,
    gpu_id: int | None = 0,
    out_model: Path | None = None,
    out_csv: Path | None = None,
) -> RuntimeModel:

    exe = Path(exe) if exe else (DEFAULT_PDSS_EXE if algo == "pdss" else DEFAULT_PSF_EXE)
    if not exe.exists():
        raise FileNotFoundError(f"{algo} binary not found at {exe}")

    ts_lengths = ts_lengths or DEFAULT_TS_LENGTHS
    segment_lengths = segment_lengths or DEFAULT_SEGMENT_LENGTHS

    rows: list[tuple[int, int, int | None, float]] = []
    with tempfile.TemporaryDirectory(prefix="pastila_calib_") as tmp:
        tmp_path = Path(tmp)
        for ts_len in ts_lengths:
            series = make_dummy_series(ts_len, seed=ts_len)
            series_path = tmp_path / f"series_{ts_len}.txt"
            np.savetxt(series_path, series, fmt="%.6f")

            for seg_len in segment_lengths:
                if seg_len > ts_len:
                    continue

                task_out = tmp_path / f"out_{ts_len}_{seg_len}"
                try:
                    result = run_task(
                        exe, algo, series_path, task_out, ts_len, seg_len,
                        num_snippets, candidate_step=None, gpu_id=gpu_id,
                    )
                except RuntimeError as e:
                    warnings.warn(
                        f"skipping calibration point ts_length={ts_len} "
                        f"segment_length={seg_len}: {e}"
                    )
                    continue
                rows.append((ts_len, seg_len, None, result.runtime_ms))

    if not rows:
        raise RuntimeError(f"every calibration point failed for {algo}; no data to fit a model on")

    X = RuntimeModel.features(algo, [r[0] for r in rows], [r[1] for r in rows])
    y = np.array([r[3] for r in rows])

    model = RuntimeModel(algo).fit(X, y)

    out_model = Path(out_model) if out_model else DEFAULT_MODEL_DIR / f"{algo}_runtime_model.joblib"
    model.save(out_model)

    if out_csv:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ts_length", "segment_length", "candidate_step", "runtime_ms"])
            writer.writerows(rows)

    return model
