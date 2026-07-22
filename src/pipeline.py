from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess

from model import RuntimeModel
from paths import DEFAULT_MODEL_DIR, DEFAULT_PDSS_EXE, DEFAULT_PSF_EXE
from profiles import pick_best_length, score_segment_length
from runner import run_task
from scheduler import ldm_partition


@dataclass
class PastilaConfig:
    algo: str  # "pdss" | "psf"
    input_file: Path
    outdir: Path
    ts_length: int
    segment_lengths: list[int]
    num_snippets: int
    candidate_step: int | None = None  # pdss only
    exe: Path | None = None
    model_path: Path | None = None
    gpu_ids: list[int] | None = None


@dataclass
class LengthResult:
    segment_length: int
    outdir: Path
    predicted_runtime_ms: float | None
    actual_runtime_ms: float
    gpu_id: int | None
    area_score: float = 0.0


def detect_gpus() -> list[int]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return [0]

    try:
        out = subprocess.run(
            [nvidia_smi, "-L"], capture_output=True, text=True, check=True, timeout=15
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return [0]

    ids = [int(m.group(1)) for m in re.finditer(r"^GPU (\d+):", out, re.MULTILINE)]
    return ids or [0]


def run_pastila(cfg: PastilaConfig) -> dict:
    if cfg.algo not in ("pdss", "psf"):
        raise ValueError(f"unknown algorithm {cfg.algo!r}, expected 'pdss' or 'psf'")
    if cfg.algo == "psf" and cfg.candidate_step is not None:
        raise ValueError("candidate_step only applies to pdss")
    if any(s > cfg.ts_length for s in cfg.segment_lengths):
        raise ValueError("segment_length must not exceed time_series_length")

    exe = Path(cfg.exe) if cfg.exe else (DEFAULT_PDSS_EXE if cfg.algo == "pdss" else DEFAULT_PSF_EXE)
    if not exe.exists():
        raise FileNotFoundError(f"{cfg.algo} binary not found at {exe}")

    gpu_ids = cfg.gpu_ids if cfg.gpu_ids else detect_gpus()

    if len(gpu_ids) <= 1:
        predicted: dict[int, float | None] = {seg_len: None for seg_len in cfg.segment_lengths}
        bins = [list(range(len(cfg.segment_lengths)))]
    else:
        model_path = Path(cfg.model_path) if cfg.model_path else DEFAULT_MODEL_DIR / f"{cfg.algo}_runtime_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"no runtime model at {model_path}; run `pastila calibrate --algo {cfg.algo}` first "
                f"(only required when running across more than one GPU)"
            )
        model = RuntimeModel.load(model_path)
        predicted = {
            seg_len: model.predict_one(cfg.ts_length, seg_len, cfg.candidate_step)
            for seg_len in cfg.segment_lengths
        }
        bins = ldm_partition([predicted[s] for s in cfg.segment_lengths], len(gpu_ids))

    series_name = Path(cfg.input_file).stem
    algo_outdir = Path(cfg.outdir) / cfg.algo / series_name
    algo_outdir.mkdir(parents=True, exist_ok=True)

    results: dict[int, LengthResult] = {}

    def run_bin(gpu_idx: int, seg_indices: list[int]) -> None:
        gpu_id = gpu_ids[gpu_idx]
        for idx in seg_indices:
            seg_len = cfg.segment_lengths[idx]
            task_outdir = algo_outdir / f"seglen_{seg_len}"
            result = run_task(
                exe, cfg.algo, cfg.input_file, task_outdir, cfg.ts_length, seg_len,
                cfg.num_snippets, candidate_step=cfg.candidate_step, gpu_id=gpu_id,
            )
            results[seg_len] = LengthResult(
                seg_len, task_outdir, predicted[seg_len], result.runtime_ms, gpu_id
            )

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(run_bin, i, indices) for i, indices in enumerate(bins) if indices]
        for f in futures:
            f.result()

    for seg_len, res in results.items():
        res.area_score = score_segment_length(res.outdir / "profiles.txt")

    best_len = pick_best_length({s: r.area_score for s, r in results.items()})

    for stale in algo_outdir.glob("best_seglen_*"):
        shutil.rmtree(stale)
    best_dir = algo_outdir / f"best_seglen_{best_len}"
    shutil.copytree(results[best_len].outdir, best_dir)

    summary = {
        "algo": cfg.algo,
        "outdir": str(algo_outdir),
        "ts_length": cfg.ts_length,
        "num_snippets": cfg.num_snippets,
        "candidate_step": cfg.candidate_step,
        "gpu_ids": gpu_ids,
        "best_segment_length": best_len,
        "lengths": {
            str(s): {
                "predicted_runtime_ms": r.predicted_runtime_ms,
                "actual_runtime_ms": r.actual_runtime_ms,
                "gpu_id": r.gpu_id,
                "area_score": r.area_score,
                "outdir": str(r.outdir),
            }
            for s, r in sorted(results.items())
        },
    }
    with (algo_outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return summary
