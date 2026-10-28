from __future__ import annotations

import json
import re
import shutil
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from merge import PatternSet, merge_patterns
from model import RuntimeModel
from paths import DEFAULT_MODEL_DIR, DEFAULT_PDSS_EXE, DEFAULT_PSF_EXE
from profiles import area_norm
from runner import run_task
from scheduler import ldm_partition


@dataclass
class PastilaConfig:
    algo: str
    input_file: Path
    outdir: Path
    ts_length: int
    segment_lengths: list[int]
    num_snippets: int
    candidate_step: int | None = None
    exe: Path | None = None
    model_path: Path | None = None
    gpu_ids: list[int] | None = None
    reserve: int = 1


@dataclass
class LengthResult:
    segment_length: int
    outdir: Path
    predicted_runtime_ms: float | None
    actual_runtime_ms: float
    gpu_id: int | None
    area_score: float = 0.0
    worker: str | None = None


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


def validate(cfg: PastilaConfig) -> Path:
    if cfg.algo not in ("pdss", "psf"):
        raise ValueError(f"unknown algorithm {cfg.algo!r}, expected 'pdss' or 'psf'")
    if cfg.algo == "psf" and cfg.candidate_step is not None:
        raise ValueError("candidate_step only applies to pdss")
    if any(s > cfg.ts_length for s in cfg.segment_lengths):
        raise ValueError("segment_length must not exceed time_series_length")
    if cfg.reserve < 1:
        raise ValueError("reserve (lambda) must be >= 1")
    searched = cfg.reserve * cfg.num_snippets
    max_len = max(cfg.segment_lengths)
    if cfg.algo == "psf" and cfg.ts_length // max_len < searched:
        raise ValueError(
            f"PSF has only {cfg.ts_length // max_len} candidate segments at length {max_len}, fewer than "
            f"lambda*K = {searched} patterns (need lambda <= n / (K * maxL))"
        )
    if cfg.algo == "pdss":
        step = cfg.candidate_step or max_len
        if (cfg.ts_length - max_len) // step + 1 < searched:
            raise ValueError(f"PDSS has fewer candidates than lambda*K = {searched} patterns at length {max_len}")
    exe = Path(cfg.exe) if cfg.exe else (DEFAULT_PDSS_EXE if cfg.algo == "pdss" else DEFAULT_PSF_EXE)
    if not exe.exists():
        raise FileNotFoundError(f"{cfg.algo} binary not found at {exe}")
    return exe


def predict_and_partition(cfg: PastilaConfig, num_bins: int, require_model: bool = True):
    if num_bins <= 1:
        return {s: None for s in cfg.segment_lengths}, [list(range(len(cfg.segment_lengths)))]
    model_path = Path(cfg.model_path) if cfg.model_path else DEFAULT_MODEL_DIR / f"{cfg.algo}_runtime_model.joblib"
    if model_path.exists():
        model = RuntimeModel.load(model_path)
        predicted = {s: model.predict_one(cfg.ts_length, s, cfg.candidate_step) for s in cfg.segment_lengths}
    elif require_model:
        raise FileNotFoundError(
            f"no runtime model at {model_path}; run `pastila calibrate --algo {cfg.algo}` first "
            f"(only required when running on more than one GPU / worker)"
        )
    else:
        warnings.warn(f"no runtime model at {model_path}: tasks are balanced by number of lengths only "
                      f"(run `pastila calibrate --algo {cfg.algo}` for runtime-balanced tasks)")
        predicted = {s: None for s in cfg.segment_lengths}
    weights = [predicted[s] if predicted[s] is not None else 1.0 for s in cfg.segment_lengths]
    return predicted, ldm_partition(weights, num_bins)


def algo_outdir_for(cfg: PastilaConfig) -> Path:
    return Path(cfg.outdir) / cfg.algo / Path(cfg.input_file).stem


def run_length(cfg: PastilaConfig, exe: Path, seg_len: int, gpu_id: int | None) -> tuple[Path, PatternSet]:
    task_outdir = algo_outdir_for(cfg) / f"seglen_{seg_len}"
    result = run_task(
        exe, cfg.algo, cfg.input_file, task_outdir, cfg.ts_length, seg_len,
        cfg.reserve * cfg.num_snippets, candidate_step=cfg.candidate_step, gpu_id=gpu_id,
    )
    return task_outdir, PatternSet.load(task_outdir, seg_len, result.runtime_ms)


def score_length(cfg: PastilaConfig, ps: PatternSet) -> float:
    return area_norm(ps.profiles[: cfg.num_snippets])


def finalize(cfg: PastilaConfig, results: dict[int, LengthResult], best: PatternSet, extra: dict) -> dict:
    algo_outdir = algo_outdir_for(cfg)
    algo_outdir.mkdir(parents=True, exist_ok=True)
    best_len = best.segment_length
    for stale in algo_outdir.glob("best_seglen_*"):
        shutil.rmtree(stale)
    best_dir = algo_outdir / f"best_seglen_{best_len}"
    groups = [[p] for p in range(len(best.starts))]
    source = results[best_len].outdir
    if cfg.reserve > 1:
        merged, groups = merge_patterns(best, cfg.num_snippets)
        merged.write(best_dir)
    elif Path(source).exists():
        shutil.copytree(source, best_dir)
    else:
        best.write(best_dir)
    with (best_dir / "merge.json").open("w") as f:
        json.dump({"reserve": cfg.reserve, "patterns_searched": cfg.reserve * cfg.num_snippets,
                   "groups": groups, "starts": [int(s) for s in best.starts]}, f, indent=2)

    summary = {
        "algo": cfg.algo,
        "outdir": str(algo_outdir),
        "ts_length": cfg.ts_length,
        "num_snippets": cfg.num_snippets,
        "reserve": cfg.reserve,
        "candidate_step": cfg.candidate_step,
        **extra,
        "best_segment_length": best_len,
        "merge_groups": groups,
        "lengths": {
            str(s): {
                "predicted_runtime_ms": r.predicted_runtime_ms,
                "actual_runtime_ms": r.actual_runtime_ms,
                "gpu_id": r.gpu_id,
                "worker": r.worker,
                "area_score": r.area_score,
                "outdir": str(r.outdir),
            }
            for s, r in sorted(results.items())
        },
    }
    with (algo_outdir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_pastila(cfg: PastilaConfig) -> dict:
    exe = validate(cfg)
    gpu_ids = cfg.gpu_ids if cfg.gpu_ids else detect_gpus()
    predicted, bins = predict_and_partition(cfg, len(gpu_ids))

    results: dict[int, LengthResult] = {}
    best: dict[str, PatternSet | None] = {"ps": None}

    def run_bin(gpu_idx: int, seg_indices: list[int]) -> None:
        gpu_id = gpu_ids[gpu_idx]
        for idx in seg_indices:
            seg_len = cfg.segment_lengths[idx]
            task_outdir, ps = run_length(cfg, exe, seg_len, gpu_id)
            results[seg_len] = LengthResult(seg_len, task_outdir, predicted[seg_len], ps.runtime_ms, gpu_id,
                                            score_length(cfg, ps), worker="local")

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(run_bin, i, indices) for i, indices in enumerate(bins) if indices]
        for f in futures:
            f.result()

    best_len = max(sorted(results), key=lambda s: results[s].area_score)
    best_ps = PatternSet.load(results[best_len].outdir, best_len, results[best_len].actual_runtime_ms)
    return finalize(cfg, results, best_ps, {"gpu_ids": gpu_ids, "mode": "threads"})
