from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

_RUNTIME_RE = re.compile(r"runtime:\s*([\d.]+)\s*ms", re.IGNORECASE)


@dataclass
class TaskResult:
    segment_length: int
    outdir: Path
    runtime_ms: float
    gpu_id: int | None


def _ensure_executable(exe: Path) -> None:
    if platform.system() == "Windows":
        return
    if not os.access(exe, os.X_OK):
        exe.chmod(exe.stat().st_mode | 0o111)


def _parse_runtime_from_log(outdir: Path) -> float:
    log_path = outdir / "log.txt"
    if not log_path.exists():
        return float("nan")
    last_line = ""
    for last_line in log_path.read_text().splitlines():
        pass
    if not last_line:
        return float("nan")
    try:
        return float(last_line.split()[-1])
    except (ValueError, IndexError):
        return float("nan")


def run_task(
    exe: Path,
    algo: str,
    input_file: Path,
    outdir: Path,
    ts_length: int,
    segment_length: int,
    num_snippets: int,
    candidate_step: int | None = None,
    gpu_id: int | None = None,
    timeout: float | None = None,
) -> TaskResult:
    
    exe = Path(exe).resolve()
    input_file = Path(input_file).resolve()
    outdir = Path(outdir).resolve()
    _ensure_executable(exe)
    args = [str(exe), str(input_file), str(outdir), str(ts_length), str(segment_length), str(num_snippets)]
    if algo == "pdss" and candidate_step is not None:
        args.append(str(candidate_step))

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    outdir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        args, cwd=str(outdir), env=env, capture_output=True, text=True, timeout=timeout
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{exe.name} failed (ts_length={ts_length}, segment_length={segment_length}, "
            f"gpu={gpu_id}, exit={proc.returncode}):\n{proc.stderr}"
        )

    match = _RUNTIME_RE.search(proc.stdout)
    runtime_ms = float(match.group(1)) if match else _parse_runtime_from_log(outdir)
    return TaskResult(segment_length, outdir, runtime_ms, gpu_id)
