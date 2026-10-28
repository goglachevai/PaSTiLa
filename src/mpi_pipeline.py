from __future__ import annotations

import socket
import time
import traceback

from merge import PatternSet
from pipeline import (LengthResult, PastilaConfig, detect_gpus, finalize, predict_and_partition, run_length,
                      score_length, validate)

TAG_TASK = 1
TAG_RESULT = 2


def _node_worker_index(comm) -> int:
    from mpi4py import MPI

    node = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.Get_rank())
    ranks = node.allgather(comm.Get_rank())
    node.Free()
    workers = [r for r in sorted(ranks) if r != 0]
    return workers.index(comm.Get_rank()) if comm.Get_rank() in workers else 0


def _payload(task_outdir, ps: PatternSet, gpu_id, host) -> dict:
    return {"m": ps.segment_length, "outdir": str(task_outdir), "starts": ps.starts, "snippets": ps.snippets,
            "profiles": ps.profiles, "fracs": ps.fracs, "labels": ps.labels, "runtime_ms": ps.runtime_ms,
            "gpu_id": gpu_id, "host": host}


def _worker(comm, cfg: PastilaConfig, exe, node_index: int) -> None:
    gpu_ids = cfg.gpu_ids if cfg.gpu_ids else detect_gpus()
    gpu_id = gpu_ids[node_index % len(gpu_ids)]
    host = f"{socket.gethostname()}:rank{comm.Get_rank()}"
    task = comm.recv(source=0, tag=TAG_TASK)
    for seg_len in task:
        try:
            task_outdir, ps = run_length(cfg, exe, seg_len, gpu_id)
            comm.send(_payload(task_outdir, ps, gpu_id, host), dest=0, tag=TAG_RESULT)
        except Exception:
            comm.send({"m": seg_len, "error": traceback.format_exc(), "host": host}, dest=0, tag=TAG_RESULT)


def run_pastila_mpi(cfg: PastilaConfig, comm=None) -> dict | None:
    from mpi4py import MPI

    comm = comm or MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    exe = validate(cfg)
    node_index = _node_worker_index(comm)

    if rank != 0:
        _worker(comm, cfg, exe, node_index)
        return None

    t0 = time.perf_counter()
    num_workers = size - 1
    predicted, bins = predict_and_partition(cfg, max(num_workers, 1), require_model=False)
    results: dict[int, LengthResult] = {}
    errors: list[str] = []
    best: PatternSet | None = None

    def accept(msg: dict) -> None:
        nonlocal best
        seg_len = msg["m"]
        if "error" in msg:
            errors.append(f"length {seg_len} on {msg['host']}:\n{msg['error']}")
            return
        ps = PatternSet(seg_len, msg["starts"], msg["snippets"], msg["profiles"], msg["fracs"], msg["labels"],
                        msg["runtime_ms"])
        score = score_length(cfg, ps)
        results[seg_len] = LengthResult(seg_len, msg["outdir"], predicted[seg_len], msg["runtime_ms"],
                                        msg["gpu_id"], score, worker=msg["host"])
        if best is None or (score, -seg_len) > (results[best.segment_length].area_score, -best.segment_length):
            best = ps

    if num_workers == 0:
        gpu_ids = cfg.gpu_ids if cfg.gpu_ids else detect_gpus()
        for seg_len in cfg.segment_lengths:
            try:
                task_outdir, ps = run_length(cfg, exe, seg_len, gpu_ids[0])
                accept(_payload(task_outdir, ps, gpu_ids[0], f"{socket.gethostname()}:rank0"))
            except Exception:
                accept({"m": seg_len, "error": traceback.format_exc(), "host": "rank0"})
    else:
        tasks = [[cfg.segment_lengths[i] for i in b] for b in bins]
        tasks += [[] for _ in range(num_workers - len(tasks))]
        for p, task in enumerate(tasks, 1):
            comm.send(task, dest=p, tag=TAG_TASK)
        for _ in range(len(cfg.segment_lengths)):
            accept(comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT))

    if errors:
        raise RuntimeError(f"{len(errors)} segment length(s) failed:\n" + "\n".join(errors))
    extra = {"mode": "mpi", "mpi_processes": size, "workers": num_workers,
             "tasks": {str(p): [cfg.segment_lengths[i] for i in b] for p, b in enumerate(bins, 1)},
             "wall_time_until_all_results_s": round(time.perf_counter() - t0, 3)}
    return finalize(cfg, results, best, extra)
