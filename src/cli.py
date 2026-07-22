from __future__ import annotations

import argparse
import sys
from pathlib import Path

from calibrate import calibrate
from paths import DEFAULT_MODEL_DIR
from pipeline import PastilaConfig, run_pastila


def parse_segment_lengths(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        raise ValueError("segment length spec must not be empty")

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"invalid range spec {spec!r}, expected MIN:MAX or MIN:MAX:STEP")
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 else 1
        if step <= 0:
            raise ValueError("STEP must be positive")
        if start <= 0 or stop < start:
            raise ValueError(f"invalid range {spec!r}: need 0 < MIN <= MAX")
        lengths = list(range(start, stop + 1, step))
    else:
        lengths = [int(x) for x in spec.split(",") if x.strip()]

    lengths = sorted(set(lengths))
    if not lengths:
        raise ValueError(f"segment length spec {spec!r} produced no lengths")
    if any(l <= 0 for l in lengths):
        raise ValueError("segment lengths must be positive")
    return lengths


def _add_run_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="discover time series snippets across a range of segment lengths")
    p.add_argument("--algo", choices=["pdss", "psf"], required=True)
    p.add_argument("--input", required=True, type=Path, help="input time series file")
    p.add_argument("--outdir", required=True, type=Path, help="output folder")
    p.add_argument("--ts-length", required=True, type=int, help="time series length")
    p.add_argument(
        "--segment-lengths", required=True,
        help="MIN:MAX[:STEP] range, or a comma-separated list, e.g. '50:400:50' or '50,100,200'",
    )
    p.add_argument("--num-snippets", required=True, type=int)
    p.add_argument("--candidate-step", type=int, default=None, help="pdss only")
    p.add_argument(
        "--gpus", default=None,
        help="comma-separated physical GPU ids to use, e.g. '0,1,2' (default: autodetect all)",
    )
    p.add_argument("--exe", default=None, type=Path, help="override path to the PDSS/PSF binary")
    p.add_argument("--model", default=None, type=Path, help="override path to the runtime model")


def _add_calibrate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("calibrate", help="fit the polynomial runtime model for pdss or psf")
    p.add_argument("--algo", choices=["pdss", "psf"], required=True)
    p.add_argument("--exe", default=None, type=Path)
    p.add_argument("--gpu", default=0, type=int, help="physical GPU id to run calibration on")
    p.add_argument("--out", default=None, type=Path, help="output model path")
    p.add_argument("--csv", default=None, type=Path, help="also dump the raw calibration data to this CSV")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pastila")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(sub)
    _add_calibrate_parser(sub)
    args = parser.parse_args(argv)

    if args.command == "run":
        if args.algo == "psf" and args.candidate_step is not None:
            parser.error("--candidate-step only applies to --algo pdss")
        cfg = PastilaConfig(
            algo=args.algo,
            input_file=args.input,
            outdir=args.outdir,
            ts_length=args.ts_length,
            segment_lengths=parse_segment_lengths(args.segment_lengths),
            num_snippets=args.num_snippets,
            candidate_step=args.candidate_step,
            exe=args.exe,
            model_path=args.model,
            gpu_ids=[int(x) for x in args.gpus.split(",")] if args.gpus else None,
        )
        summary = run_pastila(cfg)
        print(f"Best segment length: {summary['best_segment_length']}")
        print(f"Outputs written to: {summary['outdir']}")
        return 0

    if args.command == "calibrate":
        model = calibrate(
            algo=args.algo, exe=args.exe, gpu_id=args.gpu, out_model=args.out, out_csv=args.csv,
        )
        out = args.out or (DEFAULT_MODEL_DIR / f"{args.algo}_runtime_model.joblib")
        print(f"Fitted {args.algo} runtime model, saved to {out}")
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
