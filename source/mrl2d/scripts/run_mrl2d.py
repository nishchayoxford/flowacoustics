#!/usr/bin/env python3
"""Four strict-2D cases: explicit preparation and one-owner bounded execution."""

import argparse
import json
from pathlib import Path

from flowacoustics.mrl2d_case import case_specs, prepare_cases, run_case, verify_frozen
from flowacoustics.mrl2d_production import run_cases


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="create four fresh cases, no solver execution")
    prep.add_argument("--mesh-case", type=Path, required=True)
    prep.add_argument("--output-root", type=Path, default=root / "cfd")
    prep.add_argument(
        "--depth", type=float, required=True, help="numerical extrusion depth, metres"
    )
    prep.add_argument("--dt", type=float, default=0.0000625)
    prep.add_argument(
        "--end", type=float, default=10.0, help="short endpoints permitted only in cache"
    )
    prep.add_argument("--ranks", type=int, choices=range(1, 9), default=2)
    run = sub.add_parser(
        "run", help="one-owner native 10 s run, explicit wall budget and mesh proof"
    )
    run.add_argument("--case", choices=[s.name for s in case_specs()], required=True)
    run.add_argument("--output-root", type=Path, default=root / "cfd")
    run.add_argument("--wall-seconds", type=float, required=True)
    run.add_argument("--motion-evidence", type=Path, required=True)
    many = sub.add_parser("run-many", help="up to four cases, two MPI ranks each, ONE shared owner")
    many.add_argument(
        "--cases",
        nargs="+",
        choices=[s.name for s in case_specs()],
        default=[s.name for s in case_specs()],
    )
    many.add_argument("--output-root", type=Path, default=root / "cfd")
    many.add_argument("--motion-evidence", type=Path, required=True)
    many.add_argument("--wall-seconds", type=float, required=True)
    many.add_argument(
        "--native-seconds",
        type=float,
        required=True,
        help="sum of subprocess wall times, not concurrent elapsed time",
    )
    many.add_argument("--carry-forward", type=float, default=0.0)
    many.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    many.add_argument("--min-free-gib", type=float, default=20)
    many.add_argument(
        "--work-root", type=Path, required=True, help="fresh execution/log/build workspace"
    )
    verify = sub.add_parser("verify-inputs", help="read protected source/input hashes, no flow")
    verify.add_argument("case", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = {
            "status": "prepared_not_run",
            "cases": list(
                map(
                    str,
                    prepare_cases(
                        args.mesh_case,
                        args.output_root,
                        depth=args.depth,
                        dt=args.dt,
                        ranks=args.ranks,
                        end=args.end,
                    ),
                )
            ),
        }
    elif args.action == "run":
        result = run_case(
            args.output_root / args.case,
            wall_seconds=args.wall_seconds,
            motion_evidence=args.motion_evidence,
        )
    elif args.action == "run-many":
        result = run_cases(
            [args.output_root / name for name in args.cases],
            motion_evidence=args.motion_evidence,
            wall_seconds=args.wall_seconds,
            native_seconds=args.native_seconds,
            carry_forward=args.carry_forward,
            workers=args.workers,
            min_free_bytes=args.min_free_gib * 1024**3,
            work_root=args.work_root,
        )
    else:
        result = verify_frozen(args.case)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
