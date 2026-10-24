#!/usr/bin/env python3
"""Offline only: never starts or advances CFD, never writes production/site files."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flowacoustics.mrl2d_postprocess import export_study  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("diagnostic", "production"), required=True)
    parser.add_argument("--panel-chunk", type=int, default=96)
    parser.add_argument("--memory-limit-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()
    try:
        result = export_study(
            args.input,
            args.output,
            mode=args.mode,
            panel_chunk=args.panel_chunk,
            memory_limit_bytes=args.memory_limit_bytes,
        )
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f"Rejected: {error}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
