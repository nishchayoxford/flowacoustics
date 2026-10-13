#!/usr/bin/env python3
"""Build nine auditable acoustic-signature figures from completed native exports.

No solver execution. Existing output targets and incomplete runs are refused.
Historical schema-1 data require --historical-smoke and a destination beneath
results/cylinder/signature/historical_smoke/. See docs/SIGNATURE_ANALYSIS.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flowacoustics.signature_analysis import build_report, preflight_output  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["run", "forces", "observers", "case", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start-time-s", type=float, default=2.0)
    parser.add_argument("--end-time-s", type=float, default=10.0)
    parser.add_argument("--nperseg", type=int, default=3200)
    parser.add_argument("--resample-dt-s", type=float)
    parser.add_argument("--historical-smoke", action="store_true")
    args = parser.parse_args(argv)
    root = ROOT / "results/cylinder/signature"
    try:
        preflight_output(args.output, root)
        if args.historical_smoke:
            historical = root / "historical_smoke"
            if not args.output.absolute().is_relative_to(historical):
                raise ValueError(
                    "Historical smoke output must be under "
                    "results/cylinder/signature/historical_smoke"
                )
        result = build_report(
            run_path=args.run,
            forces_path=args.forces,
            observers_path=args.observers,
            case=args.case,
            project=ROOT,
            output=args.output,
            allowed_root=root,
            start_time_s=args.start_time_s,
            end_time_s=args.end_time_s,
            nperseg=args.nperseg,
            resample_dt_s=args.resample_dt_s,
            historical_smoke=args.historical_smoke,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Refused: {exc}\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "evidence_label",
                    "figure_count",
                    "counts",
                    "actual_window_s",
                    "analysis_samples",
                    "estimator",
                    "numerical_checks",
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
