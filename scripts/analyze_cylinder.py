#!/usr/bin/env python3
"""Analyze explicit cylinder handover files. Never launch CFD or invent missing observations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

# Works from a clean source checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flowacoustics.cylinder_acoustics import build_dataset  # noqa: E402
from flowacoustics.native_run_adapter import verified_exports  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--conventions",
        type=Path,
        help="Explicit metadata adapter JSON; cannot override status or evidence kind",
    )
    parser.add_argument("--forces", required=True, type=Path)
    parser.add_argument(
        "--observers",
        required=True,
        type=Path,
        help="Explicit CSV; a header-only file honestly means no native observers",
    )
    parser.add_argument("--output", type=Path, default=Path("results/cylinder/analysis"))
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--resample-dt-s", type=float)
    parser.add_argument("--start-time-s", type=float)
    parser.add_argument("--end-time-s", type=float)
    args = parser.parse_args(argv)
    try:
        if (args.output / "results.json").exists() or (args.output / "manifest.json").exists():
            raise ValueError("Analysis output exists; choose a fresh --output directory")
        run_bytes = args.run.read_bytes()
        run = json.loads(run_bytes)
        if run.get("schema_version") == 2:
            if args.conventions:
                raise ValueError("A sidecar cannot override recorded native v2 conventions")
            run, force_bytes, observer_bytes = verified_exports(run, args.forces, args.observers)
        else:
            force_bytes, observer_bytes = args.forces.read_bytes(), args.observers.read_bytes()
        if args.conventions:
            conventions = json.loads(args.conventions.read_text())
            allowed = {
                "medium",
                "geometry",
                "clock",
                "observers",
                "force_convention",
                "force_content",
            }
            if set(conventions) - allowed:
                raise ValueError("Conventions sidecar contains non-convention keys")
            run = {**run, **conventions}
        with io.StringIO(force_bytes.decode(), newline="") as stream:
            reader = csv.DictReader(stream)
            columns = ("time_s", "force_x_N", "force_y_N", "force_z_N")
            if not set(columns) <= set(reader.fieldnames or []):
                raise ValueError("Missing required force CSV columns")
            forces = np.array([[float(row[key]) for key in columns] for row in reader])
        with io.StringIO(observer_bytes.decode(), newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"time_s", "observer", "method", "pressure_Pa"}
            if not required <= set(reader.fieldnames or []):
                raise ValueError("Missing required observer CSV columns")
            observers = list(reader)
        data, aligned = build_dataset(
            run,
            forces,
            observers,
            nperseg=args.nperseg,
            resample_dt_s=args.resample_dt_s,
            start_time_s=args.start_time_s,
            end_time_s=args.end_time_s,
        )
        inputs = [
            {"role": role, "file": path.name, "sha256": hashlib.sha256(payload).hexdigest()}
            for role, path, payload in (
                ("run", args.run, run_bytes),
                ("forces", args.forces, force_bytes),
                ("observers", args.observers, observer_bytes),
            )
        ]
        if args.conventions:
            inputs.append(
                {
                    "role": "conventions",
                    "file": args.conventions.name,
                    "sha256": digest(args.conventions),
                }
            )
        data["metadata"]["inputs"] = inputs
        data["downloads"] = [
            {"label": label, "href": f"downloads/{filename}"}
            for label, filename in (
                ("Recorded forces (N)", "forces.csv"),
                ("Recorded native pressures (Pa)", "observers.csv"),
                ("Aligned signed comparison (Pa)", "aligned_pressures.csv"),
            )
        ]
        encoded = json.dumps(data, allow_nan=False, default=json_default, separators=(",", ":"))
        downloads = args.output / "downloads"
        targets = [
            downloads / name for name in ("forces.csv", "observers.csv", "aligned_pressures.csv")
        ]
        if any(path.exists() for path in targets):
            raise ValueError("A download output already exists; choose a fresh output directory")
        downloads.mkdir(parents=True, exist_ok=True)
        targets[0].write_bytes(force_bytes)
        targets[1].write_bytes(observer_bytes)
        with targets[2].open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "time_s",
                    "source_time_s",
                    "observer",
                    "method",
                    "native_pressure_Pa",
                    "compact_pressure_Pa",
                    "compact_near_Pa",
                    "compact_far_Pa",
                    "force_on_fluid_projection_N",
                ),
            )
            writer.writeheader()
            writer.writerows(aligned)
        result_path = args.output / "results.json"
        result_path.write_text(encoded + "\n")
        outputs = [
            {
                "path": str(path.relative_to(args.output)),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            for path in [result_path, *targets]
        ]
        manifest = {
            "schema_version": 1,
            "status": data["status"],
            "chart_count": len(data["charts"]),
            "aligned_row_count": len(aligned),
            "chart_counts_by_origin": {
                origin: sum(c["origin"] == origin for c in data["charts"])
                for origin in ("computed", "analytical")
            },
            "inputs": inputs,
            "outputs": outputs,
            "implementation": [
                {"file": path.name, "sha256": digest(path)}
                for path in (
                    Path(__file__),
                    Path(__file__).resolve().parents[1] / "src/flowacoustics/cylinder_acoustics.py",
                    Path(__file__).resolve().parents[1] / "src/flowacoustics/analysis.py",
                    *(
                        [
                            Path(__file__).resolve().parents[1]
                            / "src/flowacoustics/native_run_adapter.py"
                        ]
                        if run.get("source_schema_version") == 2
                        else []
                    ),
                )
            ],
            "analysis_options": {
                key: getattr(args, key)
                for key in ("nperseg", "resample_dt_s", "start_time_s", "end_time_s")
            },
        }
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "status": data["status"],
                    "chart_count": len(data["charts"]),
                    "aligned_row_count": len(aligned),
                    "output": str(args.output),
                }
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Analysis error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
