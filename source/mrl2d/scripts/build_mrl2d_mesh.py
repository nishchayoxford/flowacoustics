#!/usr/bin/env python3
"""Build and execute a fresh 2D MRL mesh admission; never creates flow cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowacoustics.mrl2d_mesh import _weights, build_base, verify_native  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True, help="Fresh temporary base case path")
    parser.add_argument(
        "--report", type=Path, default=ROOT / "docs/mrl2d/mesh", help="Compact evidence directory"
    )
    args = parser.parse_args()
    start = time.monotonic()
    build_base(args.case)
    manifest = verify_native(args.case)
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    logs = report / "native"
    logs.mkdir(exist_ok=True)
    for stage in manifest["native_stages"]:
        source = Path(stage["log"])
        destination = logs / source.name
        shutil.copy2(source, destination)
        stage["retained_log"] = str(destination)
    weights = {}
    for state in manifest["motion"]["states"]:
        for patch in state["coverage"]:
            key = f"t{state['time_s']:g}_{patch['patch']}"
            weights[key] = _weights(Path(patch["file"]))
    np.savez_compressed(report / "ami_all_faces.npz", **weights)
    manifest["retained_ami_arrays"] = str(report / "ami_all_faces.npz")
    case = Path(manifest["case"])
    # Only checkMesh diagnostic sets/region labels are discarded; core mesh is unchanged.
    diagnostics = [case / "constant/polyMesh/sets", case / "constant/cellToRegion"]
    for path in diagnostics:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    inputs = [
        case / "constant/polyMesh" / name
        for name in ("points", "faces", "owner", "neighbour", "boundary", "cellZones")
    ]
    inputs += [
        case / "constant/dynamicMeshDict",
        ROOT / "src/flowacoustics/mrl2d_mesh.py",
        Path(__file__).resolve(),
    ]
    manifest["sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
    }
    manifest["wall_seconds_build_and_admission"] = time.monotonic() - start
    manifest["scope"] = "Mesh and 0.6 s motion-only admission; no flow or production execution"
    payload = json.dumps(manifest, indent=2) + "\n"
    (case / "mesh_manifest.json").write_text(payload)
    (report / "admission.json").write_text(payload)
    print(
        json.dumps(
            {
                "admitted": manifest["admitted"],
                "case": str(case),
                "cells": manifest["cells"],
                "report": str(report / "admission.json"),
                "wall_seconds": manifest["wall_seconds_build_and_admission"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
