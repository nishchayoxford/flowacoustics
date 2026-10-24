#!/usr/bin/env python3
"""Emit the exact production mesh-policy source; do not generate or run a mesh."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flowacoustics.mrl2d_runtime import candidate_mesh_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-generator",
        type=Path,
        required=True,
        help="New source file in an existing directory (exclusive create; no native execution)",
    )
    args = parser.parse_args()
    target = args.emit_generator
    if ".." in target.parts or any(p.is_symlink() for p in (target, *target.parents)):
        parser.error("generator destination must not contain symlinks or '..'")
    if target.exists() or not target.parent.is_dir():
        parser.error("generator destination must be new with an existing parent directory")
    source = (ROOT / "src/flowacoustics/mrl2d_mesh.py").read_text()
    payload = candidate_mesh_source(source).encode()
    with target.open("xb") as stream:
        stream.write(payload)
    print(f"Generator prepared: sha256={hashlib.sha256(payload).hexdigest()}; native not executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
