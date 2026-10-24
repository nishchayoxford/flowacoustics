"""Bounded diagnostic helpers, not production admission or case generation."""

import json
import re
from pathlib import Path

from flowacoustics.mrl2d_case import case_files, native_command


def stage_kind(name):
    """Do not confuse setup logs with endpoint field-readback evidence."""
    return {
        "checkMesh": "mesh",
        "finalMesh": "mesh",
        "flow": "flow",
        "volumeReadback": "readback",
        "decomposePar": "setup",
    }[name]


def trial_files(spec, *, dt, end, ranks):
    """Cache-only diagnostic: write full fields at the bounded endpoint."""
    files = case_files(spec, depth=0.01, dt=dt, end=end, ranks=ranks)
    control = files["system/controlDict"]
    old = f"writeControl timeStep; writeInterval {round(0.2 / dt)};"
    new = f"writeControl timeStep; writeInterval {round(end / dt)};"
    assert control.count(old) == 1
    files["system/controlDict"] = control.replace(old, new)
    return files


class NativeBudget:
    """Append actual native subprocess wall; reserve cleanup margin at hard cap."""

    def __init__(self, path, limit=1200):
        self.path = Path(path)
        self.limit = limit
        self.records = json.loads(self.path.read_text())["records"] if self.path.exists() else []

    def run(self, command, *, cwd, log, timeout):
        remaining = self.limit - sum(r["wall_seconds"] for r in self.records)
        if remaining <= 5:
            raise RuntimeError("native wall budget exhausted")
        record = native_command(command, cwd=cwd, log=log, timeout=min(timeout, remaining - 5))
        self.records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "limit_seconds": self.limit,
                    "total_seconds": sum(r["wall_seconds"] for r in self.records),
                    "records": self.records,
                },
                indent=2,
            )
            + "\n"
        )
        return record


def candidate_mesh_source(original):
    """One frozen resolution-policy experiment; preserve all physical geometry."""
    replacements = {
        '"Mesh.MeshSizeMin", 0.00006': '"Mesh.MeshSizeMin", 0.0005',
        '"SizeMin": 0.0004': '"SizeMin": 0.0008',
        '"DistMax": 0.20': '"DistMax": 0.10',
        '"DistMax": 0.25': '"DistMax": 0.10',
    }
    for old, new in replacements.items():
        if original.count(old) != 1:
            raise ValueError("unexpected frozen size policy")
        original = original.replace(old, new)
    return original


def instrument_solver(original):
    """Retain original equations; insert one read-only post-solve metric call."""
    if original.count("        runTime.write();") != 1:
        raise ValueError("unexpected pimpleFoam source")
    return original.replace(
        '#include "fvCFD.H"', '#include "fvCFD.H"\n#include "runtimeMetrics.H"'
    ).replace(
        "        runTime.write();",
        "        runtimeMetrics(mesh, U, phi);\n        runTime.write();",
    )


def courant_records(text):
    """Fluid lines precede the next increment; mesh lines follow mesh update."""
    result = {"fluid": [], "mesh": [], "times": [], "normal_end": False}
    time = 0.0
    for line in text.splitlines():
        if match := re.fullmatch(r"Time = (\S+)", line):
            time = float(match[1])
            result["times"].append(time)
        if match := re.fullmatch(r"(Mesh )?Courant Number mean: (\S+) max: (\S+)", line):
            result["mesh" if match[1] else "fluid"].append(
                {"state_time": time, "mean": float(match[2]), "maximum": float(match[3])}
            )
        if line.strip() == "End":
            result["normal_end"] = True
    return result
