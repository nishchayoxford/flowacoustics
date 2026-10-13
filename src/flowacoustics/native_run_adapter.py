"""Translate completed native v2 records for the existing v1 analysis contract.

This adapter never mutates the native manifest, changes clocks, or promotes a
running calculation. The returned schema number identifies the analysis view,
not the original native record's format.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path


def normalize_run(source):
    """Return a detached analysis view of a completed reconstructed v2 run."""
    if source.get("schema_version") != 2:
        raise ValueError("Expected native schema_version 2")
    if source.get("status") != "completed_native_reconstructed":
        raise ValueError("Native run must be completed before analysis")
    if source.get("evidence_kind") != "native_execution":
        raise ValueError("Completed native execution evidence is required")
    if any(
        source.get(key, {}).get("status") != "completed" for key in ("reconstruction", "exports")
    ):
        raise ValueError("Reconstruction and exports must both be completed")
    coverage, solver = source.get("time_coverage", {}), source.get("solver_verification", {})
    end = coverage.get("end_s")
    if (
        not isinstance(end, (float, int))
        or not math.isfinite(end)
        or end <= 0
        or end != coverage.get("requested_end_s")
        or end != solver.get("end_s")
        or solver.get("solver_reached_target") is not True
    ):
        raise ValueError("Native target-time evidence is inconsistent")
    positions = source.get("observer_coordinates_m")
    if not isinstance(positions, dict) or not positions:
        raise ValueError("Actual observer positions are required")
    for name, position in positions.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(position, (list, tuple))
            or len(position) != 3
            or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in position)
        ):
            raise ValueError("Invalid observer coordinate")
    result = copy.deepcopy(source)
    result["source_schema_version"] = 2
    result["schema_version"] = 1
    result["observers"] = [
        {"name": name, "position_m": position}
        for name, position in result["observer_coordinates_m"].items()
    ]
    result["schema_adapter"] = (
        "Completed native v2 record mapped to the analysis v1 observer-list contract; "
        "clocks, force conventions and source status unchanged."
    )
    return result


def verified_exports(source, forces_path, observers_path):
    """Read once, verify against recorded export pins, and return those same bytes."""
    normalized = normalize_run(source)
    expected = source.get("exports", {}).get("sha256", {})
    payloads = []
    for path in (forces_path, observers_path):
        path = Path(path).resolve()
        identity = expected.get(str(path))
        if not isinstance(identity, str) or len(identity) != 64:
            raise ValueError(f"Missing native export identity: {path.name}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != identity:
            raise ValueError(f"Native export identity mismatch: {path.name}")
        payloads.append(payload)
    return normalized, *payloads
