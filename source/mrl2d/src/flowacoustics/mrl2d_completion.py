"""Detached, read-only adapter for original MRL2D runner reports.

This version consumes direct rank-contour and plane outputs. Reconstruction is
NOT required and is never represented by an invented native stage. No writes,
subprocesses, owner lock or automatic pin generation occur in this module.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

SCHEMA = "mrl2d-direct-native-completion-v2"
# Covers accumulated binary64 time and decimal native logging roundoff, NOT a
# missing solve. Absolute only; < 1/60000 of the fixed 62.5 us solver step.
TIME_ATOL_S = 1e-9
DT_S = 0.0000625


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _path(root, name):
    path = Path(name)
    _require(".." not in path.parts, "unsafe evidence path")
    path = path if path.is_absolute() else root / path
    _require(path.resolve().is_relative_to(root), "evidence path outside original case")
    return path


def _read(path, expected):
    _require(
        isinstance(expected, str) and re.fullmatch(r"[a-f0-9]{64}", expected),
        "missing expected identity",
    )
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == expected, f"identity mismatch: {path.name}")
    return data


def _clock(actual, cadence, count, label):
    values = np.asarray(actual, dtype=float)
    _require(
        values.shape == (count,)
        and np.isfinite(values).all()
        and np.allclose(values, np.arange(1, count + 1) * cadence, rtol=0, atol=TIME_ATOL_S),
        f"incomplete {label} clock",
    )


def _stage_evidence(root, run, ranks):
    from flowacoustics.mrl2d_case import audit_mesh_log, audit_native_run

    stages = run["stages"]
    names = {"checkMesh", "pimpleFoam", "volumeReadback"}
    if ranks > 1:
        names.add("decomposePar")
    _require(set(stages) == names, "missing/unexpected native stages")
    prefix = ["mpirun", "--bind-to", "none", "-np", str(ranks)] if ranks > 1 else []
    suffix = ["-parallel"] if ranks > 1 else []
    commands = {
        "checkMesh": ["checkMesh", "-allGeometry", "-allTopology"],
        "decomposePar": ["decomposePar"],
        "volumeReadback": prefix
        + ["postProcess"]
        + suffix
        + ["-dict", "system/readbackDict", "-fields", "(p U)", "-noZero"],
    }
    logs, evidence = {}, {}
    for name, stage in stages.items():
        # Current supervisor receipts count all reaped children, including the launcher.
        _require(
            stage.get("cleanup_confirmed") is True
            and type(stage.get("descendants_reaped")) is int
            and stage["descendants_reaped"] >= 1,
            f"missing/invalid native cleanup receipt: {name}",
        )
        _require(
            type(stage.get("returncode")) is int
            and stage["returncode"] == 0
            and stage.get("timed_out") is False
            and "monitor_error" in stage
            and stage["monitor_error"] is None
            and stage.get("status", "completed") == "completed"
            and not stage.get("error"),
            f"failed/pending native stage: {name}",
        )
        wall, limit = stage["wall_seconds"], stage["wall_limit"]
        _require(
            math.isfinite(wall) and math.isfinite(limit) and 0 < wall <= limit,
            f"native stage wall budget: {name}",
        )
        argv = stage["argv"]
        if name == "pimpleFoam":
            i = 2 + len(prefix)
            _require(
                len(argv) == i + 1 + len(suffix)
                and argv[0] == "env"
                and argv[1].startswith("LD_LIBRARY_PATH=")
                and argv[2:i] == prefix
                and Path(argv[i]).name == "productionPimpleFoam"
                and argv[i + 1 :] == suffix,
                "solver stage command/ranks mismatch",
            )
        else:
            _require(argv == commands[name], f"native stage command mismatch: {name}")
        path = _path(root, stage["log"])
        _require(path == root / ("log." + name), "stage-specific log identity required")
        _require(stage["log_sha256"] not in evidence.values(), "native stage evidence reused")
        log = _read(path, stage["log_sha256"]).decode()
        _require(
            re.search(r"^End\s*$", log, re.M)
            and not re.search(
                r"FOAM FATAL|Floating point exception(?! trapping enabled)|"
                r"Segmentation fault|\b(?:nan|inf)\b",
                log,
                re.I,
            ),
            f"native failure/End: {name}",
        )
        logs[name] = log
        evidence[path.name] = stage["log_sha256"]
    mesh = audit_mesh_log(logs["checkMesh"], returncode=0)
    _require(mesh == run["mesh"], "mesh report differs from original log")
    solver = stages["pimpleFoam"]
    log = logs["pimpleFoam"]
    flow = audit_native_run(
        log,
        returncode=0,
        dt=DT_S,
        end=10,
        elapsed=solver["wall_seconds"],
        wall_limit=solver["wall_limit"],
        require_complete_ami=True,
    )
    _clock(re.findall(r"^Time = (\S+)\s*$", log, re.M), DT_S, 160000, "solver")
    for key, value in flow.items():
        actual = run["flow"].get(key)
        _require(
            actual is not None
            and math.isclose(
                actual, value, rel_tol=0, abs_tol=TIME_ATOL_S if key == "end" else 1e-12
            ),
            "flow report/log mismatch",
        )
    rows = re.findall(
        r"^MRL2D_POST time=(\S+) fluid=(\S+) meshRaw=(\S+) "
        r"meshEffective=(\S+) finite=1$",
        log,
        re.M,
    )
    _require(
        len(rows) == 160000 and len(re.findall(r"^MRL2D_POST", log, re.M)) == 160000,
        "incomplete post-solve finite-field coverage",
    )
    post = np.asarray(rows, dtype=float)
    _clock(post[:, 0], DT_S, 160000, "post-solve")
    _require(
        np.isfinite(post).all() and np.all((post[:, 1:] >= 0) & (post[:, 1:] <= 1)),
        "post-solve Courant/finite failure",
    )
    # Coverage, ordering and every weight were checked by the shared pure gate.
    summary = {
        "post_solve_steps": 160000,
        "solved_time": float(post[-1, 0]),
        "max_fluid_courant": float(post[:, 1].max()),
        "max_mesh_courant": float(post[:, 2:].max()),
        "ami_lines": log.count("sum(weights)"),
    }
    for key, value in summary.items():
        _require(
            key in run["post_solve"]
            and math.isclose(
                run["post_solve"][key],
                value,
                rel_tol=0,
                abs_tol=TIME_ATOL_S if key == "solved_time" else 1e-12,
            ),
            "post-solve report/log mismatch",
        )
    volume = logs["volumeReadback"]
    _clock(re.findall(r"^Time = (\S+)\s*$", volume, re.M), 0.2, 50, "volume readback")
    _require(run["volume_frames_read_natively"] == 50, "50 native volume readbacks required")
    for block in re.split(r"^Time = \S+\s*$", volume, flags=re.M)[1:]:
        for field in (r"p", r"mag\(U\)"):
            for operator in ("min", "max"):
                values = re.findall(rf"\b{operator}\({field}\)\s*=\s*(\S+)", block)
                _require(
                    len(values) == 1 and math.isfinite(float(values[0])),
                    "volume readback missing finite p/U extrema",
                )
    return evidence


def _inputs(root, manifest, spec):
    from flowacoustics.mrl2d_case import CaseSpec, case_files, case_specs

    identity = CaseSpec(manifest["name"], manifest["model"], manifest["envelope"])
    _require(
        identity in case_specs()
        and type(manifest["envelope"]) is bool
        and identity.name == spec["id"],
        "original model/envelope/case identity mismatch",
    )
    _require(
        type(manifest["ranks"]) is int
        and manifest["ranks"] in (1, 2)
        and manifest["delta_t"] == DT_S
        and spec["end_s"] == 10
        and spec.get("cadence_s", 0.002) == manifest["source_cadence"] == 0.002
        and manifest["animation_cadence"] == 0.02
        and manifest["volume_cadence"] == 0.2
        and spec["envelope_recorded"] is manifest["envelope"],
        "original rank/timestep/cadence/envelope mismatch",
    )
    _require(
        manifest["physical"] == {"rho": 1000.0, "nu": 1e-6, "U": [1, 0, 0], "omega": 10.472}
        and math.isfinite(manifest["numerical_depth_m"])
        and manifest["numerical_depth_m"] > 0,
        "unsupported physical constants/depth",
    )
    expected = case_files(
        identity, depth=manifest["numerical_depth_m"], dt=DT_S, end=10, ranks=manifest["ranks"]
    )
    required = set(expected) | {
        "constant/polyMesh/" + name
        for name in ("points", "faces", "owner", "neighbour", "boundary", "cellZones")
    }
    pins = manifest["input_sha256"]
    actual = {
        str(p.relative_to(root))
        for sub in ("0", "constant", "system")
        for p in (root / sub).rglob("*")
        if p.is_file()
        and str(p.relative_to(root)) != "0/cellToRegion"
        and not str(p.relative_to(root)).startswith("constant/polyMesh/sets/")
    }
    _require(required <= set(pins) == actual, "executed input identity coverage incomplete")
    excluded = {"0/k", "0/omega", "0/nut", "constant/turbulenceProperties"}
    checked = excluded | {"system/controlDict", "system/decomposeParDict", "system/readbackDict"}
    physical = {}
    for name in sorted(pins):
        data = _read(_path(root, name), pins[name])
        if name in checked:
            _require(
                name in expected and data == expected[name].encode(),
                f"executed model/control dictionary differs from supported input: {name}",
            )
        if name in excluded:
            continue
        if name == "system/controlDict":
            data = data.replace(b"envelope true;", b"envelope false;")
        physical[name] = hashlib.sha256(data).hexdigest()
    payload = b"mrl2d-matched-inputs-v1\n" + b"".join(
        name.encode() + b"\0" + sha.encode() + b"\n" for name, sha in physical.items()
    )
    return hashlib.sha256(payload).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate original report/input key")
        result[key] = value
    return result


def direct_native_completion(spec):
    """Derive an analysis view from pinned originals; never rewrite native history."""
    root = Path(spec["root"]).resolve()
    pins = spec["native_completion"]
    _require(pins.get("schema") == SCHEMA, "versioned direct-native completion required")
    manifest = json.loads(
        _read(root / "mrl2d-input.json", pins.get("input_sha256")), object_pairs_hook=_unique_object
    )
    run = json.loads(
        _read(root / "mrl2d-run.json", pins.get("run_sha256")), object_pairs_hook=_unique_object
    )
    _require(manifest.get("schema") == "mrl2d-flow-input-v1", "unknown original input schema")
    _require(
        run.get("status") == "complete_10s_native_verified"
        and run.get("source_history_valid") is True
        and not run.get("error"),
        "original run is not complete_10s_native_verified / source_history_valid",
    )
    _require(Path(run["case"]).resolve() == root, "original run case identity mismatch")
    _require(
        manifest["name"] == spec["id"]
        and manifest["target_seconds"] == 10
        and manifest["scope"] == "production",
        "original case/10 s scope mismatch",
    )
    fingerprint = _inputs(root, manifest, spec)
    inputs = manifest["input_sha256"]
    sources = {}
    for name, sha in run["source"]["native_files_sha256"].items():
        canonical = str(_path(root, name).resolve().relative_to(root))
        _require(canonical not in sources, "duplicate canonical source identity")
        _require(
            isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha),
            "invalid source identity pin",
        )
        sources[canonical] = sha
    ranks = manifest["ranks"]
    expected_sources = {
        f"processor{i}/postProcessing/mrl2dSources/rank{i}.bin" for i in range(ranks)
    }
    if ranks == 1:
        expected_sources = {"postProcessing/mrl2dSources/rank0.bin"}
    actual_sources = {
        str(p.relative_to(root))
        for pattern in (
            "postProcessing/mrl2dSources/rank*.bin",
            "processor*/postProcessing/mrl2dSources/rank*.bin",
        )
        for p in root.glob(pattern)
    }
    _require(
        set(sources) == expected_sources == actual_sources
        and sources == spec["source_sha256"]
        and all(isinstance(h, str) and re.fullmatch(r"[a-f0-9]{64}", h) for h in sources.values()),
        "source identity/rank coverage mismatch",
    )
    _require(
        run["source"]["schema"] == "mrl2d-native-source-v1"
        and Path(run["source"]["case"]).resolve() == root
        and run["source"]["frames"] == 5000
        and run["source"]["rank_files"] == ranks,
        "original source identity/frame coverage mismatch",
    )
    frames = [
        {
            "time_s": float(Path(f["path"]).parent.name),
            "path": str(_path(root, f["path"]).relative_to(root)),
            "sha256": f["sha256"],
        }
        for f in run["animation"]
    ]
    _require(
        len(frames) == 500
        and len({f["path"] for f in frames}) == 500
        and np.allclose(
            [f["time_s"] for f in frames], np.arange(1, 501) * 0.02, rtol=0, atol=TIME_ATOL_S
        ),
        "500 chronological field frames required",
    )
    _require(
        run["animation_frames"] == 500
        and all(f.get("vtk_header_valid") is True for f in run["animation"]),
        "original animation verification incomplete",
    )
    actual_frames = {
        str(p.relative_to(root)) for p in root.glob("postProcessing/animation/*/section.vtp")
    }
    _require({f["path"] for f in frames} == actual_frames, "animation identity coverage mismatch")
    for frame in frames:
        _read(_path(root, frame["path"]), frame["sha256"])
    evidence = _stage_evidence(root, run, ranks)
    return {
        "schema": SCHEMA,
        "case_id": manifest["name"],
        "source_type": "native_cfd",
        "physical_input_fingerprint": fingerprint,
        "time_roundoff_atol_s": TIME_ATOL_S,
        "original_reports": pins,
        "input_sha256": inputs,
        "evidence_sha256": evidence,
        "source_sha256": sources,
        "field_frames": frames,
        "requested_end_s": 10,
        "cells": run["mesh"]["cells"],
        "ranks": manifest["ranks"],
        "dt_s": manifest["delta_t"],
        "stages": {
            "solver": {"steps": run["flow"]["steps"], "log": "log.pimpleFoam"},
            "volume_readback": {
                "times_read": run["volume_frames_read_natively"],
                "log": "log.volumeReadback",
            },
            "reconstruction": {
                "status": "not_required",
                "reason": "direct_native_contour_and_slice",
            },
        },
        "fields_still_require_readback": True,
    }
