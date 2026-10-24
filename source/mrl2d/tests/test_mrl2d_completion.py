"""Explicitly SYNTHETIC completion fixtures; never native-run or publication evidence.

No solver, native owner lock, existing CFD histories, reconstruction or renderer.
"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from flowacoustics.mrl2d_case import case_files, case_specs
from flowacoustics.mrl2d_postprocess import validate_completion

SCHEMA = "mrl2d-direct-native-completion-v2"
DT = 0.0000625


def digest(data):
    return hashlib.sha256(data).hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Fixtures may be hard-linked; never mutate the shared seed.
    if path.exists():
        path.unlink()
    path.write_bytes(data.encode() if isinstance(data, str) else data)


def put_json(path, data):
    put(path, json.dumps(data, sort_keys=True, indent=2) + "\n")


def seal(root, manifest, run):
    put_json(root / "mrl2d-input.json", manifest)
    put_json(root / "mrl2d-run.json", run)
    return {
        "id": manifest["name"],
        "root": str(root),
        "end_s": 10,
        "cadence_s": 0.002,
        "envelope_recorded": manifest["envelope"],
        "source_sha256": {
            str(Path(p).relative_to(root)): h
            for p, h in run["source"]["native_files_sha256"].items()
        },
        "native_completion": {
            "schema": SCHEMA,
            "input_sha256": digest((root / "mrl2d-input.json").read_bytes()),
            "run_sha256": digest((root / "mrl2d-run.json").read_bytes()),
        },
    }


def stage(root, name, argv, text):
    path = root / ("log." + name)
    put(path, text)
    return {
        "argv": argv,
        "log": str(path),
        "log_sha256": digest(text.encode()),
        "returncode": 0,
        "cleanup_confirmed": True,
        # The current supervisor counts the launcher itself among reaped children.
        "descendants_reaped": 1,
        "timed_out": False,
        "monitor_error": None,
        "wall_seconds": 20.0,
        "wall_limit": 100.0,
        "started_unix": 1.0,
        "finished_unix": 21.0,
    }


def plane(time):
    return f"""<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
<PolyData><FieldData>
<DataArray type="Float64" Name="TimeValue" NumberOfTuples="1"
format="ascii">{time:.17g}</DataArray></FieldData>
<Piece NumberOfPoints="3" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="1">
<PointData><DataArray type="Float32" Name="p" format="ascii">0 1 2</DataArray>
<DataArray type="Float32" Name="U" NumberOfComponents="3"
format="ascii">1 0 0 1 0 0 1 0 0</DataArray></PointData>
<Points><DataArray type="Float32" NumberOfComponents="3"
format="ascii">0 0 0 1 0 0 0 1 0</DataArray></Points>
<Polys><DataArray type="Int32" Name="connectivity" format="ascii">0 1 2</DataArray>
<DataArray type="Int32" Name="offsets" format="ascii">3</DataArray></Polys>
</Piece></PolyData></VTKFile>"""


@pytest.fixture(scope="module")
def seed(tmp_path_factory):
    root = tmp_path_factory.mktemp("SYNTHETIC-NOT-CFD") / "mrl2d_urans_body"
    root.mkdir()
    put(root / "SYNTHETIC.txt", "ALL files are synthetic test data, NOT native CFD evidence.\n")
    files = case_files(case_specs()[0], depth=0.01)
    files.update(
        {
            "constant/polyMesh/" + name: "SYNTHETIC mesh " + name + "\n"
            for name in ("points", "faces", "owner", "neighbour", "boundary", "cellZones")
        }
    )
    for name, text in files.items():
        put(root / name, text)
    manifest = {
        "schema": "mrl2d-flow-input-v1",
        "status": "prepared_not_run",
        "name": root.name,
        "model": "urans",
        "envelope": False,
        "scope": "production",
        "target_seconds": 10,
        "delta_t": DT,
        "ranks": 2,
        "numerical_depth_m": 0.01,
        "source_cadence": 0.002,
        "animation_cadence": 0.02,
        "volume_cadence": 0.2,
        "physical": {"rho": 1000.0, "nu": 1e-6, "U": [1, 0, 0], "omega": 10.472},
        "input_sha256": {p: digest(t.encode()) for p, t in files.items()},
    }
    times = np.arange(1, 160001) * DT
    ami = (
        "AMI: Creating addressing and weights between 127 source faces and 127 target faces\n"
        "AMI: Patch source sum(weights) min:1 max:1 average:1\n"
        "AMI: Patch target sum(weights) min:1 max:1 average:1\n"
    ) * 4
    log = "SYNTHETIC pimpleFoam\ntrapFpe: Floating point exception trapping enabled\n" + ami
    log += (
        "".join(
            f"Time = {t:.12g}\nPIMPLE: iteration 1\n" + ami + "Courant Number mean: 0.1 max: 0.2\n"
            "Mesh Courant Number mean: 0.1 max: 0.2\nPIMPLE: iteration 2\n"
            + f"MRL2D_POST time={t:.17g} fluid=0.2 meshRaw=0.2 meshEffective=0.2 finite=1\n"
            "ExecutionTime = 1 s  ClockTime = 1 s\n"
            for t in times
        )
        + "End\n"
    )
    mesh_log = (
        "SYNTHETIC checkMesh\ncells: 12\nMesh has 2 geometric directions (1 1 0)\n"
        "Mesh has 2 solution directions (1 1 0)\nMesh OK.\nEnd\n"
    )
    stages = {
        "checkMesh": stage(
            root, "checkMesh", ["checkMesh", "-allGeometry", "-allTopology"], mesh_log
        ),
        "decomposePar": stage(
            root, "decomposePar", ["decomposePar"], "SYNTHETIC decomposePar\nEnd\n"
        ),
        "pimpleFoam": stage(
            root,
            "pimpleFoam",
            [
                "env",
                "LD_LIBRARY_PATH=/synthetic",
                "mpirun",
                "--bind-to",
                "none",
                "-np",
                "2",
                "/synthetic/productionPimpleFoam",
                "-parallel",
            ],
            log,
        ),
        "volumeReadback": stage(
            root,
            "volumeReadback",
            [
                "mpirun",
                "--bind-to",
                "none",
                "-np",
                "2",
                "postProcess",
                "-parallel",
                "-dict",
                "system/readbackDict",
                "-fields",
                "(p U)",
                "-noZero",
            ],
            "SYNTHETIC volumeReadback\n"
            + "".join(
                f"Time = {t:.12g}\nmin(p) = 0\nmax(p) = 1\nmin(mag(U)) = 0\nmax(mag(U)) = 1\n"
                for t in np.arange(1, 51) * 0.2
            )
            + "End\n",
        ),
    }
    source_pins = {}
    for rank in range(2):
        path = root / f"processor{rank}/postProcessing/mrl2dSources/rank{rank}.bin"
        put(path, b"SYNTHETIC header-test only; not a binary history")
        source_pins[str(path)] = digest(path.read_bytes())
    frames = []
    for t in np.arange(1, 501) * 0.02:
        path = root / f"postProcessing/animation/{t:.12g}/section.vtp"
        put(path, plane(t))
        frames.append(
            {"path": str(path), "sha256": digest(path.read_bytes()), "vtk_header_valid": True}
        )
    run = {
        "case": str(root),
        "status": "complete_10s_native_verified",
        "source_history_valid": True,
        "mesh": {"cells": 12, "mesh_ok": True, "geometric_dimensions": 2, "solution_dimensions": 2},
        "stages": stages,
        "flow": {
            "steps": 160000,
            "end": 10.0,
            "native_returncode": 0,
            "native_clock_seconds": 1.0,
            "wall_seconds": 20.0,
            "max_courant": 0.2,
            "max_mesh_courant": 0.2,
        },
        "post_solve": {
            "solved_time": 10.0,
            "post_solve_steps": 160000,
            "max_fluid_courant": 0.2,
            "max_mesh_courant": 0.2,
            "ami_lines": 8 * 160001,
        },
        "source": {
            "schema": "mrl2d-native-source-v1",
            "case": str(root),
            "frames": 5000,
            "rank_files": 2,
            "native_files_sha256": source_pins,
        },
        "animation": frames,
        "animation_frames": 500,
        "volume_frames_read_natively": 50,
    }
    seal(root, manifest, run)
    return root


@pytest.fixture
def native_fixture(seed, tmp_path):
    root = tmp_path / seed.name
    shutil.copytree(seed, root, copy_function=os.link)
    manifest = json.loads((root / "mrl2d-input.json").read_bytes())
    run = json.loads(
        (root / "mrl2d-run.json").read_bytes().replace(str(seed).encode(), str(root).encode())
    )
    spec = seal(root, manifest, run)
    return root, manifest, run, spec


def test_direct_native_adapter_uses_original_records_without_reconstruction(native_fixture):
    root, manifest, run, spec = native_fixture
    original = {
        p.name: p.read_bytes() for p in (root / "mrl2d-input.json", root / "mrl2d-run.json")
    }
    view = validate_completion(spec)
    assert view["schema"] == SCHEMA
    assert view["stages"]["reconstruction"] == {
        "status": "not_required",
        "reason": "direct_native_contour_and_slice",
    }
    assert view["stages"]["solver"]["steps"] == 160000
    assert view["stages"]["volume_readback"]["times_read"] == 50
    assert view["fields_still_require_readback"] is True
    assert len(view["field_frames"]) == 500
    assert view["input_sha256"] == manifest["input_sha256"]
    assert view["original_reports"] == spec["native_completion"]
    assert all((root / name).read_bytes() == data for name, data in original.items())


@pytest.mark.parametrize(
    "corruption",
    [
        "timeout",
        "monitor_error",
        "nonzero",
        "pending",
        "segfault",
        "fpe",
        "fatal",
        "nan",
        "partial_clock",
        "restart_clock",
        "post_solve_missing_middle",
        "post_solve_wrong_time",
        "post_solve_nonfinite",
        "post_solve_over_courant",
        "no_End",
        "wrong_step_count",
        "failed_outer",
        "false_source_valid",
        "bounded_smoke",
        "wrong_flow_endpoint",
        "wrong_post_endpoint",
        "reused_log",
        "wrong_stage_command",
        "wrong_stage_rank",
        "volume_missing_middle",
        "volume_has_no_field_readback",
        "volume_wrong_count",
        "missing_stage",
        "empty_evidence",
        "elapsed_budget",
        "post_summary_mismatch",
    ],
)
def test_production_boundary_rejects_failure_and_partial_evidence(native_fixture, corruption):
    from flowacoustics.mrl2d_postprocess import load_native_case

    root, manifest, run, spec = native_fixture
    solver = run["stages"]["pimpleFoam"]
    log = (root / "log.pimpleFoam").read_text()
    if corruption == "timeout":
        solver["timed_out"] = True
    elif corruption == "monitor_error":
        solver["monitor_error"] = "cancelled by peer"
    elif corruption == "nonzero":
        solver["returncode"] = -15
    elif corruption == "pending":
        solver["status"] = "pending"
    elif corruption in {"segfault", "fpe", "fatal", "nan"}:
        log += {
            "segfault": "Segmentation fault",
            "fpe": "Floating point exception",
            "fatal": "FOAM FATAL",
            "nan": "nan",
        }[corruption] + "\n"
    elif corruption == "partial_clock":
        log = "Time = 9.998\nTime = 10\nEnd\n"
    elif corruption == "restart_clock":
        log = log.replace("Time = 0.5\n", "Time = 0.4999375\n", 1)
    elif corruption.startswith("post_solve_"):
        token = "MRL2D_POST time=0.5 fluid=0.2 meshRaw=0.2 meshEffective=0.2 finite=1\n"
        replacement = {
            "post_solve_missing_middle": "",
            "post_solve_wrong_time": token.replace("time=0.5 ", "time=0.5000625 "),
            "post_solve_nonfinite": token.replace("finite=1", "finite=0"),
            "post_solve_over_courant": token.replace("fluid=0.2", "fluid=1.01"),
        }[corruption]
        assert log.count(token) == 1
        log = log.replace(token, replacement)
    elif corruption == "no_End":
        log = log.removesuffix("End\n")
    elif corruption == "wrong_step_count":
        run["flow"]["steps"] -= 1
    elif corruption == "failed_outer":
        run["status"] = "failed"
    elif corruption == "false_source_valid":
        run["source_history_valid"] = False
    elif corruption == "bounded_smoke":
        run["status"] = "complete_bounded_smoke_native_verified"
    elif corruption == "wrong_flow_endpoint":
        run["flow"]["end"] = 9.999
    elif corruption == "wrong_post_endpoint":
        run["post_solve"]["solved_time"] = 9.999
    elif corruption == "reused_log":
        run["stages"]["volumeReadback"]["log"] = solver["log"]
        run["stages"]["volumeReadback"]["log_sha256"] = solver["log_sha256"]
    elif corruption == "wrong_stage_command":
        solver["argv"][-2] = "/synthetic/decomposePar"
    elif corruption == "wrong_stage_rank":
        solver["argv"][solver["argv"].index("-np") + 1] = "1"
    elif corruption in {"volume_missing_middle", "volume_has_no_field_readback"}:
        volume = root / "log.volumeReadback"
        text = volume.read_text()
        text = (
            text.replace("Time = 0.6\n", "")
            if corruption.endswith("middle")
            else text.replace("min(p) = 0\n", "")
        )
        put(volume, text)
        run["stages"]["volumeReadback"]["log_sha256"] = digest(volume.read_bytes())
    elif corruption == "volume_wrong_count":
        run["volume_frames_read_natively"] = 1
    elif corruption == "missing_stage":
        run["stages"].pop("volumeReadback")
    elif corruption == "empty_evidence":
        run["stages"] = {}
    elif corruption == "elapsed_budget":
        solver["wall_seconds"] = 101
    elif corruption == "post_summary_mismatch":
        run["post_solve"]["max_fluid_courant"] = 0.1
    put(root / "log.pimpleFoam", log)
    solver["log_sha256"] = digest(log.encode())
    spec = seal(root, manifest, run)
    with pytest.raises(ValueError):
        validate_completion(spec)
    with pytest.raises(ValueError):
        load_native_case(spec, mode="production")


def test_contradictory_duplicate_outer_status_is_not_silently_discarded(native_fixture):
    root, manifest, run, spec = native_fixture
    path = root / "mrl2d-run.json"
    text = path.read_text().replace(
        '"status": "complete_10s_native_verified"',
        '"status": "failed", "status": "complete_10s_native_verified"',
    )
    put(path, text)
    spec["native_completion"]["run_sha256"] = digest(path.read_bytes())
    with pytest.raises(ValueError, match="duplicate"):
        validate_completion(spec)


def independent_fingerprint(root, manifest):
    # Independent recipe: actual files, not adapter-returned digests or helpers.
    excluded = {"0/k", "0/omega", "0/nut", "constant/turbulenceProperties"}
    records = []
    for name in sorted(set(manifest["input_sha256"]) - excluded):
        data = (root / name).read_bytes()
        if name == "system/controlDict":
            data = data.replace(b"envelope true;", b"envelope false;")
        records.append(name.encode() + b"\0" + digest(data).encode() + b"\n")
    return digest(b"mrl2d-matched-inputs-v1\n" + b"".join(records))


def test_fingerprint_is_derived_from_actual_pinned_physics(native_fixture):
    root, manifest, run, spec = native_fixture
    view = validate_completion(spec)
    assert view["physical_input_fingerprint"] == independent_fingerprint(root, manifest)
    old = view["physical_input_fingerprint"]
    path = root / "system/fvSchemes"
    put(path, path.read_bytes().replace(b"default backward;", b"default Euler;"))
    manifest["input_sha256"]["system/fvSchemes"] = digest(path.read_bytes())
    changed = validate_completion(seal(root, manifest, run))["physical_input_fingerprint"]
    assert changed == independent_fingerprint(root, manifest) and changed != old


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_input_pin",
        "missing_run_pin",
        "input_mismatch",
        "run_mismatch",
        "empty_inputs",
        "missing_mesh_input",
        "unlisted_input",
        "changed_executed_input",
        "model",
        "envelope",
        "display_id",
        "dt",
        "ranks",
        "source_cadence",
        "animation_cadence",
        "volume_cadence",
        "spec_cadence",
        "spec_envelope",
        "executed_cadence",
        "source_frames",
        "source_rank_coverage",
        "source_empty",
        "frame_missing",
        "frame_duplicate",
        "frame_clock",
        "frame_pin",
        "animation_count",
        "source_other_case",
        "wrong_root",
    ],
)
def test_original_input_and_product_identity_binding(native_fixture, corruption):
    root, manifest, run, spec = native_fixture
    if corruption.startswith("missing_") and corruption.endswith("_pin"):
        spec["native_completion"].pop("input_sha256" if "input" in corruption else "run_sha256")
    elif corruption in {"input_mismatch", "run_mismatch"}:
        spec["native_completion"][corruption.split("_")[0] + "_sha256"] = "0" * 64
    else:
        if corruption == "empty_inputs":
            manifest["input_sha256"] = {}
        elif corruption == "missing_mesh_input":
            manifest["input_sha256"].pop("constant/polyMesh/points")
        elif corruption == "unlisted_input":
            put(root / "system/unlisted", "SYNTHETIC unpinned input")
        elif corruption == "changed_executed_input":
            put(root / "0/U", "SYNTHETIC finite but changed input")
        elif corruption == "model":
            manifest["model"] = "les"
        elif corruption == "envelope":
            manifest["envelope"] = True
        elif corruption == "display_id":
            manifest["name"] = "mrl2d_les_body"
        elif corruption == "dt":
            manifest["delta_t"] = 0.000125
        elif corruption == "ranks":
            manifest["ranks"] = 1
        elif corruption in {"source_cadence", "animation_cadence", "volume_cadence"}:
            manifest[corruption] *= 2
        elif corruption == "executed_cadence":
            path = root / "system/controlDict"
            put(path, path.read_bytes().replace(b"executeInterval 32;", b"executeInterval 64;"))
            manifest["input_sha256"]["system/controlDict"] = digest(path.read_bytes())
        elif corruption == "source_frames":
            run["source"]["frames"] = 2500
        elif corruption == "source_rank_coverage":
            run["source"]["native_files_sha256"].pop(
                next(iter(run["source"]["native_files_sha256"]))
            )
        elif corruption == "source_empty":
            run["source"]["native_files_sha256"] = {}
        elif corruption == "frame_missing":
            run["animation"].pop()
        elif corruption == "frame_duplicate":
            run["animation"][-1] = run["animation"][0]
        elif corruption == "frame_clock":
            run["animation"][40]["path"] = str(root / "postProcessing/animation/0.825/section.vtp")
        elif corruption == "frame_pin":
            run["animation"][40]["sha256"] = "0" * 64
        elif corruption == "animation_count":
            run["animation_frames"] = 1
        elif corruption == "source_other_case":
            run["source"]["case"] = str(root.parent / "other")
        elif corruption == "wrong_root":
            run["case"] = str(root.parent / "other")
        spec = seal(root, manifest, run)
        if corruption == "spec_cadence":
            spec["cadence_s"] = 0.004
        if corruption == "spec_envelope":
            spec["envelope_recorded"] = True
    with pytest.raises(ValueError):
        validate_completion(spec)
