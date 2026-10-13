"""Schema translation must not turn an unfinished native run into evidence."""

import importlib
import importlib.util

import pytest


def adapter():
    name = "flowacoustics.native_run_adapter"
    assert importlib.util.find_spec(name) is not None, "Verified native-run adapter is missing"
    return importlib.import_module(name)


def completed_fixture():
    return {
        "schema_version": 2,
        "status": "completed_native_reconstructed",
        "evidence_kind": "native_execution",
        "reconstruction": {"status": "completed", "times_s": [2, 4, 6, 8, 10]},
        "exports": {"status": "completed"},
        "solver_verification": {"solver_reached_target": True, "end_s": 10},
        "time_coverage": {"requested_end_s": 10, "end_s": 10},
        "observer_coordinates_m": {"cross_plus": [0, 1, 0], "cross_double": [0, 2, 0]},
        "clock": {
            "forces_offset_s": 0,
            "libacoustics_curle": {"offset_s": 0, "kind": "source_unretarded"},
            "libacoustics_fwh": {"offset_s": 0, "kind": "observer"},
        },
        "force_convention": "on_body",
        "force_content": "total_traction",
    }


def test_completed_v2_maps_observers_without_changing_clocks_or_source():
    import copy

    source = completed_fixture()
    before = copy.deepcopy(source)
    mapped = adapter().normalize_run(source)
    assert source == before
    assert mapped["schema_version"] == 1
    assert mapped["source_schema_version"] == 2
    assert mapped["observers"] == [
        {"name": "cross_plus", "position_m": [0, 1, 0]},
        {"name": "cross_double", "position_m": [0, 2, 0]},
    ]
    assert mapped["clock"] == source["clock"]
    assert mapped["status"] == source["status"]


@pytest.mark.parametrize("state", ["running", "failed", "reconstructing", "exporting"])
def test_incomplete_run_is_never_promoted(state):
    source = completed_fixture()
    source["status"] = state
    with pytest.raises(ValueError, match="completed"):
        adapter().normalize_run(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("evidence_kind", "native_execution_in_progress"),
        ("reconstruction", {"status": "pending"}),
        ("exports", {"status": "pending"}),
        ("solver_verification", {"solver_reached_target": False, "end_s": 10}),
        ("time_coverage", {"requested_end_s": 10, "end_s": 6}),
        ("observer_coordinates_m", {}),
    ],
)
def test_completed_label_cannot_override_contradictory_evidence(field, value):
    source = completed_fixture()
    source[field] = value
    with pytest.raises(ValueError):
        adapter().normalize_run(source)


def test_verified_exports_returns_exact_pinned_bytes_and_rejects_corruption(tmp_path):
    import hashlib

    source = completed_fixture()
    force = tmp_path / "forces.csv"
    observer = tmp_path / "observers.csv"
    force.write_bytes(b"test-only force fixture\n")
    observer.write_bytes(b"test-only observer fixture\n")
    source["exports"]["sha256"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (force, observer)
    }
    module = adapter()
    assert hasattr(module, "verified_exports"), "Hash-bound table reader is missing"
    normalized, force_bytes, observer_bytes = module.verified_exports(source, force, observer)
    assert normalized["schema_version"] == 1
    assert force_bytes == force.read_bytes()
    assert observer_bytes == observer.read_bytes()
    observer.write_bytes(b"changed finite-looking test-only values\n")
    with pytest.raises(ValueError, match="identity"):
        module.verified_exports(source, force, observer)
    del source["exports"]["sha256"][str(force.resolve())]
    with pytest.raises(ValueError, match="identity"):
        module.verified_exports(source, force, observer)


def test_analysis_cli_consumes_v2_without_rewriting_native_manifest(tmp_path):
    """The in-memory metadata/forces here are test fixtures, never CFD evidence."""
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

    import numpy as np
    from test_cylinder_acoustics import fixture_forces, fixture_run

    root = Path(__file__).resolve().parents[1]
    force, observer, run_path = (
        tmp_path / name for name in ("forces.csv", "observers.csv", "run.json")
    )
    np.savetxt(
        force,
        fixture_forces()[:5121],
        delimiter=",",
        comments="",
        header="time_s,force_x_N,force_y_N,force_z_N",
    )
    observer.write_text("time_s,observer,method,pressure_Pa\n")
    source = {**fixture_run(), **completed_fixture()}
    source["exports"]["sha256"] = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (force, observer)
    }
    run_path.write_text(json.dumps(source))
    original = run_path.read_bytes()
    output = tmp_path / "analysis"
    command = [
        sys.executable,
        str(root / "scripts/analyze_cylinder.py"),
        "--run",
        str(run_path),
        "--forces",
        str(force),
        "--observers",
        str(observer),
        "--output",
        str(output),
        "--nperseg",
        "512",
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    assert run_path.read_bytes() == original
    manifest = json.loads((output / "manifest.json").read_text())
    assert next(i for i in manifest["inputs"] if i["role"] == "run")["sha256"] == (
        hashlib.sha256(original).hexdigest()
    )
    assert "native_run_adapter.py" in {i["file"] for i in manifest["implementation"]}
    assert (output / "downloads/forces.csv").read_bytes() == force.read_bytes()


def test_native_v2_conventions_cannot_be_replaced_by_a_sidecar(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(completed_fixture()))
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/analyze_cylinder.py"),
            "--run",
            str(run_path),
            "--forces",
            str(tmp_path / "absent-forces.csv"),
            "--observers",
            str(tmp_path / "absent-observers.csv"),
            "--conventions",
            str(tmp_path / "absent-sidecar.json"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 2
    assert "cannot override" in result.stderr
    assert not output.exists()
