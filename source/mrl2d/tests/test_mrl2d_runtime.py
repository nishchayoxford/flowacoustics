"""Focused runtime diagnostics; synthetic snippets are parser unit fixtures only."""

import importlib.util

import pytest


def test_budget_refuses_launch_when_exhausted(tmp_path):
    module = runtime()
    assert hasattr(module, "NativeBudget"), "bounded native ledger missing"
    ledger = module.NativeBudget(tmp_path / "ledger.json", limit=0)
    with pytest.raises(RuntimeError, match="budget"):
        ledger.run(["false"], cwd=tmp_path, log=tmp_path / "must-not-run", timeout=1)
    assert not (tmp_path / "must-not-run").exists()


def runtime():
    assert importlib.util.find_spec("flowacoustics.mrl2d_runtime") is not None, (
        "runtime diagnostics helper missing"
    )
    from flowacoustics import mrl2d_runtime

    return mrl2d_runtime


def test_candidate_size_policy_changes_only_four_frozen_literals():
    from pathlib import Path

    module = runtime()
    assert hasattr(module, "candidate_mesh_source"), "bounded mesh size policy missing"
    original = (Path(__file__).parents[1] / "src/flowacoustics/mrl2d_mesh.py").read_text()
    candidate = module.candidate_mesh_source(original)
    assert '"Mesh.MeshSizeMin", 0.0005' in candidate
    assert '"SizeMin": 0.0008' in candidate
    assert candidate.count('"DistMax": 0.10') == 2
    restored = candidate.replace(
        '"Mesh.MeshSizeMin", 0.0005', '"Mesh.MeshSizeMin", 0.00006'
    ).replace('"SizeMin": 0.0008', '"SizeMin": 0.0004')
    restored = restored.replace('"DistMax": 0.10', '"DistMax": 0.20', 1)
    restored = restored.replace('"DistMax": 0.10', '"DistMax": 0.25', 1)
    assert restored == original


def test_instrumentation_is_observational_only():
    module = runtime()
    assert hasattr(module, "instrument_solver"), "observational solver instrumentation missing"
    original = '#include "fvCFD.H"\nint main() {\n        runTime.write();\n}\n'
    observed = module.instrument_solver(original)
    assert (
        observed.replace('#include "runtimeMetrics.H"\n', "").replace(
            "        runtimeMetrics(mesh, U, phi);\n", ""
        )
        == original
    )
    assert observed.count("runtimeMetrics(mesh, U, phi);") == 1


def test_courant_records_keep_pre_step_time_and_duplicate_initials():
    module = runtime()
    text = """Courant Number mean: 0.1 max: 11
Courant Number mean: 0.1 max: 11
Courant Number mean: 0.1 max: 11
Time = 0.00025
Mesh Courant Number mean: 0.01 max: 9
ExecutionTime = 0.4 s ClockTime = 0 s
Courant Number mean: 0.1 max: 49
Time = 0.0005
Mesh Courant Number mean: 0.01 max: 9.1
End
"""
    result = module.courant_records(text)
    assert [r["state_time"] for r in result["fluid"]] == [0, 0, 0, 0.00025]
    assert [r["state_time"] for r in result["mesh"]] == [0.00025, 0.0005]
    assert result["normal_end"]
    assert result["fluid"][-1]["maximum"] == 49
