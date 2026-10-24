"""Runtime-only continuation controls; no production-case mutation."""

import re

from flowacoustics import mrl2d_runtime as runtime
from flowacoustics.mrl2d_case import case_files, case_specs


def test_setup_stage_never_receives_endpoint_readback_gate():
    assert hasattr(runtime, "stage_kind"), "setup stage classification missing"
    assert runtime.stage_kind("decomposePar") == "setup"
    assert runtime.stage_kind("volumeReadback") == "readback"
    assert runtime.stage_kind("flow") == "flow"
    assert runtime.stage_kind("finalMesh") == "mesh"


def test_trial_files_change_only_endpoint_volume_write():
    assert hasattr(runtime, "trial_files"), "endpoint readback controls missing"
    spec = case_specs()[1]
    original = case_files(spec, depth=0.01, dt=0.0000625, end=0.02, ranks=2)
    actual = runtime.trial_files(spec, dt=0.0000625, end=0.02, ranks=2)
    assert actual.keys() == original.keys()
    for key in original:
        if key != "system/controlDict":
            assert actual[key] == original[key]
    old = original["system/controlDict"]
    new = actual["system/controlDict"]
    assert re.sub(r"writeInterval 320;", "writeInterval 3200;", new, count=1) == old
    assert "executeInterval 32;" in new
    assert "writeInterval 320;" in new
