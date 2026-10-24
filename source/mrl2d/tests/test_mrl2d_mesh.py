"""Contract tests for geometry-built one-layer mesh, not retired cases."""

import importlib
import os
from pathlib import Path

import numpy as np
import pytest


def test_native_quality_summary_keeps_unrelaxed_metrics():
    module = mesh_module()
    text = """
    cells: 54
    Mesh has 2 geometric (non-empty/wedge) directions (1 1 0)
    Mesh has 2 solution (non-empty) directions (1 1 0)
    Max aspect ratio = 3.75 OK.
    Mesh non-orthogonality Max: 44.37 average: 4.68
    Max skewness = 0.889 OK.
    Min/max edge length = 5.98e-05 0.0339 OK.
    Cell determinant (wellposedness) : minimum: 0.0141 average: 0.263
    Face interpolation weight : minimum: 0.108 average: 0.471
    Face volume ratio : minimum: 0.121 average: 0.895
    Mesh OK.
    """
    result = module._check_summary(text)
    assert "quality" in result, "Native quality metrics must be retained"
    assert result["quality"]["max_nonorthogonality_deg"] == 44.37
    assert result["quality"]["min_determinant"] == 0.0141
    with pytest.raises(ValueError, match="did not admit"):
        module._check_summary(text + "Failed 1 mesh checks.")
    with pytest.raises(ValueError, match="x-y-only"):
        module._check_summary(text.replace("2 solution", "3 solution"))


def test_builder_cli_exposes_fresh_case_and_report():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/build_mrl2d_mesh.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--case" in result.stdout
    assert "--report" in result.stdout


def mesh_module():
    return importlib.import_module("flowacoustics.mrl2d_mesh")


def test_geometry_built_base_is_one_layer_with_disjoint_regions(tmp_path):
    module = mesh_module()
    case = tmp_path / "base"
    result = module.build_base(case)
    assert 1000 < result["cells"] <= 250000
    assert result["span_layers"] == 1
    assert result["depth_m"] == 0.01
    points = module.read_points(case / "constant/polyMesh/points")
    assert set(points[:, 2]) == {-0.005, 0.005}
    assert result["patches"]["frontAndBack"]["type"] == "empty"
    assert result["patches"]["frontAndBack"]["faces"] == 2 * result["cells"]
    groups = np.load(case / "mesh_reference.npz")
    assert len(groups["point_region"]) == len(points)
    assert set(groups["point_region"]) == set(range(5))
    assert set(result["zones"]) == {"carrierZone", "blade0Zone", "blade1Zone", "blade2Zone"}
    for i in range(3):
        assert result["patches"][f"blade{i}"]["type"] == "wall"
    for interface in ["carrier", "blade0", "blade1", "blade2"]:
        for side in ["master", "slave"]:
            patch = result["patches"][f"{interface}AMI_{side}"]
            assert patch["type"] == "cyclicAMI"
            assert patch["faces"] > 50
    assert result["farfield_max_radius_error_m"] < 1e-12
    text = (case / "constant/dynamicMeshDict").read_text()
    assert "multiSolidBodyMotionSolver" in text
    assert text.count("solidBodyMotionFunction multiMotion;") == 3
    assert text.count("omega -10.472;") == 4
    assert text.count("omega 5.236;") == 3
    assert Path(result["case"]) == case


def test_native_ordering_and_elliptical_blade_angles(tmp_path):
    module = mesh_module()
    case = tmp_path / "ordered"
    result = module.build_base(case)
    import re

    mesh = case / "constant/polyMesh"

    def labels(name):
        text = (mesh / name).read_text().split("}", 1)[1]
        return np.array(list(map(int, re.findall(r"\d+", text)[1:])))

    owner, neighbour = labels("owner"), labels("neighbour")
    pairs = list(zip(owner[: len(neighbour)], neighbour, strict=True))
    assert pairs == sorted(pairs), "Native internal faces must be upper-triangular ordered"
    points = module.read_points(mesh / "points")
    faces_text = (mesh / "faces").read_text().split("}", 1)[1]
    faces = [list(map(int, line.split())) for line in re.findall(r"\d+\(([^)]*)\)", faces_text)]
    for i, angle in enumerate(module.ANGLES):
        patch = result["patches"][f"blade{i}"]
        selected = faces[patch["startFace"] : patch["startFace"] + patch["faces"]]
        ids = sorted({node for face in selected for node in face})
        xy = points[ids, :2] - module.CENTRES[i]
        radians = np.radians(angle)
        along = xy @ np.array([np.cos(radians), np.sin(radians)])
        across = xy @ np.array([-np.sin(radians), np.cos(radians)])
        residual = (along / 0.035) ** 2 + (across / 0.0035) ** 2 - 1
        assert np.max(np.abs(residual)) < 1e-10, f"Blade {i} has incorrect initial chord angle"


def test_native_mesh_and_motion_admission(tmp_path):
    module = mesh_module()
    assert callable(getattr(module, "verify_native", None)), "native verifier missing"
    case = Path(os.environ.get("MRL2D_MESH_TEST_CASE", str(tmp_path / "native")))
    module.build_base(case)
    result = module.verify_native(case)
    assert result["admitted"] is True
    assert result["native_cells"] == result["cells"]
    assert result["native_directions"] == {"geometric": [1, 1, 0], "solution": [1, 1, 0]}
    assert len(result["motion"]["states"]) == 12
    assert result["motion"]["states"][-1]["time_s"] == 0.6
    assert result["motion"]["max_xy_error_m"] < 1e-12
    assert result["motion"]["max_z_error_m"] <= 8 * np.finfo(float).eps * module.DEPTH
    assert result["motion"]["z_roundoff_tolerance_m"] == 8 * np.finfo(float).eps * module.DEPTH
    assert result["motion"]["all_faces_covered"]
    assert all(state["full_check_mesh_ok"] for state in result["motion"]["states"])
    assert all(stage["return_code"] == 0 for stage in result["native_stages"])
