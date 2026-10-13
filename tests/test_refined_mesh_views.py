"""Standalone mesh-view contracts; box fixtures are NOT native validation."""

import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def renderer():
    path = ROOT / "scripts/render_cylinder_mesh.py"
    assert path.exists(), "standalone renderer missing"
    spec = importlib.util.spec_from_file_location("render_cylinder_mesh", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def box_mesh(tmp_path):
    """Test-only closed box, not a cylinder, not evidence of native validity."""
    mesh = tmp_path / "case/constant/polyMesh"
    mesh.mkdir(parents=True)
    points = [
        (-0.01, -0.01, -0.03),
        (0.01, -0.01, -0.03),
        (0.01, 0.01, -0.03),
        (-0.01, 0.01, -0.03),
        (-0.01, -0.01, 0.03),
        (0.01, -0.01, 0.03),
        (0.01, 0.01, 0.03),
        (-0.01, 0.01, 0.03),
        (-0.24, -0.24, 0),
        (0.24, -0.24, 0),
        (0.24, 0.24, 0),
        (-0.24, 0.24, 0),
        (-0.24, -0.24, -0.08),
        (0.24, -0.24, -0.08),
        (0.24, 0.24, 0.08),
        (-0.24, 0.24, 0.08),
    ]
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (8, 9, 10, 11),
        (12, 13, 14, 15),
    ]
    (mesh / "points").write_text(
        f"{len(points)}\n(\n"
        + "\n".join("(" + " ".join(map(str, p)) + ")" for p in points)
        + "\n)\n"
    )
    (mesh / "faces").write_text(
        f"{len(faces)}\n(\n"
        + "\n".join("4(" + " ".join(map(str, f)) + ")" for f in faces)
        + "\n)\n"
    )
    (mesh / "boundary").write_text(
        "3 ( cylinder {nFaces 6; startFace 0;} "
        "spanEnds {nFaces 1; startFace 6;} "
        "farfield {nFaces 1; startFace 7;} )"
    )
    pins = {
        name: hashlib.sha256((mesh / name).read_bytes()).hexdigest()
        for name in ("points", "faces", "boundary")
    }
    return mesh, pins, np.array(points), np.array(faces)


def test_read_hash_bound_quad_mesh_without_reindexing(box_mesh):
    mesh, pins, points, faces = box_mesh
    n = renderer()
    assert hasattr(n, "load_mesh"), "hash-bound reader missing"
    data = n.load_mesh(mesh, pins)
    np.testing.assert_array_equal(data["points"], points)
    np.testing.assert_array_equal(data["faces"], faces)
    np.testing.assert_array_equal(data["patch_ids"]["cylinder"], np.arange(6))
    np.testing.assert_array_equal(data["patch_ids"]["farfield"], [7])
    assert data["native_mesh_sha256"] == pins
    assert not data["points"].flags.writeable
    assert not data["faces"].flags.writeable


@pytest.fixture
def fixture_run(box_mesh):
    base = ROOT / "results/cylinder/mesh_views"
    base.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".test-refined-", dir=base))
    try:
        mesh, pins, _, _ = box_mesh
        case = work / "case"
        shutil.copytree(mesh, case / "constant/polyMesh")
        log = work / "checkMesh.log"
        log.write_text("points: 16\nfaces: 8\ncells: 1\nMesh OK.\n")
        record = {
            "case_path": str(case),
            "cell_count": 1,
            "evidence_kind": "synthetic_test_fixture",
            "input_sha256": {str(case / "constant/polyMesh" / name): h for name, h in pins.items()},
        }
        run = work / "run.json"
        run.write_text(json.dumps(record))
        (work / "mesh_audit.json").write_text(
            json.dumps(
                {
                    "cell_count": 1,
                    "wall_geometry": {
                        "closed": True,
                        "faces": 6,
                        "cap_minus_faces": 1,
                        "cap_plus_faces": 1,
                    },
                    "checkMesh_command": {
                        "log": str(log),
                        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                    },
                }
            )
        )
        yield work, run
    finally:
        shutil.rmtree(work)


def test_preflight_and_run_binding_use_recorded_mesh_counts(fixture_run):
    work, run = fixture_run
    n = renderer()
    assert hasattr(n, "prepare_inputs"), "source/output preflight and run binding missing"
    data = n.prepare_inputs(run, work / "out")
    assert data["cell_count"] == 1  # Must not hardcode native production count.
    assert data["counts"]["points"] == 16
    assert data["counts"]["faces"] == 8
    assert not (work / "out").exists()
    for invalid in (ROOT / "cfd/not-an-output", work / "out/../escape", work):
        with pytest.raises(ValueError, match="output|parent|existing"):
            n.prepare_inputs(ROOT / "nonexistent-run.json", invalid)
    with pytest.raises(ValueError, match="source"):
        n.prepare_inputs(Path("/no-such-external-run.json"), work / "out")
    linked = work / "linked"
    linked.symlink_to(work, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        n.prepare_inputs(run, linked / "out")
    assert not (work / "out").exists()


def test_geometry_and_export_size_layout_keep_all_native_arrays(fixture_run):
    work, run = fixture_run
    n = renderer()
    assert hasattr(n, "select_geometry"), "native display selection missing"
    assert hasattr(n, "build_figures"), "readable figure builder missing"
    data = n.prepare_inputs(run, work / "out")
    geom = n.select_geometry(data)
    assert geom["info"]["wall_faces"] == 6
    assert geom["info"]["cap_minus_faces"] == geom["info"]["cap_plus_faces"] == 1
    assert geom["info"]["section_faces"] == 1
    assert geom["info"]["section_edges"] == 4
    assert geom["info"]["wall_closed_edge_incidence"] is True
    before = {k: v.copy() for k, v in geom.items() if isinstance(v, np.ndarray)}
    source_before = (data["points"].copy(), data["faces"].copy())
    figures = n.build_figures(data, geom)
    try:
        assert set(figures) == {"mesh_cross_section", "mesh_3d"}
        for fig in figures.values():
            fig.canvas.draw()
            audit = n.text_layout_audit(fig)
            assert audit["collisions"] == [], audit
            assert audit["outside_canvas"] == [], audit
            assert audit["minimum_clearance_px"] >= 4, audit
        fig = figures["mesh_3d"]
        assert tuple(fig.canvas.get_width_height()) == (2800, 1500)
        assert len(fig.axes) == 2
        np.testing.assert_allclose(
            fig.axes[0].get_box_aspect() / fig.axes[0].get_box_aspect()[0], [1, 1, 1 / 3]
        )
        np.testing.assert_allclose(
            fig.axes[1].get_box_aspect() / fig.axes[1].get_box_aspect()[0], [1, 1, 0.09 / 0.032]
        )
        for k, v in before.items():
            np.testing.assert_array_equal(geom[k], v)
        np.testing.assert_array_equal(data["points"], source_before[0])
        np.testing.assert_array_equal(data["faces"], source_before[1])
    finally:
        import matplotlib.pyplot as plt

        for fig in figures.values():
            plt.close(fig)


def test_deterministic_six_artifact_export_and_no_clobber(fixture_run):
    import time

    work, run = fixture_run
    n = renderer()
    assert hasattr(n, "render"), "staged deterministic mesh export missing"
    first = work / "first"
    report = n.render(run, first)
    expected = {
        f"{stem}.{suffix}"
        for stem in ("mesh_cross_section", "mesh_3d")
        for suffix in ("png", "pdf", "svg")
    }
    assert {p.name for p in first.iterdir()} == expected | {"manifest.json"}
    assert set(report["artifact_sha256"]) == expected
    assert report["mesh"]["cells"] == 1
    assert report["source_evidence_kind"] == "synthetic_test_fixture"
    assert report["geometry_unchanged_after_export"] is True
    for stem in ("mesh_cross_section", "mesh_3d"):
        assert (first / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        pdf = (first / f"{stem}.pdf").read_bytes()
        assert b"/CreationDate" not in pdf and b"/ModDate" not in pdf
        assert "<svg" in (first / f"{stem}.svg").read_text()
    time.sleep(1.1)  # A different timestamp second, not a false reproducibility pass.
    second = work / "replay"
    n.render(run, second)
    for name in expected | {"manifest.json"}:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    before = {p.name: p.read_bytes() for p in first.iterdir()}
    with pytest.raises(ValueError, match="existing"):
        n.render(run, first)
    assert before == {p.name: p.read_bytes() for p in first.iterdir()}


def test_cli_renders_fixture_and_rejects_outside_before_source_read(fixture_run, tmp_path):
    import os
    import subprocess
    import sys

    work, run = fixture_run
    args = [
        sys.executable,
        str(ROOT / "scripts/render_cylinder_mesh.py"),
        "--run",
        str(run),
        "--output",
        str(work / "cli"),
    ]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1")
    completed = subprocess.run(args, capture_output=True, text=True, env=env)
    assert completed.returncode == 0, completed.stderr
    assert (work / "cli/manifest.json").exists(), "CLI produced no manifest"
    rejected = subprocess.run(
        args[:2] + ["--run", str(work / "missing.json"), "--output", str(tmp_path / "forbidden")],
        capture_output=True,
        text=True,
        env=env,
    )
    assert rejected.returncode != 0 and "output" in rejected.stderr
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "empty_pins",
        "missing_pin",
        "changed_bytes",
        "nonquad",
        "bad_count",
        "bad_index",
        "patch_range",
    ],
)
def test_reader_rejection_controls(box_mesh, corruption):
    mesh, pins, _, _ = box_mesh
    n = renderer()
    if corruption == "empty_pins":
        pins = {}
    elif corruption == "missing_pin":
        pins.pop("faces")
    elif corruption == "changed_bytes":
        path = mesh / "points"
        path.write_text(path.read_text().replace("0.24", "0.25"))
    else:
        name = "boundary" if corruption == "patch_range" else "faces"
        path = mesh / name
        text = path.read_text()
        if corruption == "nonquad":
            text = text.replace("4(0 3 2 1)", "3(0 3 2)")
        elif corruption == "bad_count":
            text = text.replace("8\n", "7\n", 1)
        elif corruption == "bad_index":
            text = text.replace("4(0 3 2 1)", "4(0 3 2 99)")
        else:
            text = text.replace("startFace 7", "startFace 8")
        path.write_text(text)
        pins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        n.load_mesh(mesh, pins)


def test_projected_text_audit_catches_changed_drawn_label(fixture_run):
    work, run = fixture_run
    n = renderer()
    data = n.prepare_inputs(run, work / "out")
    figs = n.build_figures(data, n.select_geometry(data))
    fig = figs["mesh_3d"]
    assert n.text_layout_audit(fig)["collisions"] == []
    fig.axes[1].set_xlabel("x [m] changed visible axis label " * 20)
    bad = n.text_layout_audit(fig)
    assert bad["collisions"] or bad["outside_canvas"], "dormant-label false pass"


def test_accepted_log_and_cap_count_mismatch_rejected(fixture_run):
    work, run = fixture_run
    n = renderer()
    record = json.loads(run.read_text())
    record["cell_count"] = 2
    run.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="count"):
        n.prepare_inputs(run, work / "out")
    record["cell_count"] = 1
    run.write_text(json.dumps(record))
    data = n.prepare_inputs(run, work / "out")
    data["wall_audit"]["cap_plus_faces"] = 0
    with pytest.raises(ValueError, match="wall/cap"):
        n.select_geometry(data)
    assert not (work / "out").exists()


def test_drawn_3d_ticks_clear_projected_spines_and_tick_strokes(fixture_run):
    work, run = fixture_run
    n = renderer()
    data = n.prepare_inputs(run, work / "out")
    fig = n.build_figures(data, n.select_geometry(data))["mesh_3d"]
    fig.canvas.draw()
    collisions = []
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            strokes = [axis.line] + [t.tick1line for t in axis.get_major_ticks()]
            for tick in axis.get_major_ticks():
                box = tick.label1.get_window_extent(fig.canvas.get_renderer()).padded(4)
                for line in strokes:
                    path = line.get_path().transformed(line.get_transform())
                    if line.get_visible() and path.intersects_bbox(box, filled=False):
                        collisions.append((axis.axis_name, tick.label1.get_text()))
    assert not collisions, f"Visible numeric labels intersect projected axis strokes: {collisions}"
    assert n.text_layout_audit(fig)["text_axis_stroke_collisions"] == []
    fig.axes[1].tick_params(axis="z", pad=3)
    assert n.text_layout_audit(fig)["text_axis_stroke_collisions"], "spine collision not audited"


def test_atomic_publication_does_not_replace_competing_empty_directory(fixture_run):
    work, _ = fixture_run
    n = renderer()
    stage, target = work / "stage", work / "target"
    stage.mkdir()
    target.mkdir()
    (stage / "payload").write_bytes(b"retained staged bytes")
    original_inode = target.stat().st_ino
    with pytest.raises(FileExistsError):
        n._publish_no_replace(stage, target)
    assert target.stat().st_ino == original_inode
    assert (stage / "payload").read_bytes() == b"retained staged bytes"


def test_real_refined_native_mesh_and_saved_views():
    """Opt-in real integration, separate from all synthetic geometry fixtures."""
    import os

    if os.environ.get("FLOWACOUSTICS_VERIFY_REFINED_MESH") != "1":
        pytest.skip("set FLOWACOUSTICS_VERIFY_REFINED_MESH=1 for native integration")
    n = renderer()
    run = ROOT / "results/cylinder/native_circular_60s/run.json"
    target = ROOT / "results/cylinder/mesh_views/refined_10s"
    data = n.prepare_inputs(run, target.parent / "unused-verification-target")
    geom = n.select_geometry(data)
    record = json.loads(run.read_text())
    assert data["cell_count"] == record["cell_count"] == 459264
    assert data["counts"] == {"points": 473514, "faces": 1392000, "cells": 459264}
    assert geom["info"]["wall_faces"] == 8064
    assert geom["info"]["cap_minus_faces"] == geom["info"]["cap_plus_faces"] == 1344
    assert geom["info"]["wall_closed_edge_incidence"]
    assert geom["info"]["section_faces"] == 4224
    assert geom["info"]["section_edges"] == 8544
    assert geom["info"]["point_bounds_m"] == [[-0.24, -0.24, -0.08], [0.24, 0.24, 0.08]]
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["native_mesh_sha256"] == data["native_mesh_sha256"]
    assert manifest["geometry_array_sha256"] == n._geometry_hashes(data, geom)
    for name, sha in manifest["artifact_sha256"].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == sha
    assert len(manifest["artifact_sha256"]) == 6
    assert manifest["geometry_unchanged_after_export"]
    for audit in manifest["layout"].values():
        assert not audit["collisions"] and not audit["outside_canvas"]
        assert not audit["text_axis_stroke_collisions"]
        assert audit["minimum_clearance_px"] >= 4
