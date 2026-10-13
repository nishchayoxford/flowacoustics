"""Renderer-level regressions for the two native-10 s label collisions.

The default fixture is wholly synthetic; native integration is opt-in. Geometry
has the actual dimensional aspect, and the analytic polar profile added below
is a layout stress case, not fabricated solver evidence.
"""

import copy
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.text import Text
from test_cylinder_signature import geometry_fixture, table_fixture

from flowacoustics import signature_analysis as api


@pytest.fixture(scope="module", params=["synthetic", "native"])
def rendered(request, tmp_path_factory):
    root = tmp_path_factory.mktemp("signature_final_layout")
    if request.param == "native":
        if os.environ.get("FLOWACOUSTICS_VERIFY_SIGNATURE_LAYOUT") != "1":
            pytest.skip("set FLOWACOUSTICS_VERIFY_SIGNATURE_LAYOUT=1 for native integration")
        project = Path(__file__).resolve().parents[1]
        native = project / "results/cylinder/native_circular_60s"
        args = dict(
            run_path=native / "run.json",
            forces_path=native / "forces.csv",
            observers_path=native / "observers.csv",
            case=project / "cfd/cylinder_circular_60s",
            project=project,
        )
        nperseg = 3200
    else:
        args = table_fixture(root)
        geometry_fixture(args)
        nperseg = 200
    data = api.load_records(**args, start_time_s=2, end_time_s=10)
    result = api.analyze_records(data, nperseg=nperseg)
    if request.param == "synthetic":
        # Analytic tilted covariance ellipse at native-like scale. No generated
        # production tables are required by the portable layout regression.
        theta = result["directivity"]["angles_rad"] - np.deg2rad(12)
        result["directivity"]["radiating_rms_Pa"] = np.hypot(
            2.27e-7 * np.sin(theta), 8.5e-8 * np.cos(theta)
        )
    geometry = api.load_geometry(data)
    before = copy.deepcopy((data, result))
    stage = root / "figures"
    stage.mkdir()
    figures = {}
    savefig, close = Figure.savefig, plt.close

    def capture(fig, path, *args, **kwargs):
        savefig(fig, path, *args, **kwargs)
        if Path(path).suffix == ".pdf":
            figures[Path(path).parent.name] = fig

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Figure, "savefig", capture)
        patch.setattr(plt, "close", lambda *args: None)
        api._render_report(data, result, geometry, stage)
    yield dict(figures=figures, data=data, result=result, before=before)
    for fig in figures.values():
        close(fig)


def body_collisions(fig):
    """Compare drawn z text with projected axis/tick strokes and other text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = next(ax for ax in fig.axes if ax.name == "3d")
    axes = (ax.xaxis, ax.yaxis, ax.zaxis)
    strokes = [axis.line for axis in axes] + [
        tick.tick1line for axis in axes for tick in axis.get_major_ticks()
    ]
    labels = [tick.label1 for tick in ax.zaxis.get_major_ticks()]
    other_text = [axis.label for axis in axes] + [
        tick.label1 for axis in axes[:2] for tick in axis.get_major_ticks()
    ]
    collisions = []
    for i, label in enumerate(labels):
        box = label.get_window_extent(renderer).padded(1)
        for line in strokes:
            path = line.get_path().transformed(line.get_transform())
            if line.get_visible() and path.intersects_bbox(box, filled=False):
                collisions.append((label.get_text(), "axis/tick stroke"))
        for other in labels[i + 1 :] + other_text:
            if other.get_visible() and other.get_text():
                if box.overlaps(other.get_window_extent(renderer)):
                    collisions.append((label.get_text(), other.get_text()))
    return collisions


@pytest.mark.parametrize("dpi", [150, 160])
def test_body_z_labels_clear_projected_strokes_and_title(rendered, dpi):
    fig = rendered["figures"]["I01_geometry"]
    fig.set_dpi(dpi)
    assert not body_collisions(fig), body_collisions(fig)
    ax = next(ax for ax in fig.axes if ax.name == "3d")
    geometry = rendered["data"]["run"]["geometry"]
    diameter, span = geometry["diameter_m"], geometry["span_m"]
    np.testing.assert_allclose(
        ax.get_box_aspect() / ax.get_box_aspect()[0], [1, 1, span / diameter]
    )
    np.testing.assert_array_equal(ax.get_xlim(), [-diameter / 2, diameter / 2])
    np.testing.assert_array_equal(ax.get_ylim(), [-diameter / 2, diameter / 2])
    np.testing.assert_array_equal(ax.get_zlim(), [-span / 2, span / 2])


def test_body_collision_detector_rejects_original_zero_padding(rendered):
    fig = rendered["figures"]["I01_geometry"]
    ax = next(ax for ax in fig.axes if ax.name == "3d")
    original_pad = ax.zaxis.get_major_ticks()[0].get_pad()
    try:
        ax.tick_params(axis="z", pad=0)
        assert body_collisions(fig), "original projected-spine collision escaped the detector"
    finally:
        ax.tick_params(axis="z", pad=original_pad)
        fig.canvas.draw()


def radial_collisions(fig):
    """Every drawn detail radial label versus every scientific line and text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    detail = fig.axes[1]
    labels = [
        tick.label1
        for tick in detail.yaxis.get_major_ticks()
        if detail.get_ylim()[0] < tick.get_loc() <= detail.get_ylim()[1]
    ]
    dormant = {
        text
        for ax in fig.axes
        for tick in ax.yaxis.get_major_ticks()
        if not ax.get_ylim()[0] < tick.get_loc() <= ax.get_ylim()[1]
        for text in (tick.label1, tick.label2)
    }
    texts = [
        text
        for text in fig.findobj(Text)
        if text.get_visible() and text.get_text() and text not in dormant
    ]
    collisions = []
    for label in labels:
        box = label.get_window_extent(renderer)
        for ax in fig.axes:
            for line in ax.lines:
                if not line.get_visible():
                    continue
                # Treat even dashed paths as continuous: no reliance on a gap.
                path = line.get_path().transformed(line.get_transform())
                margin = 1 + line.get_linewidth() * fig.dpi / 144
                if path.intersects_bbox(box.padded(margin), filled=False):
                    collisions.append((label.get_text(), line.get_label()))
        for other in texts:
            if other is not label and box.padded(1).overlaps(other.get_window_extent(renderer)):
                collisions.append((label.get_text(), other.get_text()))
    return collisions


@pytest.mark.parametrize("dpi", [150, 160])
def test_radiating_radial_labels_clear_all_scientific_lines_and_text(rendered, dpi):
    fig = rendered["figures"]["I09_directivity"]
    fig.set_dpi(dpi)
    assert not radial_collisions(fig), radial_collisions(fig)
    main, detail = fig.axes
    direct = rendered["result"]["directivity"]
    np.testing.assert_array_equal(detail.lines[0].get_xdata(), direct["angles_rad"])
    np.testing.assert_array_equal(detail.lines[0].get_ydata(), direct["radiating_rms_Pa"])
    np.testing.assert_array_equal(main.lines[-2].get_ydata(), direct["total_rms_Pa"])
    np.testing.assert_array_equal(main.lines[-1].get_ydata(), direct["radiating_rms_Pa"])
    assert detail.get_ylim() == (0, float(direct["radiating_rms_Pa"].max()) * 1.10)
    assert detail.get_yscale() == "linear"
    assert detail.get_ylabel() == "AC RMS pressure [Pa]"
    assert main.get_rlabel_position() == 35  # The shared absolute panel is untouched.


def test_radial_collision_detector_rejects_original_35_degree_angle(rendered):
    fig = rendered["figures"]["I09_directivity"]
    ax = fig.axes[1]
    original_angle = ax.get_rlabel_position()
    try:
        ax.set_rlabel_position(35)
        collisions = radial_collisions(fig)
        assert ("1.0", "Compact radiating only") in collisions, collisions
    finally:
        ax.set_rlabel_position(original_angle)
        fig.canvas.draw()


def test_verification_only_layout_render_does_not_mutate_scientific_inputs(rendered):
    def equal(before, after):
        if isinstance(before, np.ndarray):
            np.testing.assert_array_equal(before, after)
        elif isinstance(before, dict):
            assert before.keys() == after.keys()
            for key in before:
                equal(before[key], after[key])
        elif isinstance(before, (list, tuple)):
            assert len(before) == len(after)
            for a, b in zip(before, after):
                equal(a, b)
        elif isinstance(before, (float, np.floating)) and np.isnan(before):
            assert np.isnan(after)
        else:
            assert before == after

    equal(rendered["before"], (rendered["data"], rendered["result"]))
