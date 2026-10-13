"""Display-only regression tests; synthetic inputs are never native evidence."""

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from test_cylinder_signature import geometry_fixture, table_fixture

from flowacoustics import signature_analysis as api


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    root = tmp_path_factory.mktemp("signature_display")
    args = table_fixture(root)
    geometry_fixture(args)
    data = api.load_records(**args, start_time_s=2, end_time_s=10)
    # Exercise all three support keys, independently of fixture run metadata.
    data["run"]["time_coverage"]["startup_plot_interval_s"] = [0, 1]
    data["exclusions"] = [[0.01, 0.1]]
    result = api.analyze_records(data, nperseg=200)
    geometry = api.load_geometry(data)
    before = copy.deepcopy((data, result))
    stage = root / "figures"
    stage.mkdir()
    figures = {}
    savefig = Figure.savefig
    close = plt.close

    def capture(fig, path, *args, **kwargs):
        savefig(fig, path, *args, **kwargs)
        if Path(path).suffix == ".pdf":
            figures[Path(path).parent.name] = fig

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Figure, "savefig", capture)
        patch.setattr(plt, "close", lambda *args: None)
        products = api._render_report(data, result, geometry, stage)
    yield dict(
        figures=figures, data=data, result=result, before=before, products=products, stage=stage
    )
    for fig in figures.values():
        close(fig)


def test_phase_axis_keeps_pi_boundaries_clear_without_changing_values(rendered):
    phase = rendered["figures"]["I07_coherence_phase"].axes[1]
    lo, hi = phase.get_ylim()
    assert lo < -np.pi and hi > np.pi, "phase at ±pi must not merge with a spine"
    np.testing.assert_allclose(phase.get_yticks(), [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    assert phase.get_yticklabels()[0].get_text() == "−π"
    for line, method in zip(phase.lines, api.METHODS):
        np.testing.assert_array_equal(
            line.get_ydata(), rendered["result"]["coupling"][method]["display_phase_rad"]
        )
    assert rendered["figures"]["I07_coherence_phase"].axes[0].get_ylim() == (0, 1)


def test_coincident_coupling_curves_have_true_distinct_marker_keys(rendered):
    fig = rendered["figures"]["I07_coherence_phase"]
    handles = fig.legends[0].legend_handles
    for ax, quantity in zip(fig.axes, ["coherence", "display_phase_rad"]):
        lines = ax.lines[:2]
        assert len({line.get_marker() for line in lines}) == 2
        assert all(line.get_marker() not in ("None", "", None) for line in lines)
        assert len({line.get_linestyle() for line in lines}) == 2
        for line, handle, method in zip(lines, handles, api.METHODS):
            assert handle.get_marker() == line.get_marker()
            assert handle.get_linestyle() == line.get_linestyle()
            assert handle.get_color() == line.get_color()
            np.testing.assert_array_equal(
                line.get_ydata(), rendered["result"]["coupling"][method][quantity]
            )
            np.testing.assert_array_equal(
                line.get_xdata(), rendered["result"]["force_spectra"]["lift"]["frequency_Hz"]
            )


def test_directivity_has_absolute_comparison_and_radiating_detail_with_real_keys(rendered):
    fig = rendered["figures"]["I09_directivity"]
    main, detail = fig.axes
    native = [line for line in main.lines if line.get_linestyle() == "None"]
    dense = [line for line in main.lines if line.get_linestyle() != "None"]
    assert len(native) == 8 and all(len(line.get_xdata()) == 1 for line in native)
    assert len(dense) == 2, (
        "main absolute Pa panel must retain both dense curves beside native data"
    )
    direct = rendered["result"]["directivity"]
    np.testing.assert_array_equal(dense[0].get_ydata(), direct["total_rms_Pa"])
    np.testing.assert_array_equal(dense[1].get_ydata(), direct["radiating_rms_Pa"])
    assert len(detail.lines) == 1
    np.testing.assert_array_equal(detail.lines[0].get_xdata(), direct["angles_rad"])
    np.testing.assert_array_equal(detail.lines[0].get_ydata(), direct["radiating_rms_Pa"])
    farmax = np.max(direct["radiating_rms_Pa"])
    assert farmax < detail.get_ylim()[1] < 1.25 * farmax
    assert detail.get_ylim()[1] < main.get_ylim()[1]
    assert all("[Pa]" in ax.get_ylabel() for ax in fig.axes)
    assert "radiating only" in detail.get_title().lower()
    assert "separate" in detail.get_title().lower()
    handles = fig.legends[0].legend_handles
    for mi in range(2):
        assert handles[mi].get_marker() == native[4 * mi].get_marker()
        assert handles[mi].get_linestyle() == "None"
        assert handles[mi].get_markersize() == native[4 * mi].get_markersize()
    for line, (method, observer) in zip(native, [(m, o) for m in api.METHODS for o in api.RING]):
        expected = np.std(rendered["data"]["pressure"][(observer, method)])
        assert line.get_ydata()[0] == expected


def test_record_support_shades_have_explicit_color_matched_keys(rendered):
    fig = rendered["figures"]["I02_record_support"]
    assert len(fig.legends) == 1, "analysis, exclusion and startup shades need an explicit key"
    legend = fig.legends[0]
    keys = {text.get_text(): handle for text, handle in zip(legend.texts, legend.legend_handles)}
    assert set(keys) == {"Analysis interval", "Excluded interval", "Startup context"}
    for ax in fig.axes[:2]:
        spans = dict(zip(["Analysis interval", "Excluded interval", "Startup context"], ax.patches))
        assert len(spans) == 3
        for name, span in spans.items():
            np.testing.assert_array_equal(keys[name].get_facecolor(), span.get_facecolor())


def test_force_psd_limits_use_only_positive_density_in_the_visible_frequency_band(rendered):
    for ax in rendered["figures"]["I03_force_spectra"].axes:
        visible = []
        for line in ax.lines:
            x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
            if len(np.unique(x)) <= 1:  # the candidate marker is not a PSD
                continue
            good = (x >= ax.get_xlim()[0]) & (x <= ax.get_xlim()[1]) & (y > 0)
            visible.extend(y[good])
        loglo, loghi = np.log10([min(visible), max(visible)])
        padding = 0.08 * max(loghi - loglo, 1.0)
        np.testing.assert_allclose(np.log10(ax.get_ylim()), [loglo - padding, loghi + padding])
        assert ax.get_yscale() == "log"
        assert ax.get_ylabel() == "Force PSD [N²/Hz]"


def test_spectrogram_cells_use_disclosed_seam_free_raster_fallback_with_unchanged_values(rendered):
    from matplotlib.collections import QuadMesh

    fig = rendered["figures"]["I05_spectrograms"]
    meshes = [art for ax in fig.axes for art in ax.collections if isinstance(art, QuadMesh)]
    assert len(meshes) == 4  # two scientific fields plus their colorbars
    for art in meshes:
        assert art.get_rasterized(), "vector edge fixes still show seams in the actual Poppler PDF"
        np.testing.assert_array_equal(art.get_edgecolors(), art.get_facecolors())
        assert np.all(art.get_linewidths() == 0)
        assert not np.any(art.get_antialiased())
    spectra = [
        rendered["result"]["force_spectra"]["lift"],
        rendered["result"]["pressure_spectra"][("cross_plus", api.METHODS[1])],
    ]
    for ax, spectrum in zip(fig.axes[:2], spectra):
        keep = (spectrum["frequency_Hz"] > 0) & (spectrum["frequency_Hz"] <= 20)
        np.testing.assert_array_equal(ax.collections[0].get_array(), spectrum["columns"][keep])


def test_band_method_shapes_are_distinct_and_all_marker_keys_match(rendered):
    fig = rendered["figures"]["I06_band_variability"]
    full, blocks = fig.axes
    assert len(full.lines) == 8
    method_markers = [{line.get_marker() for line in full.lines[i : i + 4]} for i in (0, 4)]
    assert method_markers[0].isdisjoint(method_markers[1]), (
        "method identification must not rely on color"
    )
    keys = {
        text.get_text(): handle
        for legend in fig.legends
        for text, handle in zip(legend.texts, legend.legend_handles)
    }
    assert len(keys) == 8
    for line in full.lines:
        key = keys[line.get_label()]
        assert key.get_linestyle() == line.get_linestyle() == "None"
        assert key.get_marker() == line.get_marker()
        assert key.get_color() == line.get_color()
    for method_index in range(2):
        for line in blocks.lines[4 * method_index : 4 * method_index + 4]:
            assert line.get_marker() == full.lines[4 * method_index].get_marker()
            assert line.get_linestyle() == "None"


def test_verification_only_render_keeps_all_inputs_and_nine_figure_coverage(rendered):
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
    assert len(rendered["products"]) == len(rendered["figures"]) == 9
    assert [p["id"][:3] for p in rendered["products"]] == [f"I{i:02d}" for i in range(1, 10)]
    for fig in rendered["figures"].values():
        fig.set_dpi(150)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for legend in fig.legends:
            box = legend.get_window_extent(renderer)
            assert fig.bbox.contains(*box.get_points()[0])
            assert fig.bbox.contains(*box.get_points()[1])
            for ax in fig.axes:
                assert not box.overlaps(ax.title.get_window_extent(renderer))
                assert not box.overlaps(ax.get_window_extent(renderer))


def test_verification_only_psd_lines_retain_full_positive_nyquist_arrays(rendered):
    axes = rendered["figures"]["I03_force_spectra"].axes
    result = rendered["result"]
    groups = [
        [result["force_spectra"][key] for key in ["lift", "drag"]],
        result["segment_sensitivities"],
        [row["spectrum"] for row in result["sensitivity_windows"] if "spectrum" in row],
        [result["force_spectra"]["lift"]],
    ]
    for ax, spectra in zip(axes, groups):
        for line, spectrum in zip(ax.lines, spectra):
            keep = (spectrum["frequency_Hz"] > 0) & (spectrum["density"] > 0)
            np.testing.assert_array_equal(line.get_xdata(), spectrum["frequency_Hz"][keep])
            np.testing.assert_array_equal(line.get_ydata(), spectrum["density"][keep])


def test_verification_only_i05_manifest_discloses_raster_fallback(rendered):
    import json

    path = rendered["stage"] / "I05_spectrograms/manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["rendering"]["pdf"] == (
        "mixed vector text/axes with rasterized QuadMesh fields and colorbars"
    )
    assert manifest["rendering"]["raster_dpi"] == 160
    assert manifest["rendering"]["scientific_arrays_changed"] is False
