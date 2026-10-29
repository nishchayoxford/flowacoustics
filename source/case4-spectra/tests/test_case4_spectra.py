"""Offline-only Case 4 spectral reductions and plotting regressions."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy.signal import periodogram, windows

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_case4_spectra.py"


def implementation():
    assert SCRIPT.is_file(), "Case 4 spectral module is missing"
    spec = spec_from_file_location("case4_spectra", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coefficient_psd_matches_independent_scipy_periodogram():
    module = implementation()
    n, dt = 4000, 0.002
    time = 2.002 + np.arange(n) * dt
    local_time = np.arange(n) * dt
    rng = np.random.default_rng(141)
    # Off-bin content and broadband deterministic-seed noise expose taper errors.
    signal = np.column_stack(
        [
            2.3 * np.cos(2 * np.pi * 5 * local_time + 0.3),
            1.7 * np.sin(2 * np.pi * 7.37 * local_time) + 0.2 * rng.normal(size=n),
        ]
    ) + np.array([6.0, -2.0])
    taper = windows.hann(n, sym=False)
    coeff = np.fft.rfft((signal - signal.mean(axis=0)) * taper[:, None], axis=0)
    coeff = coeff[1:-1] / taper.sum()
    rms, psd, estimator = module.coefficient_spectra(coeff, time)
    independent_f, independent_psd = periodogram(
        signal,
        fs=1 / dt,
        window="hann",
        detrend="constant",
        scaling="density",
        return_onesided=True,
        axis=0,
    )
    assert_allclose(psd, independent_psd[1:-1], rtol=3e-12, atol=2e-25)
    assert_allclose(rms, np.sqrt(2) * np.abs(coeff), rtol=2e-15)
    assert_allclose(estimator["sample_interval_s"], dt, rtol=1e-14)
    assert_allclose(estimator["frequency_spacing_hz"], independent_f[1], rtol=1e-14)
    assert_allclose(estimator["enbw_hz"], 0.1875, rtol=1e-14)
    assert estimator["sample_count"] == n
    assert estimator["window"] == "periodic Hann"
    # Density is not the coherent-bin mean-square divided by bare df.
    assert_allclose(psd * estimator["enbw_hz"], rms**2, rtol=3e-15)
    assert not np.allclose(psd, rms**2 / estimator["frequency_spacing_hz"])


def test_peak_table_handles_zero_exact_ties_and_band_edges():
    module = implementation()
    frequency = np.array([0.625, 1.0, 5.0, 25.0])
    angles = np.array([0.0, 30.0, 60.0, 90.0])
    rms = np.array([[0, 3, 1, 1], [0, 2, 4, 2], [0, 3, 4 - 1e-13, 3], [0, 1, 2, 4]], dtype=float)
    psd = rms**2 / 0.1875
    peaks = module.peak_table(frequency, angles, rms, psd)
    assert peaks[0] == {
        "angle_deg": 0.0,
        "index": None,
        "frequency_hz": None,
        "bin_rms_pa": 0.0,
        "psd_pa2_per_hz": 0.0,
        "is_band_edge": False,
        "tied_maximum_count": 0,
    }
    assert [p["index"] for p in peaks] == [None, 0, 1, 3]
    assert [p["tied_maximum_count"] for p in peaks] == [0, 2, 1, 1]
    assert [p["is_band_edge"] for p in peaks] == [False, True, False, True]
    for j, peak in enumerate(peaks[1:], start=1):
        index = peak["index"]
        assert peak["angle_deg"] == angles[j]
        assert peak["frequency_hz"] == frequency[index]
        assert peak["bin_rms_pa"] == rms[index, j]
        assert peak["psd_pa2_per_hz"] == psd[index, j]
        assert index == np.argmax(psd[:, j])


def test_band_levels_use_one_micropascal_and_null_zero_without_a_floor():
    module = implementation()
    import pytest

    levels = module.band_levels([0, 1e-200, 1e-6, 1e-5, 1])
    assert levels[0] is None
    assert_allclose(levels[1:], [-3880, 0, 20, 120], rtol=1e-15, atol=1e-12)
    with pytest.raises(ValueError, match="nonnegative finite"):
        module.band_levels([-1])
    with pytest.raises(ValueError, match="nonnegative finite"):
        module.band_levels([np.nan])


def test_authenticated_native_summary_checks_all_bins_and_all_twelve_observers(tmp_path):
    import hashlib
    import json
    import math
    import shutil

    import pytest

    module = implementation()
    inputs = [
        "site/data/mrl2d/mrl2d_les_permeable.npz",
        "site/data/results.json",
        "site/data/mrl2d/public-manifest.json",
        "src/flowacoustics/acoustics2d.py",
    ]
    before = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs}
    summary = module.build_summary(ROOT)
    assert summary["schema"] == "mrl2d-case4-spectral-v1"
    assert summary["case_id"] == "mrl2d_les_permeable"
    assert len(summary["peaks"]) == 12
    assert np.asarray(summary["bin_rms_pa"]).shape == (196, 12)
    assert np.asarray(summary["psd_pa2_per_hz"]).shape == (196, 12)
    assert_allclose(summary["angles_deg"], np.arange(12) * 30, atol=1e-12)
    estimator = summary["estimator"]
    native = np.load(ROOT / inputs[0], allow_pickle=False)
    assert summary["frequency_hz"] == native["frequency_hz"].tolist()
    assert summary["band_rms_pa"] == native["band_rms_pa"].tolist()
    assert estimator["sample_window_s"] == native["time_s"][[0, -1]].tolist()
    assert estimator["band_hz"] == native["frequency_hz"][[0, -1]].tolist()
    assert estimator["window"] == "periodic Hann"
    n = len(native["time_s"])
    dt = (native["time_s"][-1] - native["time_s"][0]) / (n - 1)
    # Independent scalar math: no reuse of the production vector reductions.
    weights = [0.5 - 0.5 * math.cos(2 * math.pi * i / n) for i in range(n)]
    enbw = math.fsum(w * w for w in weights) / dt / math.fsum(weights) ** 2
    df = 1 / (n * dt)
    for j in range(12):
        coherent = 0.0
        integrated = 0.0
        for k, coefficient in enumerate(native["coefficients_pa"][:, j]):
            magnitude2 = coefficient.real**2 + coefficient.imag**2
            expected_rms = math.sqrt(2 * magnitude2)
            expected_psd = 2 * magnitude2 / enbw
            assert summary["bin_rms_pa"][k][j] == pytest.approx(expected_rms, rel=5e-15)
            assert summary["psd_pa2_per_hz"][k][j] == pytest.approx(expected_psd, rel=5e-15)
            coherent += 2 * magnitude2
            integrated += expected_psd * df
        assert estimator["coherent_band_mean_square_pa2"][j] == pytest.approx(coherent)
        assert estimator["psd_band_integral_pa2"][j] == pytest.approx(integrated)
        assert integrated / coherent == pytest.approx(2 / 3, rel=1e-14)
        assert np.mean(native["pressure_pa"][:, j] ** 2) == pytest.approx(coherent, rel=1e-12)
        assert summary["band_level_db_re_1upa"][j] == pytest.approx(
            20 * math.log10(native["band_rms_pa"][j] / 1e-6), rel=1e-14
        )
        index = int(np.argmax(np.abs(native["coefficients_pa"][:, j])))
        assert summary["peaks"][j]["index"] == index
        assert summary["peaks"][j]["frequency_hz"] == native["frequency_hz"][index]
        assert index == int(np.argmax(np.asarray(summary["psd_pa2_per_hz"])[:, j]))
    assert {p["path"]: p["sha256"] for p in summary["provenance"]["sources"]} == before
    assert "single-record" in estimator["psd_definition"]
    assert "stationarity" in estimator["psd_definition"]
    assert "ENBW" in estimator["psd_definition"]
    assert all(name in estimator for name in ["bin_rms_definition", "band_level_definition"])
    assert "/home/" not in json.dumps(summary, allow_nan=False)
    assert before == {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs}
    for p in inputs:
        target = tmp_path / p
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / p, target)
    with (tmp_path / inputs[0]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        module.build_summary(tmp_path)


def test_spectral_reduction_rejects_invalid_time_or_coefficients():
    import pytest

    module = implementation()
    coeff = np.ones((2, 1), dtype=complex)
    for bad_time in ([0, 1, 2.1], [0], [0, 1, 0], [0, np.nan, 2], [[0, 1], [2, 3]]):
        with pytest.raises(ValueError, match="time"):
            module.coefficient_spectra(coeff, bad_time)
    for bad_coeff in ([1, 2], [[np.nan]], np.empty((0, 1)), np.empty((2, 0))):
        with pytest.raises(ValueError, match="coefficients"):
            module.coefficient_spectra(bad_coeff, np.arange(8) * 0.002)


def test_numerical_downloads_roundtrip_all_native_values(tmp_path):
    import csv
    import json

    module = implementation()
    summary = module.build_summary(ROOT)
    files = module.export_tables(summary, tmp_path)
    assert set(files) == {"summary.json", "spectra.csv", "peaks.csv", "directivity.csv"}
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    with (tmp_path / "spectra.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12 * 196
    assert list(rows[0]) == ["angle_deg", "frequency_hz", "bin_rms_pa", "psd_pa2_per_hz"]
    for j in range(12):
        for k in range(196):
            row = rows[j * 196 + k]
            assert float(row["angle_deg"]) == summary["angles_deg"][j]
            assert float(row["frequency_hz"]) == summary["frequency_hz"][k]
            assert float(row["bin_rms_pa"]) == summary["bin_rms_pa"][k][j]
            assert float(row["psd_pa2_per_hz"]) == summary["psd_pa2_per_hz"][k][j]
    with (tmp_path / "peaks.csv").open(newline="") as stream:
        peaks = list(csv.DictReader(stream))
    with (tmp_path / "directivity.csv").open(newline="") as stream:
        directivity = list(csv.DictReader(stream))
    assert len(peaks) == len(directivity) == 12
    for j in range(12):
        assert int(peaks[j]["index"]) == summary["peaks"][j]["index"]
        for key in ("angle_deg", "frequency_hz", "bin_rms_pa", "psd_pa2_per_hz"):
            assert float(peaks[j][key]) == summary["peaks"][j][key]
        assert int(peaks[j]["tied_maximum_count"]) == summary["peaks"][j]["tied_maximum_count"]
        assert (peaks[j]["is_band_edge"] == "True") == summary["peaks"][j]["is_band_edge"]
        assert float(directivity[j]["band_rms_pa"]) == summary["band_rms_pa"][j]
        assert float(directivity[j]["band_level_db_re_1upa"]) == summary["band_level_db_re_1upa"][j]


def synthetic_summary():
    """Self-contained plot fixture; native files are not a layout dependency."""
    module = implementation()
    frequency = np.linspace(0.625, 25, 196)
    angles = np.arange(12) * 30
    rms = np.column_stack(
        [(1 + j / 12) * (0.1 + np.exp(-(((frequency - (4 + j)) / 0.4) ** 2))) for j in range(12)]
    )
    psd = rms**2 / 0.1875
    band = np.sqrt(np.sum(rms**2, axis=0))
    return {
        "case_id": "synthetic_layout_fixture",
        "frequency_hz": frequency.tolist(),
        "angles_deg": angles.tolist(),
        "bin_rms_pa": rms.tolist(),
        "psd_pa2_per_hz": psd.tolist(),
        "band_rms_pa": band.tolist(),
        "band_level_db_re_1upa": module.band_levels(band),
        "peaks": module.peak_table(frequency, angles, rms, psd),
        "estimator": {
            "sample_window_s": [2.002, 10],
            "band_hz": [0.625, 25],
            "enbw_hz": 0.1875,
            "frequency_spacing_hz": 0.125,
        },
        "provenance": {"receiver_radius_m": 1.0},
    }


def test_dual_directivity_uses_twelve_unconnected_absolute_pa_and_db_samples():
    import matplotlib.pyplot as plt

    module = implementation()
    summary = synthetic_summary()
    figure = module.directivity_figure(summary)
    assert len(figure.axes) == 2
    figure.canvas.draw()
    for axis, key in zip(figure.axes, ["band_rms_pa", "band_level_db_re_1upa"], strict=True):
        assert axis.name == "polar"
        assert len(axis.collections) == 1
        assert len(axis.lines) == 0
        points = axis.collections[0].get_offsets()
        assert points.shape == (12, 2)
        assert_allclose(points[:, 0], np.deg2rad(summary["angles_deg"]))
        assert_allclose(points[:, 1], summary[key])
    db_axis = figure.axes[1]
    assert all("dB" in label.get_text() for label in db_axis.get_yticklabels())
    for tick, label in zip(db_axis.get_yticks(), db_axis.get_yticklabels(), strict=True):
        assert float(label.get_text().replace(" dB", "")) == tick
    assert "1 µPa" in db_axis.get_title()
    assert all("Maximum" in text.get_text() for axis in figure.axes for text in axis.texts)
    plt.close(figure)


def test_twelve_panel_spectra_preserve_samples_shared_axes_and_peak_labels():
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    module = implementation()
    summary = synthetic_summary()
    for metric, unit in [("bin_rms_pa", "Pa"), ("psd_pa2_per_hz", "Pa²/Hz")]:
        figure = module.spectra_figure(summary, metric)
        assert len(figure.axes) == 12
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for j, axis in enumerate(figure.axes):
            assert axis.get_subplotspec().get_gridspec().get_geometry() == (4, 3)
            assert axis.get_xlim() == figure.axes[0].get_xlim()
            assert axis.get_ylim() == figure.axes[0].get_ylim()
            assert_allclose(axis.lines[0].get_xdata(), summary["frequency_hz"])
            assert_allclose(axis.lines[0].get_ydata(), np.asarray(summary[metric])[:, j])
            peak = summary["peaks"][j]
            assert_allclose(
                axis.collections[0].get_offsets(), [[peak["frequency_hz"], peak[metric]]]
            )
            assert f"{peak['frequency_hz']:.5g} Hz" in axis.get_title()
            assert f"{peak[metric]:.5g} {unit}" in axis.get_title()
            assert f"{summary['angles_deg'][j]:g}°" in axis.get_title()
            assert axis.title.get_window_extent(renderer).y0 > axis.bbox.y1
        # Titles, footnotes and labels must lie within the final export canvas.
        for text in figure.findobj(Text):
            if (
                text.get_visible()
                and text.get_text()
                and text
                not in [
                    label
                    for axis in figure.axes
                    for label in [*axis.get_xticklabels(), *axis.get_yticklabels()]
                ]
            ):
                bbox = text.get_window_extent(renderer)
                assert bbox.x0 >= 0 and bbox.y0 >= 0
                assert bbox.x1 <= figure.bbox.x1 and bbox.y1 <= figure.bbox.y1
        plt.close(figure)


def test_native_bundle_has_hash_bound_tables_and_three_readable_vector_pdf_pages(tmp_path):
    import hashlib
    import json
    import subprocess
    import time

    from PIL import Image

    module = implementation()
    output = tmp_path / "first"
    manifest = module.export_bundle(ROOT, output)
    expected = {
        "summary.json",
        "spectra.csv",
        "peaks.csv",
        "directivity.csv",
        "plots.pdf",
        "directivity.png",
        "bin-rms.png",
        "psd.png",
    }
    assert set(manifest["outputs"]) == expected
    assert manifest["manifest_self_excluded"] is True
    assert manifest["counts"] == {
        "observers": 12,
        "frequencies": 196,
        "spectra_rows": 2352,
        "peaks_rows": 12,
        "directivity_rows": 12,
        "pdf_pages": 3,
    }
    assert json.loads((output / "manifest.json").read_text()) == manifest
    for name, entry in manifest["outputs"].items():
        assert entry["sha256"] == hashlib.sha256((output / name).read_bytes()).hexdigest()
        assert entry["bytes"] == (output / name).stat().st_size
    info = subprocess.run(
        ["pdfinfo", str(output / "plots.pdf")], capture_output=True, text=True, check=True
    ).stdout
    assert "Pages:           3" in info
    text = subprocess.run(
        ["pdftotext", "-layout", str(output / "plots.pdf"), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pages = [page for page in text.split("\f") if page.strip()]
    assert len(pages) == 3
    assert "Band level (dB re 1 µPa)" in pages[0]
    assert "Coherent bin RMS" in pages[1]
    assert "One-sided Hann PSD" in pages[2]
    for page in pages[1:]:
        assert page.count("° · peak") == 12
        assert "stationarity claim" in page
        assert "0.1875 Hz" in page and "0.125 Hz" in page
    image_objects = subprocess.run(
        ["pdfimages", "-list", str(output / "plots.pdf")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert len(image_objects.splitlines()) == 2  # header only: all vector
    for name in ["directivity.png", "bin-rms.png", "psd.png"]:
        with Image.open(output / name) as image:
            assert image.width >= 2000 and image.height >= 1200
            assert np.asarray(image.convert("RGB")).std() > 10
    time.sleep(1.1)  # Delayed regeneration catches creation timestamps in PDFs.
    second = module.export_bundle(ROOT, tmp_path / "second")
    assert second == manifest
    for name in expected:
        assert (output / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


def test_cli_runs_real_export_into_explicit_additive_directory(tmp_path):
    import json
    import subprocess
    import sys

    output = tmp_path / "cli-export"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(ROOT), "--output", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert (output / "summary.json").is_file(), "CLI did not generate the numerical artifact"
    assert (output / "plots.pdf").is_file()
    receipt = json.loads(result.stdout)
    assert receipt["counts"]["observers"] == 12
    assert receipt["counts"]["pdf_pages"] == 3


def test_polar_maximum_notes_do_not_collide_with_figure_footer():
    import matplotlib.pyplot as plt

    figure = implementation().directivity_figure(synthetic_summary())
    figure.set_dpi(180)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        for note in axis.texts:
            box = note.get_window_extent(renderer)
            for text in figure.texts:
                assert not box.overlaps(text.get_window_extent(renderer)), (
                    "Polar maximum note overlaps the footer or heading"
                )
    plt.close(figure)
