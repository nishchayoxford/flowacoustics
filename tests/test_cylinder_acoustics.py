"""Analytical fixtures only: none of these waveforms are CFD evidence."""

import importlib

import numpy as np
import pytest


def model():
    return importlib.import_module("flowacoustics.cylinder_acoustics")


def test_compact_curle_has_retarded_body_force_sign_and_si_units():
    t = np.arange(20001) / 10000
    omega, amplitude, c, r = 2 * np.pi * 7, 3.0, 100.0, 2.0
    force = np.column_stack((np.zeros_like(t), amplitude * np.sin(omega * t), t * 0))
    arrival = t[100:-100] + r / c
    result = model().compact_curle(t, force, arrival, [0, r, 0], c, force_convention="on_body")
    tau = arrival - r / c
    near = -amplitude * np.sin(omega * tau) / (4 * np.pi * r**2)
    far = -amplitude * omega * np.cos(omega * tau) / (4 * np.pi * c * r)
    np.testing.assert_allclose(result["near_Pa"], near, atol=1e-12)
    np.testing.assert_allclose(result["far_Pa"], far, atol=2e-7)
    np.testing.assert_allclose(result["pressure_Pa"], near + far, atol=2e-7)
    opposite = model().compact_curle(t, -force, arrival, [0, r, 0], c, force_convention="on_fluid")
    np.testing.assert_allclose(opposite["pressure_Pa"], result["pressure_Pa"])
    np.testing.assert_allclose(result["source_time_s"], tau)


@pytest.mark.parametrize(
    "change",
    [
        "past",
        "future",
        "duplicate",
        "complex",
        "nan",
        "shape",
        "zero_r",
        "negative_c",
        "bad_sign",
        "few",
    ],
)
def test_compact_curle_rejects_invalid_or_unsupported_inputs(change):
    args = dict(
        time_s=np.arange(10.0) / 10,
        force_N=np.ones((10, 3)),
        observer_time_s=np.array([0.3, 0.4]),
        observer_m=[1, 0, 0],
        sound_speed_m_s=10,
        force_convention="on_body",
    )
    if change == "past":
        args["observer_time_s"] = [0]
    elif change == "future":
        args["observer_time_s"] = [2]
    elif change == "duplicate":
        args["time_s"][3] = args["time_s"][2]
    elif change == "complex":
        args["force_N"] = args["force_N"].astype(complex)
    elif change == "nan":
        args["force_N"][1, 0] = np.nan
    elif change == "shape":
        args["force_N"] = np.ones((10, 2))
    elif change == "zero_r":
        args["observer_m"] = [0, 0, 0]
    elif change == "negative_c":
        args["sound_speed_m_s"] = -1
    elif change == "bad_sign":
        args["force_convention"] = "auto"
    elif change == "few":
        args["time_s"], args["force_N"] = [0, 1], np.ones((2, 3))
    with pytest.raises(ValueError):
        model().compact_curle(**args)


def test_near_far_distance_laws_at_identical_retarded_time():
    t = np.linspace(0, 3, 3001)
    force = np.column_stack((np.sin(2 * np.pi * t), 0 * t, 0 * t))
    tau = t[10:-10]
    a = model().compact_curle(t, force, tau + 1 / 100, [1, 0, 0], 100, force_convention="on_fluid")
    b = model().compact_curle(t, force, tau + 2 / 100, [2, 0, 0], 100, force_convention="on_fluid")
    np.testing.assert_allclose(a["near_Pa"], 4 * b["near_Pa"], atol=1e-14)
    np.testing.assert_allclose(a["far_Pa"], 2 * b["far_Pa"], atol=1e-14)


def test_spectral_bundle_window_parseval_and_explicit_resampling():
    t = np.arange(4096) / 256
    x = 2 * np.sin(2 * np.pi * 8 * t)
    spec = model().spectral_bundle(t, x, nperseg=256)
    assert spec["integrated_power"] == pytest.approx(2, rel=1e-12)
    assert spec["frequency_Hz"][np.argmax(spec["psd"])] == pytest.approx(8)
    assert spec["segments"] == 31
    assert spec["window"] == "hann"
    assert spec["spectrogram_psd"].shape == (129, 31)
    np.testing.assert_allclose(np.sum(spec["spectrogram_psd"], axis=0), 2)
    irregular = t.copy()
    irregular[1::2] += 0.0001
    with pytest.raises(ValueError, match="resampl"):
        model().spectral_bundle(irregular, x, nperseg=256)
    explicit = model().spectral_bundle(irregular, x, nperseg=512, resample_dt_s=1 / 512)
    assert explicit["resampled"] is True
    assert explicit["time_s"][-1] <= irregular[-1]
    assert explicit["dt_s"] == pytest.approx(1 / 512)
    with pytest.raises(ValueError, match="downsampl"):
        model().spectral_bundle(irregular, x, nperseg=256, resample_dt_s=0.1)
    # Boxcar exact Parseval and Hann window-weighted Parseval are distinct.
    from scipy.signal import get_window

    from flowacoustics.analysis import integrated_power, welch_psd

    y = np.random.default_rng(7).normal(size=256)
    f, density = welch_psd(y, 256, 256)
    w = get_window("hann", 256)
    expected = np.sum(((y - y.mean()) * w) ** 2) / np.sum(w**2)
    assert integrated_power(f, density) == pytest.approx(expected, rel=1e-12)


def test_cross_spectrum_phase_retardation_and_validity_gates():
    t = np.arange(4096) / 256
    x = np.sin(2 * np.pi * 8 * t)
    delay = 0.015625
    y = np.sin(2 * np.pi * 8 * (t - delay))
    cross = model().coherence_phase(t, x, y, nperseg=256)
    i = np.argmin(abs(cross["frequency_Hz"] - 8))
    assert cross["coherence"][i] == pytest.approx(1)
    assert cross["phase_rad"][i] == pytest.approx(-2 * np.pi * 8 * delay)
    assert cross["phase_valid"][i]
    assert cross["status"] == "descriptive"
    assert not cross["phase_valid"][0]
    short = model().coherence_phase(t[:256], x[:256], y[:256], nperseg=256)
    assert short["status"] == "insufficient_segments"
    assert not np.any(short["phase_valid"])
    silent = model().coherence_phase(t, x, 0 * y, nperseg=256)
    assert not np.any(silent["phase_valid"])
    assert np.all(np.isfinite(silent["coherence"]))


def fixture_run():
    return {
        "schema_version": 1,
        "status": "analytical_fixture",
        "evidence_kind": "analytical_fixture",
        "medium": {
            "density_kg_m3": 1000,
            "sound_speed_m_s": 1500,
            "velocity_m_s": 0.2,
            "kinematic_viscosity_m2_s": 1e-6,
        },
        "geometry": {"diameter_m": 0.02, "span_m": 0.06},
        "force_convention": "on_body",
        "force_content": "total_traction",
        "observers": [
            {"name": "lift", "position_m": [0, 2, 0]},
            {"name": "lift_far", "position_m": [0, 4, 0]},
            {"name": "drag", "position_m": [2, 0, 0]},
        ],
        "clock": {
            "forces_offset_s": 0,
            "libacoustics_curle": {"offset_s": 0, "kind": "source_unretarded"},
            "libacoustics_fwh": {"offset_s": 0, "kind": "observer"},
        },
    }


def fixture_forces():
    # Resolve interpolation/derivative truncation below the independent oracle tolerance.
    t = np.arange(8192) / 512
    return np.column_stack(
        (t, 0.02 + 0.001 * np.cos(4 * np.pi * 2 * t), 0.01 * np.sin(2 * np.pi * 2 * t), t * 0)
    )


def test_force_dataset_coefficients_and_measured_strouhal_not_target():
    data, aligned = model().build_dataset(fixture_run(), fixture_forces(), [], nperseg=1024)
    charts = {chart["id"]: chart for chart in data["charts"]}
    assert {"forces", "force_coefficients", "force_psd"} <= charts.keys()
    assert data["status"] == "analytical_fixture"
    assert all(c["origin"] == "analytical" for c in charts.values())
    assert data["metadata"]["shedding"]["frequency_Hz"] == pytest.approx(2)
    assert data["metadata"]["shedding"]["strouhal"] == pytest.approx(0.2)
    assert data["metadata"]["reynolds_number"] == pytest.approx(4000)
    cd = charts["force_coefficients"]["traces"][0]["y"]
    assert np.mean(cd) == pytest.approx(0.02 / (0.5 * 1000 * 0.2**2 * 0.02 * 0.06))
    assert aligned == []
    assert data["metadata"]["chart_count"] == len(data["charts"])
    assert all("samples" in chart for chart in data["charts"])


def fixture_observers(run, forces):
    rows = []
    t = forces[:, 0]
    omega, c = 4 * np.pi, run["medium"]["sound_speed_m_s"]
    for obs in run["observers"]:
        xyz = np.array(obs["position_m"])
        r = np.linalg.norm(xyz)
        n = xyz / r
        for method in ("libacoustics_curle", "libacoustics_fwh"):
            tau = t if method.endswith("curle") else t - r / c
            projected = n[0] * (0.02 + 0.001 * np.cos(2 * omega * tau)) + n[1] * 0.01 * np.sin(
                omega * tau
            )
            derivative = -n[0] * 0.002 * omega * np.sin(2 * omega * tau) + n[
                1
            ] * 0.01 * omega * np.cos(omega * tau)
            sign = 1 if method.endswith("curle") else -1
            p = sign * (projected / r**2 + derivative / (c * r)) / (4 * np.pi)
            rows.extend(
                dict(time_s=float(tt), observer=obs["name"], method=method, pressure_Pa=float(pp))
                for tt, pp in zip(t, p)
            )
    return rows


def test_observer_comparisons_preserve_signed_clocks_and_exclude_startup():
    run, force = fixture_run(), fixture_forces()
    data, aligned = model().build_dataset(run, force, fixture_observers(run, force), nperseg=1024)
    charts = {chart["id"]: chart for chart in data["charts"]}
    assert {
        "pressure_lift",
        "pressure_psd_lift",
        "spectrogram_lift",
        "coherence_lift",
        "phase_lift",
    } <= charts.keys()
    pressure = charts["pressure_lift"]
    traces = {trace["name"]: trace for trace in pressure["traces"]}
    native = traces["libacoustics_fwh"]
    compact = traces["compact_curle"]
    np.testing.assert_array_equal(native["x"], compact["x"])
    np.testing.assert_allclose(native["y"], compact["y"], atol=5e-8)
    assert np.corrcoef(traces["libacoustics_curle"]["y"], compact["y"])[0, 1] < -0.99
    assert pressure["samples"]["excluded"] > 0
    assert all(row["source_time_s"] >= force[1, 0] for row in aligned)
    assert any(check["name"] == "native_curle_convention" for check in data["checks"])
    assert data["metadata"]["aligned_row_count"] == len(aligned)
    assert data["metadata"]["comparisons"]["lift"]["gain_fitted"] is False
    assert data["metadata"]["comparisons"]["lift"]["delay_fitted"] is False


def test_directivity_uses_actual_equal_radius_observers_and_distance_retardation():
    run, force = fixture_run(), fixture_forces()
    data, _ = model().build_dataset(run, force, fixture_observers(run, force), nperseg=1024)
    charts = {chart["id"]: chart for chart in data["charts"]}
    polar = [c for c in charts.values() if c["id"].startswith("directivity_")]
    assert len(polar) == 1  # 2 m ring only; lone 4 m observer isn't a swept directivity.
    assert len(polar[0]["traces"][0]["x"]) == 2
    assert sorted(polar[0]["traces"][0]["x"]) == [0, 90]
    pairs = data["metadata"]["distance_pairs"]
    fwh = next(p for p in pairs if p["method"] == "libacoustics_fwh")
    assert (fwh["near_observer"], fwh["far_observer"]) == ("lift", "lift_far")
    assert fwh["arrival_delay_difference_s"] == pytest.approx(2 / 1500)
    assert fwh["rms_ratio_far_over_near"] == pytest.approx(0.25, rel=0.002)
    assert fwh["reference_far_1_over_r"] == pytest.approx(0.5)
    assert fwh["reference_near_1_over_r2"] == pytest.approx(0.25)
    assert fwh["alignment"] == "common_source_time"


def test_point_source_phasors_and_separate_lessons():
    ca = model()
    f, c, rho, r = 2.0, 1500.0, 1000.0, 3.0
    omega, k = 2 * np.pi * f, 2 * np.pi * f / c
    green = np.exp(-1j * k * r) / (4 * np.pi * r)
    assert ca.point_source_phasor("monopole", [r, 0, 0], f, c, rho, 1e-6) == pytest.approx(
        1j * omega * rho * 1e-6 * green
    )
    assert ca.point_source_phasor("dipole", [r, 0, 0], f, c, rho, [1, 0, 0]) == pytest.approx(
        green * (1 / r + 1j * k)
    )
    assert ca.point_source_phasor("dipole", [0, r, 0], f, c, rho, [1, 0, 0]) == 0
    tensor = np.diag([1.0, -1.0, 0.0])
    assert ca.point_source_phasor("quadrupole", [r, 0, 0], f, c, rho, tensor) == pytest.approx(
        green * (-k * k + 3j * k / r + 3 / r**2)
    )
    for kind in ("unknown", "monopole"):
        with pytest.raises(ValueError):
            ca.point_source_phasor(kind, [0, 0, 0], f, c, rho, 1)
    data, _ = ca.build_dataset(fixture_run(), fixture_forces(), [], nperseg=1024)
    lessons = [chart for chart in data["charts"] if chart["id"].startswith("lesson_")]
    assert len(lessons) == 4
    assert all(chart["origin"] == "analytical" for chart in lessons)
    assert all("not computed cylinder" in chart["description"] for chart in lessons)


def test_cli_writes_finite_contract_raw_downloads_and_hash_manifest(tmp_path):
    import csv
    import hashlib
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "analyze_cylinder.py"
    no_input = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert no_input.returncode == 2 and "required" in no_input.stderr
    run, force = fixture_run(), fixture_forces()
    run_path, force_path, obs_path = (
        tmp_path / name for name in ("run.json", "forces.csv", "observers.csv")
    )
    run_path.write_text(json.dumps(run))
    with force_path.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "force_x_N", "force_y_N", "force_z_N"])
        writer.writerows(force)
    with obs_path.open("w") as stream:
        rows = fixture_observers(run, force)
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "analysis"
    command = [
        sys.executable,
        str(script),
        "--run",
        str(run_path),
        "--forces",
        str(force_path),
        "--observers",
        str(obs_path),
        "--output",
        str(out),
        "--nperseg",
        "1024",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(
        (out / "results.json").read_text(),
        parse_constant=lambda value: pytest.fail(f"Nonfinite JSON {value}"),
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert data["status"] == "analytical_fixture"
    assert manifest["chart_count"] == len(data["charts"])
    assert manifest["aligned_row_count"] == data["metadata"]["aligned_row_count"]
    assert len(data["downloads"]) == 3
    assert (out / "downloads" / "forces.csv").read_bytes() == force_path.read_bytes()
    assert (out / "downloads" / "observers.csv").read_bytes() == obs_path.read_bytes()
    for record in manifest["outputs"]:
        assert hashlib.sha256((out / record["path"]).read_bytes()).hexdigest() == record["sha256"]
    for chart in data["charts"]:
        assert chart["kind"] in ("line", "polar", "heatmap", "scatter")
        assert "samples" in chart
        for trace in chart["traces"]:
            assert len(trace["x"]) == len(trace["y"])
        if chart["kind"] == "heatmap":
            assert np.shape(chart["z"]) == (len(chart["y"]), len(chart["x"]))
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and "exists" in again.stderr


@pytest.mark.parametrize(
    "field,value",
    [("force_convention", "auto"), ("force_content", "unknown"), ("schema_version", 2)],
)
def test_dataset_rejects_ambiguous_metadata(field, value):
    run = fixture_run()
    run[field] = value
    with pytest.raises(ValueError):
        model().build_dataset(run, fixture_forces(), [], nperseg=1024)


def test_force_on_fluid_input_converts_coefficients_but_preserves_raw_force():
    run, forces = fixture_run(), fixture_forces()
    run["force_convention"] = "on_fluid"
    forces[:, 1:] *= -1
    data, _ = model().build_dataset(run, forces, [], nperseg=1024)
    charts = {chart["id"]: chart for chart in data["charts"]}
    assert np.mean(charts["forces"]["traces"][0]["y"]) < 0
    assert np.mean(charts["force_coefficients"]["traces"][0]["y"]) > 0


def test_duplicate_observers_and_unknown_clocks_are_rejected():
    run, force = fixture_run(), fixture_forces()
    rows = fixture_observers(run, force)
    run["observers"].append(run["observers"][0])
    with pytest.raises(ValueError, match="duplicate"):
        model().build_dataset(run, force, rows, nperseg=1024)
    run = fixture_run()
    run["clock"]["libacoustics_fwh"]["kind"] = "auto"
    with pytest.raises(ValueError, match="clock"):
        model().build_dataset(run, force, rows, nperseg=1024)


def test_spectral_chart_counts_record_unused_window_tail():
    run, force = fixture_run(), fixture_forces()[:-3]
    data, _ = model().build_dataset(run, force, [], nperseg=1024)
    chart = next(c for c in data["charts"] if c["id"] == "force_psd")
    unused = data["metadata"]["spectral_estimator"]["unused_tail_samples"]
    assert chart["samples"]["welch_unused_tail_samples"] == unused
    assert chart["samples"]["welch_used_samples"] + unused == len(force)


def test_compactness_is_frequency_and_geometry_dependent_not_assumed_farfield():
    run, force = fixture_run(), fixture_forces()
    data, _ = model().build_dataset(run, force, fixture_observers(run, force), nperseg=1024)
    compactness = data["metadata"]["compactness"]
    a = np.hypot(0.01, 0.03)
    assert compactness["source_radius_bound_m"] == pytest.approx(a)
    assert compactness["k_a_at_lift_peak"] == pytest.approx(2 * np.pi * 2 * a / 1500)
    assert compactness["k_r_at_lift_peak"]["lift"] == pytest.approx(2 * np.pi * 2 * 2 / 1500)
    assert compactness["interpretation"] == "Small k*a is not equivalent to large k*r"


def test_cli_explicit_conventions_sidecar_preserves_input_identity(tmp_path):
    import csv
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

    run = fixture_run()
    conventions = {
        key: run.pop(key)
        for key in ("medium", "geometry", "force_convention", "force_content", "clock", "observers")
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    (tmp_path / "conventions.json").write_text(json.dumps(conventions))
    with (tmp_path / "forces.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "force_x_N", "force_y_N", "force_z_N"])
        writer.writerows(fixture_forces())
    (tmp_path / "observers.csv").write_text("time_s,observer,method,pressure_Pa\n")
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_cylinder.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run",
            str(tmp_path / "run.json"),
            "--forces",
            str(tmp_path / "forces.csv"),
            "--observers",
            str(tmp_path / "observers.csv"),
            "--conventions",
            str(tmp_path / "conventions.json"),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert len(manifest["inputs"]) == 4
    assert len(manifest["implementation"]) == 3
    for item in manifest["inputs"]:
        assert item["sha256"] == hashlib.sha256((tmp_path / item["file"]).read_bytes()).hexdigest()
