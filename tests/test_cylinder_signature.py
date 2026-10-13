"""Signature tests: analytic fixtures are tests, never native evidence."""

import importlib

import numpy as np
import pytest


def module():
    return importlib.import_module("flowacoustics.signature_analysis")


def test_completion_gate_rejects_running_and_requires_explicit_historical_smoke():
    api = module()
    with pytest.raises(ValueError, match="completed_native_reconstructed"):
        api.validate_status({"schema_version": 2, "status": "running"})
    old = {"schema_version": 1, "status": "completed_demo", "evidence_kind": "native_execution"}
    with pytest.raises(ValueError, match="historical"):
        api.validate_status(old)
    assert api.validate_status(old, historical_smoke=True) == "historical_smoke_only"
    new = {
        "schema_version": 2,
        "status": "completed_native_reconstructed",
        "evidence_kind": "native_execution",
        "exports": {"status": "completed"},
        "reconstruction": {"status": "completed"},
    }
    assert api.validate_status(new) == "native_completed_short_record"
    new["reconstruction"]["status"] = "pending"
    with pytest.raises(ValueError, match="reconstruction"):
        api.validate_status(new)


@pytest.mark.parametrize("length", [15, 16])
def test_spectra_match_independent_dft_parseval_and_column_mean(length):
    api = module()
    t = np.arange(6 * length) * 0.01
    x = np.sin(2 * np.pi * 7 * t) + 0.23 * np.cos(2 * np.pi * 13 * t) + t
    spec = api.spectrum(t, x, length)
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(length) / length)
    starts = np.arange(0, len(x) - length + 1, length - length // 2)
    expected = []
    energies = []
    for start in starts:
        y = x[start : start + length] - x[start : start + length].mean()
        fft = np.array(
            [
                sum(y[j] * w[j] * np.exp(-2j * np.pi * k * j / length) for j in range(length))
                for k in range(length // 2 + 1)
            ]
        )
        power = abs(fft) ** 2 * 0.01 / sum(w * w)
        power[1 : -1 if length % 2 == 0 else None] *= 2
        expected.append(power)
        energies.append(sum((y * w) ** 2) / sum(w * w))
    np.testing.assert_allclose(spec["columns"], np.array(expected).T, atol=1e-14)
    np.testing.assert_allclose(spec["density"], np.mean(expected, axis=0), atol=1e-14)
    assert spec["weighted_power"] == pytest.approx(np.mean(energies), rel=1e-13)
    assert spec["segments"] == len(starts)


def test_adaptive_sampling_and_downsampling_are_not_silent():
    api = module()
    t = np.cumsum(np.tile([0.001, 0.0011], 100))
    with pytest.raises(ValueError, match="Nonuniform"):
        api.spectrum(t, np.sin(t), 32)
    with pytest.raises(ValueError, match="downsampling"):
        api.spectrum(t, np.sin(t), 32, resample_dt_s=0.002)
    assert api.spectrum(t, np.sin(t), 32, resample_dt_s=0.0009)["resampled"]


def test_bands_are_half_open_additive_and_silence_is_minus_infinity():
    api = module()
    f = np.arange(11.0)
    bands = api.band_powers(f, np.ones(11), [0, 1, 3, 10])
    assert [b["bin_count"] for b in bands] == [1, 2, 8]
    assert sum(b["power"] for b in bands) == 11
    assert bands[-1]["bin_centers_Hz"][-1] == 10
    assert api.level(np.array([0.0, 1e-12]))[0] == -np.inf
    assert api.level(np.array([0.0, 1e-12]))[1] == 0
    with pytest.raises(ValueError):
        api.band_powers(f, -np.ones(11), [0, 10])


def test_nonoverlap_blocks_retain_tail_and_never_claim_stationarity():
    api = module()
    t = np.arange(0, 5.9, 0.1)
    b = api.block_statistics(t, t, start=0, end=5.8, duration=2)
    assert len(b["blocks"]) == 2
    assert [v["samples"] for v in b["blocks"]] == [20, 20]
    assert b["tail_samples"] == 19
    assert b["stationarity"] == "not_established"
    assert b["blocks"][0]["ac_rms"] == pytest.approx(np.std(t[:20]))


def test_compact_terms_covariance_and_distance_laws_without_density():
    api = module()
    t = np.arange(1001) * 0.001
    force = np.column_stack([t * t, 2 * t * t, -t * t])
    tau = t[2:-2]
    out = api.compact_terms(t, force, tau + 1 / 1500, [0, 1, 0], 1500)
    np.testing.assert_allclose(out["near_Pa"], -2 * tau * tau / (4 * np.pi), atol=1e-15)
    np.testing.assert_allclose(out["far_Pa"], -4 * tau / (4 * np.pi * 1500), atol=1e-15)
    double = api.compact_terms(t, force, tau + 2 / 1500, [0, 2, 0], 1500)
    np.testing.assert_allclose(double["near_Pa"], out["near_Pa"] / 4, atol=1e-15)
    np.testing.assert_allclose(double["far_Pa"], out["far_Pa"] / 2, atol=1e-15)
    with pytest.raises(ValueError, match="derivative"):
        api.compact_terms(t, force, [1 / 1500], [0, 1, 0], 1500)
    rng = np.random.default_rng(42)
    vector = rng.normal(size=(101, 3)) * [1, 0.4, 0.2]
    angles = np.linspace(0, 2 * np.pi, 73)
    rms, covariance = api.covariance_directivity(vector, angles)
    expected = [np.std(vector @ [np.cos(a), np.sin(a), 0]) for a in angles]
    np.testing.assert_allclose(rms, expected, atol=1e-15)
    np.testing.assert_allclose(covariance, np.cov(vector, rowvar=False, bias=True))


def test_coherence_short_support_delay_sign_and_nulls():
    api = module()
    t = np.arange(2048) / 128
    x = np.sin(2 * np.pi * 8 * t)
    y = np.sin(2 * np.pi * 8 * (t - 0.02))
    result = api.coupling(t, x, y, 256)
    k = np.argmin(abs(result["frequency_Hz"] - 8))
    assert result["phase_rad"][k] == pytest.approx(-2 * np.pi * 8 * 0.02, abs=1e-10)
    assert result["phase_valid"][k]
    assert np.nanmax(result["coherence"]) <= 1
    short = api.coupling(t[:384], x[:384], y[:384], 256)
    assert np.isnan(short["coherence"]).all()
    assert not short["phase_valid"].any()
    null = api.coupling(t, np.zeros_like(t), y, 256)
    assert not null["phase_valid"].any()


def test_hash_gates_bind_expected_bytes_and_repo_relative_paths(tmp_path):
    import hashlib
    import json

    api = module()
    source = tmp_path / "forces.csv"
    source.write_text("original")
    run = dict(
        schema_version=1,
        status="completed_demo",
        evidence_kind="native_execution",
        sha256={"forces.csv": hashlib.sha256(source.read_bytes()).hexdigest()},
    )
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps(run))
    result = api.verify_run(
        manifest, project=tmp_path, required_paths=[source], historical_smoke=True
    )
    assert result["verified_count"] == 1
    source.write_text("finite corruption")
    with pytest.raises(ValueError, match="identity mismatch"):
        api.verify_run(manifest, project=tmp_path, required_paths=[source], historical_smoke=True)
    run["sha256"] = {}
    manifest.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="empty"):
        api.verify_run(manifest, project=tmp_path, required_paths=[source], historical_smoke=True)


def test_v2_requires_three_nonempty_identity_maps_and_required_coverage(tmp_path):
    import hashlib
    import json

    api = module()
    source = tmp_path / "source"
    source.write_text("fixture only")
    hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    run = dict(
        schema_version=2,
        status="completed_native_reconstructed",
        evidence_kind="native_execution",
        input_sha256=hashes,
        exports=dict(status="completed", sha256=hashes),
        reconstruction=dict(status="completed", sha256=hashes),
    )
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    assert api.verify_run(path, project=tmp_path, required_paths=[source])["verified_count"] == 1
    with pytest.raises(ValueError, match="missing expected"):
        api.verify_run(path, project=tmp_path, required_paths=[tmp_path / "other"])
    run["exports"]["sha256"] = {}
    path.write_text(json.dumps(run))
    with pytest.raises(ValueError, match="empty"):
        api.verify_run(path, project=tmp_path)


def table_fixture(tmp_path):
    import csv
    import hashlib
    import json

    t = np.arange(1, 1001) / 100
    case = tmp_path / "case"
    case.mkdir()
    forces = tmp_path / "forces.csv"
    headers = ["time_s"] + [
        prefix + c + "_N"
        for prefix in ["force_", "pressure_force_", "viscous_force_"]
        for c in "xyz"
    ]
    vector = np.column_stack(
        (
            0.02 + 0.001 * np.cos(2 * np.pi * 3 * t),
            0.003 * np.sin(2 * np.pi * 1.5 * t),
            0.0002 * np.cos(t),
        )
    )
    with forces.open("w") as stream:
        w = csv.writer(stream)
        w.writerow(headers)
        w.writerows(np.column_stack((t, vector, vector * 0.9, vector * 0.1)))
    coords = dict(
        cross_plus=[0, 1, 0],
        cross_minus=[0, -1, 0],
        downstream=[1, 0, 0],
        oblique=[0.6, 0.8, 0],
        cross_double=[0, 2, 0],
    )
    observers = tmp_path / "observers.csv"
    with observers.open("w") as stream:
        w = csv.writer(stream)
        w.writerow(["time_s", "observer", "method", "pressure_Pa"])
        for name, xyz in coords.items():
            for method in ["libacoustics_curle", "libacoustics_fwh"]:
                for ti, v in zip(t, vector):
                    w.writerow(
                        [ti, name, method, np.dot(v, xyz) / np.linalg.norm(xyz) ** 3 / (4 * np.pi)]
                    )
    run = dict(
        schema_version=2,
        status="completed_native_reconstructed",
        evidence_kind="native_execution",
        test_fixture_only=True,
        case_path=str(case),
        input_sha256={},
        medium=dict(
            density_kg_m3=1000, sound_speed_m_s=1500, velocity_m_s=0.2, reference_pressure_Pa=1e-6
        ),
        geometry=dict(
            diameter_m=0.02,
            span_m=0.06,
            centre_m=[0, 0, 0],
            axis="z",
            outer_radius_m=0.24,
            finite_closed=True,
        ),
        observer_coordinates_m=coords,
        force_convention="on_body",
        force_content="total_traction",
        sampling=dict(adaptive=False, fixed_dt_s=0.01),
        time_coverage=dict(end_s=10, requested_end_s=10),
        solver_verification=dict(end_s=10, solver_reached_target=True),
        clock={
            "forces_offset_s": 0,
            "libacoustics_curle": dict(offset_s=0, kind="source_unretarded"),
            "libacoustics_fwh": dict(offset_s=0, kind="observer"),
        },
        exports=dict(
            status="completed", tables=dict(force_samples=len(t), observer_rows=len(t) * 10)
        ),
        reconstruction=dict(status="completed"),
    )
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [forces, observers]}
    run["input_sha256"] = pins
    run["exports"]["sha256"] = pins
    run["reconstruction"]["sha256"] = pins
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    return dict(
        run_path=path, forces_path=forces, observers_path=observers, case=case, project=tmp_path
    )


def test_table_loader_accepts_original_v2_without_adapter_and_preserves_counts(tmp_path):
    api = module()
    args = table_fixture(tmp_path)
    data = api.load_records(**args, start_time_s=2, end_time_s=10)
    assert data["counts"] == dict(force_rows=1000, observer_rows=10000, observer_channels=10)
    assert len(data["time_s"]) == 801
    assert data["run"]["test_fixture_only"]
    assert data["time_s"][0] == 2
    with pytest.raises(ValueError, match="support"):
        api.load_records(**args, start_time_s=2, end_time_s=11)


def test_table_loader_rejects_missing_columns_even_when_hash_is_updated(tmp_path):
    import hashlib
    import json

    api = module()
    args = table_fixture(tmp_path)
    p = args["forces_path"]
    p.write_text("time_s,force_x_N\n1,2\n")
    r = json.loads(args["run_path"].read_text())
    for pins in [r["input_sha256"], r["exports"]["sha256"], r["reconstruction"]["sha256"]]:
        pins[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    args["run_path"].write_text(json.dumps(r))
    with pytest.raises(ValueError, match="columns"):
        api.load_records(**args, start_time_s=2, end_time_s=10)


def test_signature_reductions_have_window_sensitivity_and_honest_support(tmp_path):
    api = module()
    data = api.load_records(**table_fixture(tmp_path), start_time_s=2, end_time_s=10)
    result = api.analyze_records(data, nperseg=200)
    assert result["force_spectra"]["lift"]["segments"] == 7
    assert [x["requested_start_s"] for x in result["sensitivity_windows"]] == [2, 4, 6]
    assert result["sensitivity_windows"][-1]["segments"] == 3
    assert result["stationarity"] == "not_established"
    assert len(result["blocks"]["lift"]["blocks"]) == 4
    assert result["directivity"]["covariance_max_abs_error_Pa"] < 1e-16
    assert len(result["directivity"]["angles_rad"]) == 181
    assert len(result["ratios"]) == 3
    assert result["ratios"][-1]["matched_samples"] > 0
    assert result["band_edges_Hz"] == [0, 1, 3, 10, 20]
    assert all(x["spectral_bin_count"] > 0 for x in result["band_rows"])
    assert not result["peak"]["stationary_tone_established"]
    assert len(result["pressure_summary"]) == 10
    for row in result["pressure_summary"]:
        # Independent summation order can move a last bit when all energy is in-band.
        assert row["analysed_band_power_Pa2"] <= row["whole_nyquist_welch_power_Pa2"] * (1 + 1e-14)
        assert row["global_ac_variance_Pa2"] >= 0


def geometry_fixture(args):
    import hashlib
    import json

    case = args["case"]
    mesh = case / "constant/polyMesh"
    mesh.mkdir(parents=True)
    theta = np.arange(16) * 2 * np.pi / 16
    xy = np.concatenate(
        [np.column_stack((r * np.cos(theta), r * np.sin(theta))) for r in [0.01, 0.08, 0.24]]
    )
    points = np.column_stack((xy, np.zeros(len(xy))))
    # Synthetic connectivity is confined to tests, not used in historical plots.
    faces = [[i, (i + 1) % 16, (i + 1) % 16 + 16, i + 16] for i in range(16)]
    for name, content in [
        ("points", "\n".join("(" + " ".join(map(str, row)) + ")" for row in points)),
        ("faces", "\n".join("4(" + " ".join(map(str, row)) + ")" for row in faces)),
        ("boundary", "cylinder { nFaces 16; startFace 0; }"),
    ]:
        (mesh / name).write_text(content + "\n")
    raw = case / "postProcessing/flowSlice/10/vorticity_midspan.raw"
    raw.parent.mkdir(parents=True)
    field = np.column_stack((points, np.zeros((len(points), 2)), np.sin(xy[:, 0] * 30)))
    raw.write_text(
        f"# vorticity POINT_DATA {len(field)}\n# x y z vorticity_x vorticity_y vorticity_z\n"
        + "\n".join(" ".join(map(str, row)) for row in field)
        + "\n"
    )
    run = json.loads(args["run_path"].read_text())
    for p in [*mesh.iterdir(), raw]:
        run["input_sha256"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    args["run_path"].write_text(json.dumps(run))


def test_geometry_uses_native_connectivity_and_solid_mask(tmp_path):
    api = module()
    args = table_fixture(tmp_path)
    geometry_fixture(args)
    data = api.load_records(**args, start_time_s=2, end_time_s=10)
    result = api.load_geometry(data)
    assert result["time_s"] == 10
    assert result["audit"]["solid_present"]
    assert result["audit"]["solid_intersecting_triangles_removed"] > 0
    assert result["triangulation"].get_trifinder()(0, 0) == -1
    assert result["mesh_segments_m"].shape[1:] == (2, 3)
    assert result["snapshot_values"].shape == (48, 6)


def test_report_preflight_rejects_existing_links_and_escape_before_read(tmp_path):
    api = module()
    root = tmp_path / "allowed"
    root.mkdir()
    existing = root / "existing"
    existing.mkdir()
    (existing / "keep").write_text("unchanged")
    with pytest.raises(ValueError, match="new child"):
        api.preflight_output(existing, root)
    link = root / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        api.preflight_output(link / "new", root)
    with pytest.raises(ValueError):
        api.preflight_output(tmp_path / "outside", root)
    assert (existing / "keep").read_text() == "unchanged"


def test_end_to_end_nine_figures_all_formats_numeric_provenance_and_no_clobber(tmp_path):
    import json

    api = module()
    args = table_fixture(tmp_path)
    geometry_fixture(args)
    out = tmp_path / "reports/one"
    result = api.build_report(
        **args,
        output=out,
        allowed_root=tmp_path / "reports",
        start_time_s=2,
        end_time_s=10,
        nperseg=200,
    )
    assert len(result["figures"]) == 9
    assert result["evidence_label"] == "test_fixture_only"
    assert (out / "manifest.json").is_file()
    for entry in result["figures"]:
        manifest = json.loads((out / entry["manifest"]).read_text())
        assert manifest["sources"] and manifest["equations"] and manifest["numeric_tables"]
        for ext in ["png", "svg", "pdf"]:
            assert (out / manifest["outputs"][ext]["path"]).stat().st_size > 500
        for table in manifest["numeric_tables"]:
            assert table["rows"] > 0
        assert manifest["stationarity"] == "not_established"
        assert manifest["layout_audit"]["legend_caption_overlaps"] == 0
        assert manifest["layout_audit"]["legend_title_overlaps"] == 0
        assert all(axis["left_title"] == "" for axis in manifest["axes"])
    four = json.loads((out / "I04_receiver_spectra/manifest.json").read_text())
    assert four["layout"] == [2, 2]
    seven = json.loads((out / "I07_coherence_phase/manifest.json").read_text())
    assert seven["axes"][0]["ylim"] == [0.0, 1.0]
    nine = json.loads((out / "I09_directivity/manifest.json").read_text())
    assert nine["native_angular_support"] == 4 and not nine["native_connecting_lines"]
    before = api.digest(out / "manifest.json")
    with pytest.raises(ValueError, match="new child"):
        api.build_report(
            **args,
            output=out,
            allowed_root=tmp_path / "reports",
            start_time_s=2,
            end_time_s=10,
            nperseg=200,
        )
    assert api.digest(out / "manifest.json") == before


def test_cli_help_and_running_refusal_without_output(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cli = root / "scripts/build_cylinder_signature.py"
    help_run = subprocess.run([sys.executable, str(cli), "--help"], capture_output=True, text=True)
    assert help_run.returncode == 0, help_run.stderr
    assert "--resample-dt-s" in help_run.stdout and "--historical-smoke" in help_run.stdout
    run = tmp_path / "running.json"
    run.write_text(json.dumps(dict(schema_version=2, status="running")))
    target = root / "results/cylinder/signature/reject_running_test"
    command = [
        sys.executable,
        str(cli),
        "--run",
        str(run),
        "--forces",
        str(tmp_path / "absent.csv"),
        "--observers",
        str(tmp_path / "absent2.csv"),
        "--case",
        str(tmp_path / "absent_case"),
        "--output",
        str(target),
        "--start-time-s",
        "2",
        "--end-time-s",
        "10",
        "--nperseg",
        "3200",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0 and "completed_native_reconstructed" in completed.stderr
    assert not target.exists()


def test_upstream_resampling_is_visible_in_estimator_provenance(tmp_path):
    api = module()
    args = table_fixture(tmp_path)
    data = api.load_records(**args, start_time_s=2, end_time_s=10, resample_dt_s=0.009)
    assert data["resampled"]
    result = api.analyze_records(data, nperseg=200)
    assert result["force_spectra"]["lift"]["resampled"]


def test_native_completion_requires_consistent_endpoint_before_tables(tmp_path):
    import json

    api = module()
    args = table_fixture(tmp_path)
    run = json.loads(args["run_path"].read_text())
    run["time_coverage"].update(requested_end_s=10, end_s=9)
    run["solver_verification"] = {"last_time_s": 10}
    args["run_path"].write_text(json.dumps(run))
    with pytest.raises(ValueError, match="endpoint"):
        api.load_records(**args, start_time_s=2, end_time_s=9)
