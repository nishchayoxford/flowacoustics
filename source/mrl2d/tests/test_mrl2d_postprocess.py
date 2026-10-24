"""Offline-only tests. No native solver commands or owner-lock access."""

import numpy as np
import pytest


def test_analysis_policy_refuses_short_production_and_labels_diagnostics():
    from flowacoustics.mrl2d_postprocess import analysis_plan

    short = np.arange(1, 51) * 0.002
    with pytest.raises(ValueError, match="10 s"):
        analysis_plan(short, mode="production")
    diagnostic = analysis_plan(short, mode="diagnostic")
    assert diagnostic["acoustic_band_claim_valid"] is False
    assert diagnostic["startup_excluded_s"] == 0
    assert diagnostic["frequency_spacing_hz"] == pytest.approx(10)
    assert diagnostic["requested_band_hz"] == [0.0, 25.0]
    full = analysis_plan(np.arange(1, 5001) * 0.002, mode="production")
    assert full["sample_count"] == 4000
    assert full["sample_window_s"] == [2.0020000000000002, 10.0]
    assert full["frequency_spacing_hz"] == pytest.approx(0.125)
    assert full["requested_band_hz"] == pytest.approx([0.625, 25])
    assert full["window"] == "hann"
    assert full["stationarity_verified"] is False
    with pytest.raises(ValueError, match="uniform"):
        analysis_plan(np.array([0.002, 0.004, 0.008, 0.01]), mode="diagnostic")


def circle_history(n=50, panels=19, moving=False):
    from flowacoustics.acoustics2d import ContourHistory

    time = np.arange(1, n + 1) * 0.002
    angle = np.arange(panels) * 2 * np.pi / panels
    normal = np.column_stack((np.cos(angle), np.sin(angle)))
    xy = 0.05 * normal
    if moving:
        xy = (
            xy[None, :, :]
            + np.column_stack((0.01 * np.cos(10 * time), 0.01 * np.sin(10 * time)))[:, None, :]
        )
    return ContourHistory(
        time=time,
        xy=xy,
        normal=normal,
        ds=np.full(panels, 2 * np.pi * 0.05 / panels),
        p=3 * np.cos(2 * np.pi * 10 * time)[:, None] * normal[None, :, 0],
        U=np.zeros((n, panels, 2)),
        surface_velocity=np.zeros((panels, 2)),
        kind="body",
        blade_id=np.zeros(panels, dtype=int),
        provenance={"source_type": "analytical_verification"},
    )


@pytest.mark.parametrize("moving", [False, True])
def test_chunked_reduction_matches_direct_kernel_and_force_units(moving):
    from flowacoustics.acoustics2d import contour_spectrum
    from flowacoustics.mrl2d_postprocess import reduce_history, resource_estimate

    h = circle_history(moving=moving)
    reduced = reduce_history(h, mode="diagnostic", panel_chunk=7)
    direct = contour_spectrum(h, fmax=25, method="direct")
    np.testing.assert_allclose(reduced["coefficients_pa"], direct.coefficient, atol=2e-13)
    np.testing.assert_allclose(reduced["pressure_pa"], direct.synthesize(), atol=5e-13)
    np.testing.assert_allclose(reduced["band_rms_pa"], direct.band_rms, atol=5e-13)
    expected = -3 * np.pi * 0.05 * np.cos(2 * np.pi * 10 * h.time)
    np.testing.assert_allclose(reduced["pressure_force_n_per_m"][:, 0], expected, atol=1e-14)
    np.testing.assert_allclose(reduced["pressure_force_n_per_m"][:, 1], 0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(reduced["receivers_m"], axis=1), 1)
    assert reduced["coefficients_pa"].shape == (2, 12)
    assert reduced["metadata"]["pressure_units"] == "Pa"
    assert reduced["metadata"]["loading_units"] == "N/m"
    assert reduced["metadata"]["chunk_count"] == 3
    full = resource_estimate(5000, 866, 200, panel_chunk=96)
    assert full["forbidden_dense_complex_tensor_bytes"] > 100_000_000_000
    assert full["conservative_peak_bytes"] < 2_000_000_000


def test_native_loader_rejects_short_before_reading_missing_path():
    from flowacoustics.mrl2d_postprocess import load_native_case

    with pytest.raises(ValueError, match="10 s"):
        load_native_case(
            {"id": "mrl2d_urans_body", "root": "/does-not-exist", "end_s": 0.1}, mode="production"
        )


def test_existing_native_diagnostic_is_pinned_and_not_rescaled():
    import json
    from pathlib import Path

    from flowacoustics.mrl2d_postprocess import load_native_case

    root = Path(__file__).resolve().parents[1] / "native-archives/diagnostic-urans"
    if not root.exists():
        pytest.skip("optional retained native diagnostic archive absent")
    receipt = json.loads((root / "diagnostic.json").read_text())
    pins = {
        str(Path(k).relative_to(root)): v
        for k, v in receipt["sources"]["native_files_sha256"].items()
    }
    spec = {
        "id": "mrl2d_urans_body",
        "root": str(root),
        "end_s": 0.1,
        "envelope_recorded": True,
        "source_sha256": pins,
    }
    history, audit = load_native_case(spec, mode="diagnostic")
    assert history.p.shape == (50, 546)
    assert history.p.min() == pytest.approx(-8146.88333196775)
    assert audit["native_coupling_verified"] is True
    assert audit["source_identity_basis"] == "expected hashes supplied before reading"
    assert audit["maximum_wall_normal_slip_m_s"] < 0.001
    assert "/home/" not in json.dumps(audit)
    with pytest.raises(ValueError, match="identity"):
        load_native_case({**spec, "source_sha256": {}}, mode="diagnostic")
    with pytest.raises(ValueError, match="identity"):
        load_native_case(
            {**spec, "source_sha256": dict.fromkeys(pins, "0" * 64)}, mode="diagnostic"
        )
    with pytest.raises(ValueError, match="memory"):
        load_native_case(spec, mode="diagnostic", memory_limit_bytes=10)


def test_ascii_plane_uses_actual_vtk_and_rejects_binary(tmp_path):
    import json
    import os
    import subprocess
    from pathlib import Path

    from flowacoustics.mrl2d_postprocess import read_plane

    fake = tmp_path / "binary.vtp"
    fake.write_text('<VTKFile type="PolyData"><DataArray format="binary">bad</DataArray></VTKFile>')
    with pytest.raises(ValueError, match="ASCII"):
        read_plane(fake)
    root = Path(__file__).resolve().parents[1]
    ascii_path = root / (
        "native-archives/diagnostic-urans/postProcessing/runtimeAsciiSlice/0.1/section.vtp"
    )
    if not ascii_path.exists():
        pytest.skip("optional retained ASCII native plane absent")
    command = [
        "/usr/bin/python3",
        "-c",
        (
            "import json,sys; from flowacoustics.mrl2d_postprocess import read_plane; "
            "print(json.dumps(read_plane(sys.argv[1])['metadata']))"
        ),
        str(ascii_path),
    ]
    run = subprocess.run(
        command,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr
    meta = json.loads(run.stdout)
    assert meta["reader_errors"] == 0
    assert meta["native_points"] == 135877
    assert meta["native_triangles"] == 267510
    assert meta["display_only_zero_area_triangles_removed"] == 107004
    assert meta["area_m2"] == pytest.approx(1.1294930067968791)
    assert meta["point_dtype"] == "float32"
    assert meta["pressure_conversion"] == "kinematic p times 1000 once for display"


def test_production_completion_requires_all_stages_and_pinned_fields(tmp_path):
    import hashlib
    import json

    from flowacoustics.mrl2d_postprocess import validate_completion

    # A synthetic *receipt* unit fixture, never exported as CFD data.
    pins = {}
    stages = {}
    for name in ("solver", "reconstruction", "fields"):
        file = tmp_path / (name + ".log")
        file.write_text("Time = 10\nEnd\n")
        pins[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        stages[name] = {
            "status": "completed",
            "returncode": 0,
            "log": file.name,
            "achieved_end_s": 10.0,
            "verified_end_s": 10.0,
        }
    receipt = {
        "schema": "mrl2d-postprocessing-completion-v1",
        "case_id": "mrl2d_urans_body",
        "source_type": "native_cfd",
        "requested_end_s": 10.0,
        "stages": stages,
        "evidence_sha256": pins,
        "cells": 26751,
        "ranks": 2,
        "dt_s": 0.0000625,
        "physical_input_fingerprint": "a" * 64,
        "source_sha256": {"source.bin": "b" * 64},
        "field_frames": [
            {"time_s": (i + 1) * 0.02, "path": f"planes/{i}.vtp", "sha256": "c" * 64}
            for i in range(500)
        ],
    }
    file = tmp_path / "completion.json"
    file.write_text(json.dumps(receipt))
    spec = {
        "root": str(tmp_path),
        "id": "mrl2d_urans_body",
        "end_s": 10,
        "completion": "completion.json",
        "completion_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "source_sha256": receipt["source_sha256"],
    }
    # POST-2: old detached assertions cannot establish executed-run identity.
    # Even three separately named logs do not prove three independent stages.
    with pytest.raises(ValueError, match="versioned direct-native"):
        validate_completion(spec)
    receipt["stages"]["fields"]["status"] = "pending"
    file.write_text(json.dumps(receipt))
    spec["completion_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="versioned direct-native"):
        validate_completion(spec)
    receipt["stages"]["fields"]["status"] = "completed"
    receipt["field_frames"].pop()
    file.write_text(json.dumps(receipt))
    spec["completion_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="versioned direct-native"):
        validate_completion(spec)


def test_four_case_bundle_is_site_compatible_and_comparisons_fail_closed(tmp_path):
    import json
    import subprocess
    from pathlib import Path

    from flowacoustics.mrl2d_postprocess import CASE_IDS, build_bundle, reduce_history

    reduced = {case: reduce_history(circle_history(), mode="diagnostic") for case in CASE_IDS}
    audits = {case: {"native_coupling_verified": False} for case in CASE_IDS}
    bundle = build_bundle(reduced, audits, mode="diagnostic")
    assert bundle["schema_version"] == 2 and bundle["study"] == "mrl2d"
    assert [c["id"] for c in bundle["cases"]] == list(CASE_IDS)
    assert all(c["status"] == "not_run" for c in bundle["cases"])
    assert all(c["metadata"]["acoustic_band_hz"] is None for c in bundle["cases"])
    assert len(bundle["comparisons"]) == 4
    assert all(not c["physical_interpretation_allowed"] for c in bundle["comparisons"])
    assert len({chart["id"] for case in bundle["cases"] for chart in case["charts"]}) == 4
    assert all(len(c["charts"][1]["traces"]) == 12 for c in bundle["cases"])
    file = tmp_path / "diagnostic.json"
    file.write_text(json.dumps(bundle, allow_nan=False))
    module = (Path(__file__).resolve().parents[1] / "site/mrl2d-model.mjs").as_uri()
    run = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            f"import {{adaptResults,comparisonFor}} from '{module}'; "
            "import fs from 'node:fs'; "
            "const x=adaptResults(JSON.parse(fs.readFileSync(process.argv[1]))); "
            "if(x.cases.length!==4||comparisonFor(x.cases[0],x.cases[1],'spectrum').allowed)"
            "process.exit(2); console.log(x.state);",
            str(file),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "ready"
    with pytest.raises(ValueError, match="four"):
        build_bundle({CASE_IDS[0]: reduced[CASE_IDS[0]]}, audits, mode="diagnostic")
    with pytest.raises(ValueError, match="completion"):
        build_bundle(reduced, audits, mode="production")


def test_field_readback_checks_pins_time_and_native_data():
    from pathlib import Path

    from flowacoustics.mrl2d_postprocess import verify_field_frames

    root = Path(__file__).resolve().parents[1] / "native-archives/diagnostic-urans"
    if not root.exists():
        pytest.skip("optional native diagnostic archive absent")
    frame = {
        "time_s": 0.1,
        "path": "postProcessing/runtimeAsciiSlice/0.1/section.vtp",
        "sha256": "1244ee5ff64bef213af2f3903948001dfcbe313afeb047cd80a481fda9d5b751",
    }
    summary = verify_field_frames(root, [frame])
    assert summary["frames_read_with_vtk"] == 1
    assert summary["zero_area_xy_triangles_per_frame"] == [107004]
    with pytest.raises(ValueError, match="identity"):
        verify_field_frames(root, [{**frame, "sha256": "0" * 64}])
    with pytest.raises(ValueError, match="time"):
        verify_field_frames(root, [{**frame, "time_s": 0.2}])


def test_cli_exercises_native_diagnostic_export_and_never_promotes_short_run(tmp_path):
    import hashlib
    import json
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    from flowacoustics.mrl2d_postprocess import CASE_IDS

    root = Path(__file__).resolve().parents[1]
    cache = root / ".cache/mrl2d/postprocessing"
    cache.mkdir(parents=True, exist_ok=True)
    entries = []
    for identity in CASE_IDS:
        model = "urans" if "_urans_" in identity else "wale"
        case = root / f"native-archives/diagnostic-{model}"
        if not case.exists():
            pytest.skip("optional native diagnostic archives absent")
        receipt = json.loads((case / "diagnostic.json").read_text())
        entries.append(
            {
                "id": identity,
                "root": str(case),
                "end_s": 0.1,
                "envelope_recorded": True,
                "source_sha256": {
                    str(Path(k).relative_to(case)): v
                    for k, v in receipt["sources"]["native_files_sha256"].items()
                },
            }
        )
    spec = tmp_path / "input.json"
    spec.write_text(json.dumps({"schema": "mrl2d-postprocess-input-v1", "cases": entries}))
    script = root / "scripts/postprocess_mrl2d.py"
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory(dir=cache, prefix="test-export-") as work:
        work = Path(work)
        for name in ("first", "repeat"):
            run = subprocess.run(
                [
                    str(root / ".venv/bin/python"),
                    str(script),
                    "--input",
                    str(spec),
                    "--output",
                    str(work / name),
                    "--mode",
                    "diagnostic",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            assert run.returncode == 0, run.stderr
            report = json.loads(run.stdout)
            assert report["cases"] == 4
            assert report["mode"] == "diagnostic"
        bundle = json.loads((work / "first/diagnostic-results.json").read_text())
        assert all(c["status"] == "not_run" for c in bundle["cases"])
        for identity in CASE_IDS:
            data = np.load(work / "first" / (identity + ".npz"))
            assert data["pressure_pa"].shape == (50, 12)
            assert data["coefficients_pa"].shape == (2, 12)
            np.testing.assert_allclose(
                np.sqrt(np.mean(data["pressure_pa"] ** 2, axis=0)), data["band_rms_pa"], rtol=1e-13
            )
        first = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (work / "first").iterdir()
        }
        repeat = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (work / "repeat").iterdir()
        }
        assert first == repeat
        denied = subprocess.run(
            [
                str(root / ".venv/bin/python"),
                str(script),
                "--input",
                str(spec),
                "--output",
                str(work / "denied"),
                "--mode",
                "production",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert denied.returncode != 0 and "10 s" in denied.stderr
        assert not (work / "denied").exists()


def test_production_loader_validates_receipt_identity_instead_of_blanket_rejection(tmp_path):
    from flowacoustics.mrl2d_postprocess import load_native_case

    (tmp_path / "completion.json").write_text("{}")
    spec = {
        "root": str(tmp_path),
        "id": "mrl2d_urans_body",
        "end_s": 10,
        "completion": "completion.json",
        "completion_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="^completion identity mismatch"):
        load_native_case(spec, mode="production")


def test_output_guards_run_before_input_access(tmp_path):
    import tempfile
    from pathlib import Path

    from flowacoustics.mrl2d_postprocess import OUTPUT_ROOT, export_study

    with pytest.raises(ValueError, match="output"):
        export_study("/missing-input", tmp_path / "outside", mode="diagnostic")
    assert not (tmp_path / "outside").exists()
    with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT, prefix="test-guard-") as work:
        work = Path(work)
        (work / "linked").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="symlink"):
            export_study("/missing-input", work / "linked/output", mode="diagnostic")
        with pytest.raises(ValueError, match="traversal"):
            export_study("/missing-input", work / "a/../other", mode="diagnostic")
        with pytest.raises(ValueError, match="exists"):
            export_study("/missing-input", work, mode="diagnostic")


def test_renderer_uses_native_connectivity_and_writes_only_diagnostic_assets(tmp_path):
    import json
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    from flowacoustics.mrl2d_postprocess import OUTPUT_ROOT

    root = Path(__file__).resolve().parents[1]
    source = root / (
        "native-archives/diagnostic-urans/postProcessing/runtimeAsciiSlice/0.1/section.vtp"
    )
    if not source.exists():
        pytest.skip("optional native ASCII plane absent")
    with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT, prefix="test-render-") as work:
        output = Path(work) / "plane"
        command = (
            "import sys,json; from flowacoustics.mrl2d_postprocess import render_plane; "
            "print(json.dumps(render_plane(sys.argv[1],sys.argv[2],pressure_limit_pa=4000,speed_limit_m_s=2)))"
        )
        result = subprocess.run(
            ["/usr/bin/python3", "-c", command, str(source), str(output)],
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(tmp_path / "mpl"),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        meta = json.loads(result.stdout)
        assert meta["display_only_zero_area_triangles_removed"] == 107004
        assert meta["title_prefix"] == "DIAGNOSTIC native CFD"
        assert (output / "pressure.png").stat().st_size > 10000
        assert (output / "speed.png").stat().st_size > 10000
        assert meta["triangulation"] == "native triangles only, zero x-y area removed for display"


def test_resource_budget_counts_three_native_record_residencies():
    from flowacoustics.mrl2d_postprocess import resource_estimate

    estimate = resource_estimate(5000, 866, 196)
    # Native memmaps, concatenation and sorted copy may coexist (16 float64 columns each).
    assert estimate["conservative_peak_bytes"] >= 5000 * 866 * 16 * 8 * 3 + 64 * 1024**2


def test_zero_signal_and_material_mean_loading_are_not_fabricated():
    from flowacoustics.acoustics2d import contour_spectrum
    from flowacoustics.mrl2d_postprocess import reduce_history

    h = circle_history(moving=True)
    h.p[:] = h.p[0]
    result = reduce_history(h, mode="diagnostic", panel_chunk=7)
    exact = contour_spectrum(h, fmax=25, method="direct")
    assert np.max(abs(exact.coefficient)) > 1e-7
    np.testing.assert_allclose(result["coefficients_pa"], exact.coefficient, atol=1e-13)
    h.p[:] = 0
    zero = reduce_history(h, mode="diagnostic")
    assert zero["band_level_db_re_1uPa"] == [None] * 12
    assert np.array_equal(zero["pressure_pa"], np.zeros((50, 12)))


def test_loading_chart_support_is_full_source_not_spectral_window():
    from flowacoustics.mrl2d_postprocess import CASE_IDS, build_bundle, reduce_history

    r = reduce_history(circle_history(n=5000, panels=9), mode="production")
    audits = {
        case: {
            "completion_verified": True,
            "native_coupling_verified": True,
            "physical_input_fingerprint": "a" * 64,
        }
        for case in CASE_IDS
    }
    bundle = build_bundle(dict.fromkeys(CASE_IDS, r), audits, mode="production")
    chart = bundle["cases"][0]["charts"][0]
    assert chart["window"]["sample_window_s"] == [0.002, 10]
    assert chart["window"]["window"] == "none; native source history"
    assert bundle["cases"][0]["charts"][1]["window"]["sample_window_s"][0] > 2
