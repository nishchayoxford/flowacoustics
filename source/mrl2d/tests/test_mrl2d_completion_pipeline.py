"""SYNTHETIC ONLY: full 10 s pipeline; not native production evidence."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import test_mrl2d_completion as fixtures
from test_mrl2d_completion import (
    digest,
    independent_fingerprint,
    put,
    put_json,
    seal,
)

from flowacoustics.mrl2d_case import case_files, case_specs

seed = fixtures.seed


def synthetic_binary_sources(root, envelope):
    """Manufactured closed square blades/rectangle; NOT mesh-derived CFD data."""
    t = np.arange(1, 5001) * 0.002
    initial = np.array(
        [[0.0, 0.055], [np.sqrt(3) * 0.055 / 2, -0.0275], [-np.sqrt(3) * 0.055 / 2, -0.0275]]
    )
    a = -10.472 * t
    orbit = np.stack((np.cos(a), -np.sin(a), np.sin(a), np.cos(a)), axis=-1).reshape(-1, 2, 2)
    a = a / 2
    rotation = np.stack((np.cos(a), -np.sin(a), np.sin(a), np.cos(a)), axis=-1).reshape(-1, 2, 2)
    base_normals = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    normals = np.einsum("tij,pj->tpi", rotation, base_normals)
    body = np.zeros((5000, 12, 16), dtype="<f8")
    body[:, :, 0] = t[:, None]
    for b, c in enumerate(initial):
        center = np.einsum("tij,j->ti", orbit, c)
        part = body[:, b * 4 : (b + 1) * 4]
        part[:, :, 2] = np.arange(b * 4, (b + 1) * 4)
        part[:, :, 3] = b
        part[:, :, 4:6] = center[:, None] + 0.002 * normals
        part[:, :, 7:9] = normals
        part[:, :, 9] = 0.004
        part[:, :, 10] = 20 + 5 * np.cos(2 * np.pi * 10 * t[:, None]) * normals[:, :, 0]
        part[:, :, 13] = 5.236 * (part[:, :, 5] + center[:, None, 1])
        part[:, :, 14] = -5.236 * (part[:, :, 4] + center[:, None, 0])
        part[:, :, 11:13] = part[:, :, 13:15]
    put(root / "processor0/postProcessing/mrl2dSources/rank0.bin", body.tobytes())
    env = np.zeros((5000, 320 if envelope else 0, 16), dtype="<f8")
    if envelope:
        corners = np.array([[-0.14, -0.14], [0.30, -0.14], [0.30, 0.14], [-0.14, 0.14]])
        for i in range(4):
            delta = corners[(i + 1) % 4] - corners[i]
            length = np.linalg.norm(delta)
            part = env[:, i * 80 : (i + 1) * 80]
            part[:, :, 4:6] = corners[i] + (np.arange(80) + 0.5)[:, None] / 80 * delta
            part[:, :, 7:9] = np.array([delta[1], -delta[0]]) / length
            part[:, :, 9] = length / 80
        env[:, :, 0] = t[:, None]
        env[:, :, 1] = 1
        env[:, :, 2] = np.arange(320)
        env[:, :, 3] = -1
        env[:, :, 10] = 3 * np.sin(2 * np.pi * 10 * t[:, None]) * env[:, :, 7]
    put(root / "processor1/postProcessing/mrl2dSources/rank1.bin", env.tobytes())
    return {
        str(p): digest(p.read_bytes())
        for p in root.glob("processor*/postProcessing/mrl2dSources/rank*.bin")
    }


@pytest.fixture
def synthetic_study(seed, tmp_path):
    entries = []
    original_reports = {}
    for case in case_specs():
        root = tmp_path / "SYNTHETIC-NOT-CFD" / case.name
        shutil.copytree(seed, root, copy_function=os.link)
        manifest = json.loads((root / "mrl2d-input.json").read_bytes())
        run = json.loads(
            (root / "mrl2d-run.json").read_bytes().replace(str(seed).encode(), str(root).encode())
        )
        for name in ("0/k", "0/omega"):
            if case.model == "les":
                (root / name).unlink()
                manifest["input_sha256"].pop(name)
        for name, text in case_files(case, depth=0.01).items():
            put(root / name, text)
            manifest["input_sha256"][name] = digest(text.encode())
        manifest.update(name=case.name, model=case.model, envelope=case.envelope)
        run["source"]["native_files_sha256"] = synthetic_binary_sources(root, case.envelope)
        entries.append(seal(root, manifest, run))
        for name in ("mrl2d-input.json", "mrl2d-run.json"):
            original_reports[str(root / name)] = digest((root / name).read_bytes())
    specification = tmp_path / "SYNTHETIC-input.json"
    put_json(
        specification,
        {
            "schema": "mrl2d-postprocess-input-v1",
            "cases": entries,
            "fixture": "SYNTHETIC ONLY, never native production evidence",
        },
    )
    return specification, entries, original_reports


def synthetic_cli(spec, output, output_root):
    """Real CLI; relocate its allowed private output root ONLY, not any gate."""
    repo = Path(__file__).resolve().parents[1]
    code = (
        "import runpy,sys; from pathlib import Path; "
        "from flowacoustics import mrl2d_postprocess as p; "
        "p.OUTPUT_ROOT=Path(sys.argv.pop(1)); p.REPOSITORY=p.OUTPUT_ROOT; "
        "runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(output_root),
            str(repo / "scripts/postprocess_mrl2d.py"),
            "--input",
            str(spec),
            "--output",
            str(output),
            "--mode",
            "production",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(repo / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        },
    )


def test_synthetic_complete_10s_cli_pipeline_repeat_and_fingerprint_control(
    synthetic_study, tmp_path
):
    specification, entries, originals = synthetic_study
    for name in ("SYNTHETIC-first", "SYNTHETIC-repeat"):
        output = tmp_path / name
        result = synthetic_cli(specification, output, tmp_path)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["cases"] == 4
        for entry in entries:
            detail = json.loads((output / (entry["id"] + "-analysis.json")).read_bytes())
            audit = detail["native_audit"]
            assert audit["field_readback"]["frames_read_with_vtk"] == 500
            assert audit["completion"]["stages"]["reconstruction"]["status"] == "not_required"
            m = json.loads((Path(entry["root"]) / "mrl2d-input.json").read_bytes())
            assert audit["physical_input_fingerprint"] == independent_fingerprint(
                Path(entry["root"]), m
            )
            with np.load(output / (entry["id"] + ".npz")) as arrays:
                assert arrays["pressure_pa"].shape == (4000, 12)
                assert arrays["source_time_s"].shape == (5000,)
                assert arrays["coefficients_pa"].shape == (196, 12)
    hashes = [
        {p.name: digest(p.read_bytes()) for p in (tmp_path / name).iterdir()}
        for name in ("SYNTHETIC-first", "SYNTHETIC-repeat")
    ]
    assert hashes[0] == hashes[1]
    assert all(digest(Path(p).read_bytes()) == h for p, h in originals.items())
    root = Path(entries[-1]["root"])
    m = json.loads((root / "mrl2d-input.json").read_bytes())
    r = json.loads((root / "mrl2d-run.json").read_bytes())
    path = root / "system/fvSchemes"
    put(path, path.read_bytes().replace(b"default backward;", b"default Euler;"))
    m["input_sha256"]["system/fvSchemes"] = digest(path.read_bytes())
    entries[-1] = seal(root, m, r)
    put_json(specification, {"schema": "mrl2d-postprocess-input-v1", "cases": entries})
    denied = tmp_path / "SYNTHETIC-denied"
    result = synthetic_cli(specification, denied, tmp_path)
    assert result.returncode == 2 and "unmatched production" in result.stderr
    assert not denied.exists()


def test_synthetic_native_roundoff_and_float32_frame_times(seed, tmp_path):
    from test_mrl2d_completion import plane

    from flowacoustics.mrl2d_postprocess import validate_completion, verify_field_frames

    root = tmp_path / seed.name
    shutil.copytree(seed, root, copy_function=os.link)
    m = json.loads((root / "mrl2d-input.json").read_bytes())
    r = json.loads(
        (root / "mrl2d-run.json").read_bytes().replace(str(seed).encode(), str(root).encode())
    )
    t = float(np.cumsum(np.full(160000, 0.0000625))[-1])
    assert t != 10 and abs(t - 10) < 1e-9
    log = (root / "log.pimpleFoam").read_text()
    log = log.replace("MRL2D_POST time=10 ", f"MRL2D_POST time={t:.17g} ")
    put(root / "log.pimpleFoam", log)
    r["stages"]["pimpleFoam"]["log_sha256"] = digest(log.encode())
    r["post_solve"]["solved_time"] = t
    # Genuine Float32 serialization in a synthetic VTP; keep the actual VTK gate.
    for frame in r["animation"]:
        p = Path(frame["path"])
        time = float(p.parent.name)
        text = plane(float(np.float32(time))).replace('type="Float64"', 'type="Float32"')
        put(p, text)
        frame["sha256"] = digest(p.read_bytes())
    view = validate_completion(seal(root, m, r))
    assert view["time_roundoff_atol_s"] == 1e-9
    result = verify_field_frames(root, view["field_frames"])
    assert result["frames_read_with_vtk"] == 500
    bad = dict(view["field_frames"][0], time_s=0.02001)
    with pytest.raises(ValueError, match="time"):
        verify_field_frames(root, [bad])


def test_actual_vtk_gate_rejects_pinned_corrupt_synthetic_plane_before_output(
    synthetic_study, tmp_path
):
    specification, entries, _ = synthetic_study
    root = Path(entries[0]["root"])
    m = json.loads((root / "mrl2d-input.json").read_bytes())
    r = json.loads((root / "mrl2d-run.json").read_bytes())
    p = Path(r["animation"][0]["path"])
    put(
        p,
        p.read_bytes().replace(
            b'format="ascii">0 1 2</DataArray>', b'format="ascii">0 nan 2</DataArray>', 1
        ),
    )
    r["animation"][0]["sha256"] = digest(p.read_bytes())
    entries[0] = seal(root, m, r)
    put_json(specification, {"schema": "mrl2d-postprocess-input-v1", "cases": entries})
    output = tmp_path / "SYNTHETIC-corrupt-plane-denied"
    result = synthetic_cli(specification, output, tmp_path)
    assert result.returncode == 2 and "VTK field readback failed" in result.stderr
    assert not output.exists()


def test_all_real_four004_bounded_reports_are_rejected_without_reading_native_payloads(tmp_path):
    # Read original reports ONLY. No native sources/logs/planes/reductions/exports.
    repo = Path(__file__).resolve().parents[1]
    evidence = repo / "native-archives/bounded/parent-readback.json"
    if not evidence.exists():
        pytest.skip("optional parent bounded-run report absent")
    approved = json.loads(evidence.read_bytes())
    entries = []
    for case in case_specs():
        root = evidence.parent / "cases" / case.name
        original = (root / "mrl2d-run.json").read_bytes()
        assert digest(original) == approved["cases"][case.name]["native_report_sha256"]
        run = json.loads(original)
        entries.append(
            {
                "id": case.name,
                "root": str(root),
                "end_s": 10,
                "envelope_recorded": case.envelope,
                "source_sha256": {
                    str(Path(p).relative_to(root)): h
                    for p, h in run["source"]["native_files_sha256"].items()
                },
                "native_completion": {
                    "schema": "mrl2d-direct-native-completion-v2",
                    "input_sha256": digest((root / "mrl2d-input.json").read_bytes()),
                    "run_sha256": digest(original),
                },
            }
        )
    from flowacoustics.mrl2d_postprocess import validate_completion

    for entry in entries:
        with pytest.raises(ValueError, match="not complete_10s"):
            validate_completion(entry)
    specification = tmp_path / "bounded-report-rejection.json"
    put_json(specification, {"schema": "mrl2d-postprocess-input-v1", "cases": entries})
    result = synthetic_cli(specification, tmp_path / "denied-bounded", tmp_path)
    assert result.returncode == 2 and "not complete_10s" in result.stderr
    assert not (tmp_path / "denied-bounded").exists()
