"""Pure generator preparation only: no Gmsh, OpenFOAM or native owner lock."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_production_mesh.py"


def test_emit_generator_matches_exact_four_literal_policy(tmp_path):
    assert SCRIPT.is_file(), "pure production-policy emitter is missing"
    target = tmp_path / "candidate generator ; '.py"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--emit-generator", str(target)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    original = (ROOT / "src/flowacoustics/mrl2d_mesh.py").read_text()
    expected = original
    for before, after in (
        ('"Mesh.MeshSizeMin", 0.00006', '"Mesh.MeshSizeMin", 0.0005'),
        ('"SizeMin": 0.0004', '"SizeMin": 0.0008'),
        ('"DistMax": 0.20', '"DistMax": 0.10'),
        ('"DistMax": 0.25', '"DistMax": 0.10'),
    ):
        assert expected.count(before) == 1
        expected = expected.replace(before, after)
    assert target.read_bytes() == expected.encode()
    assert sorted(tmp_path.iterdir()) == [target]
    assert "not executed" in result.stdout


def test_emitter_refuses_existing_or_linked_destinations(tmp_path):
    target = tmp_path / "preserve.py"
    target.write_text("original sentinel")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(tmp_path, target_is_directory=True)
    for destination in (target, link, linked_parent / "new.py", tmp_path / "missing/new.py"):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--emit-generator", str(destination)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 2, result.stderr
        assert target.read_text() == "original sentinel"
        assert not (tmp_path / "new.py").exists()
        assert not (tmp_path / "missing").exists()


def test_emitter_help_and_missing_action_never_prepare_or_run(tmp_path):
    for args, expected in ((["--help"], 0), ([], 2), (["--execute-native"], 2)):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == expected, result.stderr
    assert list(tmp_path.iterdir()) == []
