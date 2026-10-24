"""Packaging-only isolation checks; no native runs, locks, or numerical changes."""

import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_copied_numerical_sources_and_import_roots_are_exact():
    manifest = ROOT / "SOURCE_IDENTITY.json"
    assert manifest.is_file(), "standalone copied-source identity file is missing"
    pins = json.loads(manifest.read_text())["unchanged_source_sha256"]
    assert pins
    for name, expected in pins.items():
        path = ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if name.startswith("src/") and name.endswith(".py"):
            module = name.removeprefix("src/").removesuffix(".py").replace("/", ".")
            if module.endswith(".__init__"):
                module = module.removesuffix(".__init__")
            location = importlib.import_module(module).__file__
            assert location is not None
            assert Path(location).resolve() == path.resolve()


def test_packaged_tests_do_not_require_private_machine_paths():
    import re

    offenders = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "tests").glob("*.py")
        if re.search(rb"/home/[A-Za-z0-9_-]+/", path.read_bytes())
    ]
    assert not offenders, f"private machine locators in copied tests: {offenders}"


def test_retained_initial_dictionaries_match_explicit_generator_parameters():
    from flowacoustics.mrl2d_case import case_files, case_specs

    for case in case_specs():
        files = case_files(case, depth=0.01, dt=0.0000625, ranks=2, end=10.0)
        retained = ROOT / "examples" / "initial-inputs" / case.name
        assert retained.is_dir(), "retained initial/configuration inputs are missing"
        actual = {str(p.relative_to(retained)) for p in retained.rglob("*") if p.is_file()}
        assert actual == set(files)
        for name, text in files.items():
            assert (retained / name).read_bytes() == text.encode()


def test_postprocessing_output_is_bound_to_this_copy_before_input_access():
    from flowacoustics.mrl2d_postprocess import OUTPUT_ROOT, REPOSITORY, export_study

    assert REPOSITORY == ROOT
    assert OUTPUT_ROOT == ROOT / ".cache/mrl2d/postprocessing"
    outside = ROOT.parent / "not-an-allowed-export"
    assert not outside.exists()
    with pytest.raises(ValueError, match="output"):
        export_study("missing-input-must-not-be-read.json", outside, mode="diagnostic")
    assert not outside.exists()


@pytest.mark.parametrize("script", ["build_mrl2d_mesh.py", "run_mrl2d.py", "postprocess_mrl2d.py"])
def test_standalone_cli_help_uses_the_active_python(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
