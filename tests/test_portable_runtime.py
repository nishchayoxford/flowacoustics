"""Portable public runtime locator; these tests do not launch OpenFOAM."""
from pathlib import Path
import pytest
from test_cylinder_native import native


def test_runtime_location_comes_from_sourced_openfoam(monkeypatch, tmp_path):
    monkeypatch.setenv("WM_PROJECT_DIR", str(tmp_path / "foam"))
    assert native().FOAM == Path(tmp_path / "foam")
    monkeypatch.delenv("WM_PROJECT_DIR")
    module = native()
    assert module.FOAM is None
    with pytest.raises(RuntimeError, match="WM_PROJECT_DIR"):
        module.run_command(["true"], "not-launched", 1, ledger=tmp_path / "commands.json")
