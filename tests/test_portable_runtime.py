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


def test_longrun_rejects_missing_runtime_before_any_write(monkeypatch, tmp_path):
    from test_cylinder_longrun import runner

    module = runner()
    monkeypatch.setattr(module.native, "FOAM", None)
    with pytest.raises(RuntimeError, match="WM_PROJECT_DIR"):
        module.run_command(["true"], "not-launched", ledger=tmp_path / "commands.json")
    assert not (tmp_path / "commands.json").exists()


def test_longrun_failed_environment_does_not_launch_command(monkeypatch, tmp_path):
    import sys
    from test_cylinder_longrun import runner

    module = runner()
    foam = tmp_path / "synthetic runtime's path"
    bashrc = foam / "etc/bashrc"
    bashrc.parent.mkdir(parents=True)
    bashrc.write_text("return 7\n")
    monkeypatch.setattr(module.native, "FOAM", foam)
    marker = tmp_path / "should-not-launch"
    command = [sys.executable, "-c", "from pathlib import Path; Path(__import__('sys').argv[1]).touch()", str(marker)]
    result = module.run_command(command, "failed-environment", ledger=tmp_path / "commands.json")
    assert not marker.exists(), "Failed environment must prevent command execution"
    assert result["exit_code"] == 7 and result["state"] == "failed"
