"""Pure AST-isolated locator/shell fixtures: NOT OpenFOAM or native evidence.

The existing autouse native/owner-lock guard remains enabled. Only the case
helper's pre-launch prefix or the mesh stage definition is compiled here;
no supervisor is run and no full mesh generator is executed.
"""

import ast
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def case_setup():
    tree = ast.parse((ROOT / "src/flowacoustics/mrl2d_case.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "native_command"
    )
    prefix = []
    for statement in function.body:
        # All supervisor and output creation follow this timing assignment.
        if isinstance(statement, ast.Assign) and any(
            isinstance(n, ast.Name) and n.id == "started" for n in ast.walk(statement.targets[0])
        ):
            break
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            prefix.append(statement)
    else:
        raise AssertionError("native_command prefix boundary changed")
    function.body = prefix + [
        ast.Return(
            value=ast.Tuple(
                elts=[ast.Name(id="bashrc", ctx=ast.Load()), ast.Name(id="shell", ctx=ast.Load())],
                ctx=ast.Load(),
            )
        )
    ]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = dict(os=os, math=math, Path=Path)
    exec(compile(module, "<pure-case-environment-prefix>", "exec"), scope)
    return scope["native_command"]([], cwd=ROOT, log="not-created.log", timeout=1)


@pytest.mark.parametrize("setting", [None, "missing-fixture"])
def test_case_requires_explicit_existing_bashrc(monkeypatch, tmp_path, setting):
    if setting is None:
        monkeypatch.delenv("MRL2D_FOAM_BASHRC", raising=False)
    else:
        monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(tmp_path / setting))
    with pytest.raises(ValueError, match="MRL2D_FOAM_BASHRC"):
        case_setup()


def mesh_stage():
    tree = ast.parse((ROOT / "src/flowacoustics/mrl2d_mesh.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_native_stage"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    scope = {"Path": Path}
    exec(compile(module, "<pure-mesh-stage>", "exec"), scope)
    return scope["_native_stage"]


@pytest.mark.parametrize("setting", [None, "missing-fixture"])
def test_mesh_rejects_missing_bashrc_before_outputs(monkeypatch, tmp_path, setting):
    import subprocess

    if setting is None:
        monkeypatch.delenv("MRL2D_FOAM_BASHRC", raising=False)
    else:
        monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(tmp_path / setting))
    case = tmp_path / "case"
    case.mkdir()
    stages = []

    def forbidden(*args, **kwargs):
        pytest.fail("missing bashrc reached command launch")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="MRL2D_FOAM_BASHRC"):
        mesh_stage()(case, "pure-fixture", ["never-executed"], stages)
    assert not (case / "logs").exists()
    assert stages == []


@pytest.mark.parametrize("helper", ["case", "mesh"])
def test_failing_source_cannot_run_sentinel(monkeypatch, tmp_path, helper):
    import shlex
    import subprocess
    import sys

    bashrc = tmp_path / "fixture bashrc ; ' $()"
    bashrc.write_text("# PURE SHELL FIXTURE, NOT OPENFOAM\nreturn 19\n")
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(bashrc))
    sentinel = tmp_path / "must-not-exist"
    args = [
        sys.executable,
        "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
        str(sentinel),
    ]
    if helper == "case":
        selected, shell = case_setup()
        assert selected == str(bashrc)
        result = subprocess.run(
            ["bash", "-c", shell, "fixture", selected, *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 121
    else:
        real_run = subprocess.run

        def fixture_only(argv, **kwargs):
            assert argv[:2] == ["bash", "-c"]
            assert argv[2].startswith(f"source {shlex.quote(str(bashrc))} ")
            return real_run(argv, **{**kwargs, "timeout": 5})

        monkeypatch.setattr(subprocess, "run", fixture_only)
        stages = []
        with pytest.raises(RuntimeError, match="returned 121"):
            mesh_stage()(tmp_path, "pure-fixture", args, stages)
        assert stages[0]["return_code"] == 121
    assert not sentinel.exists()


@pytest.mark.parametrize("helper", ["case", "mesh"])
def test_valid_quoted_fixture_preserves_arguments_and_thread_limits(monkeypatch, tmp_path, helper):
    """Verification-only coverage of existing shell behavior; no native command."""
    import json
    import subprocess
    import sys

    bashrc = tmp_path / "fixture bashrc ; ' $()"
    bashrc.write_text(
        "# PURE SHELL FIXTURE, NOT OPENFOAM\n"
        "export PURE_FIXTURE=ok OMP_NUM_THREADS=17 OPENBLAS_NUM_THREADS=17\n"
    )
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(bashrc))
    payload = "literal ; ' $(must-not-run) & spaces"
    args = [
        sys.executable,
        "-c",
        "import json,os,sys; print(json.dumps([sys.argv[1], "
        "os.environ['PURE_FIXTURE'], os.environ['OMP_NUM_THREADS'], "
        "os.environ['OPENBLAS_NUM_THREADS']]))",
        payload,
    ]
    if helper == "case":
        selected, shell = case_setup()
        result = subprocess.run(
            ["bash", "-c", shell, "fixture", selected, *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0, result.stderr
        output = result.stdout
        assert "MKL_NUM_THREADS=1" in shell
    else:
        stages = []
        output = mesh_stage()(tmp_path, "pure-fixture", args, stages)
        assert stages[0]["return_code"] == 0
    assert json.loads(output) == [payload, "ok", "1", "1"]
