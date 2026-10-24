"""Current strict-2D flow contract; no retired workflow dependencies."""

import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


def api():
    spec = importlib.util.find_spec("flowacoustics.mrl2d_case")
    assert spec is not None, "new independent four-case configuration module is missing"
    from flowacoustics import mrl2d_case

    return mrl2d_case


def test_four_case_factory_has_identical_flow_within_pairs():
    m = api()
    cases = m.case_specs()
    assert [c.name for c in cases] == [
        "mrl2d_urans_body",
        "mrl2d_urans_permeable",
        "mrl2d_les_body",
        "mrl2d_les_permeable",
    ]
    for first, second in (cases[:2], cases[2:]):
        a, b = m.case_files(first, depth=0.01), m.case_files(second, depth=0.01)
        assert a.keys() == b.keys()
        assert (
            a["system/controlDict"].replace("envelope false", "envelope true")
            == b["system/controlDict"]
        )
        assert all(a[k] == b[k] for k in a if k != "system/controlDict")
        assert "endTime 10;" in a["system/controlDict"]
        assert "deltaT 6.25e-05;" in a["system/controlDict"]
        assert "formatOptions { vtk { format ascii; precision 17; } }" in a["system/controlDict"]
        assert "numberOfSubdomains 2;" in a["system/decomposeParDict"]
        assert "zSample 0;" in a["system/controlDict"]
        assert "point (0 0 0); normal (0 0 1);" in a["system/controlDict"]
        assert "empty" in a["0/U"]
        assert "movingWallVelocity" in a["0/U"]
        assert "frontAndBack" in a["0/p"]
        assert "freestreamValue uniform 0;" in a["0/p"]
    assert "kOmegaSST" in m.case_files(cases[0], depth=0.01)["constant/turbulenceProperties"]
    assert "WALE" in m.case_files(cases[2], depth=0.01)["constant/turbulenceProperties"]
    assert "cubeRootVol" in m.case_files(cases[2], depth=0.01)["constant/turbulenceProperties"]
    assert "multiMotion" in a["constant/dynamicMeshDict"]
    assert "omega -10.472" in a["constant/dynamicMeshDict"]
    assert "omega 5.236" in a["constant/dynamicMeshDict"]
    assert "carrierZone" in a["constant/dynamicMeshDict"]
    assert "blade2Zone" in a["constant/dynamicMeshDict"]
    source = Path(m.__file__).read_text()
    assert "import mrl_" not in source
    assert "cylinder" not in source


@pytest.mark.skipif(os.environ.get("MRL2D_NATIVE_TEST") != "1", reason="opt-in native coupon")
def test_native_recorder_coupon(tmp_path):
    root = Path(__file__).resolve().parents[1]
    native = root / "acoustics/mrl2d"
    assert (native / "mrl2dRecorder.C").exists(), "native compact recorder missing"
    cache = root / ".cache/mrl2d/flow"
    cache.mkdir(parents=True, exist_ok=True)
    import tempfile

    case = Path(tempfile.mkdtemp(prefix="coupon-", dir=cache))
    for name in ("0", "constant", "system"):
        (case / name).mkdir()
    foam = api()._foam
    (case / "system/blockMeshDict").write_text(
        foam(
            "blockMeshDict",
            """
scale 1;
vertices ((-.5 -.5 0) (.5 -.5 0) (.5 .5 0) (-.5 .5 0)
          (-.5 -.5 .01) (.5 -.5 .01) (.5 .5 .01) (-.5 .5 .01));
blocks (hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1));
edges ();
boundary (
 blade0 {type wall; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2));}
 frontAndBack {type empty; faces ((0 3 2 1) (4 5 6 7));}
);
mergePatchPairs ();
""",
        )
    )
    control = """
application postProcess; startFrom startTime; startTime 0; stopAt endTime; endTime 0;
deltaT .001; writeControl timeStep; writeInterval 1; writePrecision 17;
functions {sources {type mrl2dRecorder; libs ("libmrl2dRecorder.so");
 executeControl timeStep; executeInterval 1; writeControl none;
 depth .01; rho 1000; omega 10.472; envelope true; segmentsPerSide 10;
 bodyPatches (blade0); zSample .005;
}}
"""
    (case / "system/controlDict").write_text(foam("controlDict", control))
    (case / "system/fvSchemes").write_text(
        foam(
            "fvSchemes",
            """
ddtSchemes {default Euler;} gradSchemes {default Gauss linear;}
divSchemes {default none;} laplacianSchemes {default Gauss linear corrected;}
interpolationSchemes {default linear;} snGradSchemes {default corrected;}
""",
        )
    )
    (case / "system/fvSolution").write_text(foam("fvSolution", "solvers {}"))
    for name, cls, dims, value in (
        ("p", "volScalarField", "[0 2 -2 0 0 0 0]", "7"),
        ("U", "volVectorField", "[0 1 -1 0 0 0 0]", "(1 2 0)"),
    ):
        # Manufactured, spatially varying *native cell fields*: verification only.
        values = []
        for j in range(20):
            for i in range(20):
                x, y = -0.475 + 0.05 * i, -0.475 + 0.05 * j
                values.append(
                    f"{7 + 2 * x - 3 * y:.17g}" if name == "p" else f"({1 + x:.17g} {2 - y:.17g} 0)"
                )
        field_type = "scalar" if name == "p" else "vector"
        internal = f"nonuniform List<{field_type}> 400 (\n" + "\n".join(values) + "\n)"
        (case / f"0/{name}").write_text(
            foam(
                name,
                f"""
dimensions {dims}; internalField {internal};
boundaryField {{blade0 {{type fixedValue; value uniform {value};}}
frontAndBack {{type empty;}}}}
""",
                cls,
            )
        )
    commands = [
        (native, "wmake libso", "build"),
        (case, "blockMesh", "blockMesh"),
        (case, "postProcess -fields '(p U)' -time 0", "sample"),
    ]
    for directory, command, name in commands:
        result = subprocess.run(
            [
                "bash",
                "-c",
                f'source "${{MRL2D_FOAM_BASHRC:?}}" >/dev/null 2>&1 || exit 121; {command}',
            ],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=180,
        )
        (case / f"log.{name}").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (case, result.stdout, result.stderr)
    rows = np.fromfile(case / "postProcessing/mrl2dSources/rank0.bin", dtype="<f8")
    assert rows.size == (80 + 40) * 16
    rows = rows.reshape(-1, 16)
    body, envelope = rows[rows[:, 1] == 0], rows[rows[:, 1] == 1]
    np.testing.assert_allclose(body[:, 10], 7000, rtol=1e-13)
    np.testing.assert_allclose(body[:, 11:13], np.tile([1, 2], (80, 1)), atol=1e-13)
    np.testing.assert_allclose(
        envelope[:, 10], 1000 * (7 + 2 * envelope[:, 4] - 3 * envelope[:, 5]), rtol=1e-13
    )
    np.testing.assert_allclose(
        envelope[:, 11:13], np.column_stack((1 + envelope[:, 4], 2 - envelope[:, 5])), atol=1e-13
    )
    np.testing.assert_allclose(np.linalg.norm(rows[:, 7:9], axis=1), 1)
    assert np.all(rows[:, 9] > 0)
    assert np.all(rows[:, 15] >= 0)
    body, envelope = rows[rows[:, 1] == 0], rows[rows[:, 1] == 1]
    assert np.unique(body[:, 2]).size == 80
    np.testing.assert_allclose(body[:, 9].sum(), 4)
    np.testing.assert_allclose((body[:, 7:9] * body[:, 9:10]).sum(axis=0), 0, atol=1e-13)
    np.testing.assert_allclose(envelope[:, 9].sum(), 1.44)
    np.testing.assert_array_equal(envelope[:, 13:15], 0)
    # Actual native square-wall Cf lies at radius >=.5; -Sf points into its fluid.
    assert np.all((body[:, 4:6] * body[:, 7:9]).sum(axis=1) < 0)
    expected_v = np.column_stack((5.236 * (body[:, 5] + 0.055), -5.236 * body[:, 4]))
    np.testing.assert_allclose(body[:, 13:15], expected_v)
    # Verification-only extensions: real MPI decomposition and a second native reader.
    (case / "system/decomposeParDict").write_text(
        foam("decomposeParDict", "numberOfSubdomains 2; method scotch;")
    )
    for label, argv in (
        ("decompose", ["decomposePar"]),
        (
            "parallelSample",
            ["mpirun", "-np", "2", "postProcess", "-parallel", "-fields", "(p U)", "-time", "0"],
        ),
    ):
        result = api().native_command(argv, cwd=case, log=case / f"log.{label}", timeout=30)
        assert result["returncode"] == 0, Path(result["log"]).read_text()
    parallel_files = sorted(case.glob("processor*/postProcessing/mrl2dSources/rank*.bin"))
    assert len(parallel_files) == 2
    parallel = np.concatenate([np.fromfile(p, dtype="<f8").reshape(-1, 16) for p in parallel_files])
    order = np.lexsort((rows[:, 2], rows[:, 1]))
    parallel = parallel[np.lexsort((parallel[:, 2], parallel[:, 1]))]
    np.testing.assert_allclose(parallel[:, :15], rows[order, :15], atol=1e-10)
    readback = api().case_files(api().case_specs()[0], depth=0.01)["system/readbackDict"]
    (case / "system/readbackDict").write_text(readback)
    result = api().native_command(
        ["postProcess", "-dict", "system/readbackDict", "-fields", "(p U)", "-time", "0"],
        cwd=case,
        log=case / "log.volumeReadback",
        timeout=30,
    )
    assert result["returncode"] == 0, Path(result["log"]).read_text()
    assert "min(p)" in Path(result["log"]).read_text()
    (cache / "latest_coupon.txt").write_text(str(case))


def _source_fixture(case, *, bad=None):
    """Explicit synthetic binary fixture for reader tests; never a flow output."""
    records = []
    for t in (0.002, 0.004):
        for b in range(3):
            for j, (x, y, nx, ny) in enumerate(
                ((0.01, 0, 1, 0), (0, 0.01, 0, 1), (-0.01, 0, -1, 0), (0, -0.01, 0, -1))
            ):
                records.append(
                    [t, 0, b * 4 + j, b, x + b * 0.1, y, 0.005, nx, ny, 0.02, 100, 1, 2, 3, 4, j]
                )
    rows = np.array(records, dtype="<f8")
    if bad == "time":
        rows[12:, 0] = 0.005
    elif bad == "ids":
        rows[-1, 2] = 99
    elif bad == "normal":
        rows[0, 7:9] *= -1
    elif bad == "inward":
        rows[:, 7:9] *= -1
    elif bad == "nan":
        rows[0, 10] = np.nan
    directory = case / "postProcessing/mrl2dSources"
    directory.mkdir(parents=True)
    rows.tofile(directory / "rank0.bin")
    return rows


def test_reader_roundtrips_units_geometry_and_every_frame(tmp_path):
    m = api()
    assert hasattr(m, "read_sources"), "native history readback is missing"
    rows = _source_fixture(tmp_path)
    out = m.read_sources(tmp_path, envelope=False, end=0.004)
    np.testing.assert_array_equal(out["time"], [0.002, 0.004])
    assert out["position"].shape == (2, 12, 2)
    np.testing.assert_array_equal(out["pressure"], rows.reshape(2, 12, 16)[:, :, 10])
    np.testing.assert_array_equal(out["length"], 0.02)
    np.testing.assert_array_equal(out["surface_velocity"][0, 0], [3, 4])
    assert out["metadata"]["pressure_units"] == "Pa"
    assert out["metadata"]["body_depth_divisions"] == 1
    assert out["metadata"]["envelope_depth_divisions"] == 0
    assert out["metadata"]["native_files_sha256"]


def test_acoustic_adapter_preserves_frozen_api_and_rigid_panel_lengths(tmp_path):
    m = api()
    assert hasattr(m, "acoustic_source_kwargs"), "frozen acoustic API adapter missing"
    _source_fixture(tmp_path)
    data = m.read_sources(tmp_path, envelope=False, end=0.004)
    args = m.acoustic_source_kwargs(data, kind="body")
    assert args["kind"] == "body"
    assert args["ds"].shape == (12,)
    np.testing.assert_array_equal(args["p"], data["pressure"])
    np.testing.assert_array_equal(args["U"], data["fluid_velocity"])
    np.testing.assert_array_equal(args["blade_id"], data["blade"])
    assert args["provenance"]["pressure_conversion"] == "native_kinematic_times_rho_once"
    with pytest.raises(ValueError):
        m.acoustic_source_kwargs(data, kind="permeable")


@pytest.mark.parametrize("bad", ["time", "ids", "normal", "inward", "nan"])
def test_reader_rejects_corrupt_history(tmp_path, bad):
    m = api()
    assert hasattr(m, "read_sources"), "native history readback is missing"
    _source_fixture(tmp_path, bad=bad)
    with pytest.raises(ValueError):
        m.read_sources(tmp_path, envelope=False, end=0.004)


def test_native_audit_requires_all_steps_native_end_clock_exit_and_courant():
    m = api()
    assert hasattr(m, "audit_native_run"), "native execution admission gate missing"
    log = """trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).
Time = 0.00025
Courant Number mean: 0.01 max: 0.2
ExecutionTime = 0.1 s  ClockTime = 0 s
Time = 0.0005
Courant Number mean: 0.02 max: 0.3
ExecutionTime = 0.2 s  ClockTime = 1 s
End
"""
    kwargs = dict(returncode=0, dt=0.00025, end=0.0005, elapsed=1.1, wall_limit=30)
    report = m.audit_native_run(log, **kwargs)
    assert report["steps"] == 2
    assert report["max_courant"] == 0.3
    for invalid in (
        log.replace("End", ""),
        log.replace("0.0005", "0.0004"),
        log.replace("max: 0.3", "max: nan"),
        log.replace("max: 0.3", "max: 3"),
        log.replace("ClockTime", "NotAClock"),
        log + "FOAM FATAL ERROR",
    ):
        with pytest.raises(ValueError):
            m.audit_native_run(invalid, **kwargs)
    with pytest.raises(ValueError):
        m.audit_native_run(log, **{**kwargs, "returncode": 1})
    with pytest.raises(ValueError):
        m.audit_native_run(log, **{**kwargs, "elapsed": 31})


def test_preparation_writes_only_four_fresh_cases_and_freezes_inputs(tmp_path):
    m = api()
    assert hasattr(m, "prepare_cases"), "four-case preparation and source freeze missing"
    mesh = tmp_path / "mesh/constant/polyMesh"
    mesh.mkdir(parents=True)
    for name in ("points", "faces", "owner", "neighbour", "boundary", "cellZones"):
        (mesh / name).write_text("explicit nonnative unit fixture")
    output = tmp_path / "cfd"
    cases = m.prepare_cases(mesh.parent.parent, output, depth=0.01, ranks=2)
    assert len(cases) == 4
    assert sorted(x.name for x in output.iterdir()) == sorted(c.name for c in m.case_specs())
    for path in cases:
        manifest = m.verify_frozen(path)
        assert manifest["schema"] == "mrl2d-flow-input-v1"
        assert manifest["target_seconds"] == 10
        assert manifest["status"] == "prepared_not_run"
        assert manifest["model_label"] in ("2D SST URANS", "2D WALE / LES-model calculation")
    with pytest.raises(FileExistsError):
        m.prepare_cases(mesh.parent.parent, output, depth=0.01)
    (cases[0] / "0/cellToRegion").write_text("native checkMesh connectivity diagnostic")
    diagnostics = cases[0] / "constant/polyMesh/sets"
    diagnostics.mkdir()
    (diagnostics / "region0").write_text("native checkMesh cellSet diagnostic")
    m.verify_frozen(cases[0])
    (cases[0] / "0/U").write_text("changed input")
    with pytest.raises(ValueError, match="changed"):
        m.verify_frozen(cases[0])


@pytest.mark.parametrize("end", [0.02, 0.2])
def test_cache_smoke_preparation_preserves_all_production_cadences(tmp_path, end):
    import json
    import tempfile

    m = api()
    root = Path(__file__).resolve().parents[1]
    cache = root / ".cache/mrl2d/production-preflight/unit-preparations"
    cache.mkdir(parents=True, exist_ok=True)
    mesh = tmp_path / "mesh/constant/polyMesh"
    mesh.mkdir(parents=True)
    for name in ("points", "faces", "owner", "neighbour", "boundary", "cellZones"):
        (mesh / name).write_text("unit fixture")
    with tempfile.TemporaryDirectory(dir=cache) as output:
        cases = m.prepare_cases(mesh.parent.parent, output, depth=0.01, end=end)
        manifest = m.verify_frozen(cases[0])
        assert manifest["target_seconds"] == end
        assert manifest["scope"] == "bounded_smoke_not_production"
        control = (cases[0] / "system/controlDict").read_text()
        assert "writeInterval 3200;" in control
        assert "writeInterval 320;" in control
        assert "executeInterval 32;" in control
        assert json.loads((cases[0] / "mrl2d-input.json").read_text())["volume_cadence"] == 0.2
    with pytest.raises(ValueError, match="cache"):
        m.prepare_cases(
            mesh.parent.parent, Path(os.environ["TMPDIR"]) / "not-cache", depth=0.01, end=0.02
        )


def test_native_command_records_actual_exit_and_enforces_budget(tmp_path, monkeypatch):
    rc = tmp_path / "bashrc"
    rc.write_text(":\n")
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(rc))
    m = api()
    assert hasattr(m, "native_command"), "bounded native executor is missing"
    ok = m.native_command(
        ["python3", "-c", "print('coupon readback'); raise SystemExit(7)"],
        cwd=tmp_path,
        log=tmp_path / "log.exit",
        timeout=20,
    )
    assert ok["returncode"] == 7
    assert not ok["timed_out"]
    assert "coupon readback" in (tmp_path / "log.exit").read_text()
    timeout = m.native_command(
        ["python3", "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        log=tmp_path / "log.timeout",
        timeout=1,
    )
    assert timeout["timed_out"]
    assert timeout["returncode"] != 0


def test_native_command_monitor_stops_group_and_retains_failure(tmp_path, monkeypatch):
    import sys
    import time

    rc = tmp_path / "bashrc"
    rc.write_text(":\n")
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(rc))
    log = tmp_path / "log.monitor"

    def monitor():
        if "trigger" in log.read_text():
            raise ValueError("unsafe solved state")

    started = time.monotonic()
    result = api().native_command(
        [sys.executable, "-u", "-c", "import time; print('trigger'); time.sleep(20)"],
        cwd=tmp_path,
        log=log,
        timeout=10,
        monitor=monitor,
    )
    assert time.monotonic() - started < 3
    assert result["returncode"] != 0
    assert result["monitor_error"] == "unsafe solved state"
    assert not result["timed_out"]
    assert "trigger" in log.read_text()


def test_native_command_retains_interruption_receipt(tmp_path, monkeypatch):
    import sys

    rc = tmp_path / "bashrc"
    rc.write_text(":\n")
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(rc))

    def interrupt():
        raise KeyboardInterrupt("test coordinator termination")

    try:
        result = api().native_command(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            log=tmp_path / "log.interrupt",
            timeout=20,
            monitor=interrupt,
        )
    except KeyboardInterrupt:
        pytest.fail("interruption discarded native exit and metering receipt")
    assert result["returncode"] != 0
    assert "interrupted" in result["monitor_error"]
    assert (tmp_path / "log.interrupt.json").exists()


@pytest.mark.parametrize(
    "boundary",
    ["popen_return", "close_result", "first_poll", "cleanup_entry", "close_stdin", "wait"],
)
def test_real_post_spawn_sigterm_cancels_and_joins_supervisor(tmp_path, monkeypatch, boundary):
    """PROD-02-R1: actual Popen, signal handler and child; teardown is not the oracle."""
    import fcntl
    import json
    import signal
    import sys
    import time

    from flowacoustics import mrl2d_production as production

    m = api()
    (tmp_path / "bashrc").write_text(":\n")
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(tmp_path / "bashrc"))
    monkeypatch.setattr(production, "__file__", str(tmp_path / "src/flowacoustics/module.py"))
    ledger = production.NativeLedger(tmp_path / "ledger.json", limit=5)
    lock_path = tmp_path / ".cache/mrl2d/flow/native-run.lock"
    tokens = {
        "close_result": "os.close(result_fd)",
        "first_poll": "while process.poll() is None:",
        "cleanup_entry": "if process is not None:",
        "close_stdin": "process.stdin.close()",
        "wait": "process.wait()",
    }
    lines = Path(m.__file__).read_text().splitlines()
    target = (
        next(i for i, line in enumerate(lines, 1) if line.strip() == tokens[boundary])
        if boundary != "popen_return"
        else None
    )
    owned = {}
    observed = {}

    def trace(frame, event, arg):
        spawn_return = (
            boundary == "popen_return"
            and event == "return"
            and frame.f_code is subprocess.Popen.__init__.__code__
            and hasattr(frame.f_locals["self"], "pid")
        )
        boundary_line = (
            event == "line"
            and frame.f_code is m.native_command.__code__
            and frame.f_lineno == target
        )
        if not owned and (spawn_return or boundary_line):
            process = frame.f_locals["self"] if spawn_return else frame.f_locals["process"]
            native_frame = frame.f_back if spawn_return else frame
            fd = native_frame.f_locals["result_fd"]
            owned.update(process=process, result_fd=fd)
            try:
                owned["pipe_inode"] = os.fstat(fd).st_ino
            except OSError:
                pass  # The result writer is already closed at monitor/cleanup boundaries.
            until = time.monotonic() + 0.4
            while not (tmp_path / "child.pid").exists() and time.monotonic() < until:
                time.sleep(0.005)
            if (tmp_path / "child.pid").exists():
                owned["child_fd"] = os.pidfd_open(int((tmp_path / "child.pid").read_text()))
            with lock_path.open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    observed["owner_held_at_injection"] = False
                except BlockingIOError:
                    observed["owner_held_at_injection"] = True
            observed.update(injected=True, boundary=boundary, line=frame.f_lineno)
            signal.raise_signal(signal.SIGTERM)
        return trace

    def monitor():
        if (
            boundary in ("cleanup_entry", "close_stdin", "wait")
            and (tmp_path / "child.pid").exists()
        ):
            raise ValueError("fixture requests cleanup before its independent timeout")

    sys.settrace(trace)
    try:
        production.coordinate(
            [0],
            workers=1,
            setup=lambda: ledger.run(
                [
                    sys.executable,
                    "-c",
                    "import os,time; from pathlib import Path; "
                    "Path('child.pid').write_text(str(os.getpid())); time.sleep(2)",
                ],
                cwd=tmp_path,
                log=tmp_path / "log.interrupted",
                timeout=1.2,
                monitor=monitor,
            ),
            solve=lambda *args: None,
        )
    except KeyboardInterrupt as error:
        observed["caught"] = str(error)
    finally:
        sys.settrace(None)
    assert owned, "post-spawn signal boundary was not exercised"
    supervisor = owned["process"]
    try:
        observed["supervisor_alive_at_return"] = supervisor.poll() is None
        observed["stdin_open_at_return"] = not supervisor.stdin.closed
        observed["child_alive_at_return"] = False
        if "child_fd" in owned:
            try:
                signal.pidfd_send_signal(owned["child_fd"], 0)
                observed["child_alive_at_return"] = True
            except ProcessLookupError:
                pass
        receipt_path = tmp_path / "log.interrupted.json"
        observed["receipt"] = (
            json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        )
        observed["ledger"] = json.loads(ledger.path.read_text())
        (tmp_path / "launch-observed.json").write_text(json.dumps(observed, indent=2))
    finally:
        # Only exact fixture-owned handles; observations above precede this RED cleanup.
        supervisor.stdin.close()
        try:
            if os.fstat(owned["result_fd"]).st_ino == owned.get("pipe_inode"):
                os.close(owned["result_fd"])
        except OSError:
            pass
        supervisor.wait(timeout=3)
        if "child_fd" in owned:
            os.close(owned["child_fd"])
    assert observed["owner_held_at_injection"]
    assert not observed["supervisor_alive_at_return"], "SIGTERM abandoned the spawned supervisor"
    assert not observed["stdin_open_at_return"] and not observed["child_alive_at_return"]
    assert observed["receipt"]["cleanup_confirmed"]
    assert "interrupted" in observed["receipt"]["monitor_error"]
    assert observed["ledger"]["records"] or observed["ledger"]["reserved_seconds"] > 0
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_full_mesh_gate_uses_native_count_and_two_dimensions():
    m = api()
    assert hasattr(m, "audit_mesh_log"), "native mesh gate is missing"
    log = """cells: 1234
Mesh has 2 geometric (non-empty/wedge) directions (1 1 0)
Mesh has 2 solution (non-empty) directions (1 1 0)
Mesh OK.
End
"""
    assert m.audit_mesh_log(log, returncode=0)["cells"] == 1234
    for wrong in (
        log.replace("1234", "250001"),
        log + "Failed 1 mesh checks.",
        log.replace("Mesh OK.", ""),
        log.replace("2 geometric", "3 geometric"),
    ):
        with pytest.raises(ValueError):
            m.audit_mesh_log(wrong, returncode=0)


def test_runner_stops_at_native_bad_mesh_and_records_failure(tmp_path):
    m = api()
    assert hasattr(m, "run_case"), "one-owner final runner is missing"
    mesh = tmp_path / "mesh/constant/polyMesh"
    mesh.mkdir(parents=True)
    for name in ("points", "faces", "owner", "neighbour", "boundary", "cellZones"):
        (mesh / name).write_text("explicit deliberately invalid native test mesh")
    cases = m.prepare_cases(mesh.parent.parent, tmp_path / "cfd", depth=0.01, ranks=1)
    with pytest.raises(ValueError, match="motion"):
        m.run_case(cases[0], wall_seconds=30, motion_evidence=None)
    evidence = tmp_path / "motion.json"
    import json

    evidence.write_text(
        json.dumps(
            {
                "mesh_motion_accepted": True,
                "mesh_input_sha256": m.mesh_identity(cases[0]),
                "note": "unit fixture only; invalid mesh must stop before solver",
            }
        )
    )
    with pytest.raises(ValueError, match="checkMesh"):
        m.run_case(cases[0], wall_seconds=30, motion_evidence=evidence)
    report = json.loads((cases[0] / "mrl2d-run.json").read_text())
    assert report["status"] == "failed"
    assert report["stages"]["checkMesh"]["returncode"] != 0
    assert not (cases[0] / "log.pimpleFoam").exists()
    m.verify_frozen(cases[0])


def test_source_motion_gate_checks_native_positions_normals_and_surface_velocity():
    m = api()
    assert hasattr(m, "audit_source_motion"), "source-vs-native kinematic gate missing"
    omega = 10.472
    centers = np.array(
        [[0, 0.055], [np.sqrt(3) * 0.055 / 2, -0.0275], [-np.sqrt(3) * 0.055 / 2, -0.0275]]
    )
    times = np.array([0.002, 0.004])
    xy, normals, velocity = [], [], []
    offsets = np.array([[0.01, 0], [0, 0.01], [-0.01, 0], [0, -0.01]])
    for t in times:
        a, b = -omega * t, -omega * t / 2
        r1 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        r2 = np.array([[np.cos(b), -np.sin(b)], [np.sin(b), np.cos(b)]])
        c = np.repeat(centers @ r1.T, 4, axis=0)
        p = c + np.tile(offsets @ r2.T, (3, 1))
        xy.append(p)
        normals.append(np.tile(offsets / 0.01 @ r2.T, (3, 1)))
        velocity.append(
            np.column_stack((omega / 2 * (c[:, 1] + p[:, 1]), -omega / 2 * (c[:, 0] + p[:, 0])))
        )
    data = {
        "time": times,
        "position": np.array(xy),
        "normal": np.array(normals),
        "surface_velocity": np.array(velocity),
        "kind": np.zeros(12, dtype=int),
        "blade": np.repeat(np.arange(3), 4),
    }
    report = m.audit_source_motion(data)
    assert report["maximum_position_error_m"] < 1e-12
    data["surface_velocity"][1, 0, 0] += 0.01
    with pytest.raises(ValueError):
        m.audit_source_motion(data)


def test_serial_api_delegates_to_single_worker_production_path(tmp_path, monkeypatch):
    from flowacoustics import mrl2d_production

    calls = []

    def run(cases, **kwargs):
        calls.append((cases, kwargs))
        return [{"status": "unit sentinel"}]

    monkeypatch.setattr(mrl2d_production, "run_cases", run)
    result = api().run_case(tmp_path, wall_seconds=123, motion_evidence="proof.json")
    assert result == {"status": "unit sentinel"}
    assert calls[0][1]["workers"] == 1
    assert calls[0][1]["native_seconds"] == 123


def test_cli_is_explicit_and_prepare_only_by_default():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/run_mrl2d.py"
    assert script.exists(), "explicit opt-in four-case CLI missing"
    result = subprocess.run(
        [__import__("sys").executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "prepare" in result.stdout and "run" in result.stdout
    assert "run-many" in result.stdout
    batch = subprocess.run(
        [__import__("sys").executable, str(script), "run-many", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert batch.returncode == 0
    for option in ("--workers", "--native-seconds", "--work-root", "--min-free-gib"):
        assert option in batch.stdout


@pytest.mark.skipif(os.environ.get("MRL2D_NATIVE_TEST") != "1", reason="opt-in native dictionaries")
def test_all_generated_dictionaries_parse_natively_with_readback(tmp_path):
    m = api()
    files = m.case_files(m.case_specs()[0], depth=0.01)
    assert "system/readbackDict" in files, "independent native full-volume readback missing"
    root = Path(__file__).resolve().parents[1]
    import tempfile

    case = Path(tempfile.mkdtemp(prefix="dicts-", dir=root / ".cache/mrl2d/flow"))
    for name, text in files.items():
        path = case / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        result = m.native_command(
            ["foamDictionary", str(path), "-entry", "FoamFile", "-value"],
            cwd=case,
            log=case / ("log." + name.replace("/", "_")),
            timeout=20,
        )
        assert result["returncode"] == 0, Path(result["log"]).read_text()
