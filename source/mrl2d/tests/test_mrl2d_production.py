"""Production safety tests; textual log fixtures are not native evidence."""

import importlib.util
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec("flowacoustics.mrl2d_production"), (
        "production coordinator missing"
    )
    from flowacoustics import mrl2d_production

    return mrl2d_production


@pytest.mark.parametrize("failure", ["outcome", "receipt", "terminal_save", "directory_fsync"])
def test_unresolved_native_publication_poison_retains_reservation(tmp_path, monkeypatch, failure):
    """Review PROD-01: publication uncertainty must never buy another dispatch."""
    import json

    m = api()
    ledger = m.NativeLedger(tmp_path / "ledger.json", limit=20)
    native_calls = []
    original_save = ledger.save

    def command(*args, **kwargs):
        native_calls.append(args)
        if failure in ("outcome", "receipt"):
            raise OSError("unresolved " + failure)
        return {"wall_seconds": 0.25, "returncode": 0, "cleanup_confirmed": True}

    def fail_terminal():
        if ledger.records:
            if failure == "directory_fsync":
                original_save()  # Replacement is visible, durability remains uncertain.
            raise OSError("terminal publication uncertain")
        original_save()

    monkeypatch.setattr(m, "native_command", command)
    if failure in ("terminal_save", "directory_fsync"):
        monkeypatch.setattr(ledger, "save", fail_terminal)
    with pytest.raises(OSError):
        ledger.run(["fixture"], cwd=tmp_path, log=tmp_path / "first", timeout=5)
    assert ledger.reserved >= 5, "unresolved consumption released its reservation"
    assert ledger.cancel.is_set(), "publication failure did not cancel live peers"
    with pytest.raises(RuntimeError, match="poison"):
        ledger.run(["fixture"], cwd=tmp_path, log=tmp_path / "second", timeout=5)
    assert len(native_calls) == 1
    saved = json.loads(ledger.path.read_text())
    assert saved["reserved_seconds"] >= 5
    assert saved["poisoned"]
    with pytest.raises(ValueError, match="existing"):
        m.NativeLedger(ledger.path, limit=20)


@pytest.mark.parametrize("boundary", ["pop", "subtract", "append", "publish", "after_publish"])
def test_real_sigterm_terminal_transition_preserves_charge_and_cancels(
    tmp_path, monkeypatch, boundary
):
    """PROD-01-R1: real signals at semantic mutations, not fixed line numbers."""
    import json
    import signal
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    m = api()
    (tmp_path / "bashrc").write_text(":\n")
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(tmp_path / "bashrc"))
    ledger = m.NativeLedger(tmp_path / "ledger.json", limit=20)
    lines = Path(m.__file__).read_text().splitlines()
    tokens = {
        "pop": "self.pending.pop(key)",
        "subtract": 'self.reserved -= pending["seconds"]',
        "append": "self.records.append(record)",
        "publish": "self.save()",
        "after_publish": "self.save()",
    }
    start = next(i for i, line in enumerate(lines, 1) if "self.pending.pop(key)" in line)
    target = next(i for i, line in enumerate(lines, 1) if i >= start and tokens[boundary] in line)
    injected = []

    def trace(frame, event, arg):
        mutation = (
            boundary != "after_publish"
            and event == "line"
            and frame.f_code is m.NativeLedger.run.__code__
            and frame.f_lineno == target
        )
        published = (
            boundary == "after_publish"
            and event == "return"
            and frame.f_code is m.NativeLedger.save.__code__
            and any(r["log"] == str(first_log) for r in ledger.records)
        )
        if not injected and (mutation or published):
            injected.append(frame.f_lineno)
            signal.raise_signal(signal.SIGTERM)
        return trace

    monkeypatch.setattr(m, "__file__", str(tmp_path / "src/flowacoustics/module.py"))
    first_log = tmp_path / "log.first"
    with ThreadPoolExecutor(1) as pool:
        peer = pool.submit(
            ledger.run,
            [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(1.5)"],
            cwd=tmp_path,
            log=tmp_path / "log.peer",
            timeout=2,
        )
        until = time.monotonic() + 1
        while time.monotonic() < until:
            if (tmp_path / "log.peer").exists() and "ready" in (tmp_path / "log.peer").read_text():
                break
            time.sleep(0.005)
        assert (tmp_path / "log.peer").exists()
        sys.settrace(trace)
        try:
            with pytest.raises(KeyboardInterrupt):
                m.coordinate(
                    [0],
                    workers=1,
                    setup=lambda: ledger.run(
                        [sys.executable, "-c", "pass"], cwd=tmp_path, log=first_log, timeout=0.5
                    ),
                    solve=lambda *args: None,
                )
        finally:
            sys.settrace(None)
        receipt = json.loads(first_log.with_suffix(".first.json").read_text())
        saved = json.loads(ledger.path.read_text())
        observed = dict(
            injected=injected,
            reserved=ledger.reserved,
            pending=dict(ledger.pending),
            records=list(ledger.records),
            poisoned=ledger.poisoned,
            cancelled=ledger.cancel.is_set(),
            durable=saved,
        )
        (tmp_path / "terminal-observed.json").write_text(json.dumps(observed, indent=2))
        # Teardown is bounded even on RED; never substitute fixture teardown for cleanup.
        peer_result = peer.result(timeout=3)
    assert injected and receipt["cleanup_confirmed"]
    charged = any(r["log"] == str(first_log) for r in observed["records"])
    reserved = str(first_log.resolve()) in observed["pending"] and observed["reserved"] > 0
    assert charged or reserved, "SIGTERM lost both measured charge and reservation"
    assert observed["poisoned"] and observed["cancelled"], "unresolved transition did not poison"
    assert peer_result["cleanup_confirmed"] and "poison" in (peer_result["monitor_error"] or "")
    assert saved["poisoned"] and (saved["records"] or saved["reserved_seconds"] > 0)
    with pytest.raises(RuntimeError, match="poison"):
        m.coordinate(
            [0],
            workers=1,
            setup=lambda: ledger.run(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
                log=tmp_path / "log.forbidden",
                timeout=0.5,
            ),
            solve=lambda *args: None,
        )
    assert not (tmp_path / "log.forbidden").exists()


def test_reservation_is_durable_before_dispatch(tmp_path, monkeypatch):
    import json

    m = api()
    ledger = m.NativeLedger(tmp_path / "ledger.json", limit=20)

    def command(*args, **kwargs):
        data = json.loads(ledger.path.read_text())
        assert data.get("reserved_seconds", 0) >= 5, "reservation absent before launch"
        assert len(data["pending"]) == 1
        return {"wall_seconds": 0.25, "returncode": 0, "cleanup_confirmed": True}

    monkeypatch.setattr(m, "native_command", command)
    ledger.run(["fixture"], cwd=tmp_path, log=tmp_path / "first", timeout=5)
    data = json.loads(ledger.path.read_text())
    assert data["reserved_seconds"] == 0 and not data["pending"]
    assert data["charged_seconds"] == 0.25


def test_cleanup_proof_is_required_before_releasing_native_reservation(tmp_path, monkeypatch):
    m = api()
    ledger = m.NativeLedger(tmp_path / "ledger.json", limit=20)
    monkeypatch.setattr(
        m, "native_command", lambda *a, **kw: {"returncode": 0, "wall_seconds": 0.1}
    )
    with pytest.raises(ValueError, match="cleanup"):
        ledger.run(["fixture"], cwd=tmp_path, log=tmp_path / "first", timeout=5)
    assert ledger.reserved >= 5 and ledger.cancel.is_set()


def test_terminal_directory_fsync_failure_keeps_durable_poison(tmp_path, monkeypatch):
    import json
    import os
    import stat

    m = api()
    ledger = m.NativeLedger(tmp_path / "ledger.json", limit=20)
    original_fsync = os.fsync
    failures = []

    def fsync(fd):
        if ledger.records and stat.S_ISDIR(os.fstat(fd).st_mode) and not failures:
            failures.append("after replace, before directory durability")
            raise OSError("directory fsync failed")
        original_fsync(fd)

    monkeypatch.setattr(m.os, "fsync", fsync)
    monkeypatch.setattr(
        m,
        "native_command",
        lambda *a, **kw: {
            "returncode": 0,
            "wall_seconds": 0.1,
            "cleanup_confirmed": True,
        },
    )
    with pytest.raises(OSError, match="fsync"):
        ledger.run(["fixture"], cwd=tmp_path, log=tmp_path / "first", timeout=5)
    saved = json.loads(ledger.path.read_text())
    assert failures and saved["poisoned"] and saved["reserved_seconds"] >= 5
    assert not saved["records"]
    with pytest.raises(RuntimeError, match="poison"):
        ledger.run(["fixture"], cwd=tmp_path, log=tmp_path / "second", timeout=5)


@pytest.mark.parametrize("failure", ["hash", "receipt_after_replace"])
def test_real_receipt_failure_cancels_peer_and_refuses_redispatch(tmp_path, monkeypatch, failure):
    import sys
    import time
    from concurrent.futures import ThreadPoolExecutor

    from flowacoustics import mrl2d_case as flow

    m = api()
    (tmp_path / "bashrc").write_text(":\n")
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(tmp_path / "bashrc"))
    ledger = m.NativeLedger(tmp_path / "ledger.json", limit=20)
    original_sha, original_write = flow._sha, m.write_json

    def sha(path):
        if failure == "hash" and Path(path).name == "log.fail":
            raise OSError("receipt hash unavailable")
        return original_sha(path)

    def write(path, data):
        original_write(path, data)
        if failure == "receipt_after_replace" and Path(path).name == "log.fail.json":
            raise OSError("receipt directory fsync uncertain after replacement")

    monkeypatch.setattr(flow, "_sha", sha)
    monkeypatch.setattr(m, "write_json", write)
    with ThreadPoolExecutor(1) as pool:
        peer = pool.submit(
            ledger.run,
            [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(2)"],
            cwd=tmp_path,
            log=tmp_path / "log.peer",
            timeout=3,
        )
        until = time.monotonic() + 1
        while time.monotonic() < until:
            if (tmp_path / "log.peer").exists() and "ready" in (tmp_path / "log.peer").read_text():
                break
            time.sleep(0.005)
        with pytest.raises(OSError):
            ledger.run(
                [sys.executable, "-c", "pass"], cwd=tmp_path, log=tmp_path / "log.fail", timeout=3
            )
        result = peer.result(timeout=1)
    assert result["cleanup_confirmed"] is True
    assert "poison" in result["monitor_error"]
    assert ledger.reserved >= 3
    with pytest.raises(RuntimeError, match="poison"):
        ledger.run(
            [sys.executable, "-c", "pass"], cwd=tmp_path, log=tmp_path / "log.forbidden", timeout=3
        )
    assert not (tmp_path / "log.forbidden").exists()


def test_live_output_space_monitor_cancels_on_case_only_pressure(tmp_path, monkeypatch):
    import sys
    from collections import namedtuple

    m = api()
    (tmp_path / "bashrc").write_text(":\n")
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(tmp_path / "bashrc"))
    work, case = tmp_path / "work", tmp_path / "case"
    work.mkdir()
    case.mkdir()
    log = work / "log.fixture"
    Usage = namedtuple("Usage", "total used free")
    queried = []

    def usage(path):
        queried.append(Path(path))
        low = Path(path) == case and log.exists() and "ready" in log.read_text()
        return Usage(100, 99, 1) if low else Usage(100, 1, 99)

    monkeypatch.setattr(m.shutil, "disk_usage", usage)
    m.check_output_space([work, case], 10)
    result = m.native_command(
        [sys.executable, "-u", "-c", "import time; print('ready'); time.sleep(2)"],
        cwd=work,
        log=log,
        timeout=1,
        monitor=lambda: m.check_output_space([work, case], 10),
    )
    assert str(case) in result["monitor_error"]
    assert result["cleanup_confirmed"] and result["returncode"] != 0
    assert not result["timed_out"]
    assert work in queried and case in queried


def test_submission_interruption_sets_cancel_before_pool_shutdown(tmp_path, monkeypatch):
    """The reviewer's submission-boundary double, with the safe oracle."""
    m = api()
    events = []

    class Pool:
        def __init__(self, max_workers):
            pass

        def __enter__(self):
            return self

        def submit(self, solve, case, artifacts, cancel):
            events.append(cancel)
            if len(events) == 2:
                raise KeyboardInterrupt("submission")
            return object()

        def __exit__(self, *exc):
            assert events[0].is_set(), "submission shutdown left first worker uncancelled"

    monkeypatch.setattr(m, "__file__", str(tmp_path / "src/flowacoustics/module.py"))
    monkeypatch.setattr(m, "ThreadPoolExecutor", Pool)
    with pytest.raises(KeyboardInterrupt, match="submission"):
        m.coordinate([0, 1], workers=2, setup=lambda: None, solve=lambda *args: None)


def test_signal_handler_never_reenters_cancellation_event_lock(monkeypatch):
    """A signal can interrupt Event.set while its non-reentrant lock is held."""
    import signal

    m = api()
    with m.termination_guard() as cancel:

        def blocked_set():
            pytest.fail("signal handler re-entered a potentially held Event lock")

        monkeypatch.setattr(cancel, "set", blocked_set)
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
        # The first signal must latch cancellation without another asynchronous raise.
        signal.raise_signal(signal.SIGTERM)


@pytest.mark.parametrize("boundary", ["setup", "construction", "shutdown"])
def test_coordinator_signal_boundaries_keep_lock_until_shutdown(tmp_path, monkeypatch, boundary):
    import fcntl
    import signal

    m = api()
    monkeypatch.setattr(m, "__file__", str(tmp_path / "src/flowacoustics/module.py"))
    lock_path = tmp_path / ".cache/mrl2d/flow/native-run.lock"
    pool_class = m.ThreadPoolExecutor

    def check_locked():
        with lock_path.open("a+") as stream:
            with pytest.raises(BlockingIOError):
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)

    class Pool(pool_class):
        def __init__(self, *args, **kwargs):
            check_locked()
            if boundary == "construction":
                signal.raise_signal(signal.SIGTERM)
            super().__init__(*args, **kwargs)

        def __exit__(self, *args):
            if boundary == "shutdown":
                signal.raise_signal(signal.SIGTERM)
                signal.raise_signal(signal.SIGTERM)
            check_locked()
            return super().__exit__(*args)

    def setup():
        check_locked()
        if boundary == "setup":
            signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(m, "ThreadPoolExecutor", Pool)
    if boundary == "shutdown":
        assert m.coordinate([0], workers=1, setup=setup, solve=lambda *args: 7) == [7]
    else:
        with pytest.raises(KeyboardInterrupt):
            m.coordinate([0], workers=1, setup=setup, solve=lambda *args: 7)
    with lock_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("boundary", ["submission", "waiting"])
def test_repeated_sigterm_retains_owner_through_real_worker_cleanup(tmp_path, boundary):
    import subprocess
    import sys
    import time

    script = r"""
from pathlib import Path
import fcntl, sys, time
from flowacoustics import mrl2d_production as m
p=Path(sys.argv[1]); boundary=sys.argv[2]
m.__file__=str(p/'src/flowacoustics/module.py')
base=m.ThreadPoolExecutor
class Pool(base):
    def submit(self, *args, **kwargs):
        if boundary=='submission' and (p/'ready').exists():
            (p/'submitting').touch()
            time.sleep(1)
        f=super().submit(*args, **kwargs)
        while not (p/'ready').exists(): time.sleep(.005)
        return f
m.ThreadPoolExecutor=Pool
base_completed=m.as_completed
def completed(futures):
    (p/'waiting').touch()
    return base_completed(futures)
m.as_completed=completed
def solve(case, built, cancel):
    (p/'ready').touch()
    until=time.monotonic()+1.5
    while not cancel.wait(.005) and time.monotonic()<until: pass
    (p/'cleanup').touch()
    time.sleep(.15)
    with (p/'.cache/mrl2d/flow/native-run.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: (p/'locked').touch()
    (p/'finished').touch()
m.coordinate([0,1] if boundary=='submission' else [0],workers=1,
             setup=lambda:None,solve=solve)
"""
    with (tmp_path / "outer.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), boundary], stdout=log, stderr=log
        )
        try:
            until = time.monotonic() + 2
            marker = "submitting" if boundary == "submission" else "waiting"
            while not (tmp_path / marker).exists() and time.monotonic() < until:
                time.sleep(0.005)
            assert (tmp_path / marker).exists()
            process.terminate()
            until = time.monotonic() + 0.6
            while not (tmp_path / "cleanup").exists() and time.monotonic() < until:
                time.sleep(0.005)
            assert (tmp_path / "cleanup").exists(), "worker not cooperatively cancelled"
            process.terminate()
            process.wait(timeout=2)
            assert (tmp_path / "finished").exists(), "repeated SIGTERM abandoned cleanup"
            assert (tmp_path / "locked").exists(), "ownership released during cleanup"
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)


def test_sigterm_keeps_owner_until_cooperative_workers_finish(tmp_path):
    import subprocess
    import sys
    import time

    script = r"""
from pathlib import Path
import fcntl, sys, time
from flowacoustics.mrl2d_production import coordinate
root=Path(sys.argv[1]); marker=Path(sys.argv[2])
def solve(case, built, cancel):
    (marker/'ready').write_text('ready')
    try:
        while not cancel.wait(.01): pass
    finally:
        with (root/'.cache/mrl2d/flow/native-run.lock').open('a+') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: (marker/'locked-during-cleanup').write_text('yes')
        time.sleep(.05)
        (marker/'finished').write_text('yes')
coordinate([0],workers=1,setup=lambda:None,solve=solve)
"""
    root = Path(__file__).resolve().parents[1]
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(root), str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        until = time.monotonic() + 3
        while not (tmp_path / "ready").exists() and time.monotonic() < until:
            time.sleep(0.01)
        assert (tmp_path / "ready").exists()
        process.terminate()
        process.wait(timeout=3)
        assert process.returncode != 0
        assert (tmp_path / "finished").exists(), "SIGTERM abandoned workers"
        assert (tmp_path / "locked-during-cleanup").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_concurrent_mpi_launch_does_not_rebind_every_case_to_same_two_cores():
    m = api()
    assert hasattr(m, "mpi_prefix"), "explicit concurrent MPI affinity policy missing"
    assert m.mpi_prefix(1) == []
    assert m.mpi_prefix(2) == ["mpirun", "--bind-to", "none", "-np", "2"]
    with pytest.raises(ValueError):
        m.mpi_prefix(3)


def test_live_gate_reads_complete_lines_and_requires_final_solved_state(tmp_path):
    m = api()
    path = tmp_path / "log.flow"
    path.write_text(
        "trapFpe: Floating point exception trapping enabled\nTime = 0.002\n"
        "MRL2D_POST time=0.002 fluid=0.7 meshRaw=0.2 meshEffective=0.2 finite=1\n"
        "MRL2D_POST time=0.004"
    )
    gate = m.FlowWatch(path, dt=0.002, end=0.004)
    gate.poll()
    assert gate.status["solved_time"] == 0.002
    with pytest.raises(ValueError, match="post-solve"):
        gate.finish()
    with path.open("a") as stream:
        stream.write(" fluid=0.8 meshRaw=0.2 meshEffective=0.2 finite=1\nEnd\n")
    gate.poll()
    assert gate.finish()["post_solve_steps"] == 2
    assert gate.status["solved_time"] == 0.004
    assert gate.status["max_fluid_courant"] == 0.8


def test_solver_instrumentation_only_adds_read_only_post_pimple_guard(tmp_path):
    m = api()
    assert hasattr(m, "instrument_solver"), "production post-solve guard missing"
    # Labelled synthetic runtime-file bytes; not an installed/native solver audit.
    installed = (
        "// SYNTHETIC runtime-file fixture, not native CFD evidence\n"
        '#include "fvCFD.H"\nint main() {\n        runTime.write();\n}\n'
    )
    guarded = m.instrument_solver(installed)
    assert (
        guarded.replace('#include "productionGuard.H"\n', "").replace(
            "        productionGuard(mesh, U, phi);\n", ""
        )
        == installed
    )
    assert guarded.index("productionGuard(mesh, U, phi);") < guarded.index("runTime.write();")
    header = (
        Path(m.__file__).resolve().parents[2] / "acoustics/mrl2d/productionGuard.H"
    ).read_text()
    for token in (
        "fvc::surfaceSum",
        "fvc::meshPhi(U)",
        "mesh.phi()",
        "std::isfinite",
        "boundaryField()",
        "primitiveField()",
        "reduce(",
        "MRL2D_POST",
        "exit(FatalError)",
    ):
        assert token in header
    with pytest.raises(ValueError):
        m.instrument_solver("unknown solver main")


def test_shared_budget_carries_prior_and_accounts_concurrent_envelopes(tmp_path, monkeypatch):
    import json
    import sys
    from concurrent.futures import ThreadPoolExecutor

    m = api()
    assert hasattr(m, "NativeLedger"), "shared native metering missing"
    rc = tmp_path / "bashrc"
    rc.write_text(":\n")
    monkeypatch.setenv("MRL2D_FOAM_BASHRC", str(rc))
    ledger = m.NativeLedger(tmp_path / "ledger.json", limit=15, carry_forward=10)

    def job(i):
        return ledger.run(
            [sys.executable, "-c", "import time; time.sleep(.2)"],
            cwd=tmp_path,
            log=tmp_path / f"log.{i}",
            timeout=1,
        )

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(job, range(2)))
    data = json.loads(ledger.path.read_text())
    assert len(data["records"]) == 2
    assert data["charged_seconds"] == 10 + sum(r["wall_seconds"] for r in results)
    assert data["carry_forward_seconds"] == 10
    assert all(r["returncode"] == 0 for r in results)
    with pytest.raises(ValueError, match="existing"):
        m.NativeLedger(ledger.path, limit=15, carry_forward=0)


def test_coordinator_builds_once_and_holds_one_owner_through_four_jobs(tmp_path):
    import fcntl
    import threading
    import time

    m = api()
    assert hasattr(m, "coordinate"), "bounded shared-lock coordinator missing"
    active = 0
    maximum = 0
    builds = []
    mutex = threading.Lock()
    root = Path(m.__file__).resolve().parents[2]
    lock_path = root / ".cache/mrl2d/flow/native-run.lock"

    def setup():
        builds.append("once")
        return "frozen artifacts"

    def solve(case, artifacts, cancel):
        nonlocal active, maximum
        assert artifacts == "frozen artifacts"
        with lock_path.open("a+") as lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with mutex:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.05)
        with mutex:
            active -= 1
        return case

    assert m.coordinate(list(range(4)), workers=4, setup=setup, solve=solve) == list(range(4))
    assert maximum == 4
    assert builds == ["once"]
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="owner"):
            m.coordinate([0], workers=1, setup=setup, solve=solve)
    assert builds == ["once"]


def test_motion_proof_accepts_existing_coarse_worker_not_unrelated_mesh(tmp_path):
    import json

    m = api()
    assert hasattr(m, "motion_proof"), "mesh-worker proof reader missing"
    root = Path(m.__file__).resolve().parents[2]
    candidate = root / "native-archives/mesh"
    evidence = candidate / "mesh_manifest.json"
    if not evidence.is_file():
        pytest.skip("optional native mesh-motion archive absent; no evidence fabricated")
    proof = m.motion_proof(candidate, evidence)
    assert proof["kind"] == "existing_mesh_worker_manifest"
    assert proof["states"] == 12
    with pytest.raises(ValueError, match="mesh"):
        m.motion_proof(tmp_path, evidence)
    bad = tmp_path / "bad.json"
    data = json.loads(evidence.read_text())
    data["motion"]["all_faces_covered"] = False
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="motion"):
        m.motion_proof(candidate, bad)


@pytest.mark.parametrize("destination", ["all", "case"])
def test_run_cases_refuses_low_disk_without_launch_or_false_completion(
    tmp_path, monkeypatch, destination
):
    import json
    from collections import namedtuple

    from flowacoustics.mrl2d_case import mesh_identity, prepare_cases

    m = api()
    assert hasattr(m, "run_cases"), "production execution path missing"
    mesh = tmp_path / "mesh/constant/polyMesh"
    mesh.mkdir(parents=True)
    for name in ("points", "faces", "owner", "neighbour", "boundary", "cellZones"):
        (mesh / name).write_text("invalid native mesh fixture")
    cases = prepare_cases(mesh.parent.parent, tmp_path / "cases", depth=0.01)
    proof = tmp_path / "motion.json"
    proof.write_text(
        json.dumps(dict(mesh_motion_accepted=True, mesh_input_sha256=mesh_identity(cases[0])))
    )
    Usage = namedtuple("Usage", "total used free")
    queries = []

    def usage(path):
        queries.append(Path(path))
        low = destination == "all" or Path(path) == cases[0]
        return Usage(100, 99, 1) if low else Usage(100, 1, 99)

    def no_native(*args, **kwargs):
        pytest.fail("low case filesystem reached native dispatch")

    monkeypatch.setattr(m.shutil, "disk_usage", usage)
    monkeypatch.setattr(m, "native_command", no_native)
    with pytest.raises(ValueError, match="disk"):
        m.run_cases(
            cases[:2],
            motion_evidence=proof,
            wall_seconds=60,
            native_seconds=60,
            work_root=tmp_path / "run",
            workers=2,
            min_free_bytes=10,
        )
    assert not list((tmp_path / "run").glob("log.*"))
    assert not list(cases[0].glob("log.*"))
    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "failed"
    assert not (cases[0] / "mrl2d-run.json").exists()
    if destination == "case":
        assert cases[0] in queries


@pytest.mark.skipif(
    __import__("os").environ.get("MRL2D_PRODUCTION_NATIVE") != "1",
    reason="bounded explicit native production smoke",
)
def test_native_production_path_sst_wale_concurrent_ascii_and_sources():
    import json
    import os

    from flowacoustics.mrl2d_case import _sha, prepare_cases

    m = api()
    root = Path(m.__file__).resolve().parents[2]
    work = root / ".cache/mrl2d/production-preflight" / os.environ["MRL2D_PRODUCTION_ATTEMPT"]
    cases = prepare_cases(root / "native-archives/mesh", work / "cases", depth=0.01, end=0.02)
    library = Path(os.environ["FOAM_USER_LIBBIN"]) / "libmrl2dRecorder.so"
    historical_library = _sha(library)
    for a, b in (cases[:2], cases[2:]):
        for sub in ("0", "constant", "system"):
            for p in (a / sub).rglob("*"):
                if p.is_file():
                    other = b / p.relative_to(a)
                    if p.name == "controlDict":
                        assert (
                            p.read_text().replace("envelope false", "envelope true")
                            == other.read_text()
                        )
                    else:
                        assert p.read_bytes() == other.read_bytes()
    reports = m.run_cases(
        [cases[1], cases[3]],
        motion_evidence=root / "native-archives/mesh/mesh_manifest.json",
        wall_seconds=180,
        native_seconds=float(os.environ.get("MRL2D_NATIVE_LIMIT", "210")),
        carry_forward=float(os.environ.get("MRL2D_NATIVE_CARRY", "919.1972769846907")),
        work_root=work / "execution",
        workers=2,
        min_free_bytes=20 * 1024**3,
    )
    assert len(reports) == 2
    for report in reports:
        assert report["status"] == "complete_bounded_smoke_native_verified"
        assert report["post_solve"]["post_solve_steps"] == 320
        assert report["source"]["frames"] == 10
        assert report["source"]["sources"] == 866
        assert report["animation_frames"] == 1
        assert report["animation"][0]["vtk_header_valid"]
        assert report["volume_frames_read_natively"] == 0
    ledger = json.loads((work / "execution/ledger.json").read_text())
    assert sum("wmake" in r["argv"] and "libso" in r["argv"] for r in ledger["records"]) == 1
    assert _sha(library) == historical_library
    flows = [r for r in ledger["records"] if Path(r["log"]).name == "log.pimpleFoam"]
    assert max(r["started_unix"] for r in flows) < min(r["finished_unix"] for r in flows)


@pytest.mark.parametrize(
    "line",
    [
        "MRL2D_POST time=0.002 fluid=1.01 meshRaw=0.2 meshEffective=0.2 finite=1",
        "MRL2D_POST time=0.002 fluid=nan meshRaw=0.2 meshEffective=0.2 finite=1",
        "MRL2D_POST time=0.002 fluid=0.2 meshRaw=0.2 meshEffective=0.2 finite=0",
        "AMI: Patch source sum(weights) min:0 max:1 average:0.9",
        "FOAM FATAL ERROR",
        "Courant Number mean: 0.2 max: 1.1",
    ],
)
def test_live_gate_rejects_unsafe_output(tmp_path, line):
    m = api()
    path = tmp_path / "log.flow"
    path.write_text(line + "\n")
    with pytest.raises(ValueError):
        m.FlowWatch(path, dt=0.002, end=0.002).poll()
