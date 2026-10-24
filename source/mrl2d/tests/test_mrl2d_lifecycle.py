"""Real bounded Python trees only: no OpenFOAM, MPI or external process scans."""

import json
import os
import subprocess
import sys

import pytest

TREE = r"""
import os, signal, sys, time
from pathlib import Path
p=Path(sys.argv[1]); mode=sys.argv[2]
if os.fork()==0:
    os.setsid()  # Separate session as well as MPI-style separate process group.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if os.fork()==0:
        os.setpgid(0,0)
        name='grandchild'
    else: name='child'
    (p/name).write_text(str(os.getpid()))
    time.sleep(3)  # Independent hard fixture lifetime, even on a broken runner.
    os._exit(0)
until=time.monotonic()+1
while not (p/'tracked').exists() and time.monotonic()<until: time.sleep(.005)
if mode=='timeout': time.sleep(3)
os._exit(7 if mode=='nonzero' else 0)
"""

HARNESS = r"""
import ctypes, json, os, signal, subprocess, sys, time
from pathlib import Path
from flowacoustics import mrl2d_case as m
p=Path(sys.argv[1]); mode=sys.argv[2]
# Only this test subprocess adopts its own deliberately orphaned RED fixtures.
assert ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
fds={}
def monitor():
    for name in ('child','grandchild'):
        if name not in fds and (p/name).exists():
            pid=int((p/name).read_text())
            fds[name]=(pid,os.pidfd_open(pid))
    if len(fds)==2:
        (p/'tracked').touch()
        if mode=='monitor': raise ValueError('fixture cancellation')
other=subprocess.Popen([sys.executable,'-c','import time; time.sleep(3)'])
try:
    result=m.native_command([sys.executable,str(p/'tree.py'),str(p),mode],
                            cwd=p,log=p/'log.tree',timeout=.6,monitor=monitor)
    live=[]
    for name,(pid,fd) in fds.items():
        try:
            signal.pidfd_send_signal(fd,0)
            live.append(name)
        except ProcessLookupError: pass
    (p/'observed.json').write_text(json.dumps(dict(result=result,live=live,
                                    other_alive=other.poll() is None,tracked=len(fds))))
finally:
    for pid,fd in fds.values():
        try: signal.pidfd_send_signal(fd,signal.SIGKILL)
        except ProcessLookupError: pass
        os.close(fd)
    other.kill(); other.wait()
    until=time.monotonic()+1
    while time.monotonic()<until:
        try:
            pid,status=os.waitpid(-1,os.WNOHANG)
            if pid==0: time.sleep(.005)
        except ChildProcessError: break
"""


def test_signal_at_checkpoint_read_clear_boundary_is_not_lost():
    """A first signal must not be erased by a checkpoint that just read no error."""
    import inspect
    import signal

    from flowacoustics import mrl2d_lifecycle as lifecycle
    from flowacoustics import mrl2d_production as production

    lines, first = inspect.getsourcelines(lifecycle.termination_checkpoint)
    target = first + next(i for i, line in enumerate(lines) if "error = getattr" in line) + 1
    injected = []

    def trace(frame, event, arg):
        if (
            event == "line"
            and frame.f_code is lifecycle.termination_checkpoint.__code__
            and frame.f_lineno == target
            and not injected
        ):
            injected.append(target)
            signal.raise_signal(signal.SIGTERM)
        return trace

    with production.termination_guard():
        try:
            with pytest.raises(KeyboardInterrupt, match="coordinator"):
                with lifecycle.defer_termination():
                    sys.settrace(trace)
                    lifecycle.termination_checkpoint()
        finally:
            sys.settrace(None)
    assert injected


def test_coordinator_deferral_nests_and_handles_delivery_from_another_thread():
    """Verification-only: OS-thread masks would not establish this Python invariant."""
    import signal
    import threading

    from flowacoustics import mrl2d_lifecycle as lifecycle
    from flowacoustics import mrl2d_production as production

    reached = []
    before = signal.getsignal(signal.SIGTERM)
    with production.termination_guard():
        with pytest.raises(KeyboardInterrupt, match="coordinator"):
            with lifecycle.defer_termination():
                with lifecycle.defer_termination():
                    sender = threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGTERM))
                    sender.start()
                    sender.join(timeout=1)
                    assert not sender.is_alive()
                    reached.append("inner")
                reached.append("outer")
        lifecycle.termination_checkpoint()  # No stale deferred error after delivery.
    assert reached == ["inner", "outer"]
    assert signal.getsignal(signal.SIGTERM) is before


def test_worker_deferral_does_not_mask_main_thread_cancellation():
    """Verification-only: a worker's long wait must not delay coordinator SIGTERM."""
    import signal
    import threading

    from flowacoustics import mrl2d_lifecycle as lifecycle
    from flowacoustics import mrl2d_production as production

    ready, release = threading.Event(), threading.Event()

    def worker():
        with lifecycle.defer_termination():
            ready.set()
            release.wait(1)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert ready.wait(1)
        with production.termination_guard():
            with pytest.raises(KeyboardInterrupt):
                signal.raise_signal(signal.SIGTERM)
    finally:
        release.set()
        thread.join(timeout=1)
    assert not thread.is_alive()


@pytest.mark.parametrize("mode", ["success", "nonzero", "timeout", "monitor"])
def test_owned_descendants_reaped_before_native_receipt(tmp_path, mode):
    (tmp_path / "tree.py").write_text(TREE)
    (tmp_path / "bashrc").write_text(":\n")
    with (tmp_path / "harness.log").open("w") as log:
        process = subprocess.run(
            [sys.executable, "-c", HARNESS, str(tmp_path), mode],
            env=dict(os.environ, MRL2D_FOAM_BASHRC=str(tmp_path / "bashrc")),
            stdout=log,
            stderr=log,
            timeout=5,
        )
    assert process.returncode == 0, (tmp_path / "harness.log").read_text()
    observed = json.loads((tmp_path / "observed.json").read_text())
    assert observed["tracked"] == 2
    assert observed["other_alive"], "signalled an unrelated owned test sibling"
    assert not observed["live"], "native receipt returned with surviving descendants"
    assert observed["result"].get("cleanup_confirmed") is True
    if mode in ("success", "nonzero"):
        assert observed["result"]["returncode"] == (0 if mode == "success" else 7)
        assert observed["result"]["monitor_error"], "stranded children silently accepted"
    elif mode == "timeout":
        assert observed["result"]["timed_out"]
    else:
        assert observed["result"]["monitor_error"] == "fixture cancellation"


def test_supervisor_retains_owner_after_controller_death(tmp_path):
    """The controller's death must not unlock a surviving Python native tree."""
    (tmp_path / "tree.py").write_text(TREE)
    (tmp_path / "bashrc").write_text(":\n")
    controller = r"""
import sys
from pathlib import Path
from flowacoustics import mrl2d_production as m
p=Path(sys.argv[1]); m.__file__=str(p/'src/flowacoustics/module.py')
def solve(case,built,cancel):
    return m.native_command([sys.executable,str(p/'tree.py'),str(p),'timeout'],
            cwd=p,log=p/'log.tree',timeout=2)
m.coordinate([0],workers=1,setup=lambda:None,solve=solve)
"""
    (tmp_path / "controller.py").write_text(controller)
    harness = r"""
import ctypes, fcntl, json, os, signal, subprocess, sys, time
from pathlib import Path
p=Path(sys.argv[1]); fds={}
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
controller=subprocess.Popen([sys.executable,str(p/'controller.py'),str(p)])
try:
    until=time.monotonic()+1.5
    while len(fds)<2 and time.monotonic()<until:
        for name in ('child','grandchild'):
            if name not in fds and (p/name).exists():
                fds[name]=os.pidfd_open(int((p/name).read_text()))
        time.sleep(.005)
    assert len(fds)==2
    controller.kill(); controller.wait()
    held=False
    with (p/'.cache/mrl2d/flow/native-run.lock').open('a+') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: held=True
    live=True; until=time.monotonic()+1.5
    while live and time.monotonic()<until:
        live=False
        for fd in fds.values():
            try: signal.pidfd_send_signal(fd,0); live=True
            except ProcessLookupError: pass
        time.sleep(.01)
    (p/'owner-observed.json').write_text(json.dumps(dict(held=held,live=live)))
finally:
    if controller.poll() is None: controller.kill(); controller.wait()
    for fd in fds.values():
        try: signal.pidfd_send_signal(fd,signal.SIGKILL)
        except ProcessLookupError: pass
        os.close(fd)
    until=time.monotonic()+1
    while time.monotonic()<until:
        try:
            if os.waitpid(-1,os.WNOHANG)[0]==0: time.sleep(.005)
        except ChildProcessError: break
"""
    with (tmp_path / "harness.log").open("w") as log:
        proc = subprocess.run(
            [sys.executable, "-c", harness, str(tmp_path)],
            env=dict(os.environ, MRL2D_FOAM_BASHRC=str(tmp_path / "bashrc")),
            stdout=log,
            stderr=log,
            timeout=5,
        )
    assert proc.returncode == 0, (tmp_path / "harness.log").read_text()
    observed = json.loads((tmp_path / "owner-observed.json").read_text())
    assert observed["held"], "controller death released owner while descendants were alive"
    assert not observed["live"], "parent EOF did not clean owned descendants"


def test_unconfirmed_cleanup_blocks_same_process_new_coordinator(tmp_path):
    """An uncertain supervisor cannot hand its lease to another controller."""
    script = r"""
from pathlib import Path
import sys
from flowacoustics import mrl2d_production as m
p=Path(sys.argv[1]); m.__file__=str(p/'src/flowacoustics/module.py')
ledger=m.NativeLedger(p/'ledger.json',limit=20)
m.native_command=lambda *args,**kw: dict(returncode=0,wall_seconds=.1)
def solve(*args):
    return ledger.run(['stub'],cwd=p,log=p/'log.stub',timeout=5)
try: m.coordinate([0],workers=1,setup=lambda:None,solve=solve)
except ValueError as e: assert 'cleanup' in str(e)
else: raise AssertionError('unconfirmed cleanup accepted')
assert ledger.cancel.is_set() and ledger.reserved>=5
try: m.coordinate([0],workers=1,setup=lambda:None,solve=lambda *args:None)
except RuntimeError as e: assert 'owner' in str(e)
else: raise AssertionError('uncertain owner released')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=2,
    )
    assert result.returncode == 0, result.stdout + result.stderr
