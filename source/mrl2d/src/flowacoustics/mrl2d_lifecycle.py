"""Linux per-command subreaper; never enumerate or signal unrelated processes.

Run as a separate stdlib-only interpreter, NOT as a subreaper in the threaded
coordinator. Orphan descendants (including setsid/double fork) are adopted here.
Only direct, unreaped children are signalled through stable pidfds. ECHILD is the
completion proof; launcher exit or an empty process group is not.
"""

import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

owner_fd = None
_retained_owners = []
_termination = threading.local()


def termination_interrupt(error):
    """Called only by the coordinator handler; never acquire a signal-time lock."""
    if getattr(_termination, "depth", 0):
        _termination.error = error
    else:
        raise error


def termination_checkpoint():
    """Deliver a deferred coordinator interrupt at a cleanup-protected boundary."""
    error = getattr(_termination, "error", None)
    if error is not None:
        # A signal after an empty read must remain queued for the next checkpoint.
        _termination.error = None
        raise error


@contextmanager
def defer_termination():
    """Defer the coordinator's first SIGINT/TERM through an ownership handoff.

    Python handlers run on the main thread even when another OS thread receives
    a process-directed signal: pthread_sigmask in this thread alone is not enough.
    Thread-local deferral neither delays main-thread cancellation for a worker's
    critical section nor passes a blocked signal mask to the supervisor/children.
    """
    depth = getattr(_termination, "depth", 0)
    _termination.depth = depth + 1
    try:
        yield
    finally:
        _termination.depth = depth
        if not depth:
            termination_checkpoint()


@contextmanager
def owner_lease(stream):
    global owner_fd
    owner_fd = stream.fileno()
    try:
        yield
    finally:
        owner_fd = None


def retain_owner():
    """An uncertain cleanup must also refuse another same-process coordinator."""
    if owner_fd is not None:
        _retained_owners.append(os.dup(owner_fd))


def _signal_children(children_path, sig):
    # This single-threaded subreaper does not reap between enumeration and
    # pidfd_open: even exited children remain zombies, so PID reuse cannot occur.
    for token in children_path.read_text().split():
        fd = os.pidfd_open(int(token))
        try:
            try:
                signal.pidfd_send_signal(fd, sig)
            except ProcessLookupError:
                pass
        finally:
            os.close(fd)


def supervise(argv, timeout):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER (linux/prctl.h)
        raise OSError(ctypes.get_errno(), "cannot establish native subreaper")
    children = Path(f"/proc/self/task/{os.getpid()}/children")
    children.read_text()  # Check prerequisites before launching native work.
    fd = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(fd, 0)
    finally:
        os.close(fd)
    cancelled = False

    def cancel(signum, frame):
        nonlocal cancelled
        cancelled = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, cancel)
    started = time.monotonic()
    process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True)
    terminating = None
    timed_out, stranded, reaped = False, False, 0
    while True:
        try:
            while True:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
                reaped += 1
                if pid == process.pid:
                    process.returncode = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            # No descendants can remain: any orphan is adopted before waitpid
            # observes ECHILD, and no other thread here can create children.
            return dict(
                returncode=process.returncode,
                cleanup_confirmed=True,
                descendants_reaped=reaped,
                stranded_descendants=stranded,
                timed_out=timed_out,
            )
        now = time.monotonic()
        if select.select([0], [], [], 0)[0]:
            os.read(0, 1)  # Both explicit cancellation and parent EOF cancel.
            cancelled = True
        timed_out |= now - started >= timeout
        if process.returncode is not None and terminating is None:
            stranded = True
        if cancelled or timed_out or stranded:
            if terminating is None:
                terminating = now
            sig = signal.SIGKILL if now - terminating >= 0.5 else signal.SIGTERM
            _signal_children(children, sig)
        # Never abandon an unkillable descendant or release its owner lease.
        time.sleep(0.01)


if __name__ == "__main__":
    result_fd = int(sys.argv[1])
    outcome = supervise(sys.argv[3:], float(sys.argv[2]))
    os.write(result_fd, json.dumps(outcome).encode())
    os.close(result_fd)
