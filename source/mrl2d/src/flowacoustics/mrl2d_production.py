"""One native owner, bounded coordinated cases, fail-closed production checks."""

import fcntl
import json
import math
import os
import re
import shutil
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from pathlib import Path

from flowacoustics import mrl2d_lifecycle
from flowacoustics.mrl2d_case import native_command


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(data, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class NativeLedger:
    """Reserve concurrent timeouts, charge measured process walls (never elapsed union)."""

    def __init__(self, path, *, limit, carry_forward=0):
        self.path = Path(path)
        if self.path.exists():
            raise ValueError("refusing existing ledger; no budget reset")
        if not 0 <= carry_forward < limit or not math.isfinite(limit):
            raise ValueError("invalid cumulative native budget")
        self.limit, self.carry = limit, carry_forward
        self.records = []
        self.reserved = 0.0
        self.pending = {}
        self.poisoned = None
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.save()

    def save(self):
        write_json(
            self.path,
            dict(
                limit_seconds=self.limit,
                carry_forward_seconds=self.carry,
                charged_seconds=self.carry + sum(r["wall_seconds"] for r in self.records),
                records=self.records,
                reserved_seconds=self.reserved,
                pending=self.pending,
                poisoned=self.poisoned,
            ),
        )

    def _poison(self, error):
        # Called while holding the dispatch lock. Peers see cancellation before
        # any waiter can acquire the lock, even when the filesystem is unavailable.
        self.poisoned = repr(error)
        self.cancel.set()
        try:
            self.save()
        except BaseException:
            pass  # The last durable pending reservation is still unresolved.

    def run(self, command, *, cwd, log, timeout, monitor=None):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("positive finite native timeout required")
        key = str(Path(log).resolve())
        with self.lock:
            if self.poisoned is not None:
                raise RuntimeError("native ledger poisoned; reconciliation required")
            if key in self.pending:
                raise ValueError("duplicate native log reservation")
            available = (
                self.limit
                - self.carry
                - sum(r["wall_seconds"] for r in self.records)
                - self.reserved
            )
            # Reserve process-group termination overhead separately for every live child.
            allowance = min(timeout, available - 1.5)
            if allowance <= 0:
                raise RuntimeError("shared native wall budget exhausted")
            self.reserved += allowance + 1.5
            self.pending[key] = dict(argv=list(map(str, command)), seconds=allowance + 1.5)
            try:
                self.save()  # Durable intent precedes creation of any process.
            except BaseException as error:
                self._poison(error)
                raise

        def checked_monitor():
            if self.cancel.is_set():
                raise RuntimeError("native ledger poisoned; cancelling peer")
            if monitor is not None:
                monitor()

        try:
            checked_monitor()
            record = native_command(
                command, cwd=cwd, log=log, timeout=allowance, monitor=checked_monitor
            )
            if record.get("cleanup_confirmed") is not True:
                mrl2d_lifecycle.retain_owner()
                raise ValueError("native descendant cleanup unconfirmed")
            if not math.isfinite(record["wall_seconds"]) or record["wall_seconds"] < 0:
                raise ValueError("unresolved native wall charge")
            with self.lock:
                with mrl2d_lifecycle.defer_termination():
                    pending = self.pending[key]
                    reserved, count = self.reserved, len(self.records)
                    try:
                        self.pending.pop(key)
                        self.reserved -= pending["seconds"]
                        self.records.append(record)
                        self.save()
                    except BaseException as error:
                        # Restore absolute snapshots, not a partially executed inverse.
                        self.pending[key] = pending
                        self.reserved = reserved
                        del self.records[count:]
                        self._poison(error)
                        raise
        except BaseException as error:
            with self.lock:
                self._poison(error)
            raise
        return record


def mpi_prefix(ranks):
    """Avoid independently launched MPI jobs all binding to physical cores 0/1.

    Native affinity readback verified --bind-to none on this host. The OS schedules
    at most eight solver ranks; each still has one numerical thread. No oversubscribe.
    """
    if ranks not in (1, 2):
        raise ValueError("one or two ranks per coordinated case required")
    return ["mpirun", "--bind-to", "none", "-np", str(ranks)] if ranks > 1 else []


def check_output_space(paths, minimum):
    """Check destinations, not just the build workspace (mounts may differ)."""
    for path in paths:
        path = Path(path)
        while not path.exists():
            path = path.parent
        if shutil.disk_usage(path).free < minimum:
            raise ValueError(f"low disk: native execution refused at {path}")


@contextmanager
def termination_guard():
    """Convert service SIGTERM to cooperative cancellation while ownership is held."""
    cancel = threading.Event()
    if threading.current_thread() is not threading.main_thread():
        yield cancel
        return
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        # Event.set acquires a non-reentrant lock; never call it from a signal
        # handler that could have interrupted the main thread inside Event.set.
        if not interrupted and not cancel.is_set():
            interrupted = True
            mrl2d_lifecycle.termination_interrupt(
                KeyboardInterrupt(f"coordinator received signal {signum}")
            )

    for sig in previous:
        signal.signal(sig, interrupt)
    try:
        yield cancel
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def coordinate(cases, *, workers, setup, solve):
    """One owner encompasses setup, every concurrent child, and completion checks."""
    if not 1 <= workers <= 4 or not 1 <= len(cases) <= 4:
        raise ValueError("one to four cases/workers required")
    path = Path(__file__).resolve().parents[2] / ".cache/mrl2d/flow/native-run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with termination_guard() as stopping, path.open("a+") as lock, ExitStack() as stack:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another mrl2d native flow owner holds the lock") from error
        stack.enter_context(mrl2d_lifecycle.owner_lease(lock))
        cancel = stopping if stopping is not None else threading.Event()
        results, error = {}, None
        try:
            artifacts = setup()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                try:
                    futures = {
                        pool.submit(solve, c, artifacts, cancel): i for i, c in enumerate(cases)
                    }
                    for future in as_completed(futures):
                        try:
                            results[futures[future]] = future.result()
                        except Exception as caught:
                            cancel.set()
                            if error is None:
                                error = caught
                finally:
                    # Also covers submission failures and repeated termination
                    # during __exit__/join; ownership outlives every worker.
                    cancel.set()
        except BaseException:
            cancel.set()
            raise
        if error is not None:
            raise error
        return [results[i] for i in range(len(cases))]


def run_cases(
    cases,
    *,
    motion_evidence,
    wall_seconds,
    native_seconds,
    work_root,
    workers=4,
    min_free_bytes=20 * 1024**3,
    carry_forward=0.0,
):
    """Fresh cases only; native budget is the SUM of concurrent process walls.

    Failed histories remain untouched and INVALID, even if their last frame is complete.
    No resumption, repair, truncation or partial-source acoustic acceptance is performed.
    """
    import time

    import numpy as np

    from flowacoustics import mrl2d_case as flow
    from flowacoustics.mrl2d_runtime_audit import section_summary

    cases = list(map(lambda p: Path(p).resolve(), cases))
    if not 1 <= len(cases) <= 4 or len(set(cases)) != len(cases) or not 1 <= workers <= 4:
        raise ValueError("one to four unique cases/workers required")
    if any(not math.isfinite(v) or v <= 0 for v in (wall_seconds, native_seconds, min_free_bytes)):
        raise ValueError("positive finite wall/native/disk budgets required")
    root = Path(__file__).resolve().parents[2]
    work = Path(work_root).resolve()
    if work.exists():
        raise FileExistsError("refusing existing execution workspace")
    manifests = {c: flow.verify_frozen(c) for c in cases}
    if any(m["ranks"] not in (1, 2) for m in manifests.values()):
        raise ValueError("coordinated execution requires one or two ranks per case; maximum eight")
    proofs = {c: motion_proof(c, motion_evidence) for c in cases}
    for case in cases:
        if (
            (case / "mrl2d-run.json").exists()
            or list(case.glob("processor*"))
            or list(case.glob("log.*"))
        ):
            raise FileExistsError("refusing attempted flow case; no resume/overwrite")
    work.mkdir(parents=True)
    started = time.monotonic()
    status_lock = threading.Lock()
    status = dict(
        status="validating",
        cases={str(c): dict(status="queued") for c in cases},
        wall_limit_seconds=wall_seconds,
        native_limit_seconds=native_seconds,
        workers=workers,
        maximum_total_ranks=workers * 2,
        incomplete_sources="INVALID on any failure; may contain partial frames; never resume",
    )
    ledger = NativeLedger(
        work / "ledger.json", limit=carry_forward + native_seconds, carry_forward=carry_forward
    )
    sources = flow._source_hashes()
    artifacts = {}
    reports = {}
    output_paths = [work, work / "solver", work / "recorder", *cases]
    for case in cases:
        output_paths.extend([case / "postProcessing", case / "postProcessing/animation"])
        for rank in range(manifests[case]["ranks"]):
            output_paths.extend(
                [case / f"processor{rank}", case / f"processor{rank}/postProcessing"]
            )

    def save_status(case=None, **updates):
        with status_lock:
            if case is None:
                status.update(updates)
            else:
                status["cases"][str(case)].update(updates)
            status["wall_seconds"] = time.monotonic() - started
            write_json(work / "status.json", status)

    def safety(cancel=None):
        if cancel is not None and cancel.is_set():
            raise ValueError("peer failure: cancelling coordinated native work")
        if time.monotonic() - started > wall_seconds:
            raise ValueError("total execution wall timeout")
        check_output_space(output_paths, min_free_bytes)
        if flow._source_hashes() != sources:
            raise ValueError("protected execution source identity changed")
        for path, digest in artifacts.items():
            if not Path(path).is_file() or flow._sha(path) != digest:
                raise ValueError("protected native library/solver identity changed")

    def stage(name, command, directory, report=None, monitor=None, cancel=None, cap=60):
        safety(cancel)
        result = ledger.run(
            command,
            cwd=directory,
            log=directory / f"log.{name}",
            timeout=min(cap, wall_seconds - (time.monotonic() - started)),
            monitor=monitor or (lambda: safety(cancel)),
        )
        if report is not None:
            report["stages"][name] = result
        text = Path(result["log"]).read_text(errors="replace")
        if (
            result["returncode"]
            or result["timed_out"]
            or result["monitor_error"]
            or "FOAM FATAL" in text
        ):
            raise ValueError(
                f"native {name} failed: {result['monitor_error']}; see {result['log']}"
            )
        return result, text

    def setup():
        safety()
        save_status(status="setup")
        # Check each actual copied mesh before compiling or launching any solver.
        for case in cases:
            report = dict(case=str(case), status="running", stages={}, motion_evidence=proofs[case])
            reports[case] = report
            result, text = stage(
                "checkMesh", ["checkMesh", "-allGeometry", "-allTopology"], case, report
            )
            report["mesh"] = flow.audit_mesh_log(text, returncode=result["returncode"])
        frozen = work / "source-snapshot"
        frozen.mkdir()
        mapping = {}
        for i, (path, digest) in enumerate(sources.items()):
            target = frozen / f"{i}-{Path(path).name}"
            shutil.copy2(path, target)
            mapping[path] = dict(snapshot=str(target), sha256=digest)
        write_json(work / "execution-sources.json", mapping)
        _, text = stage(
            "environment",
            [
                "python3",
                "-c",
                "import os,json; print(json.dumps({k:os.environ[k] "
                'for k in ("WM_PROJECT_DIR","LD_LIBRARY_PATH")}))',
            ],
            work,
        )
        environment = json.loads(text)
        installed = (
            Path(environment["WM_PROJECT_DIR"]) / "applications/solvers/incompressible/pimpleFoam"
        )
        native = work / "solver"
        native.mkdir()
        (native / "Make").mkdir()
        original = work / "installed-pimpleFoam"
        original.mkdir()
        for path in [
            *installed.glob("*.H"),
            installed / "pimpleFoam.C",
            installed / "Make/options",
        ]:
            target = original / path.name
            shutil.copy2(path, target)
            if path.suffix == ".H":
                shutil.copy2(path, native / path.name)
        (native / "productionPimpleFoam.C").write_text(
            instrument_solver((installed / "pimpleFoam.C").read_text())
        )
        shutil.copy2(root / "acoustics/mrl2d/productionGuard.H", native / "productionGuard.H")
        shutil.copy2(installed / "Make/options", native / "Make/options")
        solver = native / "productionPimpleFoam"
        (native / "Make/files").write_text(f"productionPimpleFoam.C\nEXE = {solver}\n")
        recorder = work / "recorder"
        (recorder / "Make").mkdir(parents=True)
        shutil.copy2(root / "acoustics/mrl2d/mrl2dRecorder.C", recorder / "mrl2dRecorder.C")
        shutil.copy2(root / "acoustics/mrl2d/Make/options", recorder / "Make/options")
        library = recorder / "libmrl2dRecorder.so"
        (recorder / "Make/files").write_text(f"mrl2dRecorder.C\nLIB = {library.with_suffix('')}\n")
        stage("buildRecorder", ["wmake", "libso"], recorder)
        stage("buildSolver", ["wmake"], native)
        for path in [
            solver,
            library,
            *original.iterdir(),
            *frozen.iterdir(),
            native / "productionPimpleFoam.C",
            native / "productionGuard.H",
            native / "Make/files",
            native / "Make/options",
            recorder / "mrl2dRecorder.C",
            recorder / "Make/files",
            recorder / "Make/options",
        ]:
            artifacts[str(path)] = flow._sha(path)
        write_json(work / "artifacts.json", artifacts)
        safety()
        return dict(
            solver=str(solver),
            library=str(library),
            ld_library_path=str(recorder) + ":" + environment["LD_LIBRARY_PATH"],
        )

    def solve(case, built, cancel):
        report, manifest = reports[case], manifests[case]
        ranks = manifest["ranks"]
        prefix = mpi_prefix(ranks)
        suffix = ["-parallel"] if ranks > 1 else []
        watch = FlowWatch(
            case / "log.pimpleFoam", dt=manifest["delta_t"], end=manifest["target_seconds"]
        )
        last_identity = 0.0

        def monitor():
            nonlocal last_identity
            # Disk/cancel/time every .1s; hashes and durable status at least every second.
            if cancel.is_set() or time.monotonic() - last_identity >= 1:
                safety(cancel)
                last_identity = time.monotonic()
                save_status(case, status="solving", **watch.status)
            else:
                check_output_space(output_paths, min_free_bytes)
            watch.poll()

        try:
            safety(cancel)
            flow.verify_frozen(case)
            save_status(case, status="decomposing")
            if ranks > 1:
                stage("decomposePar", ["decomposePar"], case, report, cancel=cancel)
            report["artifacts_sha256"] = artifacts
            command = [
                "env",
                "LD_LIBRARY_PATH=" + built["ld_library_path"],
                *prefix,
                built["solver"],
                *suffix,
            ]
            result, text = stage(
                "pimpleFoam",
                command,
                case,
                report,
                monitor,
                cancel,
                cap=min(wall_seconds, native_seconds / workers - 5),
            )
            report["flow"] = flow.audit_native_run(
                text,
                returncode=result["returncode"],
                dt=manifest["delta_t"],
                end=manifest["target_seconds"],
                elapsed=result["wall_seconds"],
                wall_limit=result["wall_limit"],
                require_complete_ami=True,
            )
            report["post_solve"] = watch.finish()
            if watch.status["ami_lines"] < 8 * (watch.steps + 1):
                raise ValueError("incomplete native AMI coverage summaries")
            save_status(case, status="verifying_sources", **watch.status)
            sources_data = flow.read_sources(
                case, envelope=manifest["envelope"], end=manifest["target_seconds"]
            )
            report["source"] = sources_data["metadata"]
            report["source_motion"] = flow.audit_source_motion(sources_data)
            del sources_data
            volume_count = math.floor((manifest["target_seconds"] + 1e-10) / 0.2)
            if volume_count:
                result, text = stage(
                    "volumeReadback",
                    [
                        *prefix,
                        "postProcess",
                        *suffix,
                        "-dict",
                        "system/readbackDict",
                        "-fields",
                        "(p U)",
                        "-noZero",
                    ],
                    case,
                    report,
                    cancel=cancel,
                    cap=wall_seconds,
                )
                observed = np.array(
                    [float(t) for t in re.findall(r"^Time = (\S+)\s*$", text, re.M)]
                )
                if (
                    observed.shape != (volume_count,)
                    or not np.allclose(
                        observed, np.arange(1, volume_count + 1) * 0.2, rtol=0, atol=1e-8
                    )
                    or not re.search(r"^End\s*$", text, re.M)
                    or re.search(r"\b(?:nan|inf)\b", text, re.I)
                ):
                    raise ValueError("incomplete/nonfinite native volume readback")
            report["volume_frames_read_natively"] = volume_count
            paths = sorted(
                (case / "postProcessing/animation").glob("*/section.vtp"),
                key=lambda p: float(p.parent.name),
            )
            expected = (
                np.arange(1, math.floor((manifest["target_seconds"] + 1e-10) / 0.02) + 1) * 0.02
            )
            if len(paths) != len(expected) or not np.allclose(
                [float(p.parent.name) for p in paths], expected, rtol=0, atol=1e-8
            ):
                raise ValueError("incomplete animation cadence")
            report["animation"] = []
            for path in paths:
                safety(cancel)
                if (
                    b'format="binary"' in path.read_bytes()
                    or b'format="appended"' in path.read_bytes()
                ):
                    raise ValueError("production VTP must use verified native ASCII")
                report["animation"].append(
                    dict(
                        section_summary(path),
                        path=str(path),
                        sha256=flow._sha(path),
                        native_float_type="Float32; precision17 is formatting, not binary64 data",
                    )
                )
            report["animation_frames"] = len(paths)
            safety(cancel)
            flow.verify_frozen(case)
            report.update(
                status="complete_10s_native_verified"
                if manifest["target_seconds"] == 10
                else "complete_bounded_smoke_native_verified",
                source_history_valid=True,
                wall_seconds=time.monotonic() - started,
            )
            save_status(case, status=report["status"], **watch.status)
            return report
        except BaseException as error:
            cancel.set()
            report.update(
                status="failed",
                error=str(error),
                source_history_valid=False,
                incomplete_source_policy="retain untouched; never resume or use for acoustics",
            )
            save_status(case, status="failed", error=str(error), **watch.status)
            raise
        finally:
            write_json(case / "mrl2d-run.json", report)

    try:
        result = coordinate(cases, workers=workers, setup=setup, solve=solve)
        save_status(
            status="complete",
            native_charged_seconds=ledger.carry + sum(r["wall_seconds"] for r in ledger.records),
        )
        return result
    except BaseException as error:
        save_status(status="failed", error=str(error))
        for case, report in reports.items():
            if not (case / "mrl2d-run.json").exists():
                report.update(status="failed", error=str(error), source_history_valid=False)
                write_json(case / "mrl2d-run.json", report)
        raise


def motion_proof(case, evidence_path):
    """Read existing worker evidence. Never manufacture parent admission."""
    from flowacoustics.mrl2d_case import _sha, mesh_identity

    if evidence_path is None:
        raise ValueError("native dual-motion/AMI evidence required")
    path = Path(evidence_path).resolve()
    data = json.loads(path.read_text())
    if data.get("mesh_motion_accepted") is True:
        if data.get("mesh_input_sha256") != mesh_identity(case):
            raise ValueError("motion evidence not bound to this mesh")
        return dict(path=str(path), sha256=_sha(path), kind="explicit_mesh_motion_evidence")
    motion = data.get("motion", {})
    states = motion.get("states", [])
    if (
        data.get("admitted") is not True
        or motion.get("all_faces_covered") is not True
        or len(states) != 12
    ):
        raise ValueError("mesh-worker motion evidence incomplete")
    names = ("points", "faces", "owner", "neighbour", "boundary", "cellZones")
    meshes = [Path(case), Path(data["case"]), Path(motion["case"])]
    hashes = []
    for mesh in meshes:
        if not all((mesh / "constant/polyMesh" / name).is_file() for name in names):
            raise ValueError("mesh identity missing")
        hashes.append({name: _sha(mesh / "constant/polyMesh" / name) for name in names})
    if not hashes[0] == hashes[1] == hashes[2]:
        raise ValueError("mesh-worker evidence belongs to a different mesh")
    retained = {}
    for stage in data["native_stages"]:
        if stage["returncode"] or stage["timed_out"] or _sha(stage["log"]) != stage["log_sha256"]:
            raise ValueError("mesh-worker native evidence changed or failed")
        retained[stage["log"]] = stage["log_sha256"]
    for i, state in enumerate(states):
        if (
            not math.isclose(state["time_s"], (i + 1) * 0.05, abs_tol=1e-10)
            or state["full_check_mesh_ok"] is not True
            or len(state["coverage"]) != 8
        ):
            raise ValueError("mesh-worker motion states incomplete")
        if state["xy_error_m"] > 1e-10 or state["z_error_m"] > motion["z_roundoff_tolerance_m"]:
            raise ValueError("mesh-worker motion error")
        for coverage in state["coverage"]:
            if not 0.99 <= coverage["min"] <= coverage["max"] <= 1.01:
                raise ValueError("mesh-worker AMI coverage failed")
            retained[coverage["file"]] = _sha(coverage["file"])
    return dict(
        path=str(path),
        sha256=_sha(path),
        kind="existing_mesh_worker_manifest",
        states=len(states),
        mesh_sha256=hashes[0],
        retained_evidence_sha256=retained,
    )


def instrument_solver(original):
    """Only two insertions; native equations and PIMPLE ordering remain unchanged."""
    if original.count('#include "fvCFD.H"') != 1 or original.count("        runTime.write();") != 1:
        raise ValueError("unexpected installed pimpleFoam source")
    return original.replace(
        '#include "fvCFD.H"', '#include "fvCFD.H"\n#include "productionGuard.H"'
    ).replace(
        "        runTime.write();",
        "        productionGuard(mesh, U, phi);\n        runTime.write();",
    )


class FlowWatch:
    """Incremental whole-log checks. Partial last lines wait for the next poll."""

    def __init__(self, path, *, dt, end):
        self.path, self.dt, self.end = Path(path), dt, end
        self.offset = 0
        self.steps = 0
        self.status = dict(
            solved_time=0.0, max_fluid_courant=0.0, max_mesh_courant=0.0, ami_lines=0
        )

    def poll(self):
        if not self.path.exists():
            return
        with self.path.open() as stream:
            stream.seek(self.offset)
            while True:
                line = stream.readline()
                if not line.endswith("\n"):
                    break
                self.offset = stream.tell()
                if re.search(
                    r"FOAM FATAL|Floating point exception(?! trapping enabled)|"
                    r"Segmentation fault|\b(?:nan|inf)\b",
                    line,
                    re.I,
                ):
                    raise ValueError("fatal/nonfinite native output")
                if m := re.match(r"(?:Mesh )?Courant Number mean: \S+ max: (\S+)", line):
                    if not math.isfinite(float(m[1])) or float(m[1]) > 1:
                        raise ValueError("Courant exceeds one")
                if "sum(weights)" in line:
                    values = re.search(r"min:\s*(\S+) max:\s*(\S+) average:\s*(\S+)", line)
                    if values is None or any(
                        not math.isfinite(float(v)) or not 0.99 <= float(v) <= 1.01
                        for v in values.groups()
                    ):
                        raise ValueError("AMI weight coverage failed")
                    self.status["ami_lines"] += 1
                if line.startswith("MRL2D_POST "):
                    m = re.fullmatch(
                        r"MRL2D_POST time=(\S+) fluid=(\S+) "
                        r"meshRaw=(\S+) meshEffective=(\S+) finite=1\n",
                        line,
                    )
                    if m is None:
                        raise ValueError("invalid post-solve finite-field check")
                    t, fluid, raw, effective = map(float, m.groups())
                    if (
                        not all(math.isfinite(v) for v in (t, fluid, raw, effective))
                        or max(fluid, raw, effective) > 1
                    ):
                        raise ValueError("post-solve Courant failed")
                    if not math.isclose(t, (self.steps + 1) * self.dt, rel_tol=0, abs_tol=1e-9):
                        raise ValueError("post-solve time sequence failed")
                    self.steps += 1
                    self.status.update(
                        solved_time=t,
                        max_fluid_courant=max(self.status["max_fluid_courant"], fluid),
                        max_mesh_courant=max(self.status["max_mesh_courant"], raw, effective),
                    )

    def finish(self):
        self.poll()
        if self.steps != round(self.end / self.dt):
            raise ValueError("incomplete post-solve endpoint evidence")
        return dict(self.status, post_solve_steps=self.steps)
