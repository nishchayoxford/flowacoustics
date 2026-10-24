"""Independent strict-2D four-case factory and native-source tools."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class CaseSpec:
    name: str
    model: str
    envelope: bool


def case_specs():
    return tuple(
        CaseSpec(f"mrl2d_{model}_{source}", model, source == "permeable")
        for model in ("urans", "les")
        for source in ("body", "permeable")
    )


def _foam(name, content, cls="dictionary"):
    return f"FoamFile\n{{\n version 2.0; format ascii; class {cls}; object {name};\n}}\n{content}\n"


def case_files(spec, *, depth, dt=0.0000625, end=10.0, ranks=2):
    """Return native inputs, not a run. Depth is numerical, never physical span."""
    if depth <= 0 or dt <= 0 or end <= 0 or not 1 <= ranks <= 8:
        raise ValueError("positive depth/dt/end and 1..8 ranks required")
    steps = [round(c / dt) for c in (0.002, 0.02, 0.2, end)]
    if any(
        not math.isclose(s * dt, c, abs_tol=1e-12)
        for s, c in zip(steps, (0.002, 0.02, 0.2, end), strict=True)
    ):
        raise ValueError("all cadences and endpoint must be integer multiples of dt")
    if spec not in case_specs():
        raise ValueError("unknown final case specification")
    ami = '\n "(carrier|blade[0-2])AMI_(master|slave)" {type cyclicAMI;}\n'
    empty = " frontAndBack {type empty;}\n"

    def field(name, dims, initial, wall, farfield, cls="volScalarField"):
        body = f' "blade[0-2]" {{{wall}}}\n'
        return _foam(
            name,
            f"dimensions {dims};\ninternalField uniform {initial};\n"
            f"boundaryField\n{{\n{body}farfield {{{farfield}}}\n{empty}{ami}}}",
            cls,
        )

    files = {}
    files["0/U"] = field(
        "U",
        "[0 1 -1 0 0 0 0]",
        "(1 0 0)",
        "type movingWallVelocity; value uniform (0 0 0);",
        "type freestreamVelocity; freestreamValue uniform (1 0 0);",
        "volVectorField",
    )
    files["0/p"] = field(
        "p",
        "[0 2 -2 0 0 0 0]",
        "0",
        "type zeroGradient;",
        "type freestreamPressure; freestreamValue uniform 0; value uniform 0;",
    )
    nut_wall = (
        "type nutkWallFunction; value uniform 0;"
        if spec.model == "urans"
        else "type fixedValue; value uniform 0;"
    )
    files["0/nut"] = field(
        "nut", "[0 2 -1 0 0 0 0]", "0", nut_wall, "type calculated; value uniform 0;"
    )
    if spec.model == "urans":
        files["0/k"] = field(
            "k",
            "[0 2 -2 0 0 0 0]",
            "0.00015",
            "type kqRWallFunction; value uniform 0.00015;",
            "type inletOutlet; inletValue uniform 0.00015; value uniform 0.00015;",
        )
        files["0/omega"] = field(
            "omega",
            "[0 0 -1 0 0 0 0]",
            "3.194",
            "type omegaWallFunction; value uniform 3.194;",
            "type inletOutlet; inletValue uniform 3.194; value uniform 3.194;",
        )
        turbulence = "simulationType RAS; RAS {RASModel kOmegaSST; turbulence on; printCoeffs on;}"
    else:
        turbulence = (
            "simulationType LES; LES {LESModel WALE; turbulence on; printCoeffs on; "
            "delta cubeRootVol; cubeRootVolCoeffs {deltaCoeff 1;}}"
        )
    files["constant/turbulenceProperties"] = _foam("turbulenceProperties", turbulence)
    files["constant/transportProperties"] = _foam(
        "transportProperties", "transportModel Newtonian; nu [0 2 -1 0 0 0 0] 1e-6;"
    )
    centers = (
        (0.0, 0.055),
        (math.sqrt(3) * 0.055 / 2, -0.0275),
        (-math.sqrt(3) * 0.055 / 2, -0.0275),
    )
    orbit = (
        "solidBodyMotionFunction rotatingMotion; rotatingMotionCoeffs "
        "{origin (0 0 0); axis (0 0 1); omega -10.472;}"
    )
    zones = f"carrierZone {{{orbit}}}\n"
    for i, (x, y) in enumerate(centers):
        zones += (
            f"blade{i}Zone {{solidBodyMotionFunction multiMotion; multiMotionCoeffs {{\n"
            f"orbit {{{orbit}}}\nrelativePitch {{solidBodyMotionFunction rotatingMotion; "
            f"rotatingMotionCoeffs {{origin ({x:.17g} {y:.17g} 0); "
            "axis (0 0 1); omega 5.236;}}}}\n"
        )
    files["constant/dynamicMeshDict"] = _foam(
        "dynamicMeshDict",
        "dynamicFvMesh dynamicMotionSolverFvMesh;\n"
        'motionSolverLibs ("libfvMotionSolvers.so");\n'
        "solver multiSolidBodyMotionSolver;\nmultiSolidBodyMotionSolverCoeffs {\n" + zones + "}\n",
    )
    files["system/controlDict"] = _foam(
        "controlDict",
        f"""
application pimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {end:g};
deltaT {dt:g}; adjustTimeStep no;
writeControl timeStep; writeInterval {steps[2]};
writeFormat binary; writePrecision 17; writeCompression off;
timeFormat general; timePrecision 10; purgeWrite 0; runTimeModifiable false;
functions
{{
 sources
 {{
  type mrl2dRecorder; libs ("libmrl2dRecorder.so");
  executeControl timeStep; executeInterval {steps[0]}; writeControl none;
  depth {depth:.17g}; rho 1000; omega 10.472;
  envelope {str(spec.envelope).lower()}; segmentsPerSide 80;
  zSample 0;
 }}
 animation
 {{
  type surfaces; libs ("libsampling.so");
  writeControl timeStep; writeInterval {steps[1]};
  surfaceFormat vtk; fields (p U); interpolationScheme cellPoint;
  formatOptions {{ vtk {{ format ascii; precision 17; }} }}
  surfaces (section {{type cuttingPlane; planeType pointAndNormal;
    pointAndNormalDict {{point (0 0 0); normal (0 0 1);}}
    interpolate true; regularise false; snap false;
  }});
 }}
}}
""",
    )
    files["system/fvSchemes"] = _foam(
        "fvSchemes",
        """
ddtSchemes {default backward;}
gradSchemes {default cellLimited Gauss linear 1;}
divSchemes
{
 default none;
 div(phi,U) Gauss linearUpwind grad(U);
 div(phi,k) Gauss upwind;
 div(phi,omega) Gauss upwind;
 div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes {default Gauss linear limited 0.5;}
interpolationSchemes {default linear;}
snGradSchemes {default limited 0.5;}
wallDist {method meshWave;}
fluxRequired {default no; p;}
""",
    )
    files["system/fvSolution"] = _foam(
        "fvSolution",
        """
solvers
{
 p {solver GAMG; tolerance 1e-8; relTol 0.05; smoother GaussSeidel;}
 pFinal {$p; relTol 0;}
 "(U|k|omega)" {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.05;}
 "(U|k|omega)Final" {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0;}
 pcorr {solver GAMG; tolerance 1e-8; relTol 0; smoother GaussSeidel;}
 pcorrFinal {$pcorr;}
}
PIMPLE
{
 nOuterCorrectors 2; nCorrectors 2; nNonOrthogonalCorrectors 1;
 pRefCell 0; pRefValue 0; correctPhi true; checkMeshCourantNo true;
}
relaxationFactors {equations {".*" 1;}}
""",
    )
    files["system/decomposeParDict"] = _foam(
        "decomposeParDict", f"numberOfSubdomains {ranks}; method scotch;"
    )
    files["system/readbackDict"] = _foam(
        "readbackDict",
        """
functions
{
 extrema {type fieldMinMax; libs ("libfieldFunctionObjects.so");
 fields (p U); mode magnitude; log true; location true;
 executeControl timeStep; executeInterval 1; writeControl timeStep; writeInterval 1;}
}
""",
    )
    return files


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _input_hashes(case):
    return {
        str(p.relative_to(case)): _sha(p)
        for sub in ("0", "constant", "system")
        for p in sorted((case / sub).rglob("*"))
        if p.is_file()
        and str(p.relative_to(case)) != "0/cellToRegion"
        and not str(p.relative_to(case)).startswith("constant/polyMesh/sets/")
    }


def _source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = [
        Path(__file__).resolve(),
        root / "scripts/run_mrl2d.py",
        root / "acoustics/mrl2d/mrl2dRecorder.C",
        root / "acoustics/mrl2d/Make/files",
        root / "acoustics/mrl2d/Make/options",
        root / "acoustics/mrl2d/productionGuard.H",
        root / "src/flowacoustics/mrl2d_production.py",
        root / "src/flowacoustics/mrl2d_lifecycle.py",
        root / "src/flowacoustics/mrl2d_runtime_audit.py",
    ]
    return {str(p): _sha(p) for p in paths if p.exists()}


def prepare_cases(mesh_case, output_root, *, depth, dt=0.0000625, ranks=2, end=10.0):
    """Create exactly four new cases, preserving the supplied mesh byte-for-byte.

    This stage is explicitly not native admission. ``run_case`` checks the copied mesh.
    Refuse existing targets and unknown extra mrl2d final directories; never delete/reuse.
    """
    mesh = Path(mesh_case).resolve() / "constant/polyMesh"
    required = ("points", "faces", "owner", "neighbour", "boundary", "cellZones")
    if not all((mesh / name).is_file() for name in required):
        raise ValueError("mesh handoff requires complete native constant/polyMesh and cellZones")
    root = Path(output_root).resolve()
    cache = Path(__file__).resolve().parents[2] / ".cache/mrl2d"
    if end != 10 and (not root.is_relative_to(cache) or not 0 < end <= 0.2):
        raise ValueError("bounded smoke endpoints require cache-only output and end<=.2")
    targets = [root / spec.name for spec in case_specs()]
    if any(p.exists() for p in targets) or list(root.glob("mrl2d_*")):
        raise FileExistsError("refusing existing final cases; no overwrite or hidden fifth case")
    generated = [case_files(s, depth=depth, dt=dt, ranks=ranks, end=end) for s in case_specs()]
    source_hashes = _source_hashes()
    for spec, case, files in zip(case_specs(), targets, generated, strict=True):
        case.mkdir(parents=True)
        shutil.copytree(mesh, case / "constant/polyMesh")
        for name, content in files.items():
            path = case / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        manifest = {
            "schema": "mrl2d-flow-input-v1",
            "status": "prepared_not_run",
            "name": spec.name,
            "model": spec.model,
            "envelope": spec.envelope,
            "model_label": "2D SST URANS"
            if spec.model == "urans"
            else "2D WALE / LES-model calculation",
            "numerical_depth_m": depth,
            "delta_t": dt,
            "target_seconds": end,
            "scope": "production" if end == 10 else "bounded_smoke_not_production",
            "source_cadence": 0.002,
            "animation_cadence": 0.02,
            "volume_cadence": 0.2,
            "ranks": ranks,
            "mesh_origin": str(mesh),
            "physical": {"rho": 1000.0, "nu": 1e-6, "U": [1, 0, 0], "omega": 10.472},
            "input_sha256": _input_hashes(case),
            "source_sha256": source_hashes,
        }
        (case / "mrl2d-input.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return targets


def verify_frozen(case):
    case = Path(case)
    manifest = json.loads((case / "mrl2d-input.json").read_text())
    if manifest["input_sha256"] != _input_hashes(case):
        raise ValueError("protected native inputs changed after preparation")
    if manifest["source_sha256"] != _source_hashes():
        raise ValueError("protected generator/recorder sources changed after preparation")
    return manifest


def mesh_identity(case):
    mesh = Path(case) / "constant/polyMesh"
    return {str(p.relative_to(mesh)): _sha(p) for p in sorted(mesh.rglob("*")) if p.is_file()}


def run_case(case, *, wall_seconds, motion_evidence):
    """Backward-compatible single-case API; shares the coordinated native owner."""
    from flowacoustics.mrl2d_production import run_cases

    case = Path(case).resolve()
    return run_cases(
        [case],
        wall_seconds=wall_seconds,
        native_seconds=wall_seconds,
        motion_evidence=motion_evidence,
        work_root=case / "execution",
        workers=1,
    )[0]


def native_command(command, *, cwd, log, timeout, monitor=None):
    """Retain exit/output only after a dedicated Linux subreaper proves ECHILD."""
    from flowacoustics import mrl2d_lifecycle as lifecycle
    from flowacoustics.mrl2d_production import write_json

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("positive explicit wall budget required")
    bashrc = os.environ.get("MRL2D_FOAM_BASHRC")
    if not bashrc or not Path(bashrc).is_file():
        raise ValueError("MRL2D_FOAM_BASHRC must name an existing OpenFOAM v2206 bashrc")
    shell = (
        'foamrc=$1; shift; cmd=("$@"); set --; '
        'source "$foamrc" >/dev/null 2>&1 || exit 121; '
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; "
        'exec "${cmd[@]}"'
    )
    started, timed_out, monitor_error = time.monotonic(), False, None
    started_unix = time.time()
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    process = None
    with lifecycle.defer_termination(), log.open("x") as stream:
        read_fd, result_fd = os.pipe()
        with os.fdopen(read_fd, "rb") as outcome_stream:
            try:
                try:
                    inherited = (result_fd,)
                    if lifecycle.owner_fd is not None:
                        inherited += (lifecycle.owner_fd,)
                    lifecycle.termination_checkpoint()
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(lifecycle.__file__).resolve()),
                            str(result_fd),
                            str(timeout),
                            "bash",
                            "-c",
                            shell,
                            "mrl2d",
                            bashrc,
                            *map(str, command),
                        ],
                        cwd=cwd,
                        stdin=subprocess.PIPE,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=inherited,
                    )
                finally:
                    os.close(result_fd)
                while process.poll() is None:
                    lifecycle.termination_checkpoint()
                    if time.monotonic() - started >= timeout:
                        timed_out = True
                        break
                    if monitor is not None:
                        monitor()
                    time.sleep(0.05)
                lifecycle.termination_checkpoint()
                if not timed_out and monitor is not None:
                    monitor()
            except KeyboardInterrupt as error:
                if process is None:
                    raise
                monitor_error = f"interrupted: {error}"
            except Exception as error:
                if process is None:
                    raise
                monitor_error = str(error)
            finally:
                # The guard starts before Popen and spans entry to this finally.
                # EOF and join therefore also cover a signal before Popen returns
                # its successfully created process to the caller's local variable.
                if process is not None:
                    while True:
                        try:
                            process.stdin.close()
                            process.wait()
                            lifecycle.termination_checkpoint()
                            break
                        except KeyboardInterrupt as error:
                            monitor_error = f"interrupted during cleanup: {error}"
            try:
                outcome = json.loads(outcome_stream.read())
                if process.returncode or outcome.get("cleanup_confirmed") is not True:
                    raise ValueError("native descendant cleanup unconfirmed")
            except BaseException:
                lifecycle.retain_owner()
                raise
    timed_out |= outcome["timed_out"]
    if outcome["stranded_descendants"] and monitor_error is None:
        monitor_error = "launcher exited with surviving descendants; terminated and reaped"
    record = {
        "argv": list(map(str, command)),
        "returncode": outcome["returncode"],
        "cleanup_confirmed": True,
        "descendants_reaped": outcome["descendants_reaped"],
        "wall_seconds": time.monotonic() - started,
        "started_unix": started_unix,
        "finished_unix": time.time(),
        "wall_limit": timeout,
        "timed_out": timed_out,
        "monitor_error": monitor_error,
        "log": str(log),
        "log_sha256": _sha(log),
    }
    write_json(log.with_suffix(log.suffix + ".json"), record)
    return record


def audit_mesh_log(log, *, returncode):
    """Native checkMesh allGeometry/allTopology acceptance and whole-case cap."""
    counts = re.findall(r"^\s*cells:\s*(\d+)\s*$", log, re.M)
    if returncode or "Mesh OK." not in log or re.search(r"Failed \d+ mesh checks|FOAM FATAL", log):
        raise ValueError("native full checkMesh did not pass")
    if len(counts) != 1 or not 0 < int(counts[0]) <= 250000:
        raise ValueError("native complete-case cell count missing/above cap")
    for direction in ("geometric", "solution"):
        if not re.search(rf"Mesh has 2 {direction}[^\n]+\(1 1 0\)", log):
            raise ValueError("native mesh is not the required strict-2D x-y topology")
    return {
        "cells": int(counts[0]),
        "mesh_ok": True,
        "geometric_dimensions": 2,
        "solution_dimensions": 2,
    }


def audit_native_ami(log):
    """Pure whole-log AMI coverage in native event order, including the last step.

    Four interfaces require four explicit addressing events, each with source
    then target, at initialization and in every time block before MRL2D_POST,
    then ClockTime. Every nonempty update group must contain a multiple of four
    complete events; no-update PIMPLE iterations need no new evidence. These
    native lines do not identify unique interfaces. Later/trailing summaries
    cannot fill an earlier gap. No native I/O occurs.
    """
    state, current = "initialization", None
    block_count = update_count = total = 0
    address_count = None
    weights = re.compile(
        r"^AMI: Patch (source|target) sum\(weights\)\s+"
        r"min:\s*(\S+) max:\s*(\S+) average:\s*(\S+)\s*$"
    )

    def complete_addressing():
        if address_count is not None and address_count != 2:
            raise ValueError("incomplete AMI source/target addressing summaries")

    def complete_update():
        complete_addressing()
        if update_count % 4:
            raise ValueError("incomplete AMI mesh-update addressing-event coverage")

    for match in re.finditer(r"[^\n]+", log):
        line = match.group().strip()
        if "sum(weights)" in line:
            row = weights.search(line)
            if state not in ("initialization", "update"):
                raise ValueError("AMI summaries outside initialization/pre-solve update")
            if (
                row is None
                or address_count not in (0, 1)
                or row[1] != ("source" if address_count == 0 else "target")
            ):
                raise ValueError("AMI addressing requires exactly source then target summaries")
            if not all(
                math.isfinite(float(x)) and 0.99 <= float(x) <= 1.01 for x in row.groups()[1:]
            ):
                raise ValueError("malformed/nonfinite/failed AMI weights")
            total += 1
            address_count += 1
            if address_count == 2:
                block_count += 1
                update_count += 1
        elif line.startswith("AMI: Creating addressing and weights"):
            if state not in ("initialization", "update"):
                raise ValueError("AMI addressing outside active mesh-update block")
            complete_addressing()
            address_count = 0
        elif line.startswith("Time = "):
            if state == "initialization":
                if block_count < 4:
                    raise ValueError("incomplete AMI initialization coverage")
                complete_update()
            elif state != "clock":
                raise ValueError("incomplete AMI time-block evidence/order")
            current = float(line.split("=", 1)[1])
            state, block_count, update_count = "update", 0, 0
            address_count = None
        elif line.startswith("PIMPLE: iteration "):
            if state != "update":
                raise ValueError("AMI PIMPLE update outside active time block")
            complete_update()
            update_count = 0
            address_count = None
        elif line.startswith("MRL2D_POST"):
            row = re.match(r"MRL2D_POST time=(\S+) ", line)
            if (
                state != "update"
                or block_count < 4
                or row is None
                or current is None
                or not math.isclose(float(row[1]), current, rel_tol=0, abs_tol=1e-9)
            ):
                raise ValueError("incomplete/misordered AMI coverage for solved time")
            complete_update()
            state = "post"
        elif "ClockTime = " in line:
            if state != "post":
                raise ValueError("AMI clock precedes complete post-solve evidence")
            state = "clock"
        elif line == "End":
            if state != "clock":
                raise ValueError("incomplete AMI final time block before End")
            state = "ended"
    if state != "ended":
        raise ValueError("incomplete AMI evidence/End")
    return total


def audit_native_run(
    log, *, returncode, dt, end, elapsed, wall_limit, max_courant=1.0, require_complete_ami=False
):
    """Gate the entire log; current producer/exporter opt in to per-step AMI.

    The default retains the legacy non-AMI log API, not production admission.
    """
    if returncode != 0 or not 0 < elapsed <= wall_limit:
        raise ValueError("native nonzero exit or wall-clock budget exceeded")
    if not re.search(r"^End\s*$", log, re.M) or re.search(
        r"FOAM FATAL|Floating point exception(?! trapping enabled)|"
        r"Segmentation fault|\b(?:nan|inf)\b",
        log,
        re.I,
    ):
        raise ValueError("missing native End or fatal/nonfinite output")
    times = np.array([float(x) for x in re.findall(r"^Time = (\S+)\s*$", log, re.M)])
    expected = np.arange(1, round(end / dt) + 1) * dt
    if times.shape != expected.shape or not np.allclose(times, expected, rtol=0, atol=1e-8):
        raise ValueError("native timesteps missing, repeated, or wrong endpoint")
    clocks = np.array([float(x) for x in re.findall(r"ClockTime = (\S+) s", log)])
    if len(clocks) != len(times) or not np.isfinite(clocks).all() or np.any(clocks < 0):
        raise ValueError("native clock evidence must be complete, finite and nonnegative")
    if np.any(np.diff(clocks) < 0) or clocks[-1] > wall_limit:
        raise ValueError("native clock evidence incomplete or exceeds budget")
    courant = np.array(
        [float(x) for x in re.findall(r"^Courant Number mean: \S+ max: (\S+)", log, re.M)]
    )
    if len(courant) < len(times) or not np.isfinite(courant).all() or courant.max() > max_courant:
        raise ValueError("Courant stability admission failed/incomplete")
    mesh_courant = [
        float(x) for x in re.findall(r"^Mesh Courant Number mean: \S+ max: (\S+)", log, re.M)
    ]
    if mesh_courant and max(mesh_courant) > max_courant:
        raise ValueError("mesh Courant admission failed")
    if require_complete_ami:
        audit_native_ami(log)
    return {
        "steps": len(times),
        "end": float(times[-1]),
        "native_returncode": returncode,
        "native_clock_seconds": float(clocks[-1]),
        "wall_seconds": elapsed,
        "max_courant": float(courant.max()),
        "max_mesh_courant": max(mesh_courant, default=None),
    }


def _check_closed_curve(curve):
    """Admit oriented straight segments as one closed 2D curve, not just zero net normal."""
    xy, n, ds = curve[:, 4:6], curve[:, 7:9], curve[:, 9]
    if len(ds) < 3 or np.linalg.norm((n * ds[:, None]).sum(axis=0)) > 1e-8 * ds.sum():
        raise ValueError("incomplete/nonclosing contour normals")
    if np.sum(xy * n * ds[:, None]) <= 0:
        raise ValueError("inward contour normals or nonpositive enclosed area")
    tangent = np.column_stack((-n[:, 1], n[:, 0]))
    starts = xy - 0.5 * ds[:, None] * tangent
    ends = xy + 0.5 * ds[:, None] * tangent
    distance, indices = cKDTree(starts).query(ends)
    if distance.max() > 1e-8 or np.unique(indices).size != len(ds):
        raise ValueError("contour segment endpoints do not close uniquely")
    visited, node = set(), 0
    while node not in visited:
        visited.add(node)
        node = int(indices[node])
    if len(visited) != len(ds) or node != 0:
        raise ValueError("contour is not a single connected closed curve")


def audit_source_motion(data):
    """Compare all native Cf/normals and sampled kinematics with the prescribed law."""
    omega = 10.472
    initial = np.array(
        [[0.0, 0.055], [np.sqrt(3) * 0.055 / 2, -0.0275], [-np.sqrt(3) * 0.055 / 2, -0.0275]]
    )
    errors = np.zeros(3)
    times = data["time"]
    for b in range(3):
        select = (data["kind"] == 0) & (data["blade"] == b)
        if not np.any(select):
            raise ValueError("three complete blades required for native motion audit")
        c, s = np.cos(omega * times), np.sin(omega * times)
        center = np.column_stack(
            (c * initial[b, 0] + s * initial[b, 1], -s * initial[b, 0] + c * initial[b, 1])
        )
        reference = data["position"][0, select] - center[0]
        normals = data["normal"][0, select]
        for i, t in enumerate(times):
            a = -omega / 2 * (t - times[0])
            rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            expected = reference @ rotation.T + center[i]
            actual = data["position"][i, select]
            vs = np.column_stack(
                (
                    omega / 2 * (actual[:, 1] + center[i, 1]),
                    -omega / 2 * (actual[:, 0] + center[i, 0]),
                )
            )
            errors = np.maximum(
                errors,
                [
                    np.max(np.abs(expected - actual)),
                    np.max(np.abs(normals @ rotation.T - data["normal"][i, select])),
                    np.max(np.abs(vs - data["surface_velocity"][i, select])),
                ],
            )
    if errors[0] > 1e-8 or errors[1] > 1e-7 or errors[2] > 1e-9:
        raise ValueError(f"native body source motion/normal/surface velocity mismatch: {errors}")
    return dict(
        zip(
            (
                "maximum_position_error_m",
                "maximum_normal_error",
                "maximum_surface_velocity_error_m_per_s",
            ),
            map(float, errors),
            strict=True,
        )
    )


def acoustic_source_kwargs(data, *, kind):
    """Arguments for acoustics2d.ContourHistory without importing that module."""
    if kind not in ("body", "permeable"):
        raise ValueError("kind must be body or permeable")
    select = data["kind"] == (0 if kind == "body" else 1)
    if not np.any(select):
        raise ValueError("requested source kind was not recorded")
    lengths = data["length"][:, select]
    if not np.allclose(lengths, lengths[0], rtol=1e-9, atol=1e-12):
        raise ValueError("frozen acoustic API requires rigid, time-invariant panel lengths")
    provenance = {
        **data["metadata"],
        "source_type": "native_cfd",
        "pressure_conversion": "native_kinematic_times_rho_once",
        "normal_convention": "out_of_solid" if kind == "body" else "out_of_envelope",
        "interpolation": "native_boundary" if kind == "body" else "native_cellPoint",
        "complete_panel_coverage": True,
    }
    return {
        "time": data["time"],
        "xy": data["position"][:, select],
        "normal": data["normal"][:, select],
        "ds": lengths[0],
        "p": data["pressure"][:, select],
        "U": data["fluid_velocity"][:, select],
        "surface_velocity": data["surface_velocity"][:, select],
        "kind": kind,
        "blade_id": data["blade"][select] if kind == "body" else None,
        "rho0": 1000.0,
        "c0": 1480.0,
        "provenance": provenance,
    }


def read_sources(case, *, envelope, end=10.0, cadence=0.002, segments_per_side=80):
    """Read and validate every record; pressure/length already have final 2D units.

    Deliberately rejects partial runs, resumes, duplicate IDs and truncated writes.
    Use an explicit shorter ``end`` only for clearly labelled smoke verification.
    """
    case = Path(case)
    serial = sorted(case.glob("postProcessing/mrl2dSources/rank*.bin"))
    parallel = sorted(case.glob("processor*/postProcessing/mrl2dSources/rank*.bin"))
    if bool(serial) == bool(parallel):
        raise ValueError("require exactly one serial or parallel native source collection")
    files = serial or parallel
    expected = np.arange(1, round(end / cadence) + 1) * cadence
    if len(expected) == 0 or not np.isclose(expected[-1], end, rtol=0, atol=1e-10):
        raise ValueError("end must be a positive integer number of source intervals")
    chunks = []
    for path in files:
        size = path.stat().st_size
        if size % (16 * 8):
            raise ValueError(f"truncated native binary record: {path}")
        if not size:
            continue
        rows = np.memmap(path, dtype="<f8", mode="r").reshape(-1, 16)
        if not np.isfinite(rows).all():
            raise ValueError("nonfinite native source")
        count = np.count_nonzero(rows[:, 0] == rows[0, 0])
        if len(rows) != len(expected) * count:
            raise ValueError("missing/extra source rows or frames")
        cube = rows.reshape(len(expected), count, 16)
        if not np.allclose(cube[:, :, 0], expected[:, None], rtol=0, atol=1e-8):
            raise ValueError("nonuniform/incorrect source timestamps")
        if not np.array_equal(
            cube[:, :, 1:4], np.broadcast_to(cube[0, :, 1:4], cube[:, :, 1:4].shape)
        ):
            raise ValueError("source identity changed during run")
        chunks.append(cube)
    if not chunks:
        raise ValueError("empty native source history")
    data = np.concatenate(chunks, axis=1)
    order = np.lexsort((data[0, :, 2], data[0, :, 1]))
    data = data[:, order, :]
    kinds, ids, blades = (data[0, :, k].astype(np.int64) for k in (1, 2, 3))
    if not np.array_equal(data[0, :, 1:4], np.column_stack((kinds, ids, blades))):
        raise ValueError("nonintegral source identity")
    if len(set(zip(kinds, ids, strict=True))) != len(ids) or not set(kinds) <= {0, 1}:
        raise ValueError("duplicate or invalid native source IDs")
    if np.any(data[:, :, 9] <= 0) or np.any(data[:, :, 15] < 0):
        raise ValueError("nonpositive segment length or missing native containing cell")
    if not np.allclose(np.linalg.norm(data[:, :, 7:9], axis=2), 1, rtol=0, atol=1e-9):
        raise ValueError("nonunit contour normal")
    if set(blades[kinds == 0]) != {0, 1, 2}:
        raise ValueError("all three native blade curves required")
    for frame in data:
        for blade in range(3):
            _check_closed_curve(frame[(kinds == 0) & (blades == blade)])
    env = kinds == 1
    if envelope:
        if not np.array_equal(ids[env], np.arange(4 * segments_per_side)):
            raise ValueError("incomplete native envelope coverage")
        if not np.all(blades[env] == -1) or not np.all(data[:, env, 13:15] == 0):
            raise ValueError("envelope must be stationary and independent of blades")
        if not np.array_equal(
            data[:, env, 4:10], np.broadcast_to(data[0, env, 4:10], data[:, env, 4:10].shape)
        ):
            raise ValueError("stationary envelope geometry moved")
        _check_closed_curve(data[0, env])
    elif np.any(env):
        raise ValueError("unexpected envelope recording in body-only case")
    return {
        "time": expected,
        "position": data[:, :, 4:6],
        "normal": data[:, :, 7:9],
        "length": data[:, :, 9],
        "pressure": data[:, :, 10],
        "fluid_velocity": data[:, :, 11:13],
        "surface_velocity": data[:, :, 13:15],
        "kind": kinds,
        "id": ids,
        "blade": blades,
        "metadata": {
            "schema": "mrl2d-native-source-v1",
            "pressure_units": "Pa",
            "length_units": "m",
            "frames": len(expected),
            "sources": len(ids),
            "rank_files": len(files),
            "body_depth_divisions": 1,
            "envelope_depth_divisions": 0,
            "case": str(case.resolve()),
            "native_files_sha256": {str(p.resolve()): _sha(p) for p in files},
        },
    }
