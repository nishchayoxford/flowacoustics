"""Fresh finer finite-cylinder LES, now targeted to 10 s by user instruction.

The already allocated *_60s paths are labels only; no 60 s production is launched.
Historical generators, native library and prior records remain immutable.
"""
from pathlib import Path
import math
import json
import os
import time
import re
import signal
import shlex
import hashlib
import subprocess
import fcntl
import cylinder_circular as circular
import cylinder_native as native

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/'cfd/cylinder_circular_60s'
PILOT = ROOT/'cfd/cylinder_circular_60s_pilot'
RESULTS = ROOT/'results/cylinder/native_circular_60s'
DOCS = ROOT/'docs/cylinder_60s/native'


def mesh_description():
    """Refine all directions, retaining topology and modest angular clustering.

    Radial total expansion 16 (old 24) reduces farfield anisotropy; uniform
    axial blocks avoid increasing extreme outer-cell aspect ratios at tips.
    No mesh-quality threshold is changed.
    """
    m = circular.mesh_description()
    for b in m['blocks']:
        nr, nt, nz = b['counts']
        b['counts'] = [{32:44, 6:8, 16:24}[nr], 24, {12:20, 36:56}[nz]]
        if nr == 32:
            b['grading'][0] = 16
    m['cells'] = sum(math.prod(b['counts']) for b in m['blocks'])
    if m['cells'] >= 500000:
        raise ValueError('strict cell ceiling exceeded')
    return m


def write_case(case, *, end=10., dt=.0005, ranks=8):
    """Create a new standalone clock; refuse an existing case before any write."""
    case = Path(case)
    cadences = [end, min(2., end), .02, .5]
    if (not math.isfinite(dt) or dt <= 0 or dt > .002 or
            not math.isfinite(end) or end <= 0 or ranks not in (6,8) or
            any(abs(x/dt-round(x/dt)) > 1e-7 for x in cadences)):
        raise ValueError('invalid fixed timestep/cadence/end/rank contract')
    if case.exists():
        raise ValueError('refusing existing case')
    circular.write_case(case)
    (case/'system/blockMeshDict').write_text(circular.block_mesh_text(mesh_description()))
    p = case/'system/controlDict'
    c = p.read_text()
    c = c.replace('endTime 0.1;', f'endTime {end:g};')
    c = c.replace('deltaT 0.001; adjustTimeStep yes; maxCo 0.7; maxDeltaT 0.003;',
                  f'deltaT {dt:g}; adjustTimeStep no;')
    c = c.replace('writeControl adjustableRunTime; writeInterval 0.1; purgeWrite 3;',
                  f'writeControl runTime; writeInterval {min(2.,end):g}; purgeWrite 0;')
    c = c.replace('writeFormat ascii; writePrecision 10; writeCompression off;',
                  'writeFormat binary; writePrecision 12; writeCompression off;')
    c = c.replace('runTimeModifiable true;', 'runTimeModifiable false;')
    c = c.replace('executeControl writeTime; writeControl writeTime;',
                  'executeControl runTime; executeInterval 0.5; writeControl writeTime;')
    c = c.replace('writeControl timeStep; writeInterval 20;',
                  'writeControl runTime; writeInterval 0.02;')
    c = c.replace('flowSlice {type surfaces; libs ("libsampling.so"); writeControl writeTime;',
                  'flowSlice {type surfaces; libs ("libsampling.so"); writeControl runTime; writeInterval 0.5;')
    c = c.replace('functions {', 'functions {\nyPlus {type yPlus; libs ("libfieldFunctionObjects.so"); '
                  'executeControl runTime; executeInterval 0.5; writeControl writeTime; log false;}')
    p.write_text(c)
    (case/'system/decomposeParDict').write_text(
        'FoamFile {version 2.0; format ascii; class dictionary; object decomposeParDict;}\n'
        f'numberOfSubdomains {ranks}; method scotch;\n')


def run_command(argv, label, *, ledger=RESULTS/'commands.json', native_env=True,
                timeout=None, progress=60., watch_solver=False):
    """New transparent ledger, shared nonblocking native lock, bounded polling.

    No inherited 1800 s cap; elapsed native time and every failure are retained.
    The external launcher owns the wall-time lifetime, not a fake demo budget.
    """
    if 'mpirun' in argv and ('-np' not in argv or int(argv[argv.index('-np')+1]) not in (6,8)):
        raise ValueError('native ranks must be 6 or 8')
    if native_env and (native.FOAM is None or not (native.FOAM / 'etc/bashrc').is_file()):
        raise RuntimeError('Source OpenFOAM v2206 first: WM_PROJECT_DIR must identify its installation')
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    lock = ROOT/'.cache/cylinder_native.lock' if native_env else ledger.with_suffix('.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if native_env:
            busy = subprocess.run(['pgrep','-a','-x','pimpleFoam|mpirun|blockMesh|checkMesh|decomposePar|reconstructPar'],
                                  text=True, capture_output=True)
            if busy.returncode == 0:
                raise RuntimeError('native conflict: '+busy.stdout)
        records = json.loads(ledger.read_text()) if ledger.exists() else []
        log = ledger.parent/f'{len(records):03}_{label}.log'
        actual = argv
        if native_env:
            actual = ['bash','-c', f'source {shlex.quote(str(native.FOAM / "etc/bashrc"))} >/dev/null 2>&1 && '
                      f'export LD_LIBRARY_PATH={shlex.quote(str(native.LIB.parent))}:$LD_LIBRARY_PATH && exec '+shlex.join(argv)]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        record = dict(argv=argv, actual_argv=actual, label=label, state='running',
                      started_unix_s=time.time(), log=str(log.resolve()), timed_out=False,
                      elapsed_seconds=0., exit_code=None, timeout_s=timeout)
        records.append(record)
        start = time.monotonic()
        proc = None
        try:
            with log.open('w') as stream:
                proc = subprocess.Popen(actual, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                record['pid'] = proc.pid
                while True:
                    record['elapsed_seconds'] = time.monotonic()-start
                    ledger.write_text(json.dumps(records, indent=2)+'\n')
                    try:
                        remaining = max(.001, timeout-record['elapsed_seconds']) if timeout else progress
                        record['exit_code'] = proc.wait(timeout=min(progress, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        unstable = False
                        if watch_solver:
                            # Incremental bounded reading; never rescan an ever-growing solver log.
                            with log.open() as reader:
                                reader.seek(record.get('monitor_offset',0))
                                chunk = reader.read()
                                record['monitor_offset'] = reader.tell()
                            courant = [float(x) for x in re.findall(r'Courant Number mean: [^\n]+ max: ([0-9.eE+-]+)',chunk)]
                            times = re.findall(r'^Time = ([0-9.eE+-]+)$',chunk,re.M)
                            if times:
                                record['last_time_s'] = float(times[-1])
                            if courant:
                                record['observed_max_Co'] = max(record.get('observed_max_Co',0),max(courant))
                            if courant and max(courant)>1:
                                record['abort_reason'] = 'Courant number exceeded hard limit 1.0'
                                unstable = True
                            if re.search(r'FOAM FATAL|\b(?:nan|inf)\b|[Ff]loating point exception(?! trapping enabled)',chunk):
                                record['abort_reason'] = 'fatal/nonfinite solver output'
                                unstable = True
                        print(json.dumps(dict(progress=label, elapsed_seconds=time.monotonic()-start,
                                              last_time_s=record.get('last_time_s'),
                                              maxCo=record.get('observed_max_Co'), log=str(log))), flush=True)
                        if unstable or (timeout and time.monotonic()-start >= timeout):
                            record['timed_out'] = not unstable
                            os.killpg(proc.pid, signal.SIGTERM)
                            try:
                                proc.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(proc.pid, signal.SIGKILL)
                            record['exit_code'] = proc.wait()
                            break
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            record['elapsed_seconds'] = time.monotonic()-start
            record['state'] = 'completed' if record['exit_code'] == 0 else 'failed'
            record['log_sha256'] = hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None
            ledger.write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        return record


def mesh_gate(text):
    """OpenFOAM may return zero even for failed checks: inspect complete text."""
    match = re.search(r'cells:\s+(\d+)', text)
    if not match or 'Mesh OK.' not in text or re.search(r'Failed|FOAM FATAL|\*\*\*', text):
        raise ValueError('native mesh quality gate failed')
    cells = int(match[1])
    if not 0 < cells < 500000:
        raise ValueError('strict cell ceiling failed')
    return dict(cell_count=cells, checkMesh_allTopology_allGeometry=True)


def solver_gate(text, *, end, dt):
    """Require all fixed steps, final target, End, finite log and Co <= 1."""
    times = [float(x) for x in re.findall(r'^Time = ([0-9.eE+-]+)$',text,re.M)]
    co = [float(x) for x in re.findall(r'Courant Number mean: [^\n]+ max: ([0-9.eE+-]+)',text)]
    if (not re.search(r'^End\s*$',text,re.M) or
            re.search(r'FOAM FATAL|\b(?:nan|inf)\b|[Ff]loating point exception(?! trapping enabled)',text) or
            len(times) != round(end/dt) or not co or max(co) > 1 or
            any(abs(t-(i+1)*dt)>1e-8 for i,t in enumerate(times))):
        raise ValueError('solver completion/fixed-step/finite/Courant gate failed')
    return dict(steps=len(times), end_s=times[-1], observed_max_Co=max(co), fixed_dt_s=dt,
                solver_reached_target=True)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def input_identities(case):
    """Explicit essential-input coverage; missing mesh/config/runtime is an error."""
    case = Path(case)
    files = [case/x for x in ['0/U','0/p','0/nut','system/controlDict','system/blockMeshDict',
        'system/decomposeParDict','system/fvSchemes','system/fvSolution',
        'constant/transportProperties','constant/turbulenceProperties']]
    files += [case/'constant/polyMesh'/x for x in ['points','faces','owner','neighbour','boundary']]
    files += [Path(__file__), ROOT/'scripts/cylinder_native.py', ROOT/'scripts/cylinder_circular.py',native.LIB,
              native.FOAM/'platforms/linux64GccDPInt32Opt/bin/pimpleFoam']
    files += [p for pattern in ['processor[0-9]*/0/*','processor[0-9]*/constant/polyMesh/*']
              for p in sorted(case.glob(pattern)) if p.is_file()]
    return {str(p.resolve()): digest(p) for p in files}


def verify_identities(pins):
    if not pins:
        raise ValueError('empty identity map')
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError('input identity mismatch: '+str(path))


def solve_reconstruct(case, out, *, end, dt, ranks=8, execute=run_command):
    """One fresh solver invocation, followed only on verified success by reconstruction."""
    case, out = Path(case), Path(out)
    if (case/'acousticData').exists() or (case/'postProcessing').exists() or (out/'run.json').exists():
        raise ValueError('existing run: refusing duplicate solver or native clock overwrite')
    inputs = input_identities(case)
    out.mkdir(parents=True, exist_ok=True)
    status = dict(schema_version=2, status='running', case_path=str(case), results_path=str(out),
        evidence_kind='native_execution_in_progress', input_sha256=inputs,
        medium=dict(density_kg_m3=1000.,kinematic_viscosity_m2_s=1e-6,sound_speed_m_s=1500.,velocity_m_s=.2,reference_pressure_Pa=1e-6),
        geometry=dict(diameter_m=.02,span_m=.06,centre_m=[0,0,0],axis='z',outer_radius_m=.24,
                      domain_bounds_m=[[-.24,-.24,-.08],[.24,.24,.08]],finite_closed=True),
        cell_count=mesh_description()['cells'],
        mesh_grading=dict(radial_cells=44,circumferential_cells=96,span_cells=[20,56,20],radial_expansion_ratio=16),
        solver=dict(name='pimpleFoam',openfoam='v2206',les_model='WALE',delta='cubeRootVol',ranks=ranks,
                    threads_per_rank=1,time_scheme='backward',advection='linearUpwind grad(U)',mesh_adequacy_validated=False),
        sampling=dict(adaptive=False,fixed_dt_s=dt,forces='every step',acoustics='every step; five native observers, both methods',
                      wall_pressure_interval_s=.02,flow_interval_s=.5,full_fields_interval_s=min(end,2.),purgeWrite=0,
                      fields_format='binary uncompressed; v2206 disables binary compression'),
        clock=dict(forces_offset_s=0,libacoustics_curle=dict(offset_s=0,kind='source_unretarded'),
                   libacoustics_fwh=dict(offset_s=0,kind='observer')),
        force_convention='on_body',force_content='total_traction',observer_coordinates_m=native.OBSERVERS,
        time_coverage=dict(start_s=0,requested_end_s=end,end_s=None,statistically_developed=False),
        reconstruction=dict(status='pending',decompositions_retained=True),exports=dict(status='pending'),
        native_accounting=dict(previous_1800s_budget_applies=False,maximum_mpi_ranks=8,
                               ledger=str(out/'commands.json'),authorization='new longer-run authorization subsequently shortened to exactly 10 s; independent ledger'),
        unsupported_claims=['mesh/time independence','resolved LES adequacy','statistical stationarity',
                            'acoustic validation','radiation-far-field-only at 1m','exact serial/MPI equivalence'],
        acoustic_conventions=dict(curle='native on-body total traction; unretarded source clock; fixed-step derivative',
            fwh='native rigid closed-wall pressure-only Farassat1A; facewise observer clock; no viscous or volume quadrupole terms',
            restart_history='none; this case starts from 0 without restart or splicing',
            finite_tail='native receiver history ends with solver; no invented received tail'))
    status_file = out/'run.json'
    status_file.write_text(json.dumps(status,indent=2)+'\n')
    try:
        r = execute(['mpirun','--bind-to','core','-np',str(ranks),'pimpleFoam','-case',str(case),'-parallel'],
                    'solver',ledger=out/'commands.json',watch_solver=True,progress=30.)
        status['solver_command'] = r
        return finalize_run(case,out,status,end=end,dt=dt,ranks=ranks,execute=execute)
    except BaseException as exc:
        status['status'] = 'failed'
        status['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        status_file.write_text(json.dumps(status,indent=2)+'\n')


def finalize_run(case,out,status,*,end,dt,ranks=8,execute=run_command):
    """Replayable completion stage: NEVER launches or restarts a solver."""
    case,out = Path(case),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    status_file = out/'run.json'
    r = status['solver_command']
    try:
        if r['exit_code'] != 0:
            raise RuntimeError('solver command failed')
        status['solver_verification'] = solver_gate(Path(r['log']).read_text(),end=end,dt=dt)
        status['time_coverage']['end_s'] = end
        verify_identities(status['input_sha256'])
        status['status'] = 'reconstructing'
        status_file.write_text(json.dumps(status,indent=2)+'\n')
        rr = execute(['reconstructPar','-case',str(case),'-noZero'], 'reconstruct', ledger=out/'commands.json')
        status['reconstruction_command'] = rr
        text = Path(rr['log']).read_text()
        if rr['exit_code'] != 0 or 'FOAM FATAL' in text or not re.search(r'^End\s*$',text,re.M):
            raise RuntimeError('reconstructPar did not complete successfully')
        interval = min(end,2.)
        times = [interval*i for i in range(1,round(end/interval)+1)]
        hashes, field_checks = {}, {}
        for t in times:
            for base in [case]+[case/f'processor{i}' for i in range(ranks)]:
                for field in ['p','U','nut','vorticity','yPlus']:
                    p = base/f'{t:g}'/field
                    if not p.is_file() or not p.stat().st_size:
                        raise ValueError('missing retained/reconstructed field: '+str(p))
                    if base == case:
                        field_checks[str(p)] = finite_field(p,expected_cells=mesh_description()['cells'])
                        hashes[str(p)] = digest(p)
        status['reconstruction'] = dict(status='completed',times_s=times,sha256=hashes,
                                         field_finite_checks=field_checks,decompositions_retained=True)
        status['status'] = 'exporting'
        status_file.write_text(json.dumps(status,indent=2)+'\n')
        status['exports'] = export_outputs(case,out,expected_steps=round(end/dt))
        status['status'] = 'completed_native_reconstructed'
        status['evidence_kind'] = 'native_execution'
        return status
    except BaseException as exc:
        status['status'] = 'failed'
        status['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        status_file.write_text(json.dumps(status,indent=2)+'\n')


def export_outputs(case, out, *, expected_steps):
    """Full native tables and arrays; complete raw flow planes stay in the case.

    No old build_manifest is used: its budget, ranks and restarts are historical.
    """
    import csv
    case, out = Path(case), Path(out)
    segments = [(case/f'acousticData/{name}-time.dat', 'libacoustics_'+name, 0.) for name in ['curle','fwh']]
    table = native.export_tables(case,out,segments)
    if table['force_samples'] != expected_steps:
        raise ValueError('native samples do not match solver step count')
    g = native.wall_geometry(case/'constant/polyMesh')
    wall = native.export_wall_history(case,out,g)
    wall['sample_cadence'] = 'runTime 0.02 s; exact fixed-step timestamps retained'
    wall['geometry_audit'] = g['audit']
    verification = circular.verify_exports(out)
    nonzero = {'libacoustics_curle':False,'libacoustics_fwh':False}
    with (out/'observers.csv').open() as stream:
        for row in csv.DictReader(stream):
            nonzero[row['method']] |= float(row['pressure_Pa']) != 0
    if not all(nonzero.values()):
        raise ValueError('native acoustic method remained zero')
    hashes = {str(out/name):digest(out/name) for name in ['forces.csv','observers.csv','wall_geometry.csv','wall_pressure.npz']}
    return dict(status='completed',tables=table,wall=wall,verification=verification,native_nonzero=nonzero,sha256=hashes)


def finite_field(path, *, expected_cells):
    """Validate v2206 little-endian binary scalar/vector volume and boundary lists."""
    import numpy as np
    data = Path(path).read_bytes()
    if not re.search(rb'format\s+binary;',data[:1000]) or b'LSB;label=32;scalar=64' not in data[:1000]:
        raise ValueError('unsupported native binary field format')
    pattern = re.compile(rb'nonuniform\s+List<(scalar|vector)>\s+(\d+)\s*\(')
    offset, checked, maximum, first = 0, 0, 0., True
    while match := pattern.search(data,offset):
        count = int(match[2]); components = 3 if match[1] == b'vector' else 1
        if first and b'internalField' in data[:match.start()] and not re.search(rb'internalField\s+uniform',data[:match.start()]):
            if count != expected_cells:
                raise ValueError('native volume cell count mismatch')
        stop = match.end()+count*components*8
        if stop >= len(data) or data[stop:stop+1] != b')':
            raise ValueError('truncated native binary list')
        values = np.frombuffer(data[match.end():stop],dtype='<f8')
        if not np.isfinite(values).all():
            raise ValueError('nonfinite native volume or boundary field')
        checked += len(values)
        if len(values):
            maximum = max(maximum,float(np.abs(values).max()))
        offset, first = stop+1, False
    # Uniform internal and boundary values also need finite checks.
    for uniform in re.finditer(rb'\buniform\s+([^;]+);',data):
        values = [float(x) for x in uniform[1].replace(b'(',b' ').replace(b')',b' ').split()]
        if not all(math.isfinite(x) for x in values):
            raise ValueError('nonfinite uniform internal or boundary field')
        checked += len(values); maximum = max(maximum,max(map(abs,values)))
    if not checked or b'internalField' not in data:
        raise ValueError('missing native field values')
    return dict(checked_values=checked,max_abs_value=maximum,finite=True)


def main(argv=None):
    """Only the explicitly owned production case can be launched by this CLI."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args(argv)
    prep = json.loads((RESULTS/'prepared.json').read_text())
    if prep['target_s'] != 10:
        raise ValueError('current user authorization is exactly 10 s, not the former 60 s target')
    control = (CASE/'system/controlDict').read_text()
    if not re.search(r'\bendTime\s+10(?:\.0*)?;',control):
        raise ValueError('executed controlDict must target exactly 10 s')
    current = input_identities(CASE)
    if set(prep['input_sha256']) != set(current):
        raise ValueError('prepared essential-input identity coverage mismatch')
    verify_identities(prep['input_sha256'])
    if digest(prep['checkMesh_log']) != prep['checkMesh_sha256']:
        raise ValueError('checkMesh log identity mismatch')
    mesh_gate(Path(prep['checkMesh_log']).read_text())
    if prep['pilot']['observed_max_Co'] > prep['pilot']['acceptance_max_Co']:
        raise ValueError('pilot Courant acceptance failed')
    if args.check_only:
        print('Preflight identities, native mesh quality and pilot Courant evidence verified.',flush=True)
        return 0
    solve_reconstruct(CASE,RESULTS,end=prep['target_s'],dt=prep['dt_s'],ranks=prep['ranks'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
