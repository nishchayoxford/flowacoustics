"""Refined 10 s case: synthetic unit fixtures, no native solver execution."""
from pathlib import Path
import importlib.util
import sys
import math
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


@pytest.fixture(autouse=True)
def synthetic_runtime_identities(tmp_path, monkeypatch):
    """Hash real fixture bytes without requiring any installed native binaries."""
    import cylinder_native

    foam = tmp_path / 'synthetic-runtime'
    solver = foam / 'platforms/linux64GccDPInt32Opt/bin/pimpleFoam'
    library = tmp_path / 'synthetic-libAcoustics.so'
    solver.parent.mkdir(parents=True)
    solver.write_bytes(b'TEST ONLY identity fixture; not an executable')
    library.write_bytes(b'TEST ONLY library identity fixture; not a shared library')
    monkeypatch.setattr(cylinder_native, 'FOAM', foam)
    monkeypatch.setattr(cylinder_native, 'LIB', library)


def runner():
    path = ROOT/'scripts/cylinder_longrun.py'
    assert path.exists(), 'new longrun implementation missing'
    spec = importlib.util.spec_from_file_location('cylinder_longrun', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_finer_mesh_all_directions_under_strict_ceiling():
    n = runner()
    m = n.mesh_description()
    assert m['cells'] == sum(math.prod(b['counts']) for b in m['blocks']) == 459264
    assert m['cells'] < 500000
    assert len(m['blocks']) == 22
    assert {tuple(b['counts']) for b in m['blocks']} == {(44,24,20),(44,24,56),(8,24,20),(24,24,20)}
    assert all(b['grading'][0] == 16 for b in m['blocks'] if b['counts'][0] == 44)
    assert set(m['patches']) == {'cylinder','spanEnds','farfield'}
    assert len(m['patches']['cylinder']) == 14


def test_case_retains_all_fixed_step_fields_and_native_sampling(tmp_path):
    n = runner()
    assert hasattr(n, 'write_case'), 'longrun case writer missing'
    case = tmp_path/'case'
    n.write_case(case, dt=.001, ranks=8)
    c = (case/'system/controlDict').read_text()
    for token in ['startFrom startTime; startTime 0;', 'endTime 10;',
                  'adjustTimeStep no;', 'deltaT 0.001;', 'writeControl runTime; writeInterval 2;',
                  'purgeWrite 0;', 'writeFormat binary;', 'writeCompression off;',
                  'runTimeModifiable false;', 'type yPlus;',
                  'executeControl runTime; executeInterval 0.5;',
                  'writeControl runTime; writeInterval 0.02;',
                  'writeControl runTime; writeInterval 0.5;']:
        assert token in c
    assert c.count('writeControl timeStep; writeInterval 1;') >= 3
    assert 'numberOfSubdomains 8;' in (case/'system/decomposeParDict').read_text()
    assert 'fixedResponseDelay false' in c and 'timeStart 0' in c
    assert 'point (0 0 0.035)' in c and 'point (0 0 0)' in c
    before = (case/'system/controlDict').read_bytes()
    with pytest.raises(ValueError, match='existing'):
        n.write_case(case, end=60, dt=.001, ranks=8)
    assert before == (case/'system/controlDict').read_bytes()
    for kwargs in [dict(dt=.003), dict(ranks=16), dict(end=-1), dict(dt=float('nan'))]:
        with pytest.raises(ValueError):
            n.write_case(tmp_path/'bad', **kwargs)
        assert not (tmp_path/'bad').exists()


def test_command_ledger_records_failure_timeout_progress_and_lock(tmp_path, capsys):
    import json
    import fcntl
    n = runner()
    assert hasattr(n, 'run_command'), 'longrun command ledger missing'
    ledger = tmp_path/'commands.json'
    r = n.run_command([sys.executable, '-c', 'print("evidence"); raise SystemExit(3)'],
                      'failure', ledger=ledger, native_env=False, timeout=5, progress=.01)
    assert r['exit_code'] == 3 and r['state'] == 'failed'
    assert Path(r['log']).read_text().strip() == 'evidence'
    assert r['elapsed_seconds'] > 0 and r['log_sha256']
    r = n.run_command([sys.executable, '-c', 'import time; time.sleep(5)'],
                      'timeout', ledger=ledger, native_env=False, timeout=.1, progress=.02)
    assert r['timed_out'] and r['exit_code'] != 0
    assert len(json.loads(ledger.read_text())) == 2
    assert 'progress' in capsys.readouterr().out
    with ledger.with_suffix('.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            n.run_command(['true'], 'locked', ledger=ledger, native_env=False)
    with pytest.raises(ValueError, match='ranks'):
        n.run_command(['mpirun','-np','16','true'], 'bad', ledger=ledger)


def test_command_aborts_live_solver_on_instability(tmp_path):
    n = runner()
    script = 'import time; print("Courant Number mean: .1 max: 1.1",flush=True); time.sleep(5)'
    r = n.run_command([sys.executable,'-c',script], 'unstable', ledger=tmp_path/'commands.json',
                      native_env=False, progress=.02, watch_solver=True)
    assert r['exit_code'] != 0 and not r['timed_out']
    assert 'Courant' in r['abort_reason']
    assert r['elapsed_seconds'] < 2


def test_mesh_and_solver_acceptance_require_positive_evidence():
    n = runner()
    assert hasattr(n, 'mesh_gate'), 'mesh gate missing'
    assert n.mesh_gate('cells: 459264\nMesh OK.\n')['cell_count'] == 459264
    for text in ['cells: 500000\nMesh OK.', 'cells: 459264\nMesh OK.\nFailed 1 mesh checks.', 'cells: 459264']:
        with pytest.raises(ValueError):
            n.mesh_gate(text)
    log = 'trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\nCourant Number mean: 0.1 max: 0.5\nTime = 0.001\nTime = 0.002\nEnd\n'
    assert n.solver_gate(log, end=.002, dt=.001)['steps'] == 2
    for bad in [log.replace('End', ''), log.replace('0.002','0.0015'), log+'FOAM FATAL ERROR', log+'\nresidual nan\n', log.replace('max: 0.5','max: 1.2')]:
        with pytest.raises(ValueError):
            n.solver_gate(bad, end=.002, dt=.001)


def test_identity_requires_nonempty_complete_inputs_and_detects_change(tmp_path):
    n = runner()
    assert hasattr(n, 'input_identities'), 'input identities missing'
    case = tmp_path/'case'
    n.write_case(case)
    with pytest.raises(FileNotFoundError):
        n.input_identities(case)
    mesh = case/'constant/polyMesh'
    mesh.mkdir()
    for name in ['points','faces','owner','neighbour','boundary']:
        (mesh/name).write_text('synthetic identity fixture only')
    proc = case/'processor0/0'; proc.mkdir(parents=True)
    (proc/'U').write_text('synthetic decomposed initial field')
    pins = n.input_identities(case)
    assert str(proc/'U') in pins, 'actual decomposed inputs must be identity-bound'
    assert len(pins) >= 16
    n.verify_identities(pins)
    with pytest.raises(ValueError, match='empty'):
        n.verify_identities({})
    (case/'0/U').write_text('changed finite value')
    with pytest.raises(ValueError, match='identity'):
        n.verify_identities(pins)


def test_pipeline_never_reconstructs_failed_or_short_solver(tmp_path):
    import json
    n = runner()
    assert hasattr(n, 'solve_reconstruct'), 'solve/reconstruct lifecycle missing'
    for ending, code in [('',0), ('End\n',3), ('Time = 0.001\nEnd\n',0)]:
        case = tmp_path/str(len(list(tmp_path.iterdir())))
        n.write_case(case, end=.002, dt=.001)
        mesh = case/'constant/polyMesh'; mesh.mkdir()
        for name in ['points','faces','owner','neighbour','boundary']:
            (mesh/name).write_text('synthetic identity fixture')
        out = case/'evidence'; out.mkdir()
        calls = []
        def execute(argv, label, **kw):
            calls.append(argv[0])
            p = out/(label+'.log'); p.write_text(ending)
            return dict(exit_code=code, log=str(p), elapsed_seconds=.01)
        with pytest.raises((ValueError,RuntimeError)):
            n.solve_reconstruct(case,out,end=.002,dt=.001,execute=execute)
        assert calls == ['mpirun']
        status = json.loads((out/'run.json').read_text())
        assert status['status'] == 'failed'
        assert status['reconstruction']['status'] == 'pending'
        assert not status['sampling']['adaptive']
        assert status['solver']['ranks'] == 8
        assert status['native_accounting']['previous_1800s_budget_applies'] is False


def test_pipeline_reconstructs_every_saved_time_after_success(tmp_path, monkeypatch):
    n = runner()
    def exported(case,out,*,expected_steps):
        assert expected_steps == 4
        assert (case/'0.004/U').exists()
        return dict(status='completed',sha256={},verification={'test_fixture':True})
    monkeypatch.setattr(n, 'export_outputs', exported)
    checked = []
    def check_field(path, *, expected_cells):
        checked.append(path.name)
        assert expected_cells == 459264
        return dict(finite=True)
    monkeypatch.setattr(n,'finite_field',check_field)
    case, out = tmp_path/'case', tmp_path/'out'
    n.write_case(case,end=.004,dt=.001)
    mesh = case/'constant/polyMesh'; mesh.mkdir()
    for name in ['points','faces','owner','neighbour','boundary']:
        (mesh/name).write_text('synthetic identity fixture')
    out.mkdir()
    calls = []
    def execute(argv,label,**kw):
        calls.append(argv)
        p = out/(label+'.log')
        if argv[0] == 'mpirun':
            assert kw.get('watch_solver') is True, 'production live safety monitor not enabled'
            p.write_text('Courant Number mean: .1 max: .5\n'+''.join(f'Time = {i*.001:g}\n' for i in range(1,5))+'End\n')
            for rank in range(8):
                target = case/f'processor{rank}/0.004'; target.mkdir(parents=True)
                for field in ['p','U','nut','vorticity','yPlus']:
                    (target/field).write_text('synthetic field')
        else:
            p.write_text('Time = 0.004\nEnd\n')
            target = case/'0.004'; target.mkdir()
            for field in ['p','U','nut','vorticity','yPlus']:
                (target/field).write_text('synthetic field')
        return dict(exit_code=0, log=str(p), elapsed_seconds=.01)
    status = n.solve_reconstruct(case,out,end=.004,dt=.001,execute=execute)
    assert [a[0] for a in calls] == ['mpirun','reconstructPar']
    assert '-noZero' in calls[-1] and '-latestTime' not in calls[-1]
    assert status['reconstruction']['status'] == 'completed'
    assert status['reconstruction']['times_s'] == [.004]
    assert set(checked) == {'p','U','nut','vorticity','yPlus'}
    assert len(status['reconstruction']['sha256']) == 5
    assert status['exports']['status'] == 'completed'
    assert status['status'] == 'completed_native_reconstructed'
    assert all((case/f'processor{r}/0.004/U').exists() for r in range(8))
    with pytest.raises(ValueError,match='existing'):
        n.solve_reconstruct(case,out,end=.004,dt=.001,execute=execute)


def test_native_exports_preserve_absolute_clocks_and_check_all_channels(tmp_path):
    import numpy as np
    n = runner()
    assert hasattr(n, 'export_outputs'), 'new native export handover missing'
    case, out = tmp_path/'case', tmp_path/'exports'
    mesh = case/'constant/polyMesh'; mesh.mkdir(parents=True)
    pts = [(-.01,-.01,-.03),(.01,-.01,-.03),(.01,.01,-.03),(-.01,.01,-.03),
           (-.01,-.01,.03),(.01,-.01,.03),(.01,.01,.03),(-.01,.01,.03)]
    faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    (mesh/'points').write_text('8\n(\n'+'\n'.join('('+' '.join(map(str,p))+')' for p in pts)+'\n)')
    (mesh/'faces').write_text('6\n(\n'+'\n'.join('4('+' '.join(map(str,f[::-1]))+')' for f in faces)+'\n)')
    (mesh/'boundary').write_text('cylinder {nFaces 6; startFace 0;}')
    d = case/'postProcessing/forces/0'; d.mkdir(parents=True)
    (d/'force.dat').write_text('.02 -.0012 0 0 -.0012 0 0 0 0 0\n.04 -.0024 0 0 -.0024 0 0 0 0 0\n')
    d = case/'acousticData'; d.mkdir()
    for method in ['curle','fwh']:
        (d/f'{method}-time.dat').write_text('Time '+' '.join(x+'_pFluct' for x in n.native.OBSERVERS)+'\n'+
            '.02 1e-5 -1e-5 2e-5 3e-5 4e-5\n.04 2e-5 -2e-5 3e-5 4e-5 5e-5\n')
    centres = np.array(pts)[np.array(faces)].mean(1)
    for t,factor in [(.02,1),(.04,2)]:
        d = case/f'postProcessing/wallHistory/{t:g}'; d.mkdir(parents=True)
        pressure = np.array([1,1,1,2,1,1])*factor/1000
        np.savetxt(d/'p_body.raw',np.column_stack([centres,pressure]))
    report = n.export_outputs(case,out,expected_steps=2)
    assert report['tables']['force_samples'] == 2
    assert report['verification']['observer_rows'] == 20
    assert report['verification']['pressure_force_max_component_error_N'] < 1e-12
    assert report['native_nonzero']['libacoustics_curle'] and report['native_nonzero']['libacoustics_fwh']
    assert len(report['sha256']) == 4
    with pytest.raises(ValueError,match='step'):
        n.export_outputs(case,tmp_path/'bad',expected_steps=3)


def test_reconstructed_binary_fields_reject_nonfinite_boundaries(tmp_path):
    import numpy as np
    n = runner()
    assert hasattr(n, 'finite_field'), 'binary field finite verification missing'
    p = tmp_path/'p'
    header = b'FoamFile {format binary; arch "LSB;label=32;scalar=64";}\n'
    internal = b'internalField nonuniform List<scalar>\n2\n('+np.array([1.,2.],dtype='<f8').tobytes()+b');\n'
    boundary = b'boundaryField {cylinder {value nonuniform List<scalar>\n1\n('
    p.write_bytes(header+internal+boundary+np.array([3.],dtype='<f8').tobytes()+b');}}')
    report = n.finite_field(p,expected_cells=2)
    assert report['max_abs_value'] == 3 and report['checked_values'] == 3
    p.write_bytes(header+internal+boundary+np.array([float('nan')],dtype='<f8').tobytes()+b');}}')
    with pytest.raises(ValueError,match='nonfinite'):
        n.finite_field(p,expected_cells=2)
    p.write_bytes(header+b'internalField uniform 0; boundaryField {cylinder {value uniform nan;}}')
    with pytest.raises(ValueError,match='nonfinite'):
        n.finite_field(p,expected_cells=2)


def test_cli_preflight_binds_complete_prepared_inputs_before_launch(tmp_path, monkeypatch):
    import json
    n = runner()
    assert hasattr(n, 'main'), 'production CLI missing'
    case, out = tmp_path/'case', tmp_path/'out'
    n.write_case(case,dt=.000625)
    mesh = case/'constant/polyMesh'; mesh.mkdir()
    for name in ['points','faces','owner','neighbour','boundary']:
        (mesh/name).write_text('synthetic identity fixture')
    out.mkdir()
    monkeypatch.setattr(n,'CASE',case)
    monkeypatch.setattr(n,'RESULTS',out)
    with pytest.raises(FileNotFoundError):
        n.main(['--check-only'])
    log = out/'checkMesh.log'; log.write_text('cells: 459264\nMesh OK.\n')
    prep = dict(input_sha256=n.input_identities(case),checkMesh_log=str(log),
                checkMesh_sha256=n.digest(log),dt_s=.000625,ranks=8,target_s=10,
                pilot=dict(observed_max_Co=.7096,acceptance_max_Co=.75))
    (out/'prepared.json').write_text(json.dumps(prep))
    assert n.main(['--check-only']) == 0
    assert not (out/'run.json').exists()
    prep['target_s'] = 60
    (out/'prepared.json').write_text(json.dumps(prep))
    with pytest.raises(ValueError,match='10'):
        n.main(['--check-only'])
    prep['target_s'] = 10
    prep['input_sha256'].pop(str(mesh/'points'))
    (out/'prepared.json').write_text(json.dumps(prep))
    with pytest.raises(ValueError,match='coverage'):
        n.main(['--check-only'])
