"""Small native cylinder contracts; no solver is launched by tests."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def native():
    path = ROOT / 'scripts/cylinder_native.py'
    assert path.exists(), 'native mesh generator is missing'
    spec = importlib.util.spec_from_file_location('cylinder_native', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closed_finite_cylinder_mesh_contract():
    m = native().mesh_description()
    assert 100000 <= m['cells'] < 500000
    assert len(m['blocks']) == 22
    p = np.array(m['vertices'])
    assert p[:, 2].min() < -0.03 and p[:, 2].max() > 0.03
    # Positive local block Jacobians, with caps on genuine fluid cores.
    for b in m['blocks']:
        v = p[b['vertices']]
        assert np.linalg.det(np.array([v[1]-v[0], v[3]-v[0], v[4]-v[0]])) > 0
    assert len(m['patches']['cylinder']) == 14
    text = native().block_mesh_text(m)
    assert 'arc ' in text and 'type wall' in text
    assert 'empty' not in text and 'cyclic' not in text


def test_runner_accounts_failures_and_enforces_budget(tmp_path):
    import sys
    import pytest
    n = native()
    assert hasattr(n, 'run_command'), 'bounded runner missing'
    ledger = tmp_path/'commands.json'
    r = n.run_command([sys.executable, '-c', 'raise SystemExit(3)'], 'failure', 2, ledger=ledger, native_env=False)
    assert r['exit_code'] == 3 and r['elapsed_seconds'] > 0
    r = n.run_command([sys.executable, '-c', 'import time; time.sleep(5)'], 'timeout', .1, ledger=ledger, native_env=False)
    assert r['timed_out'] and r['exit_code'] != 0
    with pytest.raises(ValueError, match='budget'):
        n.run_command(['true'], 'denied', 2, ledger=ledger, budget=.01, native_env=False)


def test_case_configuration_has_native_write_sampling(tmp_path):
    n = native()
    assert hasattr(n, 'write_case'), 'case writer missing'
    n.write_case(tmp_path)
    control = (tmp_path/'system/controlDict').read_text()
    assert 'CurleAnalogy' in control and 'FfowcsWilliamsHawkings' in control
    assert control.count('writeControl timeStep; writeInterval 1;') >= 2
    assert 'cleanFreq 20' in control and 'fixedResponseDelay false' in control
    assert 'WALE' in (tmp_path/'constant/turbulenceProperties').read_text()
    assert 'symmetry' in (tmp_path/'0/U').read_text()
    assert 'endTime 0.1;' in control


def test_native_tables_preserve_units_names_clock_and_reject_bad_data(tmp_path):
    import pytest
    n = native()
    assert hasattr(n, 'read_forces'), 'force and observer parser missing'
    f = tmp_path/'force.dat'
    f.write_text('# native N\n0.1 1 2 3 .9 1.8 2.7 .1 .2 .3\n0.2 4 5 6 3.6 4.5 5.4 .4 .5 .6\n')
    a = n.read_forces(f)
    np.testing.assert_equal(a[:,1:4], [[1,2,3],[4,5,6]])
    p = tmp_path/'fwh.dat'
    p.write_text('Time downstream_pFluct cross_plus_pFluct\n0.001 1e-5 -2e-5\n0.002 2e-5 -3e-5\n')
    r = n.read_observers(p, 'libacoustics_fwh', offset=.1)
    assert r[0] == {'time_s': .101, 'observer':'downstream', 'method':'libacoustics_fwh','pressure_Pa':1e-5}
    p.write_text('Time x_pFluct\n0.1 nan\n')
    with pytest.raises(ValueError, match='finite'):
        n.read_observers(p, 'libacoustics_fwh')


def test_wall_geometry_proves_closed_edges_and_body_outward_normals(tmp_path):
    n = native()
    assert hasattr(n, 'wall_geometry'), 'wall geometry extraction missing'
    points = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    outward = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    (tmp_path/'points').write_text('8\n(\n'+'\n'.join('('+' '.join(map(str,p))+')' for p in points)+'\n)\n')
    (tmp_path/'faces').write_text('6\n(\n'+'\n'.join('4('+' '.join(map(str,f[::-1]))+')' for f in outward)+'\n)\n')
    (tmp_path/'boundary').write_text('1 ( cylinder { type wall; nFaces 6; startFace 0; } )')
    g = n.wall_geometry(tmp_path)
    assert g['audit']['closed'] and g['audit']['unique_edges'] == 12
    assert np.all(np.sum(g['centres']*g['area_vectors_body_out'],axis=1) > 0)
    np.testing.assert_allclose(np.linalg.norm(g['area_vectors_body_out'],axis=1), 4)
    np.testing.assert_allclose(np.sum(g['area_vectors_body_out'],axis=0),0)


def test_export_tables_carries_real_samples_and_pressure_only_force(tmp_path):
    import csv
    n = native()
    assert hasattr(n, 'export_tables'), 'table exporter missing'
    case,out = tmp_path/'case',tmp_path/'out'
    d = case/'postProcessing/forces/0'
    d.mkdir(parents=True)
    (d/'force.dat').write_text('0.1 1 2 3 .9 1.8 2.7 .1 .2 .3\n0.2 4 5 6 3.6 4.5 5.4 .4 .5 .6\n')
    p = case/'native.dat'
    p.write_text('Time cross_plus_pFluct\n0.1 1e-5\n0.2 -1e-5\n')
    summary = n.export_tables(case,out,[(p,'libacoustics_curle',0.)])
    rows = list(csv.DictReader((out/'forces.csv').read_text().splitlines()))
    assert len(rows)==2 and float(rows[0]['force_x_N'])==1
    assert float(rows[0]['pressure_force_x_N'])==.9
    assert summary['force_samples']==2 and summary['observer_rows']==2


def test_wall_history_matches_geometry_not_mpi_output_order(tmp_path):
    n = native()
    assert hasattr(n, 'export_wall_history'), 'wall history exporter missing'
    case,out=tmp_path/'case',tmp_path/'out'
    g={'centres':np.array([[0,0,0],[1,0,0]]), 'area_vectors_body_out':np.array([[1,0,0],[-1,0,0]]), 'global_face_ids':np.array([10,11])}
    for t in [.1,.2]:
        d=case/'postProcessing/wallHistory'/str(t)
        d.mkdir(parents=True)
        (d/'p_body.raw').write_text('# kinematic pressure\n1 0 0 .002\n0 0 0 .001\n')
    summary=n.export_wall_history(case,out,g)
    with np.load(out/'wall_pressure.npz') as a:
        np.testing.assert_equal(a['pressure_Pa'],[[1,2],[1,2]])
        np.testing.assert_equal(a['time_s'],[.1,.2])
    assert summary['snapshots']==2 and summary['max_mapping_error_m']==0


def test_flow_export_uses_native_velocity_vorticity_and_plots(tmp_path):
    n=native()
    assert hasattr(n,'export_flow'), 'flow exporter missing'
    case,out=tmp_path/'case',tmp_path/'out'
    d=case/'postProcessing/flowSlice/1.5'
    d.mkdir(parents=True)
    for plane,z in [('midspan',0),('tipplane',.035)]:
        rows=np.array([[-.05,-.05,z,.2,.01,.03],[.15,-.05,z,.1,-.02,-.03],[.15,.05,z,.1,.02,.04],[-.05,.05,z,.2,-.01,-.04]])
        np.savetxt(d/f'U_{plane}.raw',rows)
        rows[:,3:]*=100
        np.savetxt(d/f'vorticity_{plane}.raw',rows)
    info=n.export_flow(case,out)
    assert info['last_snapshot_s']==1.5 and info['rows']==8
    assert info['latest_max_abs_spanwise_velocity_m_s']==.04
    assert (out/'flow_vorticity.png').read_bytes().startswith(b'\x89PNG')
    assert info['colour_scale'] == 'symmetric logarithmic, linear within +/-1 per second; no clipping'


def test_manifest_requires_real_solver_end_and_matching_sample_counts():
    n=native()
    assert hasattr(n,'build_manifest'), 'native completion manifest missing'
    commands=[{'exit_code':0,'timed_out':False,'elapsed_seconds':1.5,'argv':['pimpleFoam']}]
    mesh='cells: 138240\nMesh OK.\n'
    tables={'force_samples':2,'observer_rows':20,'force_end_s':.2,'dt_min_s':.1,'dt_max_s':.1}
    run=n.build_manifest(commands,mesh,['Time = 0.1\nTime = 0.2\nEnd\n'],tables,{}, {}, {},target_end_s=.2)
    assert run['status']=='completed_demo' and run['cell_count']==138240
    np.testing.assert_allclose(run['time_coverage']['convective_times_D_over_U'],2,rtol=1e-15)
    assert run['native_budget']['used_wall_seconds']==1.5
    bad=n.build_manifest(commands,mesh,['Time = 0.1\nTime = 0.2\n'],tables,{}, {}, {},target_end_s=.2)
    assert bad['status']=='incomplete' and not bad['verification']['all_solver_end_markers']
    # OpenFOAM Time::run uses endTime - 0.5*deltaT for adaptive stopping.
    near_end=n.build_manifest(commands,mesh,['Time = 0.1\nTime = 0.199\nEnd\n'],dict(tables,force_end_s=.199),{}, {}, {},target_end_s=.2)
    assert near_end['status']=='completed_demo'
    assert near_end['time_coverage']['end_s']==.199


def test_actual_mesh_crosssection_uses_native_planar_faces(tmp_path):
    n=native()
    assert hasattr(n,'export_mesh_section'), 'native mesh cross-section plot missing'
    points=[(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    (tmp_path/'points').write_text('8\n(\n'+'\n'.join('('+' '.join(map(str,p))+')' for p in points)+'\n)\n')
    (tmp_path/'faces').write_text('2\n(\n4(0 1 2 3)\n4(4 5 6 7)\n)\n')
    info=n.export_mesh_section(tmp_path,tmp_path/'mesh.png')
    assert info['section_faces']==1 and info['section_edges']==4
    assert info['section_z_m']==0
    assert info['xy_bounds_m']==[[-1.,-1.],[1.,1.]]
    assert (tmp_path/'mesh.png').read_bytes().startswith(b'\x89PNG')
