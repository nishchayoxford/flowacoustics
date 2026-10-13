"""Circular-domain feature contracts; tests never launch native solvers."""
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))


def circular():
    path = ROOT/'scripts/cylinder_circular.py'
    assert path.exists(), 'circular outer-domain implementation is missing'
    spec = importlib.util.spec_from_file_location('cylinder_circular', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_circular_outer_mesh_has_true_arcs_wake_clustering_and_finite_caps():
    n = circular()
    m = n.mesh_description()
    assert 100000 <= m['cells'] <= 160000
    assert len(m['blocks']) == 22
    p = np.array(m['vertices'])
    outer_faces = m['patches']['farfield']
    outer_ids = np.unique(np.array(outer_faces))
    np.testing.assert_allclose(np.linalg.norm(p[outer_ids,:2], axis=1), .24, atol=1e-12)
    assert len([a for a in m['arcs'] if np.isclose(np.linalg.norm(a[2][:2]), .24)]) == 16
    assert p[:,2].min() == -.08 and p[:,2].max() == .08
    assert len(m['patches']['cylinder']) == 14
    for b in m['blocks']:
        v = p[b['vertices']]
        assert np.linalg.det(np.array([v[1]-v[0],v[3]-v[0],v[4]-v[0]])) > 0
    text = n.block_mesh_text(m)
    assert 'farfield {type patch;' in text
    assert 'inlet' not in text and 'outlet' not in text
    assert text.count('((0.5 0.5 0.65) (0.5 0.5 1.53846153846))') >= 6
    assert 'empty' not in text and 'cyclic' not in text


def test_case_uses_open_curved_farfield_and_protects_completed_case(tmp_path):
    import pytest
    n = circular()
    assert hasattr(n, 'write_case'), 'circular case writer missing'
    case = tmp_path/'fresh'
    n.write_case(case)
    u = (case/'0/U').read_text()
    p = (case/'0/p').read_text()
    assert 'farfield {type freestreamVelocity; freestreamValue uniform (0.2 0 0);' in u
    assert 'farfield {type freestreamPressure;' in p
    assert 'sides' not in u and 'inlet {' not in u
    assert 'internalField uniform (0.2 0.002 0.001)' in u
    assert 'spanEnds {type symmetry;}' in u
    assert 'WALE' in (case/'constant/turbulenceProperties').read_text()
    c = (case/'system/controlDict').read_text()
    assert 'CurleAnalogy' in c and 'FfowcsWilliamsHawkings' in c
    assert c.count('writeControl timeStep; writeInterval 1;') >= 2
    assert 'cleanFreq 20' in c and 'U0 (0 0 0)' in c
    assert n.LEDGER == ROOT/'results/cylinder/native/commands.json'
    before = (case/'system/controlDict').read_bytes()
    with pytest.raises(ValueError, match='existing'):
        n.write_case(case)
    assert (case/'system/controlDict').read_bytes() == before


def test_mesh_previews_use_native_connectivity_and_include_3d(tmp_path):
    n = circular()
    assert hasattr(n, 'export_mesh_views'), 'actual circular mesh previews missing'
    mesh = tmp_path/'mesh'
    mesh.mkdir()
    points = [(-1,-1,0),(1,-1,0),(1,1,0),(-1,1,0),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    (mesh/'points').write_text('8\n(\n'+'\n'.join('('+' '.join(map(str,p))+')' for p in points)+'\n)\n')
    (mesh/'faces').write_text('6\n(\n'+'\n'.join('4('+' '.join(map(str,f))+')' for f in faces)+'\n)\n')
    (mesh/'boundary').write_text('3 ( cylinder {nFaces 1; startFace 0;} spanEnds {nFaces 1; startFace 1;} farfield {nFaces 4; startFace 2;} )')
    output = tmp_path/'out'
    info = n.export_mesh_views(mesh,output)
    assert info['section_faces'] == 1 and info['section_edges'] == 4
    assert info['wall_faces'] == 1 and info['farfield_faces'] == 4
    assert info['xy_bounds_m'] == [[-1.,-1.],[1.,1.]]
    for name in ['mesh_cross_section.png','mesh_3d.png']:
        assert (output/name).read_bytes().startswith(b'\x89PNG')
    assert set(info['native_mesh_sha256']) == {'points','faces','boundary'}


def test_circular_manifest_keeps_shared_budget_actual_geometry_and_clocks():
    n = circular()
    assert hasattr(n, 'build_manifest'), 'circular manifest missing'
    commands = [dict(argv=['pimpleFoam'],label='old',elapsed_seconds=850,exit_code=0,timed_out=False),
                dict(argv=['pimpleFoam'],label='circular_main',elapsed_seconds=100,exit_code=0,timed_out=False)]
    tables = dict(force_samples=2,observer_rows=20,force_end_s=.2,dt_min_s=.1,dt_max_s=.1)
    audit = dict(closed=True,cap_minus_faces=640,cap_plus_faces=640)
    run = n.build_manifest(commands,'cells: 138240\nMesh OK.\n',['Time = 0.1\nTime = 0.2\nEnd\n'],tables,{}, {}, {},geometry_audit=audit,target_end_s=.2)
    assert run['status'] == 'completed_demo'
    assert run['geometry']['outer_domain_shape'] == 'circular_cylinder'
    assert run['geometry']['outer_radius_m'] == .24
    assert run['geometry']['domain_bounds_m'] == [[-.24,-.24,-.08],[.24,.24,.08]]
    assert run['native_budget']['used_wall_seconds'] == 950
    assert run['native_budget']['prior_rectangular_wall_seconds'] == 850
    assert run['evidence_kind'] == 'native_execution'
    assert run['medium']['velocity_m_s'] == .2
    assert run['force_convention'] == 'on_body' and run['force_content'] == 'total_traction'
    assert run['clock']['libacoustics_curle'] == dict(offset_s=0,kind='source_unretarded')
    assert run['clock']['libacoustics_fwh'] == dict(offset_s=0,kind='observer')
    assert run['clock']['forces_offset_s'] == 0
    assert not run['inflow']['synthetic_inlet_turbulence']
    assert run['case_path'] == 'cfd/cylinder_circular'
    bad = n.build_manifest(commands,'cells: 138240\nMesh OK.\n',['Time = 0.1\nTime = 0.2\nEnd\n'],tables,{}, {}, {},geometry_audit=dict(audit,cap_plus_faces=0),target_end_s=.2)
    assert bad['status'] == 'incomplete'


def test_export_verifier_checks_all_channels_and_wall_pressure_force(tmp_path):
    import csv
    import pytest
    n = circular()
    assert hasattr(n, 'verify_exports'), 'circular export verification missing'
    forces = np.array([[.1,3,0,0,3,0,0,0,0,0],[.2,5,0,0,5,0,0,0,0,0]])
    np.savetxt(tmp_path/'forces.csv', forces,delimiter=',',header='time,x,y,z,px,py,pz,vx,vy,vz',comments='')
    rows = [[t,name,method,1e-5] for t in [.1,.2] for name in n.native.OBSERVERS for method in ['libacoustics_curle','libacoustics_fwh']]
    with (tmp_path/'observers.csv').open('w',newline='') as stream:
        w=csv.writer(stream); w.writerow(['time_s','observer','method','pressure_Pa']); w.writerows(rows)
    np.savez(tmp_path/'wall_pressure.npz',time_s=np.array([.1,.2]),pressure_Pa=np.array([[1,4],[2,7]]),area_vectors_body_out_m2=np.array([[1,0,0],[-1,0,0]]))
    verified=n.verify_exports(tmp_path)
    assert verified['observer_channels'] == 10
    assert verified['observer_rows'] == 20
    assert verified['pressure_force_max_component_error_N'] == 0
    assert verified['force_samples'] == 2
    forces[0,4] = 2
    np.savetxt(tmp_path/'forces.csv', forces,delimiter=',',header='time,x,y,z,px,py,pz,vx,vy,vz',comments='')
    with pytest.raises(ValueError,match='traction|pressure force'):
        n.verify_exports(tmp_path)
