"""Circular outer-domain finite-cylinder native demonstration.

Reuse immutable, tested native helpers without rewriting the rectangular case.
"""
from pathlib import Path
import math
import cylinder_native as native

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/'cfd/cylinder_circular'
RESULTS = ROOT/'results/cylinder/native_circular'
LEDGER = ROOT/'results/cylinder/native/commands.json'
RADIUS = .24
WAKE_GRADING = '((0.5 0.5 0.65) (0.5 0.5 1.53846153846))'


def mesh_description():
    """Circle-to-circle O-grid; real fluid cores beyond both closed tips.

    The +/-x sectors use symmetric angular multi-grading toward their
    centreline. Opposite central-square edges inherit exactly that spacing.
    """
    m = native.mesh_description()
    vertices = m['vertices']
    for k in range(4):
        for i in range(4):
            index = k*12 + 4+i
            theta = math.pi/4+i*math.pi/2
            z = vertices[index][2]
            vertices[index] = (RADIUS*math.cos(theta),RADIUS*math.sin(theta),z)
            j = k*12+4+(i+1)%4
            mid = theta+math.pi/4
            m['arcs'].append((index,j,(RADIUS*math.cos(mid),RADIUS*math.sin(mid),z)))
    farfield = []
    for name in ['inlet','outlet','sides']:
        farfield.extend(m['patches'].pop(name))
    m['patches']['farfield'] = farfield
    for b in m['blocks']:
        if b['counts'][0] == 32:
            b['grading'][0] = 24
        if b['counts'][:2] == [16,16]:
            b['grading'][1] = WAKE_GRADING
        else:
            theta = math.atan2(vertices[b['vertices'][0]][1],vertices[b['vertices'][0]][0])
            if math.cos(theta)*math.sin(theta) < 0:
                b['grading'][1] = WAKE_GRADING
    return m


def block_mesh_text(m):
    def vec(v):
        return '('+' '.join(x if isinstance(x,str) else f'{x:.12g}' for x in v)+')'
    out = ['FoamFile {version 2.0; format ascii; class dictionary; object blockMeshDict;}',
           'scale 1;', 'vertices (']
    out += [vec(v) for v in m['vertices']]+[');','blocks (']
    out += ['hex '+vec(b['vertices'])+' '+vec(b['counts'])+' simpleGrading '+vec(b['grading']) for b in m['blocks']]
    out += [');','edges (']+[f'arc {a} {b} {vec(c)}' for a,b,c in m['arcs']]+[');','boundary (']
    for name,faces in m['patches'].items():
        kind = 'wall' if name == 'cylinder' else ('symmetry' if name == 'spanEnds' else 'patch')
        out += [f'{name} {{type {kind}; faces (']+[vec(f) for f in faces]+[');}']
    return '\n'.join(out+[');','mergePatchPairs ();',''])


def write_case(case=CASE):
    """Prepare a fresh pilot only; refuse to overwrite any existing case."""
    case = Path(case)
    if case.exists():
        raise ValueError('refusing to overwrite existing case')
    native.write_case(case)
    (case/'system/blockMeshDict').write_text(block_mesh_text(mesh_description()))
    fields = {
        'U': ('volVectorField', '[0 1 -1 0 0 0 0]', '(0.2 0.002 0.001)',
              'freestreamVelocity; freestreamValue uniform (0.2 0 0); value uniform (0.2 0 0);', 'noSlip;'),
        'p': ('volScalarField', '[0 2 -2 0 0 0 0]', '0',
              'freestreamPressure; freestreamValue uniform 0; value uniform 0;', 'zeroGradient;'),
        'nut': ('volScalarField', '[0 2 -1 0 0 0 0]', '0',
                'calculated; value uniform 0;', 'fixedValue; value uniform 0;'),
    }
    for name,(cls,dimension,value,farfield,wall) in fields.items():
        (case/'0'/name).write_text(
            f'FoamFile {{version 2.0; format ascii; class {cls}; object {name};}}\n'
            f'dimensions {dimension}; internalField uniform {value};\n'
            'boundaryField {\n'
            f'farfield {{type {farfield}}}\n'
            f'cylinder {{type {wall}}}\n'
            'spanEnds {type symmetry;}\n}\n')


def export_mesh_views(mesh_dir, output):
    """Draw only actual native edges: full midspan and 3D cutaway/detail."""
    import re
    import hashlib
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    mesh_dir, output = Path(mesh_dir), Path(output)
    points = np.array([list(map(float,m.split())) for m in re.findall(r'^\(([^()]+)\)$',(mesh_dir/'points').read_text(),re.M)])
    faces = np.array([list(map(int,m.split())) for m in re.findall(r'^4\(([^()]+)\)$',(mesh_dir/'faces').read_text(),re.M)])
    if not len(faces) or not np.isfinite(points).all():
        raise ValueError('finite native points and quadrilateral faces required')
    boundary = (mesh_dir/'boundary').read_text()
    patches = {}
    for name in ['cylinder','spanEnds','farfield']:
        body = re.search(r'\b'+name+r'\s*\{([^}]+)\}',boundary).group(1)
        start = int(re.search(r'startFace\s+(\d+)',body).group(1))
        count = int(re.search(r'nFaces\s+(\d+)',body).group(1))
        patches[name] = faces[start:start+count]
    section = faces[np.all(np.isclose(points[faces,2],0,atol=1e-10),axis=1)]
    def edges(selected):
        unique = sorted({tuple(sorted((int(a),int(b)))) for face in selected for a,b in zip(face,np.roll(face,-1))})
        return points[np.array(unique,dtype=int)] if unique else np.empty((0,2,3))
    segments = edges(section)
    if not len(segments):
        raise ValueError('no native z=0 faces: do not fabricate a mesh section')
    output.mkdir(parents=True,exist_ok=True)
    bounds = [points[:,:2].min(0).tolist(),points[:,:2].max(0).tolist()]
    fig,ax = plt.subplots(figsize=(8,8),layout='constrained')
    ax.add_collection(LineCollection(segments[:,:,:2],colors='#223963',linewidths=.42))
    ax.set(xlim=(bounds[0][0],bounds[1][0]),ylim=(bounds[0][1],bounds[1][1]),
           aspect='equal',xlabel='streamwise x [m]',ylabel='cross-flow y [m]',
           title='Actual native midspan mesh | circular outer domain\nO-grid with modest +/-x angular concentration')
    fig.savefig(output/'mesh_cross_section.png',dpi=200)
    plt.close(fig)
    fig = plt.figure(figsize=(14,7),layout='constrained')
    axes = [fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
    wall = patches['cylinder']
    for ax in axes:
        ax.add_collection3d(Poly3DCollection(points[wall],facecolor='#d8e3ec',edgecolor='#314f6a',linewidths=.22,alpha=1))
        ax.view_init(elev=24,azim=-55)
        ax.set(xlabel='x [m]',ylabel='y [m]',zlabel='z [m]')
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.zaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=8,pad=1)
        ax.zaxis.labelpad = 10
    far = patches['farfield']
    far = far[points[far,1].mean(axis=1)>0]
    bottom = patches['spanEnds']
    bottom = bottom[points[bottom,2].mean(axis=1)<0]
    if len(far):
        axes[0].add_collection3d(Line3DCollection(edges(far),colors='#8d9eac',linewidths=.2,alpha=.6))
    if len(bottom):
        axes[0].add_collection3d(Line3DCollection(edges(bottom),colors='#8d9eac',linewidths=.2,alpha=.5))
    axes[0].add_collection3d(Line3DCollection(segments,colors='#2b5479',linewidths=.25,alpha=.7))
    axes[0].set(xlim=(bounds[0][0],bounds[1][0]),ylim=(bounds[0][1],bounds[1][1]),
                zlim=(points[:,2].min(),points[:,2].max()),title='3D cutaway: rear farfield, bottom, midspan, complete body')
    axes[0].set_box_aspect(np.ptp(points,axis=0))
    near = faces[np.all(np.linalg.norm(points[faces][:,:,:2],axis=2)<.016,axis=1)]
    # Front half removed only for display; complete wall remains visible.
    near = near[(points[near,1].mean(axis=1)>0)&(np.abs(points[near,2].mean(axis=1))<.045)]
    if len(near):
        axes[1].add_collection3d(Line3DCollection(edges(near),colors='#608cac',linewidths=.17,alpha=.4))
    axes[1].set(xlim=(-.016,.016),ylim=(-.016,.016),zlim=(-.045,.045),
                title='Finite cylinder detail: genuine caps and fluid beyond tips')
    axes[1].set_box_aspect((.032,.032,.09))
    fig.suptitle('Generated OpenFOAM points/faces connectivity — no schematic mesh')
    fig.savefig(output/'mesh_3d.png',dpi=180,bbox_inches='tight',pad_inches=.25)
    plt.close(fig)
    return dict(section_z_m=0.,section_faces=len(section),section_edges=len(segments),
                wall_faces=len(wall),farfield_faces=len(patches['farfield']),xy_bounds_m=bounds,
                provenance='actual constant/polyMesh points/faces/boundary; 3D cutaway omits front farfield for visibility',
                native_mesh_sha256={name:hashlib.sha256((mesh_dir/name).read_bytes()).hexdigest() for name in ['points','faces','boundary']})


def build_manifest(commands, mesh_log, solver_logs, tables, wall, flow, file_hashes,
                   *, geometry_audit, target_end_s=6.):
    """Keep cumulative accounting while identifying only this circular case."""
    run = native.build_manifest(commands,mesh_log,solver_logs,tables,wall,flow,file_hashes,
                                target_end_s=target_end_s)
    run['evidence_kind'] = 'native_execution'
    run['case_path'] = 'cfd/cylinder_circular'
    run['results_path'] = 'results/cylinder/native_circular'
    run['medium']['velocity_m_s'] = .2
    run['force_convention'] = 'on_body'
    run['force_content'] = 'total_traction'
    run['clock'] = dict(forces_offset_s=0,
                        libacoustics_curle=dict(offset_s=0,kind='source_unretarded'),
                        libacoustics_fwh=dict(offset_s=0,kind='observer'))
    run['geometry'].update(outer_domain_shape='circular_cylinder',outer_radius_m=RADIUS,
                           outer_radius_over_D=12.,domain_bounds_m=[[-.24,-.24,-.08],[.24,.24,.08]],
                           outer_boundary='circular xy boundary centered on body axis; remote z planes beyond finite caps')
    run['inflow']['boundary'] = 'uniform freestream target; freestreamVelocity/freestreamPressure orientation-blended open curved farfield'
    run['boundary_conditions'] = dict(farfield_U='freestreamVelocity',farfield_p='freestreamPressure',
                                      freestream_velocity_m_s=[.2,0,0],freestream_kinematic_pressure_m2_s2=0,
                                      cylinder_U='noSlip',cylinder_p='zeroGradient',spanEnds='symmetry at z=+/-0.08m')
    run['mesh_type'] = '22-block all-hexahedral circular O-grid; real disc fluid cores beyond tips; moderate +/-x angular multi-grading'
    run['mesh_grading'] = dict(radial_expansion_ratio=24,angular_centre_to_sector_edge_ratio=.65,
                               circumferential_cells=64,radial_cells=32,span_cells=[12,36,12],
                               limitation='modest angular concentration, not an independently refined long wake strip or an exact copy of reference mesh density')
    run['mesh_candidates'] = sum(c.get('label','').startswith('circular_blockMesh') for c in commands)
    run['case_commands'] = [c for c in commands if c.get('label','').startswith('circular_')]
    run['native_budget']['ledger_path'] = str(LEDGER.relative_to(ROOT))
    run['native_budget']['prior_rectangular_wall_seconds'] = sum(c['elapsed_seconds'] for c in commands if not c.get('label','').startswith('circular_'))
    run['native_budget']['circular_wall_seconds'] = sum(c['elapsed_seconds'] for c in run['case_commands'])
    run['wall']['geometry_audit'] = geometry_audit
    run['verification']['finite_closed_capped_surface'] = bool(geometry_audit.get('closed') and geometry_audit.get('cap_minus_faces',0)>0 and geometry_audit.get('cap_plus_faces',0)>0)
    run['status'] = 'completed_demo' if all(run['verification'].values()) else 'incomplete'
    run['time_coverage']['end_rule'] = 'OpenFOAM Time::run stops at current time >= endTime - 0.5*deltaT; actual last timestamp retained'
    run['acoustic_conventions']['native_pFluct_label'] = 'raw native expression, not mean-subtracted; steady near-field offsets preserved'
    run['files'].update(mesh_cross_section='mesh_cross_section.png',mesh_3d='mesh_3d.png',mesh_views='mesh_views.json')
    return run


def verify_exports(output):
    """Independently reconcile pressure integrals and every native clock row."""
    import csv
    import numpy as np
    from collections import defaultdict
    output = Path(output)
    forces = np.loadtxt(output/'forces.csv',delimiter=',',skiprows=1,ndmin=2)
    if forces.shape[1]!=10 or not np.isfinite(forces).all() or np.any(np.diff(forces[:,0])<=0):
        raise ValueError('invalid finite ordered force table')
    if not np.allclose(forces[:,1:4],forces[:,4:7]+forces[:,7:10],rtol=0,atol=1e-9):
        raise ValueError('total traction does not equal pressure plus viscous force')
    channels = defaultdict(list)
    with (output/'observers.csv').open() as stream:
        for row in csv.DictReader(stream):
            channels[row['observer'],row['method']].append([float(row['time_s']),float(row['pressure_Pa'])])
    expected = {(name,method) for name in native.OBSERVERS for method in ['libacoustics_curle','libacoustics_fwh']}
    if set(channels)!=expected:
        raise ValueError('incomplete observer channel membership')
    time_error = 0.
    for values in channels.values():
        a = np.array(values)
        if a.shape!=(len(forces),2) or not np.isfinite(a).all() or np.any(np.diff(a[:,0])<=0):
            raise ValueError('incomplete/nonfinite/reversed observer samples')
        time_error=max(time_error,float(np.max(np.abs(a[:,0]-forces[:,0]))))
    if time_error>1e-9:
        raise ValueError('observer clocks do not match native solver force clock')
    with np.load(output/'wall_pressure.npz') as data:
        times,pressure,area = data['time_s'],data['pressure_Pa'],data['area_vectors_body_out_m2']
        if not all(np.isfinite(x).all() for x in [times,pressure,area]) or np.any(np.diff(times)<=0):
            raise ValueError('nonfinite/reversed wall samples')
        index=np.abs(forces[:,0,None]-times[None,:]).argmin(axis=0)
        wall_time_error=float(np.max(np.abs(forces[index,0]-times)))
        predicted = -pressure@area
        error = predicted-forces[index,4:7]
        maximum=float(np.abs(error).max())
        if wall_time_error>1e-9 or maximum>1e-9:
            raise ValueError('wall pressure force does not reconcile at native timestamps')
        snapshots,wall_faces=pressure.shape
    return dict(force_samples=len(forces),observer_channels=len(channels),observer_rows=sum(map(len,channels.values())),
                max_observer_force_clock_error_s=time_error,wall_snapshots=snapshots,wall_faces=wall_faces,
                max_wall_force_clock_error_s=wall_time_error,pressure_force_max_component_error_N=maximum,
                pressure_force_relative_vector_L2=float(np.linalg.norm(error)/max(np.linalg.norm(forces[index,4:7]),1e-300)))
