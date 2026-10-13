"""Bounded native finite-cylinder demonstration helpers (OpenFOAM v2206)."""
from pathlib import Path
import math
import json
import os
import signal
import subprocess
import time
import hashlib
import shlex
import fcntl
import numpy as np
import re
from collections import Counter
import csv


def export_mesh_section(mesh_dir,output):
    """Plot actual z=0 native faces/connectivity, never a schematic replacement."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    mesh_dir,output=Path(mesh_dir),Path(output)
    points=np.array([list(map(float,m.split())) for m in re.findall(r'^\(([^()]+)\)$',(mesh_dir/'points').read_text(),re.M)])
    faces=np.array([list(map(int,m.split())) for m in re.findall(r'^4\(([^()]+)\)$',(mesh_dir/'faces').read_text(),re.M)])
    faces=faces[np.all(np.isclose(points[faces,2],0,atol=1e-10),axis=1)]
    edges=set()
    for f in faces:
        for a,b in zip(f,np.roll(f,-1)):
            edges.add(tuple(sorted((int(a),int(b)))))
    edges=sorted(edges)
    segments=points[np.array(edges),:2]
    bounds=[segments.reshape(-1,2).min(0).tolist(),segments.reshape(-1,2).max(0).tolist()]
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax in axes:
        ax.add_collection(LineCollection(segments,colors='#253c6b',linewidths=.35))
        ax.set_aspect('equal')
        ax.set(xlabel='streamwise x [m]',ylabel='cross-flow y [m]')
    axes[0].set(xlim=(bounds[0][0],bounds[1][0]),ylim=(bounds[0][1],bounds[1][1]),title='Executed outer domain: rectangle')
    axes[1].set(xlim=(-.025,.10),ylim=(-.045,.045),title='Cylinder / near-wake detail')
    fig.suptitle('Actual native mesh at z = 0 — inner O-grid, not a circular outer boundary')
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180)
    plt.close(fig)
    return dict(section_z_m=0.,section_faces=len(faces),section_edges=len(edges),xy_bounds_m=bounds,
                provenance='native constant/polyMesh points/faces; only faces lying on z=0, unique actual edges',
                matches_new_circular_outer_domain_request=False)


def build_manifest(commands,mesh_log,solver_logs,tables,wall,flow,file_hashes,*,target_end_s=6.):
    """Completion is verified against native logs/counts, not wrapper return alone."""
    cells=int(re.search(r'cells:\s+(\d+)',mesh_log).group(1))
    times=[float(t) for text in solver_logs for t in re.findall(r'^Time = ([0-9.eE+-]+)$',text,re.M)]
    elapsed=sum(c['elapsed_seconds'] for c in commands)
    ranks=max([int(c['argv'][c['argv'].index('-np')+1]) if '-np' in c['argv'] else 1 for c in commands])
    checks=dict(mesh_ok='Mesh OK.' in mesh_log and 'Failed' not in mesh_log,
                below_cell_ceiling=cells<500000,
                all_solver_end_markers=all(bool(re.search(r'^End\s*$',text,re.M)) for text in solver_logs),
                commands_successful=all(c['exit_code']==0 and not c['timed_out'] for c in commands),
                samples_match_solver_steps=len(times)==tables['force_samples'],
                both_native_methods_all_observers=tables['observer_rows']==tables['force_samples']*len(OBSERVERS)*2,
                reached_requested_time=bool(times) and abs(times[-1]-target_end_s)<=.5*(times[-1]-times[-2] if len(times)>1 else tables['dt_max_s'])+1e-10,
                within_budget=elapsed<=1800 and ranks<=4)
    courants=[float(x) for text in solver_logs for x in re.findall(r'Courant Number mean: [^\n]+ max: ([0-9.eE+-]+)',text)]
    return dict(schema_version=1,status='completed_demo' if all(checks.values()) else 'incomplete',
        scope='single-phase finite closed 3D circular-cylinder WALE LES demonstration, not research validation',
        medium=dict(name='water demonstration assumption',density_kg_m3=1000.,kinematic_viscosity_m2_s=1e-6,sound_speed_m_s=1500.,reference_pressure_Pa=1e-6),
        geometry=dict(diameter_m=.02,span_m=.06,centre_m=[0,0,0],axis='z',axes={'x':'streamwise','y':'crossflow/lift','z':'span'},domain_bounds_m=[[-.1,-.1,-.08],[.24,.1,.08]],finite_closed=True),
        inflow=dict(mean_velocity_m_s=[.2,0,0],reynolds_number=4000.,boundary='uniform fixedValue',
                    startup_velocity_m_s=[.2,.002,.001],startup_perturbation='uniform initial +1% crossflow and +0.5% spanwise offsets, removed at inlet; finite tips break spanwise invariance',synthetic_inlet_turbulence=False),
        cell_count=cells,mesh_candidates=1,mesh_type='22-block conformal all-hexahedral O-grid, with square/disc fluid cores beyond real tips',
        solver=dict(name='pimpleFoam',openfoam='v2206',les_model='WALE',delta='cubeRootVol',ranks=ranks,threads_per_rank=1,pressure='kinematic m2/s2',maxCo_target=.7,observed_max_Co=max(courants) if courants else None,advection='linearUpwind grad(U)',time_scheme='backward',mesh_adequacy_validated=False),
        sampling=dict(adaptive=True,acoustics='writeControl timeStep, writeInterval=1, probeFrequency=1',forces='every solver step',dt_min_s=tables['dt_min_s'],dt_max_s=tables['dt_max_s'],wall_pressure='every 20 solver steps',flow='every 0.5s from main restart at 0.1s plus pilot at 0.1s; actual snapshot times in CSV',native_fft=False),
        time_coverage=dict(start_s=0.,end_s=tables['force_end_s'],requested_end_s=target_end_s,convective_time_D_over_U_s=.02/.2,convective_times_D_over_U=tables['force_end_s']/(.02/.2),startup_plot_interval_s=[0,1],later_demonstration_interval_s=[2,tables['force_end_s']],statistically_developed=False),
        observers=[dict(name=k,position_m=v) for k,v in OBSERVERS.items()],observer_coordinates_m=OBSERVERS,observer_plane='xy at z=0',
        acoustic_conventions=dict(mean_flow_m_s=[0,0,0],propagation='free space homogeneous; small-Mach convection neglected',
            methods=['libacoustics_curle','libacoustics_fwh'],
            curle='native forceEff (total pressure+viscous force ON body); current CFD time, no retarded-time shift; includes 1/r2 near term and 1/(c0*r) derivative term',
            fwh='native Farassat1A on complete rigid wall; pressure-only loading, body-out normals; facewise retarded arrival on solver receiver clock; no viscous/exterior quadrupole volume terms',
            pressure='native observer outputs already Pa, no extra rho conversion',force='N, fluid traction ON body; total, pressure and viscous exported separately',
            time_csv='absolute CFD clock; pilot raw times +0; main acousticData raw relative times +0.1s',
            restart_s=.1,restart_history='native acoustic derivatives/history restart at 0.1s; both original segments retained',
            suggested_warmup_exclusions_s=[[0,.01],[.1,.11]],
            finite_tail='receiver evaluations stop with solver; last emitted source segment has not fully arrived, no fabricated tail',
            range='1m and 2m are not radiation-far-field at low shedding frequencies in water; near terms retained',
            mpi_equivalence='not established for this case; actual 4-rank demo coupling, not exact MPI validation'),
        tables=tables,wall=wall,flow=flow,verification=checks,
        native_budget=dict(limit_wall_seconds=1800.,used_wall_seconds=elapsed,remaining_wall_seconds=1800.-elapsed,maximum_mpi_ranks=4,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,accounting='all native commands including failures, process termination and checks; pure Python exports/tests not native runtime'),
        commands=commands,files={'forces':'forces.csv','observers':'observers.csv','wall_geometry':'wall_geometry.csv','wall_pressure':'wall_pressure.npz','flow_slices':'flow_slices.csv','flow_vorticity':'flow_vorticity.png'},sha256=file_hashes,
        observed_claims=['actual 3D native WALE execution','native checkMesh allTopology/allGeometry Mesh OK','both Curle and Farassat1A function objects sampled during solver writes','real closed end caps with fluid beyond both tips'],
        unsupported_claims=['mesh/time convergence','resolved turbulent LES adequacy','statistical stationarity','experimental acoustic validation','exact serial/MPI acoustic equivalence','complete Lighthill volume quadrupoles','far-field-only radiation at 1m'])


def export_flow(case,out):
    """Export the native interpolated cutting-plane values, not generated vortices."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.colors import SymLogNorm
    case,out=Path(case),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    files=sorted((case/'postProcessing/flowSlice').glob('*/U_*.raw'),key=lambda p:(float(p.parent.name),p.name))
    rows=[]
    latest={}
    for p in files:
        plane=p.stem.removeprefix('U_')
        t=float(p.parent.name)
        u=np.loadtxt(p,ndmin=2)
        w=np.loadtxt(p.with_name('vorticity_'+plane+'.raw'),ndmin=2)
        if u.shape!=w.shape or u.shape[1]!=6 or not np.isfinite(u).all() or not np.isfinite(w).all() or not np.allclose(u[:,:3],w[:,:3],rtol=0,atol=1e-10):
            raise ValueError('native velocity/vorticity slices are not finite and aligned')
        for a,b in zip(u,w):
            rows.append([t,plane,*a,*b[3:]])
        latest[plane]=(t,u,w)
    with (out/'flow_slices.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['time_s','plane','x_m','y_m','z_m','U_x_m_s','U_y_m_s','U_z_m_s','vorticity_x_s-1','vorticity_y_s-1','vorticity_z_s-1'])
        writer.writerows(rows)
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    vmax=max(np.max(np.abs(w[:,5])) for t,u,w in latest.values())
    vmax=max(vmax,1e-12)
    for ax,plane in zip(axes,['midspan','tipplane']):
        t,u,w=latest[plane]
        tri=mtri.Triangulation(u[:,0],u[:,1])
        xy=np.column_stack([u[:,0],u[:,1]])[tri.triangles].mean(axis=1)
        if plane=='midspan':
            tri.set_mask(np.linalg.norm(xy,axis=1)<.0101)
        image=ax.tripcolor(tri,w[:,5],shading='gouraud',cmap='RdBu_r',norm=SymLogNorm(linthresh=1,vmin=-vmax,vmax=vmax))
        if plane=='midspan':
            ax.add_patch(plt.Circle((0,0),.01,color='#222222'))
        ax.set(xlim=(-.035,.15),ylim=(-.06,.06),aspect='equal',xlabel='streamwise x [m]',ylabel='cross-flow y [m]',title=f'{plane}: z={u[0,2]:g} m, t={t:g} s')
        fig.colorbar(image,ax=ax,label='spanwise vorticity [1/s], symlog colour')
    fig.suptitle('Actual 3D WALE cylinder demonstration — native flow slices')
    fig.savefig(out/'flow_vorticity.png',dpi=160)
    plt.close(fig)
    return dict(rows=len(rows),snapshots=len(set(r[0] for r in rows)),last_snapshot_s=max(r[0] for r in rows),
                latest_max_abs_spanwise_velocity_m_s=float(max(np.abs(u[:,5]).max() for t,u,w in latest.values())),
                latest_max_abs_spanwise_vorticity_s_inv=float(vmax),
                colour_scale='symmetric logarithmic, linear within +/-1 per second; no clipping',
                origin='OpenFOAM native cuttingPlane cellPoint interpolation of U and vorticity',
                chart='piecewise linear triangulation of native plane points; rigid midspan disk masked')


def export_wall_history(case, out, geometry):
    """A bijection per native raw sample, including both actual cap pressure fields."""
    from scipy.spatial import cKDTree
    case,out=Path(case),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    files=sorted((case/'postProcessing/wallHistory').glob('*/p_body.raw'),key=lambda p:float(p.parent.name))
    centres=geometry['centres']
    tree=cKDTree(centres)
    pressures,times,errors=[],[],[]
    for path in files:
        a=np.loadtxt(path,comments='#',ndmin=2)
        if a.shape!=(len(centres),4) or not np.isfinite(a).all():
            raise ValueError('incomplete/nonfinite wall pressure snapshot')
        distance,index=tree.query(a[:,:3])
        if distance.max()>1e-8 or len(np.unique(index))!=len(centres):
            raise ValueError('raw wall face geometry is not a bijection')
        p=np.empty(len(centres))
        p[index]=a[:,3]*1000.
        pressures.append(p)
        times.append(float(path.parent.name))
        errors.append(float(distance.max()))
    np.savez_compressed(out/'wall_pressure.npz',time_s=np.array(times),pressure_Pa=np.array(pressures),
                        centres_m=centres,area_vectors_body_out_m2=geometry['area_vectors_body_out'])
    table=np.column_stack([np.arange(len(centres)),geometry['global_face_ids'],centres,geometry['area_vectors_body_out']])
    np.savetxt(out/'wall_geometry.csv',table,delimiter=',',header='face_id,global_face_id,x_m,y_m,z_m,area_x_m2,area_y_m2,area_z_m2',comments='',fmt='%.12g')
    return dict(snapshots=len(times),faces_per_snapshot=len(centres),max_mapping_error_m=max(errors),
                start_s=min(times),end_s=max(times),sample_cadence='every 20 solver steps; adaptive times retained',
                pressure_unit='Pa; native kinematic pressure multiplied by rho=1000 exactly once',
                normal='outward from rigid body into fluid; negative of native boundary Sf')


def export_tables(case, out, segments):
    """Export unchanged native force values and explicit acoustic clock offsets."""
    case,out = Path(case),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    source = sorted((case/'postProcessing/forces').glob('*/force.dat'), key=lambda p:float(p.parent.name))
    forces = np.concatenate([read_forces(p) for p in source])
    if np.any(np.diff(forces[:,0]) <= 0):
        raise ValueError('overlapping/reversed force segments')
    names = ['time_s']+[prefix+axis+'_N' for prefix in ['force_','pressure_force_','viscous_force_'] for axis in 'xyz']
    np.savetxt(out/'forces.csv',forces,delimiter=',',header=','.join(names),comments='',fmt='%.12g')
    rows = []
    for path,method,offset in segments:
        rows.extend(read_observers(path,method,offset))
    rows.sort(key=lambda r:(r['time_s'],r['method'],r['observer']))
    keys=[(r['time_s'],r['method'],r['observer']) for r in rows]
    if len(set(keys))!=len(keys):
        raise ValueError('duplicate observer sample')
    with (out/'observers.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['time_s','observer','method','pressure_Pa'])
        writer.writeheader()
        writer.writerows(rows)
    return dict(force_samples=len(forces),observer_rows=len(rows),
                force_start_s=float(forces[0,0]),force_end_s=float(forces[-1,0]),
                observer_start_s=min(r['time_s'] for r in rows),observer_end_s=max(r['time_s'] for r in rows),
                dt_min_s=float(np.diff(forces[:,0]).min()),dt_max_s=float(np.diff(forces[:,0]).max()))


def wall_geometry(mesh_dir):
    """Extract every cylinder face; OpenFOAM fluid-out normals point into body."""
    mesh_dir = Path(mesh_dir)
    points = np.array([list(map(float,m.split())) for m in re.findall(r'^\(([^()]+)\)$', (mesh_dir/'points').read_text(), re.M)])
    all_faces = [list(map(int,m.split())) for m in re.findall(r'^\d+\(([^()]+)\)$', (mesh_dir/'faces').read_text(), re.M)]
    patch = re.search(r'\bcylinder\s*\{([^}]+)\}', (mesh_dir/'boundary').read_text()).group(1)
    start = int(re.search(r'startFace\s+(\d+)',patch).group(1))
    count = int(re.search(r'nFaces\s+(\d+)',patch).group(1))
    faces = all_faces[start:start+count]
    centres, areas = [], []
    edges, directed = Counter(), Counter()
    for face in faces:
        v = points[face]
        average = v.mean(axis=0)
        cross = np.cross(v-average, np.roll(v,-1,axis=0)-average)/2
        weights = np.linalg.norm(cross,axis=1)
        centres.append(np.sum(((v+np.roll(v,-1,axis=0)+average)/3)*weights[:,None],axis=0)/weights.sum())
        areas.append(-cross.sum(axis=0))
        for a,b in zip(face,face[1:]+face[:1]):
            edges[tuple(sorted((a,b)))] += 1
            directed[a,b] += 1
    closed = len(faces)==count and all(n==2 for n in edges.values()) and all(directed[a,b]==directed[b,a] for a,b in directed)
    if not closed:
        raise ValueError('cylinder surface has open/nonmanifold/inconsistently oriented edges')
    centres, areas = np.array(centres), np.array(areas)
    audit = dict(closed=True, faces=count, unique_edges=len(edges),
                 cap_minus_faces=int(np.isclose(centres[:,2],-.03,atol=1e-10).sum()),
                 cap_plus_faces=int(np.isclose(centres[:,2],.03,atol=1e-10).sum()),
                 area_m2=float(np.linalg.norm(areas,axis=1).sum()),
                 area_vector_sum_m2=areas.sum(axis=0).tolist(),
                 minimum_body_out_dot_position=float(np.sum(centres*areas,axis=1).min()))
    return dict(centres=centres, area_vectors_body_out=areas, audit=audit,
                points=points, faces=faces, global_face_ids=np.arange(start,start+count))


def read_forces(path):
    """OpenFOAM forces output is already dimensional N (never multiply by rho)."""
    a = np.loadtxt(path, comments='#', ndmin=2)
    if a.shape[1] != 10 or not np.isfinite(a).all():
        raise ValueError('expected finite time, total, pressure, viscous force columns')
    if np.any(np.diff(a[:,0]) <= 0):
        raise ValueError('strictly increasing force time required')
    return a


def read_observers(path, method, offset=0.):
    """Preserve native observer header ordering and explicitly restore global time."""
    path = Path(path)
    names = path.read_text().splitlines()[0].split()[1:]
    a = np.loadtxt(path, skiprows=1, ndmin=2)
    if a.shape[1] != len(names)+1 or not np.isfinite(a).all():
        raise ValueError('expected finite observer columns')
    if np.any(np.diff(a[:,0]) <= 0):
        raise ValueError('strictly increasing observer time required')
    return [dict(time_s=float(row[0]+offset), observer=name.removesuffix('_pFluct'),
                 method=method, pressure_Pa=float(row[j+1]))
            for row in a for j,name in enumerate(names)]


def run_command(argv, label, deadline, *, ledger=None, budget=1800, native_env=True):
    """Exclusive cumulative accounting, including failed/terminated native commands."""
    if native_env and (FOAM is None or not (FOAM / 'etc/bashrc').is_file()):
        raise RuntimeError('Source OpenFOAM v2206 first: WM_PROJECT_DIR must identify its installation')
    ledger = Path(ledger) if ledger else RESULTS/'commands.json'
    ledger.parent.mkdir(parents=True, exist_ok=True)
    lock = ROOT/'.cache/cylinder_native.lock' if native_env else ledger.with_suffix('.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = json.loads(ledger.read_text()) if ledger.exists() else []
        remaining = budget - sum(r['elapsed_seconds'] for r in records)
        if remaining <= 1:
            raise ValueError('native budget exhausted')
        deadline = min(deadline, remaining - 1)
        if 'mpirun' in argv and ('-np' not in argv or int(argv[argv.index('-np')+1]) > 4):
            raise ValueError('maximum 4 MPI ranks')
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        actual = argv
        if native_env:
            assert FOAM is not None
            actual = ['bash','-c', f'source {shlex.quote(str(FOAM / "etc/bashrc"))} >/dev/null 2>&1 && export LD_LIBRARY_PATH={shlex.quote(str(LIB.parent))}:$LD_LIBRARY_PATH && exec '+shlex.join(argv)]
        log = ledger.parent/f'{len(records):03}_{label}.log'
        start = time.monotonic()
        timed_out = False
        with log.open('w') as stream:
            p = subprocess.Popen(actual, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    rc = p.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    rc = p.wait()
        record = dict(argv=argv, label=label, exit_code=rc, timed_out=timed_out,
                      elapsed_seconds=time.monotonic()-start, deadline_seconds=deadline,
                      log=str(log.relative_to(ROOT)) if log.is_relative_to(ROOT) else str(log),
                      log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
        records.append(record)
        ledger.write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        return record

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'cfd/cylinder_demo'
RESULTS = ROOT / 'results/cylinder/native'
FOAM = Path(os.environ['WM_PROJECT_DIR']) if os.environ.get('WM_PROJECT_DIR') else None
LIB = ROOT / '.build/acoustics/platforms/linux64GccDPInt32Opt/lib/libAcoustics.so'


def mesh_description():
    """Conformal O-grid with true disc cores in the fluid beyond both tips."""
    zs = [-0.08, -0.03, 0.03, 0.08]
    vertices, blocks, arcs = [], [], []
    patches = {k: [] for k in ['cylinder', 'inlet', 'outlet', 'sides', 'spanEnds']}
    rings = {}
    for k, z in enumerate(zs):
        for name in ['circle', 'outer', 'core']:
            ring = []
            for i in range(4):
                theta = math.pi / 4 + i * math.pi / 2
                x, y = 0.01 * math.cos(theta), 0.01 * math.sin(theta)
                if name == 'outer':
                    x, y = [(0.24, .1), (-.1, .1), (-.1, -.1), (.24, -.1)][i]
                elif name == 'core':
                    x, y = x / 2, y / 2
                ring.append(len(vertices))
                vertices.append((x, y, z))
            rings[k, name] = ring
        for i in range(4):
            theta = math.pi / 2 + i * math.pi / 2
            arcs.append((rings[k, 'circle'][i], rings[k, 'circle'][(i+1)%4],
                         (.01*math.cos(theta), .01*math.sin(theta), z)))
    for k, nz in enumerate([12, 36, 12]):
        for name_in, name_out, nr, grade in [('circle', 'outer', 32, 16)]:
            for i in range(4):
                j = (i+1)%4
                a, b = rings[k, name_in], rings[k, name_out]
                c, d = rings[k+1, name_in], rings[k+1, name_out]
                v = [a[i], b[i], b[j], a[j], c[i], d[i], d[j], c[j]]
                blocks.append(dict(vertices=v, counts=[nr,16,nz], grading=[grade,1,1]))
                patches[['sides','inlet','sides','outlet'][i]].append([v[t] for t in [1,2,6,5]])
                if k == 1:
                    patches['cylinder'].append([v[t] for t in [0,4,7,3]])
                if k == 0:
                    patches['spanEnds'].append([v[t] for t in [0,3,2,1]])
                if k == 2:
                    patches['spanEnds'].append([v[t] for t in [4,5,6,7]])
        if k == 1:
            continue
        for i in range(4):
            j = (i+1)%4
            a,b,c,d = rings[k,'core'],rings[k,'circle'],rings[k+1,'core'],rings[k+1,'circle']
            v = [a[i],b[i],b[j],a[j],c[i],d[i],d[j],c[j]]
            blocks.append(dict(vertices=v, counts=[6,16,nz], grading=[1,1,1]))
            patches['cylinder'].append([v[t] for t in ([4,5,6,7] if k==0 else [0,3,2,1])])
            patches['spanEnds'].append([v[t] for t in ([0,3,2,1] if k==0 else [4,5,6,7])])
        v = [rings[k,'core'][i] for i in [2,3,0,1]] + [rings[k+1,'core'][i] for i in [2,3,0,1]]
        blocks.append(dict(vertices=v, counts=[16,16,nz], grading=[1,1,1]))
        patches['cylinder'].append([v[t] for t in ([4,5,6,7] if k==0 else [0,3,2,1])])
        patches['spanEnds'].append([v[t] for t in ([0,3,2,1] if k==0 else [4,5,6,7])])
    return dict(vertices=vertices, blocks=blocks, arcs=arcs, patches=patches,
                cells=sum(math.prod(b['counts']) for b in blocks))


def block_mesh_text(m):
    def vec(v):
        return '(' + ' '.join(f'{x:.12g}' for x in v) + ')'
    out = ['FoamFile {version 2.0; format ascii; class dictionary; object blockMeshDict;}',
           'scale 1;', 'vertices (']
    out += [vec(v) for v in m['vertices']] + [');', 'blocks (']
    out += ['hex '+vec(b['vertices'])+' '+vec(b['counts'])+' simpleGrading '+vec(b['grading']) for b in m['blocks']]
    out += [');', 'edges ('] + [f'arc {a} {b} {vec(c)}' for a,b,c in m['arcs']] + [');', 'boundary (']
    for name, faces in m['patches'].items():
        kind = 'wall' if name=='cylinder' else ('symmetryPlane' if name in ['sides','spanEnds'] else 'patch')
        # The two opposite side planes must be distinct patches for symmetryPlane.
        kind = 'symmetry' if name in ['sides','spanEnds'] else kind
        out += [f'{name} {{type {kind}; faces ('] + [vec(f) for f in faces] + [');}']
    return '\n'.join(out + [');','mergePatchPairs ();',''])


OBSERVERS = {'cross_plus': [0.,1.,0.], 'cross_minus':[0.,-1.,0.],
             'downstream':[1.,0.,0.], 'oblique':[.6,.8,0.], 'cross_double':[0.,2.,0.]}


def write_case(case=CASE):
    """Write the executed single-phase WALE and native acoustic dictionaries."""
    case = Path(case)
    def put(name, body, cls='dictionary'):
        p = case/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'FoamFile {{version 2.0; format ascii; class {cls}; object {p.name};}}\n'+body+'\n')
    (case/'system').mkdir(parents=True, exist_ok=True)
    (case/'system/blockMeshDict').write_text(block_mesh_text(mesh_description()))
    common = '''libs ("libAcoustics.so"); patches (cylinder); p p; U U;
        rho rhoInf; rhoInf 1000; CofR (0 0 0); log false;
        timeStart 0; timeEnd 100; probeFrequency 1; writeFft false; c0 1500;
        executeControl timeStep; executeInterval 1;
        writeControl timeStep; writeInterval 1;
        observers {'''
    common += '\n'.join(f'{k} {{position ({v[0]} {v[1]} {v[2]}); pRef 1e-6; fftFreq 1024;}}' for k,v in OBSERVERS.items()) + '}'
    put('system/controlDict', '''application pimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime 0.1;
deltaT 0.001; adjustTimeStep yes; maxCo 0.7; maxDeltaT 0.003;
writeControl adjustableRunTime; writeInterval 0.1; purgeWrite 3;
writeFormat ascii; writePrecision 10; writeCompression off;
timeFormat general; timePrecision 10; runTimeModifiable true;
functions {
forces {type forces; libs ("libforces.so"); patches (cylinder); rho rhoInf; rhoInf 1000; CofR (0 0 0); log false; writeControl timeStep; writeInterval 1;}
curle {type CurleAnalogy; '''+common+'''}
fwh {type FfowcsWilliamsHawkings; '''+common+'''
formulationType Farassat1AFormulation; pInf 0; Ufwh (0 0 0); U0 (0 0 0);
fixedResponseDelay false; nonUniformSurfaceMotion false; cleanFreq 20;
interpolationScheme cell;
surfaces (body {type patch; patches (cylinder); interpolate false; triangulate false;});
}
vorticity {type vorticity; libs ("libfieldFunctionObjects.so"); executeControl writeTime; writeControl writeTime;}
wallHistory {type surfaces; libs ("libsampling.so"); writeControl timeStep; writeInterval 20;
surfaceFormat raw; fields (p); interpolationScheme cell;
surfaces (body {type patch; patches (cylinder); interpolate false; triangulate false;});
}
flowSlice {type surfaces; libs ("libsampling.so"); writeControl writeTime;
surfaceFormat raw; fields (U vorticity); interpolationScheme cellPoint;
surfaces (
midspan {type cuttingPlane; planeType pointAndNormal; pointAndNormalDict {point (0 0 0); normal (0 0 1);} interpolate true;}
tipplane {type cuttingPlane; planeType pointAndNormal; pointAndNormalDict {point (0 0 0.035); normal (0 0 1);} interpolate true;}
);
}
}''')
    put('system/fvSchemes', '''ddtSchemes {default backward;}
gradSchemes {default Gauss linear;}
divSchemes {default none; div(phi,U) Gauss linearUpwind grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear;}
laplacianSchemes {default Gauss linear corrected;}
interpolationSchemes {default linear;}
snGradSchemes {default corrected;}
wallDist {method meshWave;}''')
    put('system/fvSolution', '''solvers {
p {solver GAMG; tolerance 1e-7; relTol 0.05; smoother GaussSeidel;}
pFinal {$p; relTol 0;}
U {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.05;}
UFinal {$U; relTol 0;}
}
PIMPLE {nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0;}
relaxationFactors {equations {U 1;}}''')
    put('system/decomposeParDict', 'numberOfSubdomains 4; method scotch;')
    put('constant/transportProperties', 'transportModel Newtonian; nu [0 2 -1 0 0 0 0] 1e-6;')
    put('constant/turbulenceProperties', '''simulationType LES;
LES {LESModel WALE; turbulence on; printCoeffs on; delta cubeRootVol; WALECoeffs {Cw 0.325;}}''')
    put('0/U', '''dimensions [0 1 -1 0 0 0 0]; internalField uniform (0.2 0.002 0.001);
boundaryField {
inlet {type fixedValue; value uniform (0.2 0 0);}
outlet {type inletOutlet; inletValue uniform (0.2 0 0); value uniform (0.2 0 0);}
cylinder {type noSlip;}
sides {type symmetry;} spanEnds {type symmetry;}
}''', 'volVectorField')
    put('0/p', '''dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;
boundaryField {
inlet {type zeroGradient;} outlet {type fixedValue; value uniform 0;}
cylinder {type zeroGradient;} sides {type symmetry;} spanEnds {type symmetry;}
}''', 'volScalarField')
    put('0/nut', '''dimensions [0 2 -1 0 0 0 0]; internalField uniform 0;
boundaryField {
inlet {type calculated; value uniform 0;} outlet {type calculated; value uniform 0;}
cylinder {type fixedValue; value uniform 0;} sides {type symmetry;} spanEnds {type symmetry;}
}''', 'volScalarField')
