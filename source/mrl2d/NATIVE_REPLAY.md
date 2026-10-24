# Native prerequisites and deliberately unexecuted commands

## Scope and environment selection

This private source candidate has completed local, non-native checks. The two
copied runtime helpers now require an explicit existing `MRL2D_FOAM_BASHRC`;
neither falls back to a machine-local installation. Mesh validation happens
before log-directory creation. Both shell setups stop with exit 121 if sourcing
fails. Quoting and thread limits were checked using labelled harmless shell
fixtures, including spaces and metacharacters in the bashrc path. This does not
verify that an arbitrary existing bashrc actually belongs to v2206.

Only `mrl2d_case.native_command` and `mrl2d_mesh._native_stage` changed in copied
runtime modules. Original sources, simulations and historical pins are unchanged.
`SOURCE_IDENTITY.json` distinguishes original hashes, adapted hashes and the new
preparation-only wrapper. No native execution, build or publication was performed.

## External tools

- Linux (the lifecycle supervisor uses subreaper/process APIs and owner locks).
- OpenCFD **OpenFOAM v2206**, working bashrc, compiler/wmake, solver headers and
  `applications/solvers/incompressible/pimpleFoam`, dynamic mesh libraries, MPI.
- An explicit `MRL2D_FOAM_BASHRC`; normal `WM_PROJECT_DIR`, `FOAM_USER_LIBBIN`,
  `LD_LIBRARY_PATH` etc. are supplied by that OpenFOAM installation. Native
  recorder/guard sources are bundled, installed libraries and executables are not.
- Gmsh Python bindings and system libraries for mesh generation. Gmsh is not in
  the retained Python lock; the precise original Gmsh build is not pinned here.
- VTK Python bindings plus NumPy/SciPy for actual plane readback. If the active
  interpreter lacks VTK, the unchanged completion code explicitly delegates to
  `/usr/bin/python3`, which must have compatible dependencies. No VTK readback
  of completed CFD is claimed by the source unit checks.
- Adequate disk, wall time and explicitly authorized cumulative native budget.
  Unit-check resource limits do not establish native solver feasibility.

## Mesh identity is not a default-generator claim

`build_mrl2d_mesh.py` uses the original fine generator defaults. The executed
26,751-cell candidate instead used the unchanged
`mrl2d_runtime.candidate_mesh_source()` transformation:

| Literal role | Original | Executed candidate |
|---|---:|---:|
| Mesh.MeshSizeMin | 0.00006 | 0.0005 |
| wall SizeMin | 0.0004 | 0.0008 |
| wall DistMax | 0.20 | 0.10 |
| interface DistMax | 0.25 | 0.10 |

The original candidate source is pinned at
`cfd8737377c69c0b63f7f0db4f7fb7150c7173191f155c80d706a37c2d7112c1`.
Applying the unchanged transformation to this portable copy yields
`d1ce42c648267ce1015a8ad7880335195144fa6d92735bdf0ef18ccd306a4646`.
AST and source-segment comparisons prove every candidate definition is identical
except `_native_stage`; every byte outside that helper is also identical to the
retained historical candidate. The public file must **not** claim the historical
whole-file hash.

### Tested preparation-only route

```sh
PYTHON=${PYTHON:-.venv/bin/python}
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p .verification
"$PYTHON" scripts/build_production_mesh.py \
  --emit-generator .verification/production_mesh.py
```

The wrapper creates only a new source file in an existing directory. It refuses
existing targets, symlinks and `..` components. It has **no native execution
mode**. Gmsh is not imported and no mesh, motion, CFD case or admission receipt
is created. Repeating the emission at a different fresh path yields identical
source bytes. The inherited generic `build_mrl2d_mesh.py` CLI remains unchanged;
its default resolution is different and it must not be used as this policy.

### Conditional native route — documented, never executed here

The following is a future build/motion recipe, **not part of source-package
verification**. Obtain separate authorization and an exclusive native-work
window first. Use an installed Gmsh module, a verified v2206 environment and
fresh case/ledger paths. Account for earlier charged attempts when selecting
`NATIVE_SECONDS`; `MESH_WALL_SECONDS` is the separately authorized outer wall
cap (including meshing). Do not use a shared live production case.

```sh
: "${MRL2D_ALLOW_NATIVE:?explicitly set to yes after separate authorization}"
[ "$MRL2D_ALLOW_NATIVE" = yes ] || exit 2
: "${MRL2D_FOAM_BASHRC:?set the verified v2206 bashrc}"
: "${MESH_CASE:?set a fresh absolute mesh-case path}"
: "${MESH_LEDGER:?set a fresh absolute ledger path in an existing parent}"
: "${NATIVE_SECONDS:?set the authorized cumulative native budget}"
: "${MESH_WALL_SECONDS:?set the authorized whole-build wall cap}"
source "$MRL2D_FOAM_BASHRC" || exit 121
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MESH_CASE MESH_LEDGER NATIVE_SECONDS
# PYTHON, PYTHONPATH and emitted generator are set by the preparation route.
timeout --signal=TERM --kill-after=10 "$MESH_WALL_SECONDS" "$PYTHON" - <<'PY'
import hashlib
import json
import math
import os
import runpy
from pathlib import Path
from flowacoustics.mrl2d_runtime import NativeBudget

source = Path('.verification/production_mesh.py').resolve()
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    'd1ce42c648267ce1015a8ad7880335195144fa6d92735bdf0ef18ccd306a4646'
)
case = Path(os.environ['MESH_CASE'])
ledger = Path(os.environ['MESH_LEDGER'])
limit = float(os.environ['NATIVE_SECONDS'])
assert case.is_absolute() and ledger.is_absolute()
assert not case.exists() and not case.with_name(case.name + '-motion').exists()
assert not ledger.exists() and ledger.parent.is_dir()
assert math.isfinite(limit) and limit > 5
module = runpy.run_path(str(source))
budget = NativeBudget(ledger, limit=limit)

def tracked_stage(case, name, argv, stages):
    record = budget.run([*argv, '-case', str(case)], cwd=case,
                        log=case / 'logs' / f'{name}.log', timeout=60)
    stages.append(record)
    if (record['returncode'] or record['timed_out']
            or record.get('monitor_error') or not record.get('cleanup_confirmed')):
        raise RuntimeError(str(record))
    return Path(record['log']).read_text()

# Use the case supervisor's genuine budgeted stage receipts, as in the retained
# candidate orchestration, not the generic mesh helper's older receipt schema.
module['verify_native'].__globals__['_native_stage'] = tracked_stage
module['build_base'](case)
report = module['verify_native'](case)
assert report['admitted'] is True
print(json.dumps({'admitted': report['admitted'], 'cells': report['cells'],
                  'generator_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}))
PY
```

Retain the emitted generator, complete native logs, ledger, base/motion cases,
`mesh_reference.npz`, all AMI arrays and `mesh_manifest.json`. Native quality
and 0.6 s motion admission must pass unrelaxed. The tracked-stage recipe uses
supervised `native_command` receipts required by the existing flow runner;
the generic mesh helper's `return_code` receipts are not that same schema.
This recipe was not run or integration-validated here. Failed/time-limited
attempts remain charged and are not converted into admission.

This is exact **source-policy** provenance, not a recreated 26,751-cell mesh.
The exact historical Gmsh binary is not pinned; newly generated connectivity
and counts must be measured rather than assumed, even with identical policy.

## Fresh flow rerun, not replay of saved source histories

The following recipe requires an external authentic admitted mesh and matching
motion evidence. It is documented, **not exercised** here. The copied initial
input examples have no mesh and cannot replace those prerequisites. Use a new
output root; the runner refuses attempted cases rather than resuming them.

```sh
: "${MRL2D_FOAM_BASHRC:?set the installed v2206 bashrc}"
: "${MESH_CASE:?set an authentic admitted mesh case}"
: "${MOTION_EVIDENCE:?set its genuine matching motion manifest}"
: "${WALL_SECONDS:?set an authorized finite wall budget}"
: "${NATIVE_SECONDS:?set an authorized cumulative native budget}"
source "$MRL2D_FOAM_BASHRC" || exit 121
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=${PYTHON:-.venv/bin/python}
"$PYTHON" scripts/run_mrl2d.py prepare --mesh-case "$MESH_CASE"   --output-root "$PWD/cfd" --depth 0.01 --dt 0.0000625 --end 10 --ranks 2
"$PYTHON" scripts/run_mrl2d.py run-many --output-root "$PWD/cfd"   --motion-evidence "$MOTION_EVIDENCE" --wall-seconds "$WALL_SECONDS"   --native-seconds "$NATIVE_SECONDS" --carry-forward 0 --workers 1   --min-free-gib 20 --work-root "$PWD/.cache/mrl2d/execution-001"
```

`--carry-forward 0` is only for a genuinely new independent budget; include
previously charged attempts when continuing an authorized campaign. Four cases
are processed by default; `--workers 1` avoids assuming an eight-core allocation.
The runner builds its recorder/instrumented solver in the work root, freezes
its actual inputs, logs every stage and checks native completion. Success of a
wrapper, unit suite or mesh build is not accepted completion of the CFD/acoustics.

## Offline replay

Offline replay reruns the reduction on the **same saved original native inputs**;
it is not a new CFD realization. Keep original reports and their trusted digest
index, complete native source binaries, executed dictionaries/mesh, distinct
stage logs, animation planes and field-readback evidence available at compatible
original archive roots. Public reduced CSVs and initial dictionaries are not a
substitute. A relocation adapter would need its own explicit contract; never
rewrite original reports or historical pins merely to make path checks pass.

Use the README export command only with those real inputs and a trusted
`mrl2d-postprocess-input-v1` specification. The completion adapter treats native
contours/slices as a direct pathway with `reconstruction: not_required`, never
as reconstructed or validated by an invented log. Production export requires
all four complete original 10 s cases and the unchanged completion/identity,
clock, AMI, field and source gates.

Optional historical integration test archives may be supplied under
`native-archives/diagnostic-urans`, `diagnostic-wale`, `mesh` or `bounded`.
These names are public-local handles, not included archives or download claims.
Historical 0.1 s diagnostics are not the completed four-case 10 s study.
No optional absence is filled with synthetic native evidence.
