# Four-case strict-2D MRL demonstration — private source candidate

**LOCAL PORTABILITY CHECKS COMPLETE; PRIVATE, NOT PUBLISHED.** Numerical
algorithms, geometry and physical parameters are unchanged. Only the copied
`native_command` and `_native_stage` runtime setup helpers were adapted to an
explicit fail-closed environment locator. Whole-file adapted hashes are declared
separately from historical originals. No additional reuse permission is granted;
existing GPL notices remain in force. See [LICENSES/README.md](LICENSES/README.md).
Publication remains a separate owner decision, not an implied result of testing.

## What this contains

- Original Python acoustic kernel, native-history reader, completion gate, mesh
  mathematics, flow-case factory and lifecycle code; two declared runtime-only
  helper adaptations remove private machine locators.
- Three original CLI scripts plus a new **preparation-only** production-policy
  emitter, native recorder/guard source and Make instructions. No
  native binaries or generated mesh are bundled.
- 48 unchanged initial/configuration files from the four executed cases in
  `examples/initial-inputs/`. These are **not runnable CFD case archives**:
  `constant/polyMesh`, native histories, field volumes and reports are excluded.
- Original numerical tests, a safe explicit unit selection, and copy-isolation
  tests. Optional native-archive tests skip only when their archives are absent.
- Existing `pyproject.toml` and public-PyPI `uv.lock`, method/API docs, result
  schema, and the small schema-v2 frontend model required by a contract test.
  This is not a standalone webpage or a publication bundle.

`SOURCE_IDENTITY.json` pins unchanged, runtime-adapted and new executable sources. The companion
private audit directory records originals, adaptations, test commands and
manifest hashes; **do not bundle that audit directory or test scratch data**.
No original source, historical receipt, native case or release pin was edited.

## Study boundary

The separately retained native reports record four actual 10 s cases, each
26,751 cells, two MPI ranks, dt=0.0000625 s, 0.002 s source histories, 0.02 s
animation and 0.2 s volume retention:

| ID | Flow model | Acoustic surface |
|---|---|---|
| mrl2d_urans_body | SST URANS | moving body contours |
| mrl2d_urans_permeable | SST URANS | stationary permeable contour |
| mrl2d_les_body | 2D WALE / LES-model | moving body contours |
| mrl2d_les_permeable | 2D WALE / LES-model | stationary permeable contour |

This is educational strict 2D, **not resolved 3D turbulent LES**, a finite-turbine
absolute noise prediction, or a convergence/experimental validation campaign.
Native completion is separate from final independent post-run acoustic
acceptance and webpage publication, which remain parent-owned/pending here.

The acoustic method uses the true 2D moving-contour Green function, not a 3D
stationary loading formula. The configured analysis excludes t<=2 s, keeps
2.002..10 s, uses a coherent-gain-normalized periodic Hann window and the
0.625..25 Hz positive-bin band (0.125 Hz spacing). This is an analysis choice,
not a demonstrated CFD bandwidth or stationarity claim. Synthesis is periodic,
window-conditioned and band-limited, not a causal first-arrival waveform.
Pressure is Pa; contour loading is N/m. Body/permeable estimates overlap: never
sum them or call their difference an isolated quadrupole. Omitted viscous
traction, exterior volume sources, acoustic mean-flow propagation and finite-
span physics remain limitations. See [METHOD](docs/mrl2d/acoustics/METHOD.md)
and [API](docs/mrl2d/acoustics/API.md).

## Python setup and unit verification (no CFD)

Run from this directory. Python **3.12** is required. The unchanged lock includes
NumPy, SciPy, Matplotlib, pandas and development dependencies. pandas/websockets
are retained for lock compatibility, although the selected numerical path does
not import them. Node.js is required for one real frontend-schema test. No
OpenFOAM, Gmsh or native archive is needed by the safe default unit suite.

With uv available, install the declared lock in a fresh checkout:

```sh
uv sync --locked --extra dev
```

Dependency installation itself was not replayed for this candidate: verification
used an existing Python 3.12 environment with versions matching `uv.lock` for
NumPy/SciPy/Matplotlib/pytest/Ruff. No copied virtualenv is supplied.

Use the installed Python, or set `PYTHON` to a Python 3.12 environment with those
dependencies. All imports, plotting caches and test writes stay copy-local:

```sh
PYTHON=${PYTHON:-.venv/bin/python}
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$PWD/.verification/pycache"
export MPLBACKEND=Agg MPLCONFIGDIR="$PWD/.verification/mpl"
export TMPDIR="$PWD/.verification/tmp"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MRL2D_NATIVE_TEST=0 MRL2D_PRODUCTION_NATIVE=0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
mkdir -p "$TMPDIR"

# Linux address-space limit: 4 GiB. Use a NEW --basetemp on each retained replay.
prlimit --as=4294967296 -- "$PYTHON" -m pytest -c "$PWD/pytest.ini"   --basetemp="$PWD/.verification/unit-001" -q
prlimit --as=4294967296 -- "$PYTHON" -m pytest -c "$PWD/pytest.ini"   @tests/unit-contract-nodes.txt --basetemp="$PWD/.verification/contracts-001" -q
prlimit --as=4294967296 -- "$PYTHON" -m pytest -c "$PWD/pytest.ini"   tests/test_public_portability.py tests/test_production_mesh_recipe.py   --basetemp="$PWD/.verification/portability-001" -q
```

The explicit config prevents accidental parent-repository import resolution.
Collection and session-end checks assert imported module roots and source hashes.
The unit fixture forbids native launch helpers and native-owner locks. Tests are
synthetic/analytical unless explicitly marked optional archive integration.
They neither reconstruct fields nor advance a solver. Do not replace these
commands with `pytest tests/`: retained native/lifecycle suites are intentionally
outside this bounded verification and some historical tests launch commands
without an opt-in flag. They are not silently declared passed.

The safe suite exercises all three original CLI `--help` routes; the new suite
also exercises the preparation-only emitter, quoting/fail-closed shell fixtures
and output refusal. Across three disjoint suites, **173 passed and 7 optional
archive-absence checks skipped**. The safe suite additionally covers source-copy
output binding, signed analytic acoustic controls, truncation bounds, source
units, corruption rejection, and the exact initial dictionaries with explicit
accepted parameters. The extended selection includes a real complete synthetic
completion gate, not manufactured native evidence. A broader synthetic
completion-file run exceeded the 240 s cap and is explicitly **incomplete**.
See [verification](docs/mrl2d/acoustics/VERIFICATION.md) for final counts.

## Native replay is a separate operation

See [NATIVE_REPLAY.md](NATIVE_REPLAY.md). No CFD, mesh generation, native
postprocessing, solver-owner acquisition, build or native archive replay was
performed for this package. External native prerequisites are not converted
into fake receipts or replaced by passing unit fixtures. The generic mesh CLI's
default resolution is not the executed 26,751-cell candidate; use the tested
`build_production_mesh.py --emit-generator` route in NATIVE_REPLAY for its exact
four-literal policy. Policy identity does not guarantee recreated connectivity.

A completed external archive is needed for the actual offline export. It must
retain the original authenticated input/run reports, logs, contour binaries,
planes and required input dictionaries/mesh. A webpage's reduced CSVs alone are
not sufficient. The accepted completion adapter binds original report paths;
relocating native cases is not supported by casually rewriting those reports.
Do not rehash edited reports to call them the historical original.

The exporter only writes a **fresh** child of this copy's
`.cache/mrl2d/postprocessing/`. It never publishes the site or edits source cases:

```sh
"$PYTHON" scripts/postprocess_mrl2d.py --input "$TRUSTED_REPLAY_SPEC"   --output "$PWD/.cache/mrl2d/postprocessing/replay-001" --mode production   --panel-chunk 96 --memory-limit-bytes 2000000000
```

That command is conditional on real external evidence and VTK; it was not run
against the completed native histories by this worker. The input schema and
expected outputs are documented in
[postprocessing](docs/mrl2d/postprocessing/README.md).
