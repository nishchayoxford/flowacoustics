# Reproduce the refined 10 s demonstration

## Analysis replay — no CFD

From a checkout with the **refined** public downloads, Python 3.12 and uv:

```sh
uv sync --locked --extra dev
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/analyze_cylinder.py \
  --run site/downloads/run.json \
  --forces site/downloads/forces.csv \
  --observers site/downloads/observers.csv \
  --output results/reproduced \
  --start-time-s 2 --end-time-s 10 --nperseg 3200
```

Use a fresh output directory. Do not substitute the historical 6 s download bundle. The refined CSVs retain native uniform **0.000625 s** spacing: **no `--resample-dt-s` and no +0.1 s offset**. The fresh run started at zero without restarting libAcoustics. The 2–10 s window is a developing-flow selection, not a stationarity claim. Periodic Hann segments of 3,200 samples, 50% overlap and segment-mean removal give seven overlapping segments and 0.5 Hz bins; overlapping segments are not independent realizations.

The public `run.json` is a sanitized **schema-1 analysis view**, not the original schema-2 execution manifest. Its provenance must retain the original native SHA256 `6340ad86ade01d00752b884419b0e207d678e0253aaa4ab13ef9ff597a38bb2f`. The force/observer CSV bytes are unchanged. Metadata/input identities and therefore the reproduced manifest hash differ from the local original; compare physical chart arrays, statistics and aligned CSV values separately. The original execution record must never be rewritten to make public paths match.

The analysis emits full-resolution `results.json`, a hash manifest and downloadable CSVs. Public display traces may retain extrema or limit the visible frequency band; neither changes the complete analysis arrays. The native computation completed, but scientific analysis status remains `partial` and the primary lift peak is unresolved.

## Tests and viewing

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests -q -rs
python3 -m http.server 8513 --bind 127.0.0.1 --directory site
```

Browser tests need Chromium/Chrome and Node with built-in WebSocket support. ParaView's `pvpython` and Poppler's `pdfimages`/`pdftotext` are additional converter/figure prerequisites. Synthetic numerical, geometry, binary-field and pipeline fixtures do not require OpenFOAM binaries. Optional frozen historical analysis/report tests skip on archive absence; the native refined-mesh integration additionally requires `FLOWACOUSTICS_VERIFY_REFINED_MESH=1` and its original hash-bound archive. Those skipped checks are not native verification of a fresh clone. Never copy private-path manifests or fabricate native records just to remove skips.

## Native dependencies

The executed runtime was **OpenCFD OpenFOAM v2206**, double precision, 32-bit labels, GCC, eight MPI ranks, one thread per rank. Source your installed OpenFOAM `etc/bashrc`; `scripts/cylinder_native.py` locates it through `WM_PROJECT_DIR`, not an author-specific path. MPI and FFTW3 development libraries are separate dependencies, not supplied by Python.

libAcoustics numerical source is unchanged from commit `b9cd1d62ad8c5b445240a506a945231d12454b73`. Build a separate copy:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FOAM_USER_LIBBIN="$PWD/.build/acoustics/platforms/$WM_OPTIONS/lib"
mkdir -p "$FOAM_USER_LIBBIN" .build/acoustics/libAcoustics/source
cp -R third_party/libAcoustics-v2212/upstream/Sources .build/acoustics/libAcoustics/source/
(cd .build/acoustics/libAcoustics/source/Sources/lib && wmake libso)
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"
```

The supplied `Make/options` expects `FFTW_ARCH_PATH` from the installation. Helpers retain the tested `linux64GccDPInt32Opt` layout; other variants need explicit compatibility checks. Portability guards and fixture adaptations do not imply that another runtime is numerically equivalent.

## Native rerun — separate resource authorization required

The accepted dictionaries are in `cfd/cylinder_circular_60s/`. Despite the preserved name, `controlDict` starts at zero and ends at **10 s**, using **0.000625 s**, no adaptive stepping and eight-way decomposition. This is **not** the old pilot/continuation workflow. The generator's default timestep is not the executed setting: if generating another case, explicitly pass `end=10, dt=0.000625, ranks=8` and compare every dictionary to the shipped inputs.

The following is a manual sequence, **not performed by opening the site or running analysis tests**. Obtain new compute/storage permission and use a fresh case with no previous outputs. Save logs and check every stage before proceeding:

```sh
case=cfd/cylinder_circular_60s
blockMesh -case "$case" > log.blockMesh 2>&1
checkMesh -case "$case" -allTopology -allGeometry > log.checkMesh 2>&1
# STOP and inspect log.checkMesh: require 459264 cells, 'Mesh OK.', and no failed checks.
# Confirm eight ranks, fixed deltaT=0.000625, endTime=10 and available disk/RAM.
decomposePar -case "$case" > log.decomposePar 2>&1
mpirun --bind-to core -np 8 pimpleFoam -case "$case" -parallel > log.pimpleFoam 2>&1
# STOP unless exit status, End marker, all 16000 times, finiteness and Courant gates pass.
reconstructPar -case "$case" -noZero > log.reconstructPar 2>&1
# Verify p/U/nut/vorticity/yPlus at 2,4,6,8,10; retain processor directories.
```

`checkMesh` exit zero alone is insufficient. Do not continue a failed or short solve, silently relax quality gates or increase the cell ceiling/ranks/duration. Binary fields are **uncompressed**: this v2206 build disables binary compression. New native runs need their own hash-bound configuration, mesh/runtime identities, logs and export verification; bitwise equality across parallel installations is not promised.

`scripts/cylinder_longrun.py` is the recorded guarded orchestration code, **not a one-command fresh-checkout launcher**. Its CLI requires a locally prepared `results/cylinder/native_circular_60s/prepared.json`, including current essential-input identities, native mesh-check log and accepted pilot Courant evidence. That machine-bound preflight record is intentionally not shipped. Do not invent it or reuse the old machine's identity pins.

## Signature and raw-data reproduction boundary

The [nine-signature producer](SIGNATURE_ANALYSIS.md), native mesh renderer and raw-sample converter require original hash-bound mesh/raw records beyond the public compact CSVs. Public downloads support main acoustic replay; they are not a substitute for the full execution archive. Large fields and converted raw trees stay local. See [outputs and limitations](OUTPUTS_AND_LIMITS.md).
