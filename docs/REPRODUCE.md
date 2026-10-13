# Reproduce the circular-cylinder demonstration

## Reproduce the published acoustic analysis without CFD

The website has no external JavaScript dependencies and performs no solver launches. From a checkout with Python 3.12 and uv:

```sh
UV_CACHE_DIR="$PWD/.uv-cache" uv sync --locked --extra dev
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/analyze_cylinder.py \
  --run site/downloads/run.json \
  --forces site/downloads/forces.csv \
  --observers site/downloads/observers.csv \
  --output results/reproduced \
  --start-time-s 2 --resample-dt-s 0.0009 --nperseg 1536
```

This uses the actual circular-domain records, not an analytical fixture. `run.json` is an explicit public subset of the local manifest, with its original SHA256 recorded in `provenance`; it is not claimed byte-identical to the full local manifest. Its canonical conventions and physical data are unchanged. CSV clocks are already absolute: do not add the 0.1 s restart offset again.

The 2–6 s interval is a later **developing-flow** window, not an assertion of statistical stationarity. The uniform 0.0009 s grid uses declared linear interpolation and is finer than the minimum recorded interval. Welch uses 1,536-sample periodic Hann windows, 50% overlap and segment-mean removal. Four overlapping segments give coarse frequency resolution; phase/coherence gates are descriptive, not significance tests.

The browser displays original local extrema for long time histories and a 0–20 Hz frequency window. These are presentation choices only: statistics use the full analysis record. `analysis-full.json.gz` contains full-band, full-resolution plot arrays; `aligned_pressures.csv.gz` contains the signed comparison samples.

## Native inputs and dependency build

The executed runtime was **OpenCFD OpenFOAM v2206**, double precision, 32-bit labels, GCC, four MPI ranks, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. A compatible OpenFOAM installation, MPI and FFTW3 development headers/libraries must already exist. Python installation does not provide those dependencies.

Source your installed OpenFOAM `etc/bashrc` before native work. The public native wrapper reads `WM_PROJECT_DIR`; it contains no author-specific installation path. This environment locator and quoting guard are portability-only adaptations to the originally executed helper; numerical settings, native source and exported data are unchanged. The circular generator uses the common cylinder helper; its unused rectangular generator remains an implementation dependency/test reference, not a second public result set.

The libAcoustics numerical source is unchanged from commit `b9cd1d62ad8c5b445240a506a945231d12454b73`. Build a separate copy, keeping the pinned upstream export intact:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FOAM_USER_LIBBIN="$PWD/.build/acoustics/platforms/$WM_OPTIONS/lib"
mkdir -p "$FOAM_USER_LIBBIN" .build/acoustics/libAcoustics/source
cp -R third_party/libAcoustics-v2212/upstream/Sources .build/acoustics/libAcoustics/source/
(cd .build/acoustics/libAcoustics/source/Sources/lib && wmake libso)
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"
```

The existing `Make/options` uses `FFTW_ARCH_PATH`, supplied by the OpenFOAM environment. Configure that installation for its actual FFTW location if necessary; no numerical-source patch was used in the executed build. The helper's library path corresponds to the executed `linux64GccDPInt32Opt` build. Other OpenFOAM variants/platforms need explicit compatibility work rather than an assumption of equivalence.

## Mesh and solver sequence

The accepted `system/blockMeshDict` is shipped. It creates a circle of radius 0.24 m, a D=0.02 m / span=0.06 m body, and genuine fluid beyond both tips. The free-stream target is 0.2 m/s in water with ν=1e-6 m²/s. The curved boundary uses `freestreamVelocity`/`freestreamPressure`; remote span planes use symmetry. Initial weak cross-flow/spanwise offsets are a startup perturbation, not synthetic inlet turbulence.

The public `system/controlDict` initially equals `controlDict.pilot`, ending at 0.1 s. The continuation dictionary is preserved as `controlDict.main`, starting at 0.1 s and targeting 6 s. A reproduction must run the pilot first:

```sh
case=cfd/cylinder_circular
blockMesh -case "$case"
checkMesh -case "$case" -allTopology -allGeometry
# Inspect the log: require 'Mesh OK.', not merely exit code zero.
decomposePar -case "$case"
mpirun -np 4 pimpleFoam -case "$case" -parallel
mkdir -p results/reproduction-pilot
cp "$case/acousticData/curle-time.dat" results/reproduction-pilot/
cp "$case/acousticData/fwh-time.dat" results/reproduction-pilot/
cp "$case/system/controlDict.main" "$case/system/controlDict"
mpirun -np 4 pimpleFoam -case "$case" -parallel
```

Native acoustic callbacks restart their histories and output files at continuation. Preserve pilot outputs before restarting. During export, pilot clocks have offset zero and the main raw acoustic clock receives +0.1 s exactly once. Never extend the missing acoustic tail with zeros. The helper functions in `scripts/cylinder_native.py` export native forces, acoustic records, closed-wall geometry and pressure, and actual sampled flow. `scripts/cylinder_circular.py` supplies the accepted circular mesher, native mesh views and export verification.

The original campaign used 1,711.7285358211957 s of a cumulative 1,800 s native wall allowance, including the earlier rectangular iteration and two rejected circular mesh candidates. Those failed checks were retained locally, not relabelled passes. A fresh reproduction is a new computation with its own recorded resources and outcomes; no claim is made that it will take identical wall time or return bitwise-identical parallel results.

## Interpretation boundary

The fixed closed body's moving-surface thickness term is zero. Native FW-H here evaluates retarded pressure loading, not external quadrupole volumes or viscous acoustic loading. Native Curle uses instantaneous total force on the body; the independent compact calculation uses retarded force on the fluid. No fitted gain, delay or sign correction is used to conceal these implementation differences.

The mesh is modest and quality-checked, not demonstrated to resolve all LES scales. No convergence, stationarity, exact serial/MPI equivalence or experimental acoustic validation is claimed. The website is an educational explanation supported by an actual 3D execution.
