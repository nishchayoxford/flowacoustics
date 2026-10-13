# Cylinder acoustic methods

## Scope and clocks

A finite, fixed, closed cylinder in a homogeneous water demonstration is simulated with OpenCFD OpenFOAM v2206, WALE and unmodified libAcoustics. The refined run uses 459,264 cells, eight MPI ranks and fixed Δt = 0.000625 s from zero to 10 s. There is no restart and **all exported clock offsets are zero**. The older adaptive approximately 6 s case is historical and has different restart semantics.

The native computation completed; scientific adequacy did not. The main 2–10 s analysis remains `partial`. Its largest lift PSD value is at the first nonzero 0.5 Hz bin, so a resolved shedding frequency, stationary tone and precise Strouhal number are not established. Analytical fixtures and ideal-source lessons are not CFD evidence.

Replay commands are in [REPRODUCE.md](REPRODUCE.md). The main CLI accepts canonical schema-1 metadata or a completed original schema-2 record through `native_run_adapter.py`. The detached adapter preserves source status, clocks and traction conventions, requires completed reconstruction/exports and consistent endpoint evidence, and hash-verifies the CSV bytes against the original export pins. It never rewrites the native manifest. A public sanitized schema-1 view is a different metadata artifact whose provenance must retain the original schema-2 digest.

## Input conventions

- x: streamwise; y: lift; z: cylinder axis. Source origin: (0,0,0).
- Diameter 0.02 m, span 0.06 m, outer radius 0.24 m; genuine fluid beyond the closed end caps.
- Free-stream target 0.2 m/s; density 1,000 kg/m³; kinematic viscosity 1e-6 m²/s; sound speed 1,500 m/s. Initial weak transverse offsets are startup perturbations, not synthetic inlet turbulence.
- `forces.csv`: `time_s,force_x_N,force_y_N,force_z_N` plus separately retained pressure and viscous components. Main force vectors are fluid traction **on the body**, in N, with `force_content: total_traction`.
- `observers.csv`: `time_s,observer,method,pressure_Pa`. Acoustic pressure is already Pa; do not multiply it by density again.
- Observers: `cross_plus` (0,1,0), `cross_minus` (0,-1,0), `downstream` (1,0,0), `oblique` (0.6,0.8,0), `cross_double` (0,2,0), all in metres. Use recorded coordinates, not a fitted angular pattern.
- Native Curle is `source_unretarded`; FW-H is on the physical `observer` clock with `fixedResponseDelay false`. A time coordinate must not be shifted merely to fit agreement.

The selected refined timestamps are uniform: no temporal resampling is needed. For adaptive historical records the CLI requires an explicit interpolation interval and refuses unfiltered downsampling. Interpolation does not add physical bandwidth or independent observations. Duplicate/nonfinite records and unsupported time windows are rejected; missing native observations are never synthesized.

## Independent compact loading model

Let **F** be the force **on the fluid**, **n** = **x**/r and τ = t − r/c. In a stationary homogeneous medium,

\[
p'(\mathbf{x},t)=\frac{1}{4\pi}\left[
\frac{\mathbf n\cdot\mathbf F(\tau)}{r^2}
+\frac{\mathbf n\cdot\dot{\mathbf F}(\tau)}{c r}\right].
\]

The sign follows the point-force wave equation `c^-2 p_tt − ∇²p = −∇·f`, with force acting on fluid. Exported body force therefore enters with **F_fluid = −F_body**, not an empirically selected sign. A force in N needs no additional density factor: both terms have units Pa.

The **main analysis** uses total body force. The **signature report** separately uses pressure-only body force to form its offline compact near/radiating terms. These are declared different comparisons, not interchangeable products. Neither is a copy of native FW-H.

The main independent implementation differentiates actual timestamps using second-order nonuniform/edge stencils, then linearly interpolates force and derivative to the retarded time. It excludes unsupported derivative/propagation startup and never extrapolates a missing tail. Mean pressure remains in the waveform; AC RMS and spectra remove their stated means. Native histories retain their own clocks even when compared on a common supported interval.

Compactness requires small source extent relative to both range and wavelength (`a/r`, `ka`); it does not imply `kr >> 1`. For a harmonic force, near/radiating amplitudes scale as 1/r² and 1/r, have ratio 1/(kr), and are in quadrature. Arbitrary histories require the near/far covariance; their RMS values must not simply be added.

## What native libAcoustics computes

Pinned upstream commit: `b9cd1d62ad8c5b445240a506a945231d12454b73`, retained under `third_party/libAcoustics-v2212/upstream/Sources/lib/analogies/`.

| Implementation | Consequence |
|---|---|
| `CurleAnalogy/CurleAnalogy.C`, `calcForcesMoments()` and `forceEff()` | Total **on-body** force, evaluated at current source time with near and derivative terms; no propagation retardation. Preserve its sign/clock mismatch relative to the independent force-on-fluid model. |
| `AcousticAnalogy/AcousticAnalogy.H`, `tdPrimitive::dot` | Later fixed-step calls use `(3 new − 4 old + oldOld)/(2 dt)`. Curle's current constant dt matches that assumption, unlike the old adaptive case. This is not a time-convergence result; startup uses other stencils. |
| `AcousticAnalogy/AcousticAnalogy.C` | Text acoustic clocks subtract the relevant start origin. This fresh zero-start run needs no offset; historical continuation offsets must not be reused. |
| `FfowcsWilliamsHawkings/Farassat1AFormulation.C` | At a stationary no-slip wall, moving-surface thickness is zero and loading is `p n`; explicit viscous traction and external quadrupole-volume terms are absent. |
| `FfowcsWilliamsHawkings/fwhFormulation.C` | Outward-from-surface normals and facewise arrival times retain geometry, distance and retardation. This differs from collapsing all traction to one point. |
| `FfowcsWilliamsHawkings/FfowcsWilliamsHawkings.C` | Kinematic CFD pressure is converted using density once; native observer output is already Pa. |

Native FW-H and Curle are alternative predictions, **never summed**. Finite extent, source content, normals/sign, clocks, differentiation and interpolation all affect differences. Fixed-body thickness zero does not prove omitted turbulent quadrupoles zero. The unused/lagged inherited auxiliary load files are explained in [OUTPUTS_AND_LIMITS.md](OUTPUTS_AND_LIMITS.md).

## Spectral and statistical definitions

- Periodic Hann Welch, 50% overlap, per-segment mean removal, arithmetic averaging in linear one-sided density. Force PSD: N²/Hz; pressure PSD: Pa²/Hz.
- Primary native window 2–10 s; 3,200 samples per segment; seven overlapping segments; Δf = 0.5 Hz. They are descriptive, not independent realizations or a stationarity proof.
- `sum(PSD)*Δf` is discrete bin power; with one segment it matches the **window-weighted, segment-detrended** mean square, not necessarily the unweighted record variance.
- AC RMS is separately mean-subtracted. Underwater SPL is `20 log10(p_rms / 1e-6 Pa)`. PSD density level uses `10 log10(PSD / ((1e-6 Pa)^2/Hz))`; it is not band SPL. No finite floor replaces true zero in reductions.
- Candidate `St = f_lift D/U` is inferred only if resolution gates pass. Lowest-positive-bin/Nyquist peaks or insufficient candidate cycles are unresolved. No prescribed St is presented as observed shedding.
- Coherence/phase require common supported clocks. `CSD = conj(FFT(force))*FFT(pressure)` gives negative phase for a delayed pressure. At least four overlapped segments are required; phase also uses coherence ≥0.5 and disclosed power gates. These are conditioning/display rules, not significance or causal-source tests.
- Native angular support is four 1 m observers, shown as discrete points. The 2 m receiver is a range comparison, not another point on that ring. Any dense offline compact sweep is explicitly modelled, not interpolated native observers.
- Signature bands use a shared cadence/bin contract and half-open interior bands, with the final upper edge inclusive. Boundary-bin roundoff is handled narrowly; bins are not double-counted. Native-to-uniform, retarded-time and local spectral interpolation are separately disclosed.

## Sources and limits

Foundations: [Curle (1955)](https://royalsocietypublishing.org/doi/10.1098/rspa.1955.0191); [Ffowcs Williams & Hawkings (1969)](https://doi.org/10.1098/rsta.1969.0031); [Farassat, NASA/TM-2007-214853 (2007)](https://ntrs.nasa.gov/citations/20070010579); [MF207 course notes](https://perso.ensta.fr/~cotte/files/2023-course-notes-MF207.pdf), especially force-on-fluid conventions and the free-space Green function. Code interpretations are tied to the included pinned upstream source, not a claimed new full-text audit of every paper.

No mesh/time/domain independence, statistical stationarity, exact serial/MPI equivalence, resolved LES adequacy, experimental acoustic validation or complete Lighthill source decomposition is claimed. No Amiet, cavitation, airfoil, propeller or turbine model is included in this cylinder demonstration.
