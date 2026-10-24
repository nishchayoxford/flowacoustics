# Four-case 2D turbine demonstration — current contract

## Current user authorization

The latest request is four cases: two 2D URANS cases and two 2D LES cases, with each pair comparing on-body and additional permeable-contour acoustics. This supersedes the older full-3D requirement and the preceding two-case choice. The user explicitly authorized deletion of all previous CFD cases and irrelevant old files, delegated design decisions without further suggestion prompts, and requested the webpage be updated. Maximum **250,000 fluid cells per complete case**, not per rank; this is a ceiling, not a target. Retain the previously selected **10 s simulated duration** unless native failure makes it impossible; do not silently call a short smoke a completed demonstration.

Exactly four final case directories under `cfd/`:

1. `mrl2d_urans_body`
2. `mrl2d_urans_permeable`
3. `mrl2d_les_body`
4. `mrl2d_les_permeable`

Temporary mesh/smoke candidates belong only under `.cache/mrl2d/` and must be removed after their useful compact reports are retained. Do not leave fifth cases or replicas in final CFD storage. Native logs/manifests are evidence, not a reason to retain all retired case trees. No git-index/history rewrite, commit or push without publication-specific authorization; the existing public-site deployment route must be discovered before updating it.

## Geometry and flow

Retain a circular open farfield, radius 0.6 m, in x-y; inflow `(1,0,0)` m/s. Water rho=1000 kg/m^3, nu=1e-6 m^2/s; acoustic c0=1480 m/s, reference pressure 1e-6 Pa. No cavitation, free-surface, imposed broadband inflow or shaft/support geometry.

Three elliptical blades: chord 0.070 m, thickness 0.007 m. Initial centres `(0,0.055)`, `(sqrt(3)*0.055/2,-0.055/2)`, `(-sqrt(3)*0.055/2,-0.055/2)`; chord angles 90,30,-30 degrees from +x. Omega=10.472 rad/s exactly. Centre(t)=Rz(-Omega*t)centre(0); each point relative to its blade centre rotates at absolute -Omega/2. Retain genuine orbit plus pitch, not a rigid rotor substitute.

A one-layer finite-volume extrusion with `empty` front/back represents a strictly two-dimensional section. Choose and record numerical extrusion depth; forces per unit span and contour integration must remove that arbitrary depth exactly once. Do not fabricate finite-span end-cap pressures or relabel this as the retired 0.220 m finite-span turbine. Use disjoint carrierZone and blade0Zone/blade1Zone/blade2Zone. Default carrier radius 0.105 m and blade-zone radii 0.040 m. Patch names blade0/1/2, farfield, frontAndBack; carrierAMI_master/slave and blade0/1/2AMI_master/slave. Final names must be communicated to all owners before generation.

Prefer a modest interface-fitted 2D mesh (tens of thousands of cells if adequate), not maximum-budget refinement. Require full native `checkMesh -allGeometry -allTopology`, Mesh OK, no failed checks, expected 2D directions, actual count <=250000, and native dual-motion/AMI coverage verification. Gmsh is available and the isolated `.cache/mrl-gmsh-netgen` Python environment can be reused as a dependency; no need to carry old CFD meshes.

URANS: time-dependent k-omega SST, with explicit justified wall and inlet settings. LES: requested WALE model on the same 2D topology. Label it **2D WALE / LES-model calculation**, not resolved 3D turbulence or valid broadband turbine LES. Preserve the user's requested distinction; never silently turn the LES pair into RANS. Do not promise the models' ranking or agreement. Record timestep/Courant/stability evidence; choose shared controls across all four if stable. Never use adaptive output directly in a uniform FFT. Up to eight MPI ranks, one numerical thread each; benchmark a small rank count before long execution because the retired 3D case was AMI-bound.

The permeable contour is an imaginary recording contour in stationary fluid, not a mesh wall or porous resistance. Both cases in each pair have identical physical inputs and mesh; only acoustic recording/analysis differs. Prefer identical native flow histories or verify their agreement. Actually run/verify the requested cases; do not rename a synthetic or incomplete output as production.

## Acoustic dimensional contract

Use a genuinely **2D free-space wave Green function / contour acoustic analogy** for the paired calculations, unless an alternative is explicitly documented and independently verified before coupling. The old closed 3D libAcoustics callback is NOT automatically valid on `empty` patches. No invented span/coherence correction or fake caps. Results represent an idealized span-invariant source, not finite-turbine absolute sound levels.

The acoustic implementation owner must derive and validate signs, Fourier/time convention, dimensions, moving contour velocity and normal orientation against analytical monopole/dipole controls before accepting native results. On-body includes moving-boundary thickness and loading as supported by the implementation. Permeable uses sampled p and U on a closed stationary contour enclosing all AMIs, with complete native sampling. Account explicitly for any omitted viscous traction and exterior volume sources; do not call body-minus-envelope an isolated quadrupole measurement. Never add the two overlapping acoustic estimates.

Native source contract: physical times; positions `(x,y)`; outward contour normals; positive segment lengths in metres; kinematic p converted to Pa exactly once; fluid U; surface velocity; per-blade identity. Prefer compact append-only native histories rather than one file per face/frame. Record complete body curves and a fixed closed contour in stationary fluid (e.g. x=[-.14,.30], y=+-.14), with actual native p/U interpolation and coverage. Suggested uniform acoustic cadence .002 s, to be confirmed against time step and analysis band. Twelve common observers on r=1 m, angles 0,30,...330 degrees; no automatic far-field claim. Source sampling, analysis-band limits, Fourier-window/taper assumptions and startup exclusion must be explicit. A chosen finite-frequency synthesis is band-limited, not a recovered full acoustic waveform.

## Ownership and delivery

- Mesh worker: `src/flowacoustics/mrl2d_mesh.py`, `scripts/build_mrl2d_mesh.py`, `tests/test_mrl2d_mesh.py`, `docs/mrl2d/mesh/`, `.cache/mrl2d/mesh/`. No final flow-case generation or deletion.
- Acoustic worker: `src/flowacoustics/acoustics2d.py`, `tests/test_acoustics2d.py`, `docs/mrl2d/acoustics/`. Derive/test actual 2D method and define native-source/API contract. No CFD or old-case writes.
- Flow/sampling worker: `src/flowacoustics/mrl2d_case.py`, `scripts/run_mrl2d.py`, `tests/test_mrl2d_case.py`, `docs/mrl2d/flow/`, optional native recorder under `acoustics/mrl2d/`. Configure four models and efficient genuine p/U contour capture; initial independent work only until mesh handoff.
- Parent: deletion manifest and guarded cleanup; current contract; integration and final run orchestration; webpage/data/publication. Do not modify another owner's files while they execute.

All new numerical/configuration behavior follows vertical RED-GREEN tests. Check scoped Ruff lint and format as well as pytest before final source freeze. Keep native returns/endpoints separate from wrapper success. Use succinct user-facing reporting and a coherent geometry → real flow → source histories → receiver/spectrum comparison. A full website of diagnostics alone is not delivery. Show actual computed data only, with missing/pending states rather than fabricated curves. Preserve files needed by new workers; remove obsolete workflows only after dependency inventory.
