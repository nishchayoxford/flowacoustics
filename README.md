# FlowAcoustics — turbine / flow to sound

**[Open the interactive demonstration](https://nishchayoxford.github.io/flowacoustics/)** · **[Browse the verified source](source/mrl2d/)** · **[Download the source archive](https://nishchayoxford.github.io/flowacoustics/downloads/flowacoustics-mrl2d-source.tar.gz)**

Three elliptical blades orbit and pitch in a circular domain. Follow their motion,
inspect the actual computed flow, then compare acoustic recording surfaces and
receiver directions. This is an educational strict-2D demonstration, not an
experimentally validated prediction of a finite turbine's noise.

## Start with one comparison

1. Open the demonstration and choose **SST URANS / On-body**.
2. Play the recorded flow. The two panels show native gauge pressure and speed,
   not a propagating acoustic wave.
3. Inspect the body and permeable recording contours, then the loading, receiver
   history, spectrum and directivity.
4. Choose **Other surface · same flow model**. Change receiver direction to see
   how the two alternative acoustic estimates compare. Never add them together.
5. Compare **Other flow model · same surface** to inspect SST URANS versus the
   two-dimensional WALE model under matched support.

## What actually ran

| Case | Flow model | Acoustic surface |
|---|---|---|
| `mrl2d_urans_body` | SST URANS | Moving blade contours |
| `mrl2d_urans_permeable` | SST URANS | Stationary permeable contour |
| `mrl2d_les_body` | Strict-2D WALE / LES model | Moving blade contours |
| `mrl2d_les_permeable` | Strict-2D WALE / LES model | Stationary permeable contour |

All four runs reached **10 s**, with **26,751 fluid cells** per case, two MPI
ranks and a fixed timestep of **0.0000625 s**. The original native runtime was
OpenCFD **OpenFOAM v2206**. Each flow movie contains 500 actual saved frames,
covering 0.02–10 s; no synthetic flow frames replace the calculation.

The webpage includes all four numerical NPZ files and sixteen acoustic/source
CSVs, plus field-history data. The receiver selector changes the display, not
the downloadable arrays. Its source bundle retains all 108 chart traces.

## Verification and limits

- All four native calculations completed, with source and field readback.
- Independent acoustic implementation checks covered every signed complex
  coefficient and synthesized receiver-pressure value. These checks establish
  consistency with the implemented equations, not physical validation.
- The integrated webpage passed **14 tests**, including real-production desktop
  and mobile browser interactions, video playback/seeking and actual downloads.
- The sealed source archive was extracted into a fresh directory and its
  documented selections independently rerun: **173 passed, 7 skipped**. The
  seven checks require native archives that are deliberately not distributed.
- The broader historical completion suite had an incomplete timed run; it is
  not represented as passing. Native replay and a fresh dependency installation
  were not performed for this source release.

The WALE cases do **not** resolve three-dimensional turbulence or blade-tip
physics. The acoustic pressure is in Pa for an ideal infinite-span 2D source,
not a finite-turbine absolute noise prediction. Source loading is N/m. The
plotted spectrum is coherent-window bin RMS, not a power spectral density;
synthesized histories are band-limited and window-conditioned. Stationarity,
mesh/time convergence and useful acoustic bandwidth have not been established.
Body/permeable estimates overlap and are never summed; their difference is not
an isolated quadrupole measurement.

## Source and reproducibility

The browsable [source snapshot](source/mrl2d/) and downloadable tarball contain
identical, manifest-pinned package bytes. Start with its
[setup and safe test commands](source/mrl2d/README.md),
[verification scope](source/mrl2d/docs/mrl2d/acoustics/VERIFICATION.md),
[method](source/mrl2d/docs/mrl2d/acoustics/METHOD.md) and
[conditional native replay instructions](source/mrl2d/NATIVE_REPLAY.md).

**Source archive SHA-256**

```text
5f8157ef46d55572a1517905b8e0432567ac2910c1dea280696f4af01874e3dc
```

The archive includes 48 exact initial/configuration files, not full runnable CFD
case archives: generated meshes, native histories and volume fields remain
local. The preparation-only mesh-policy emitter reproduces the executed source
policy, not a verified new mesh. Gmsh binary/connectivity identity is not claimed.
The two declared runtime-helper changes require an explicit environment locator
and fail closed on setup failure; numerical algorithms are unchanged.

**Snapshot status note:** the sealed package deliberately retains its original
“private/not published” preparation notices and historical manifest flags.
Those describe its sealing stage. Public distribution is authorized for this
release; the original package has not been rewritten to falsify that history.
This does not grant a new license or change its unverified native-replay limits.

Python 3.12 and the declared dependencies are required for the documented safe
unit selections; Node.js is used by a frontend-schema check. Do not substitute
an unrestricted full test run: retained native/lifecycle tests are outside the
safe default. Viewing the page does not launch a solver.

To view the static site from a checkout:

```sh
python3 -m http.server 8513 --bind 127.0.0.1 --directory site
```

## Distribution boundaries

- `site/`: the tested teaching page, derived data, real movies and source-archive
  download. Numerical page/data files retain their accepted bytes.
- `source/mrl2d/`: the complete sealed source package, with its original manifest.
- `RELEASE_MANIFEST.json`: hashes of the distributed tree, excluding itself.
- `.github/workflows/pages.yml`: publishes only the static `site/` directory.

No private audit logs, credentials, machine-specific settings, downloaded papers,
compiled solver libraries or raw CFD fields are distributed. The previous
cylinder demonstration remains in Git history, not in this release's active
file tree. No Git history was rewritten.

Existing native GPL notices and license text are retained. No project-wide
license or additional reuse permission is invented; see the
[licensing record](source/mrl2d/LICENSES/README.md).
