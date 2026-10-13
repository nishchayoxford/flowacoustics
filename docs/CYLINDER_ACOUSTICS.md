# Cylinder acoustic analysis

## Scope and execution

This is an **educational demonstration**, not validation of a resolved turbulent wake, LES convergence, or an experiment. The analysis never runs OpenFOAM, modifies libAcoustics, creates substitute CFD, or publishes site files. It requires explicit input paths. Analytical test fixtures and ideal-source lessons remain labelled `analytical`; they are not cylinder evidence.

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python scripts/analyze_cylinder.py \
  --run site/downloads/run.json \
  --forces site/downloads/forces.csv \
  --observers site/downloads/observers.csv \
  --output results/reproduced \
  --start-time-s 2 --resample-dt-s 0.0009 --nperseg 1536
```

The published command explicitly resamples the adaptive timestamps at 0.0009 s. Omitting an explicit resampling interval rejects adaptive timestamps. The interval must not exceed the minimum selected source/observer interval: downsampling needs an anti-alias filter and is rejected, not approximated. Linear interpolation changes the high-frequency content; it does not increase physical bandwidth or create independent observations. Choose `nperseg` from the resulting sample rate and the desired physical segment duration, not a fixed aesthetic chart length.

Use `--start-time-s` and `--end-time-s` to select a documented physical-clock window. No automatic transient removal is claimed. **If the native library was restarted, choose a contiguous window after the last restart or otherwise independently verify its history continuity.** The code removes initial propagation/derivative support, but cannot infer undeclared library-history resets from a force CSV. Do not resample across an unrecorded gap or interpret native startup zeros as silence.

If native `run.json` uses different field names, supply `--conventions <explicit-adapter.json>` containing the canonical fields below. The adapter replaces only the six convention fields, is hashed with the original manifest, and cannot overwrite status/evidence kind. It is not an automatic guesser. A missing file, metadata field, unsupported time, duplicate clock sample, too-short force record, or ambiguous sign causes an error. A header-only observer CSV is explicitly allowed: force charts and analytical lessons are emitted, and native observer analysis is marked unavailable. Existing result/download files are not overwritten; use a fresh output directory.

Tests use only analytical histories and temporary files:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  tests/test_cylinder_acoustics.py tests/test_analysis.py -q -p no:cacheprovider
```

## Canonical metadata and CSV contract

`run.json` retains `schema_version: 1`, actual `status`, and preferably `evidence_kind: "native_execution"`. Tests instead explicitly use `evidence_kind: "analytical_fixture"`, which forces the entire result to keep that status and all data-derived charts to retain analytical origin. Unknown evidence kind is retained as unverified metadata, not upgraded to a native execution claim. Native-derived analysis is conservatively `partial`, because producing plots does not establish scientific adequacy; `metadata.native_status` retains the supplied native status.

The six convention fields, either in `run.json` or in the explicit sidecar, are:

```json
{
  "medium": {
    "density_kg_m3": 1000,
    "sound_speed_m_s": 1500,
    "velocity_m_s": 0.2,
    "kinematic_viscosity_m2_s": 1e-6
  },
  "geometry": {"diameter_m": 0.02, "span_m": 0.06},
  "force_convention": "on_body",
  "force_content": "total_traction",
  "observers": [
    {"name": "cross_plus", "position_m": [0, 1, 0]},
    {"name": "cross_minus", "position_m": [0, -1, 0]},
    {"name": "downstream", "position_m": [1, 0, 0]},
    {"name": "oblique", "position_m": [0.6, 0.8, 0]},
    {"name": "cross_double", "position_m": [0, 2, 0]}
  ],
  "clock": {
    "forces_offset_s": 0,
    "libacoustics_curle": {"offset_s": 0, "kind": "source_unretarded"},
    "libacoustics_fwh": {"offset_s": 0, "kind": "observer"}
  }
}
```

These positions match the inspected cylinder worker dictionary, **but verify the final run manifest rather than treating this example as evidence that those observers executed**. The source is fixed at the origin, x is streamwise, y lift and z the cylinder axis. `force_content` accepts `total_traction` or `pressure_only`; `force_convention` accepts `on_body` or `on_fluid`. Coefficients always convert to force-on-body; raw force charts/CSV preserve the supplied sign.

Required force columns: `time_s,force_x_N,force_y_N,force_z_N`. Extra pressure/viscous-force columns are preserved in the raw CSV download but are not silently substituted into the compact prediction. Required observer columns: `time_s,observer,method,pressure_Pa`. The only accepted native method names are `libacoustics_curle` and `libacoustics_fwh`. A name must have an explicit coordinate. Duplicate or reversed timestamps within a history are rejected, not sorted away.

`physical_time = CSV_time + offset_s`. An exporter that already restored the global clock must use offset zero; adding the restart offset again is wrong. With `fixedResponseDelay false`, FW-H uses physical observer time. A producer using a response-delay shift must restore the physical clock itself; per-observer shifted clocks must not be passed as if they were physical observer time. Curle is deliberately labelled `source_unretarded`: its native samples are retained on the simulation/source clock, not silently shifted to make agreement.

## Independent compact Curle loading

The classical stationary-surface context is Curle's *The influence of solid boundaries upon aerodynamic sound* (1955). FW-H generalizes the acoustic analogy to moving surfaces; Farassat's NASA/TM-2007-214853 develops formulations 1 and 1A.[1][2][3]

For a stationary homogeneous medium, let **F** be the force exerted **on the fluid**, **n** = **x**/r, and τ = t − r/c. The point-force Green-function solution is

\[
p'(\mathbf x,t)=\frac{1}{4\pi}\left[
 \frac{\mathbf n\cdot\mathbf F(\tau)}{r^2}
 +\frac{\mathbf n\cdot\dot{\mathbf F}(\tau)}{cr}\right].
\]

This sign follows from the momentum equation and source wave equation `c^-2 p_tt − ∇²p = −∇·f`, not from an amplitude match. MF207 equations (2.27)–(2.28) define the external force as acting **on the fluid**, and (2.38) gives the outgoing free-space Green function.[4]

An OpenFOAM force **on the body** is opposite: **F_fluid = −F_body**. Hence the independent compact pressure has a minus sign when supplied body forces. A positive static fluid force produces positive on-axis near pressure; reversing the force reverses both pressure terms. Density must **not** multiply a force already in N. The dimensions are `N/r² = Pa` and `(N/s)/(c r) = Pa`. This is a loading model; it is not a complete acoustic-source decomposition.

The implementation:

- Uses actual source timestamps, second-order central nonuniform finite differences internally and second-order one-sided edge stencils.
- Differentiates before interpolating the force and derivative to τ. Interpolation is linear and no gain/delay/phase fitting occurs.
- Rejects τ outside recorded support rather than zero-extending or holding endpoints. Only floating-point boundary round-off is clipped.
- In dataset comparisons, discards the source edge stencil, uses a conservative cylinder bounding radius `a = hypot(D/2, span/2)` for complete startup arrival support, and restricts every method at an observer to the same physical clock/window.
- Preserves the mean in pressure waveforms; AC RMS and spectra remove their stated means. The static near term must not be confused with radiated sound.

Compact-source conditions concern both geometry (`a/r` small) and wavelength (`k a` small). A geometrically remote observer can still have `k r << 1`, so **small source size is not an acoustic far-field condition**. Metadata reports `a`, `k a`, `k r` at the measured lift-peak candidate, and `a/r` for each used observer. Frequency-conditioned values are null if the lift peak is unresolved. Mean-flow acoustic convection is neglected explicitly in this low-Mach, free-space demonstration.

For a sinusoidal force, near and far terms have amplitudes proportional to `1/r²` and `1/r`, with their ratio `1/(k r)`; they are in quadrature. At identical emission times, doubling radius divides the near pressure by four and far pressure by two. Total harmonic RMS is the root-sum-square of these two components, not their linear sum. For arbitrary histories, retain the covariance; do not apply the harmonic shortcut.

## What the retained libAcoustics actually computes

Read-only source: `third_party/libAcoustics-v2212/upstream/Sources/lib`, upstream commit `b9cd1d62ad8c5b445240a506a945231d12454b73`. These are code interpretations, not claims that the new CFD has already executed.

| Source and inspected lines | Consequence |
|---|---|
| `CurleAnalogy.C:145–164` calls `calcForcesMoments()`, sets `F=forceEff()`, then uses `l·(dotF/c + F/r)/(4πr²)` at the **current time** | This is a time-domain compact approximation containing both near and radiation terms, **not** a far-term-only frequency-domain model. It contains no propagation retardation. Its `+forceEff` sign is opposite to the physical force-on-fluid convention if `forceEff` is exported as force-on-body. Retain the signed mismatch instead of fitting it away.[5] |
| `CurleAnalogy.C:109–125` calculates a face-centre-based centroid | This is not an area-weighted acoustic centre. `ni += ci.size()` is cumulative when multiple patches are appended, a potential multi-patch denominator issue. The cylinder uses one closed patch; the independent model uses the contractual origin, not an estimated/fitted centre.[5] |
| `AcousticAnalogy.H:150–185`, `tdPrimitive::dot` | First call uses `new/dt`; next call uses backward first difference; later fixed-step calls use `(3 new − 4 old + oldOld)/(2 dt)`. Variable-step mode uses first difference. Curle does not set this flag in its inspected constructor; Farassat sets it from `isAdjustTimeStep()`. Adaptive native Curle derivatives and startup transients need qualification; independent central derivatives are not an exact replica.[7][8] |
| `AcousticAnalogy.C:267–275` | Text output subtracts the greater of the function-object timeStart and run start time. Restore that origin exactly once before comparing histories.[6] |
| `Farassat1AFormulation.C:218–259` | Builds `U=(1−rho/rho0)v + rho*u/rho0` and `L=(p I + rho u (u−v))·n`, then thickness and loading surface terms. At a stationary no-slip wall `u=v=0`, `U_n=0`, `L=p n`; the native loading contains pressure, **not an explicit viscous traction term**. A total-force compact model consequently need not equal it.[8] |
| `fwhFormulation.C:114–144, 289–303` | Reorients normals away from a face-centre average and deposits each face at arrival time `source_time + R/c`; optional fixedResponseDelay subtracts `tauMin − responseDelay`. This outward-from-surface normal means stationary pressure loading is force-on-fluid, opposite to pressure force-on-body.[9] |
| `Farassat1AFormulation.C:98–110, 164–166` | Interpolates stored face contributions at observer time (with a tiny `dt*1e-6` internal increment), sums and divides by `4π`; before supported arrivals the native implementation can return zero. Analysis must not count these zeros as measured silence.[8] |
| `FfowcsWilliamsHawkings.C:474–486` | If CFD pressure is not dimensional pressure, multiplies sampled p by `rhoRef` once. Exported acoustic output is Pa; multiplying it by density again is wrong.[10] |

The distributed formulation uses each face's distance, direction and retarded time, whereas the compact model collapses loading to one point. Finite source extent, different force content, derivative schemes, interpolation, sign and clock conventions can all cause differences. Neither a pressure PSD match nor a freely fitted amplitude proves agreement. No sum of Curle and FW-H predictions is formed: they overlap as alternative predictions.

The wall's stationary impermeability makes the moving-surface thickness source zero **in this specific setup**. This is not evidence that a volume quadrupole contribution is zero. The inspected Farassat implementation is a **surface-only** thickness/loading calculation; there is no evaluated Lighthill quadrupole volume integral in that prediction. Do not label the native output a complete monopole/dipole/quadrupole decomposition or a complete Lighthill solution.[2][8]

## Spectra, measured shedding, coherence and directivity

The existing `analysis.py` helpers are reused without modification:

- Welch: periodic Hann, 50% overlap, segment-mean removal, one-sided **density**. Force PSD is N²/Hz; pressure PSD is Pa²/Hz.
- `sum(PSD)*Δf` is bin power, including the correct DC/Nyquist convention. For a single segment it equals the **window-weighted, segment-detrended** mean square. It does not generally equal the unweighted global variance. Tests cover a bin-centred sinusoid, random-data boxcar Parseval and exact Hann-weighted Parseval.
- Spectrogram: same Hann density convention. JSON `z[frequency_index][time_index]` is linear Pa²/Hz, not an invented coloured flow field.
- Mean-subtracted record RMS is reported separately from integrated Welch power. SPL uses `20 log10(p_rms / 1e-6 Pa)`. A zero RMS has null level, not an arbitrary finite floor. A PSD level would use `10 log10(PSD / (1e-6 Pa)^2 per Hz)`, not band SPL.

The measured Strouhal candidate is `St = f_lift_peak D/U`, selected from the recorded lift spectrum, never a prescribed 0.2. It is unresolved if power is zero, the peak is in the lowest positive bin or Nyquist bin, or fewer than four candidate cycles fit the record. This is a descriptive resolution gate, **not** a proof of vortex shedding, statistical development or mesh adequacy. The bin width and candidate cycle count accompany the result.

Force/pressure coherence uses the force-on-fluid component toward the actual observer, evaluated on the same physical clock. `CSD = conj(FFT(force))*FFT(pressure)`, so a delayed pressure has negative delay phase. At least four overlapping Hann segments are required before coherence is displayed; phase additionally requires coherence ≥0.5 and both auto-spectra above `1e-10` of their respective maxima. DC is masked. These thresholds are transparent display/conditioning rules, not confidence limits: overlapped segments are not independent, and high coherence does not by itself identify a unique physical source. Phase is wrapped in radians; unsupported values are null. The raw signed pressure/force records remain downloadable.

Native directivity uses **only actual supplied observers** on a common z=0 equal-radius ring, common time support, and AC RMS. It does not synthesize native observers, join a singleton into a polar sweep, or combine different radii. `connect_points:false` requests discrete markers and the azimuth unit is degrees from +x toward +y. Sparse observations do not establish a continuous angular pattern. Nonplanar observer histories still receive temporal analyses but are not silently projected into this xy polar plot.

Distance ratios compare positive collinear observer pairs on their overlapping **source-time** support. Retarded methods subtract each radius/c before interpolation; native unretarded Curle uses its source clock without pretending it had a propagation delay. Ratios, time coverage, applied delay difference and independent `1/r`/`1/r²` expectations appear in metadata. No fitted decay exponent or arbitrary waveform shift is used.

## Independent ideal-source lessons

MF207 equations (2.17), (2.21)–(2.22) and (2.38) give monopole, compact dipole directivity and the free-space Green function.[4] With `exp(+iωt)` and `G=e^(−ikr)/(4πr)`, the reusable `point_source_phasor` implements:

- Monopole: `P = iω rho Q G`, with volume-velocity amplitude `Q` in m³/s.
- Dipole: `P = (F·n)(1/r + ik)G`, with force-on-fluid amplitude **F** in N.
- Quadrupole: `P = Qij ∂i∂j G`, with moment tensor **Q** in N m. Direct differentiation gives
  `P = G[(n·Q·n)(−k² + 3ik/r + 3/r²) + tr(Q)(−ik/r − 1/r²)]`.

The quadrupole expression is an explicit differentiation of the cited Green function, not an assertion that this lecture contains that exact tensor formula. The signed angular illustration selects `Qxx=−Qyy`, other components zero, producing `cos(2θ)` in the xy plane. Other quadrupole tensors have other patterns. Individual far-field normalized RMS patterns are `1`, `|cos θ|`, `|cos 2θ|`; normalization is not an equal-strength or equal-power comparison across differently dimensioned sources.

The four lesson charts show RMS directivity, signed angular factors, exact harmonic dipole near/far RMS, and retardation of an infinite-duration illustrative sinusoid. Every one states **not computed cylinder**. The analytical sinusoid may exist at negative source times by definition; finite CFD history may not be extended that way.

## Plot-ready output and publication boundary

Output files:

- `results.json`: exact schema_version=1 site contract (`status`, `summary`, `metadata`, `charts`, `checks`, `downloads`).
- `manifest.json`: input and implementation SHA256 identities, actual options, output hashes/bytes and programmatically derived chart/origin/aligned-row counts. The manifest does not hash itself.
- `downloads/forces.csv`, `downloads/observers.csv`: byte-preserved input CSVs, including unplotted samples/extra force columns.
- `downloads/aligned_pressures.csv`: physical observer time, retarded source time, observer/method, signed native pressure, independent compact total/near/far pressure and contemporaneous force-on-fluid projection. There is no fitted or amplitude-normalized waveform.

The CSV raw/download clock is the producer's original clock; plotted and aligned times include the declared offset, which is retained in metadata. The aligned `source_time_s` column belongs to the independent compact model; it must not be interpreted as a claim that native Curle used retardation.

Every chart includes `samples`: time/range/counts or the corresponding analytical sampling definition. Spectral charts state the unused final-window samples. Resampling explicitly records a new sample count and retains original per-method counts. JSON has no NaN/Infinity. Null is an unsupported diagnostic or zero-level value; zero PSD is retained and omitted on a log display rather than replaced by a made-up floor. Frequency/time/value arrays are not decimated for appearance. If a consumer downsamples rendering, it must not modify the raw downloads or analytical record counts.

The parent/site integrator owns publication. Copy `results.json` to `site/data/results.json` only after exercising the real native inputs, reviewing clocks/windows and checks. The `href` values are relative public-safe paths `downloads/...`; copy the download directory next to the JSON or resolve those links against the JSON's base URL. Do not copy analytical test fixture data as native CFD results. The separate `citation-ledger.json` and `verification.json` are analysis-development evidence, not runtime input or a claim of native solver execution.

## Reference access and attribution

Primary citations: N. Curle (1955), *The influence of solid boundaries upon aerodynamic sound*; J. E. Ffowcs Williams and D. L. Hawkings (1969), *Sound generation by turbulence and surfaces in arbitrary motion*; F. Farassat (2007), NASA/TM-2007-214853.[1][2][3] The present implementation inspected retained local library source and the full verified MF207 extraction, specifically equations (2.17), (2.21)–(2.28), (2.38); it did not newly download or claim a new full-text audit of Curle/FW-H. Reference identifiers follow the retained repository's verified source records. Preserve upstream GPL source notices; this work changes no upstream source.

## Sources

[1] https://royalsocietypublishing.org/doi/10.1098/rspa.1955.0191
[2] https://api.crossref.org/works/10.1098%2Frsta.1969.0031
[3] https://ntrs.nasa.gov/citations/20070010579
[4] https://perso.ensta.fr/~cotte/files/2023-course-notes-MF207.pdf
[5] https://raw.githubusercontent.com/unicfdlab/libAcoustics/b9cd1d62ad8c5b445240a506a945231d12454b73/Sources/lib/analogies/CurleAnalogy/CurleAnalogy.C
[6] https://raw.githubusercontent.com/unicfdlab/libAcoustics/b9cd1d62ad8c5b445240a506a945231d12454b73/Sources/lib/analogies/AcousticAnalogy/AcousticAnalogy.C
[7] https://raw.githubusercontent.com/unicfdlab/libAcoustics/b9cd1d62ad8c5b445240a506a945231d12454b73/Sources/lib/analogies/AcousticAnalogy/AcousticAnalogy.H
[8] https://raw.githubusercontent.com/unicfdlab/libAcoustics/b9cd1d62ad8c5b445240a506a945231d12454b73/Sources/lib/analogies/FfowcsWilliamsHawkings/Farassat1AFormulation.C
[9] https://raw.githubusercontent.com/unicfdlab/libAcoustics/b9cd1d62ad8c5b445240a506a945231d12454b73/Sources/lib/analogies/FfowcsWilliamsHawkings/fwhFormulation.C
[10] https://raw.githubusercontent.com/unicfdlab/libAcoustics/b9cd1d62ad8c5b445240a506a945231d12454b73/Sources/lib/analogies/FfowcsWilliamsHawkings/FfowcsWilliamsHawkings.C
