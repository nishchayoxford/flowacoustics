# Cylinder acoustic field guide — static site

## Delivery and boundaries

`site/index.html` is a dependency-free Learn surface with a secondary native-results explorer. CSS and browser ES modules are local. There is no Node build, CDN, framework, audio, solver launcher or remote service. Serve the directory through HTTP (ES modules and `fetch` are not reliably supported by `file://`) or deploy **the contents of `site/`** as a GitHub Pages artifact. `.nojekyll` disables Jekyll processing.

`site/data/results.json` contains the checked circular-domain analysis: 35 computed charts and four additional analytical reference charts. No test fixtures are shipped in `site/data/`. The 13 interactive figures remain independent from the CFD dataset.
Actual circular-domain midspan/3D mesh previews and native flow samples appear before the source lessons. No procedural mesh or rectangular CFD domain is presented as the executed mesh. The analytical acoustic map labels its square plotting window as unrelated to CFD geometry; its central exclusion disk is a point-source-model cutoff, not a rendered cylinder wall.

Local preview from the repository root:

```sh
python -m http.server 8513 --bind 127.0.0.1 --directory site
```

Then open `http://127.0.0.1:8513/`. Publishing requires no build command. GitHub operations were not performed by the site worker.

## The complete learning loop

Start with the dipole, rotate the observer to 90°, inspect the ideal null, then move range and frequency. Numerical feedback changes with the plots. Native results never change with analytical controls.

| Figure | Calculated purpose | Effective controls |
|---|---|---|
| 01 | Source magnitude directivity plus observer ray | Family, angle |
| 02 | Signed radiative acoustic field, optional harmonic dipole near term | Family, frequency, near term, play/pause |
| 03 | Prescribed force versus its derivative | Frequency |
| 04 | Independent pulsating-sphere volume velocity and derivative | Frequency |
| 05 | Causal source emission and delayed, attenuated arrival | Family, frequency, angle, range |
| 06 | Radiating, reactive and quadrature-total dipole amplitudes | Frequency; selected-range feedback |
| 07 | Reactive/radiating amplitude ratio versus kr | Frequency and range move observer marker |
| 08 | Compact versus distributed coherent dipole pair | Separation and frequency |
| 09 | Equal-source power scaling with real pair coherence | Coherence |
| 10 | Sampling a known tone and the aliased frequency | Sample rate and frequency |
| 11 | Actual Welch estimate of known prescribed samples | Signal type, segment length, density units, frequency |
| 12 | Segment-by-segment densities from the same samples | Signal type, segment length, frequency |
| 13 | Explicitly chosen SI pressure, RMS SPL and range conversion | RMS pressure and Pa/µPa/dB units |

The loading/thickness concept check gives corrective feedback for both answers. Native range/select controls support keyboard input; outputs have labels, changing explanatory text has live status where appropriate, SVGs have titles/descriptions, and a skip link bypasses navigation. The page supports a 390 px viewport without horizontal overflow. Motion initially pauses under `prefers-reduced-motion`; play is always an explicit option, and animation stops when offscreen or the document is hidden.

## Analytical model contract

`physics.mjs` contains DOM-independent functions. These are unit lessons, **not source reconstruction or substitutes for missing CFD**.

- Coordinates use arbitrary length ℓ: `r*=r/ℓ`, `t*=c0*t/ℓ`, `f*=f*ℓ/c0`. The pressure scale is independent of the native cylinder. The field lab holds on-axis radiating amplitude at unit range fixed across frequency/family; the force-derivative lesson instead holds prescribed force amplitude fixed. They do not share a physical calibration.
- Angular examples: monopole `D=1`, dipole `D=cos(theta)`, quadrupole `D=cos(2theta)`. The last is a trace-free illustrative tensor orientation, not a universal four-lobe pattern or a computed volume-source decomposition. The analytical dipole axis is +x; cylinder lift is along y.
- The radiating field is `D*sin(2*pi*f*(t-r))/r`. The optional dipole term is `D*cos(2*pi*f*(t-r))/(2*pi*f*r^2)`. Its reactive/radiating amplitude ratio is `1/(kr)`; harmonic total amplitude is the quadrature sum, not the sum of magnitudes. This ideal point model is excluded within `r* < 0.3` in the wave map. Map colors clip at ±1, stated on its color key. Spatial axes retain equal scale.
- The causal arrival lesson uses a smoothly started prescribed tone with zero pre-emission history and delay `r*`, with no circular shift or extrapolated native tail.
- Compact pair model: equal coherent y-dipoles at x = ±d/2. Distributed normalized gain is `cos(pi*f*d*cos(theta))`, multiplying `sin(theta)`. This demonstrates phase retention, not an implementation of FW-H.
- Coherence model: equal autospectra, real identical pair cross coherence γ, zero relative observer phase. `Ssum/Sone = N + gamma*N*(N-1)`. γ is not magnitude-squared coherence; arbitrary complex coherence is outside this lesson.
- Welch: actual DFT per segment, segment-mean removal, periodic Hann window, 50% overlap, normalization by `fs*sum(window^2)`, one-sided doubling except DC/Nyquist, arithmetic averaging in linear density. The full discrete `sum(PSD)*df` is shown. Displaying dB relative to unit normalized density applies a disclosed −80 dB display floor only; integration retains unmodified linear density.
- Spectral prescribed input: 1,024 samples at normalized sample rate 16; unit first tone and 0.35-amplitude second tone at 2.3 times the chosen frequency. The optional first-tone chirp increases frequency by approximately 3 normalized frequency units over the 64-unit record. These histories are explicitly analytical and have no CFD meaning.
- SI unit lesson: chosen positive RMS pressure at 1 m, `p(r)=p(1m)/r`. It always uses **water reference `p_ref=1e-6 Pa`** and `20*log10(p_rms/p_ref)`. This is a chosen conversion example, not an absolute cylinder prediction. Numerical silence is −Infinity; invalid negative/nonfinite RMS is rejected.
- The fixed rigid impermeable cylinder's moving-surface thickness contribution is zero. This does not prove the omitted volume quadrupoles are zero. The surface library is not advertised as a complete Lighthill solution. Curle and FW-H predictions must not be added where they overlap.

The page also distinguishes hydrodynamic wall pressure from propagated acoustics; startup from statistical development; low-Mach/free-space assumptions from universal propagation; and force-on-body from force-on-fluid signs.

## Results JSON integration (parent-owned)

The read-only adapter requests **`./data/results.json`** once, with `cache: no-store` and an 8-second timeout. There is no background polling. A visible reload button repeats that request. A 404 gives `pending`, with empty chart/check/download arrays. Network, parsing or schema errors give `unavailable`, with the error explained and **no generated substitute history**.

Required top-level contract:

```json
{
  "schema_version": 1,
  "status": "pending",
  "summary": "Native cylinder results have not yet been integrated.",
  "metadata": {},
  "charts": [],
  "checks": [],
  "downloads": []
}
```

This is the empty contract, not a shipped results fixture. Actual status strings are preserved, not upgraded to completion. Metadata supports arbitrary nested SI-keyed objects and is displayed verbatim as escaped text. No adapter claim verifies the CFD; the supplied checks/manifest must do that.

Each chart requires:

- `id`, `title`, `description`, `x_label`, `y_label` strings;
- `origin: "computed"` or `"analytical"`, independently shown per chart;
- `kind: "line"`, `"scatter"`, `"polar"`, or `"heatmap"`;
- `traces: [{"name": "...", "x": [...], "y": [...]}]` (use `[]` for a pure heatmap);
- optional `x_scale` / `y_scale`: `"linear"` or `"log"`;
- heatmaps: top-level `x`, `y`, `z` with `z[yIndex][xIndex]`. Use increasing coordinate arrays. Nonuniform center spacing is supported by midpoint cell edges;
- polar traces: **x is angle in degrees**, y is radial magnitude (the renderer takes absolute values; signed pressure belongs in a line plot);
- explicit sample coverage. Recommended keys are `included_samples`, `excluded_samples`, `sample_policy` and `window`. Keys containing `sample`, `coverage`, `include`, `exclude`, `floor` or `window` are displayed beside the chart. If none is supplied the UI says sample coverage was not supplied;
- optional heatmap `color_label` for the physical units of z, `color_limit` for a disclosed symmetric color limit, `diverging: false` for sequential nonnegative density, and `equal_aspect: true` for maps whose axes share physical units. State any clipping in the chart description.

Finite numbers and `null` only. Nulls make line gaps or missing heatmap cells; isolated non-null line samples remain visible. On line/scatter logarithmic axes, nonpositive samples are omitted rather than assigned a fictitious positive floor. Provide positive coordinates for logarithmic heatmap axes. Keep plot-ready arrays compact; raw CFD files are not meant to be embedded in the web page.

Checks require `name`, `status`, `detail` strings. Downloads require `label` and a **relative public-safe** `href` such as `data/forces.csv`. URLs, root-absolute paths, parent traversal, encoded path tricks, backslashes, query strings and fragments are rejected. Parent integration must copy actual public-safe assets at those paths; a displayed link alone does not establish that its target exists.

All titles, metadata, descriptions, labels and paths are escaped before display. Controls do not edit the bundle. Tests inject schema fixtures into the browser's in-memory fetch boundary only; they never create or overwrite `site/data/results.json`.

## Verification

```sh
python -m pytest tests/test_cylinder_site.py -q
node --check site/app.mjs
node --check site/physics.mjs
node --check site/plots.mjs
node --check site/data.mjs
```

Tests require Python/pytest, a Node runtime with built-in `WebSocket` (verified with Node 24), and installed `google-chrome` or `chromium`. These are test dependencies only. No npm install is needed.

The browser tests launch an isolated headless Chromium profile inside `.cache/cylinder-site-verification/`, serve the real site on an ephemeral localhost port, use CDP from Node's built-in WebSocket, and terminate their own browser/server on exit. They do not modify global browser settings, copy a personal profile, or control another running browser. The named Hermes browser attempt (`cylinder-site`) was blocked by a real-profile SQLite lock; this fresh-profile path supplied the actual acceptance checks.

Verified layers:

- Numerical directivity identities, retarded/range scaling, force differentiation, compact/distributed phase cancellation, coherent/independent powers, alias folding, full-record boxcar Parseval, zero signal and underwater pressure-reference conversions.
- All 13 analytical SVG figures rendered; source/frequency/range/angle, near term, coherence, separation, sampling, signal type, segment length and units produced real calculated feedback changes.
- Native keyboard ArrowRight changed observer angle. Reduced motion paused initially; explicit play advanced field values and pause held the exact SVG unchanged.
- Actual 390 px rendering of all sections had no horizontal overflow. Desktop source lab, mobile polar chart and mobile PSD screenshots were visually inspected.
- Data boundary covered 404, network errors, bad JSON, malformed arrays, unsafe downloads, mixed analytical/computed origins, all four chart types, metadata escaping, coverage disclosure, and independence from analytical control changes. Browser fixture output remained visibly test-only.
- Runtime exception list remained empty. An expected missing `results.json` is an honest 404, not an application exception.
- Text contrast against warm paper / plot surface: ink 12.66 / 13.67, muted 5.58 / 6.03, teal 5.63 / 6.08, rust 4.55 / 4.92. These are computed WCAG contrast ratios.

Acceptance captures (local only, not public assets):

- `.cache/cylinder-site-verification/desktop-initial.png`
- `.cache/cylinder-site-verification/desktop-source-lab.png`
- `.cache/cylinder-site-verification/mobile-directivity.png`
- `.cache/cylinder-site-verification/mobile-spectrum.png`
- `.cache/cylinder-site-verification/test-only-results-adapter.png`

Design self-audit: Learn-first composition, editorial serif headings, restrained navy/teal/rust on warm paper, real field plots, sidebar navigation; no feature tiles, fake metrics, generic marketing cards, glass, or decorative gradients. The only gradient is the meaningful pressure color key. No compositional slop flags remained in the inspected surfaces.

## Interpretation and public integration

Native circular-domain data have been integrated, and the public CSVs/run-summary reproduce all 39 full-resolution analysis charts and pressure statistics exactly. Long displayed time traces retain local extrema; displayed frequency plots use 0–20 Hz. Full-band arrays and raw CSVs are downloadable. Numerical metadata is expandable, and analytical controls do not rebuild or alter computed evidence. The 390 px checks include the real result bundle, long provenance labels and all three native images. These implementation checks do not establish LES convergence, stationarity or experimental acoustic accuracy.
