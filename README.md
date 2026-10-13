# FlowAcoustics — a cylinder acoustic field guide

**[Open the interactive demonstration](https://nishchayoxford.github.io/flowacoustics/)**

A working educational demonstration of **3D cylinder WALE LES, Curle and FW-H**, using OpenFOAM v2206 and the unmodified OpenFOAM-related libAcoustics library. This is a demonstration, not an experimentally validated noise-prediction model.

## What is here

- A finite closed cylinder in a **circular outer domain**, with fluid beyond both genuine end caps.
- **138,240 hexahedral cells**; native topology/geometry checks passed.
- A completed four-rank solve ending at **5.999717956 s**: 5,734 force samples and 57,340 native acoustic observer rows.
- Actual midspan/3D mesh views and native velocity/vorticity sampling.
- **13 interactive analytical figures**, plus **35 computed charts and four additional analytical reference charts**.
- Recorded force and pressure CSVs, full-resolution analysis, explicit units, clocks, source conventions and limitations.

The lessons cover monopoles, dipoles and illustrative quadrupoles; loading versus thickness; retardation; near/radiating terms; source coherence; spectra and spectrograms; directivity; and pressure/SPL units. Analytical controls never modify the precomputed CFD.

## Main physical lesson

The later force record has a lift-spectrum candidate around 1.45 Hz (St ≈ 0.145), but the Welch bin spacing is about 0.72 Hz. It is an oscillatory developing-flow example, not a precise stationary shedding benchmark. At a 1 m water receiver, kr is approximately 0.006 at that candidate frequency: distance from the body alone does not establish acoustic far-field conditions.

Native Curle and FW-H are **compared, not added**. Their signs, time conventions and traction content differ in the inspected implementation; the comparison preserves those differences rather than fitting agreement. Surface-only FW-H does not compute the omitted volume quadrupoles.

**Reading the native files:** the FW-H object's inherited force/moment files are unused zero diagnostics; Curle's inherited load files lag by one timestep. The website uses the separate, correctly timestamped `forces.csv`. Both methods produced nonzero acoustic pressure. See [output diagnosis, accuracy limits and sample viewing](docs/OUTPUTS_AND_LIMITS.md). The vorticity figure now displays the full circular domain, not a rectangular crop.

## Use and reproduce

```sh
uv sync --locked --extra dev
.venv/bin/python -m pytest tests -q
python3 -m http.server 8513 --directory site
```

Python tests require Python 3.12, and browser tests additionally require an installed Chromium browser and Node with built-in WebSocket support. Sample-converter tests additionally need ParaView (`/usr/bin/pvpython` in the tested Linux setup) and Poppler's `pdfimages`/`pdftotext`. Native OpenFOAM is not launched by the tests or the website.

See [reproduction instructions](docs/REPRODUCE.md), [acoustic definitions](docs/CYLINDER_ACOUSTICS.md), and [site model documentation](docs/CYLINDER_SITE.md).

### Layout

- `site/`: dependency-free GitHub Pages site and compact display data.
- `site/downloads/`: actual recorded CSVs, a public run summary, and gzip-compressed full analysis/aligned CSVs.
- `cfd/cylinder_circular/`: accepted case inputs, with explicit pilot and main control dictionaries. Generated mesh and large raw flow fields are omitted.
- `scripts/`, `src/`, `tests/`: mesh/native helpers, independent acoustic analysis and exercised tests.
- `third_party/libAcoustics-v2212/`: selective pinned upstream source, original notices and GPL text.

The public copy excludes credentials, local configuration, downloaded papers, personal browser data, compiled binaries, large raw CFD fields and unrelated historical workflows. See [third-party provenance](third_party/libAcoustics-v2212/PROVENANCE.md).
