# FlowAcoustics — cylinder acoustics

**[Interactive field guide](https://nishchayoxford.github.io/flowacoustics/)** · [Reproduce](docs/REPRODUCE.md) · [Methods](docs/CYLINDER_ACOUSTICS.md)

An educational demonstration of a finite, closed 3D cylinder in water: WALE LES, native Curle and surface FW-H, and independent acoustic analysis. OpenCFD OpenFOAM **v2206** loads unmodified libAcoustics. This is not an experimentally validated noise-prediction model.

## Current calculation

- Circular outer domain; **459,264 hexahedral cells**, below the strict 500,000-cell ceiling.
- A fresh **0–10 s** calculation on **eight MPI ranks**, with fixed **Δt = 0.000625 s** and no restart/spliced clock.
- **16,000 force records** and **160,000 native pressure values**: five observers, two alternative methods.
- Full fields reconstructed and read back in ParaView at **2, 4, 6, 8 and 10 s**; large fields remain local, not in this repository.
- All **580 sampled raw exports** have verified CSV/VTP conversions and five time collections; the website shows the actual full-circle flow at 10 s.
- Main analysis: **35 computed charts and four analytical reference charts**. The separate nine-figure signature workflow supplies spectra, variability, coupling and directivity; analytical lessons are not CFD evidence.

The primary **2–10 s** lift spectrum is largest at the first positive Welch bin, **0.5 Hz**. Its shedding peak is unresolved: no precise shedding frequency or new fixed Strouhal number is claimed. The analysis remains `partial` even though the native solve and reconstruction completed. A longer record, mesh/time/domain sensitivity and experimental comparison are still needed.

**Curle and FW-H are compared, never added.** They differ in signs, clocks and traction content. The main independent compact comparison uses **total force**; the signature report's offline compact model uses **pressure force only**. At low frequency in water, geometric distance alone does not establish acoustic far-field conditions.

## Use

```sh
uv sync --locked --extra dev
.venv/bin/python -m pytest tests -q
python3 -m http.server 8513 --bind 127.0.0.1 --directory site
```

Python 3.12 is required. Browser tests need Chromium and Node with built-in WebSocket support; sample/figure tests also use ParaView and Poppler. Optional historical/native-archive integration tests explicitly skip when their non-public inputs are absent. Tests and the website do not launch CFD. See [reproduction and test prerequisites](docs/REPRODUCE.md).

## Layout

- `site/`: static field guide, display data and compact downloadable evidence.
- `cfd/cylinder_circular_60s/`: **current 10 s inputs**; the old directory identifier is retained, not a 60 s result.
- `cfd/cylinder_circular/`: historical, coarser approximately 6 s inputs; not the current result.
- `scripts/`, `src/`, `tests/`: mesh/run helpers, analyses, renderers and regression tests.
- [Output interpretation](docs/OUTPUTS_AND_LIMITS.md), [signature guide](docs/SIGNATURE_ANALYSIS.md), [site contract](docs/CYLINDER_SITE.md).
- [Editable project workflow](docs/structure/flowacoustics-map.svg), with its [reading guide](docs/structure/README.md).
- `third_party/libAcoustics-v2212/`: selective pinned upstream source and original GPL notices; the directory label is not the solver version. See [provenance](third_party/libAcoustics-v2212/PROVENANCE.md).

No downloaded papers, credentials, personal configuration, compiled libraries, generated volume mesh/fields or converted raw archives are included. Re-running a solver is a new resource-consuming computation, not part of viewing or replaying the analysis.
