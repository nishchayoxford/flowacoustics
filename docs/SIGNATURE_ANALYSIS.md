# Nine-figure short-record signature analysis

The signature workflow reduces the saved 10 s cylinder record; it does not run CFD. Its primary window is **2–10 s**, fixed native cadence **0.000625 s**, periodic Hann Welch with **3,200 samples**, 50% overlap and **0.5 Hz** bins. There is no new resampling or restart offset. The primary lift peak is unresolved and stationarity is not established.

| ID | Content and boundary |
|---|---|
| I01 | Actual mesh connectivity, saved vorticity samples and receiver geometry; display interpolation is masked at the body. |
| I02 | Record support, startup context and selected analysis window. |
| I03 | Force spectra with window/segment sensitivity; no prescribed shedding tone. |
| I04 | Receiver spectra and declared compact pressure-only comparison. |
| I05 | Force/pressure spectrograms; finite-window time-frequency resolution. |
| I06 | Band levels and blockwise variability; not uncertainty from independent realizations. |
| I07 | Force/pressure coherence and gated wrapped phase; not unique source identification. |
| I08 | Range/near–radiating comparisons with supported clocks and actual receiver geometry. |
| I09 | Four native 1 m angular samples versus a dense **offline** compact pressure-force sweep, with separately scaled radiating detail. |

The independent compact signature model uses **pressure force on fluid = minus exported pressure force on body**. It differs intentionally from the main explorer's **total-force** compact calculation. Native Curle itself uses total on-body force without retardation; native FW-H uses retarded wall-pressure loading. Do not sum the analogies or fit their signs, gains or clocks to force agreement.

Band edges are 0, 1, 3, 10 and 20 Hz. A common validated cadence assigns bins consistently across time blocks; interior bands are half-open and the last includes its upper edge. Parseval checks are against weighted/detrended segment energy, not an assumed equality with global variance. Interpolation provenance distinguishes upstream resampling, retarded-time interpolation and local spectral operations.

## Rebuild only with the original archive

`scripts/build_cylinder_signature.py` needs the original run, force and observer files plus original case connectivity/raw-plane inputs and their identity pins. The public compact downloads deliberately cannot impersonate that complete archive. For an independently retained matching archive, use explicit paths:

```sh
.venv/bin/python scripts/build_cylinder_signature.py \
  --run results/cylinder/native_circular_60s/run.json \
  --forces results/cylinder/native_circular_60s/forces.csv \
  --observers results/cylinder/native_circular_60s/observers.csv \
  --case cfd/cylinder_circular_60s \
  --output results/cylinder/signature/reproduced \
  --start-time-s 2 --end-time-s 10 --nperseg 3200
```

This is an **archive-dependent** command, not a fresh-clone promise. It requires a completed, reconstructed schema-2 record and an absent output directory. Schema-1 historical smoke is separately gated and is not eligible for publication as the refined record. Synthetic tests remain labelled test-only.

The report emits PNG/SVG/PDF figures, numeric CSV tables and local provenance manifests. The public preparation adapter verifies the run/asset identities, keeps safe relative download paths and emits allowlisted JSON rather than copying machine-bound source maps. Use its help and a fresh staging destination; do not copy raw `run_metadata.json` or unsanitized local manifests into `site/`.

**PDF format:** I05 has vector text/axes and four rasterized spectrogram/colorbar layers at 160 dpi to avoid rendering seams. The other eight are vector PDFs. No scientific arrays are changed for this rendering choice.

## Design context, not physical validation

Zhu et al. (2026), [DOI 10.1016/j.oceaneng.2026.128155](https://doi.org/10.1016/j.oceaneng.2026.128155), informed presentation/diagnostic organization only. No turbine or Amiet physics is adopted, and its paper is not redistributed.

The public [Applied Scientist – Acoustic Signatures – Sonardyne](https://www.sonardyne.com/job-opening/applied-scientist-acoustic-signatures-blackbushe) description supplies application context for explicit units, spectra, provenance and limits. This demonstration makes no hiring, endorsement, operational performance or validated detection claim.
