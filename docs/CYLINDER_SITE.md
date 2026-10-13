# Static cylinder field guide

`site/` is a dependency-free browser application: local ES modules/CSS, no build step, CDN, solver launcher, remote compute or audio. Serve it over HTTP; `file://` is unsuitable for module/fetch loading. GitHub Pages deploys the contents of `site/` with `.nojekyll`.

```sh
python3 -m http.server 8513 --bind 127.0.0.1 --directory site
```

## Three distinct layers

1. **Analytical lessons:** 13 interactive figures about source families, retardation, near/radiating terms, coherence, sampling, Welch spectra and underwater pressure units. Controls change only these prescribed examples.
2. **Computed results:** the native-data explorer reads `data/results.json`, with 35 computed and four additional analytical reference charts. The refined source record is 0–10 s; the primary analysis selects 2–10 s. Its `partial` status and unresolved lift peak must remain visible.
3. **Signature report:** nine run-bound figures and downloads, described in [SIGNATURE_ANALYSIS.md](SIGNATURE_ANALYSIS.md). This is static analysis of the saved record, not extra CFD or an acoustic-source decomposition.

Actual mesh previews must come from retained connectivity and bind to the refined run. A rectangular analytical plotting window is not the CFD boundary. Flow views use real saved samples and disclose interpolation/masks. Full-volume and converted raw archives remain local.

## Data and identity contract

The main JSON schema is version 1 with `status`, `summary`, `metadata`, `charts`, `checks`, `downloads`. Preserve supplied evidence status. Missing data produce a visible pending/unavailable state, not a generated substitute. Chart origin is independently `computed` or `analytical`; units, time support and estimator coverage accompany plots. Numerical arrays permit finite values and `null`, never NaN/Infinity. Null values remain gaps; log plots must not turn zero into fictitious positive energy.

Downloads use safe relative public paths; no absolute filesystem paths, parent traversal, query tricks or private source-key maps. The signature index is accepted only when its original native-run SHA256 agrees with the main results' native identity. The public schema-1 summary has different bytes from the original schema-2 manifest, so preserve both identities and their roles rather than equating their digests. Do not publish historical smoke/test fixtures as refined evidence.

Long display traces may preserve local extrema and spectra may show a bounded frequency range. These presentation reductions do not replace full-resolution downloadable arrays. Analytical controls must never mutate computed data.

## Scientific and UI checks

`physics.mjs` contains independent lesson equations. Its illustrative quadrupole angular factor is not a native volume-source calculation. Welch lessons use actual DFT samples, periodic Hann windows, one-sided density normalization and arithmetic averaging in linear power. Underwater levels use `p_ref = 1e-6 Pa`; PSD density levels are not band SPL.

After integrating the exact new site bundle, run the site, publication, plot-correction and signature tests, plus `node --check` for each module. Browser tests use isolated Chromium profiles and ephemeral local servers; check desktop and 390 px layouts, keyboard controls, reduced motion, run-identity rejection, missing-data states and every download. Optional historical archive tests are explicitly separate from current-site validation. Source preparation alone is not proof of an updated live deployment.
