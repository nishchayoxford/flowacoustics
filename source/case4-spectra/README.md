# Case 4 spectral representation supplement

This additive supplement derives representations of the **existing single-phase strict-2D WALE / stationary-permeable-contour result** (`mrl2d_les_permeable`). It does not run CFD, repeat the acoustic-source reduction, change the original coefficients, or replace the sealed `source/mrl2d` snapshot.

## Definitions

For positive retained bins with coherent periodic-Hann coefficients `c`:

- coherent bin RMS: `sqrt(2) * abs(c)`, Pa;
- equivalent noise bandwidth: `ENBW = fs * sum(w*w) / sum(w)**2`, Hz;
- one-sided propagated density: `2 * abs(c)**2 / ENBW`, Pa²/Hz;
- band level: `20*log10(original_band_rms / 1e-6)`, dB re 1 µPa.

The 4,000-sample window is 2.002–10 s; the 196 retained bins cover 0.625–25 Hz. Bin spacing is 0.125 Hz and ENBW is 0.1875 Hz. There is no second taper, Welch averaging, frequency interpolation or new bandwidth claim. Integrating this selected-band PSD gives the coherent synthesis mean square multiplied by `df/ENBW = 2/3`, not the original coherent band RMS squared. The original band RMS is preserved for directivity.

Largest sampled bins are identified separately for all twelve receivers at 1 m radius. They are finite-record peaks, not proof of a blade-passing mechanism. The model is infinite-span 2D, not validated absolute finite-turbine noise. Neither stationarity nor far-field conditions are established by these plots.

## Reproduce without modifying the published results

From the public repository root, create a separate working layout. The two source copies have the same verified acoustic-kernel identity; that file is inspected for provenance, not executed by this spectral extension.

```sh
mkdir -p reproduction/src/flowacoustics
cp -R source/case4-spectra/scripts reproduction/scripts
cp -R source/case4-spectra/tests reproduction/tests
cp -R site reproduction/site
cp source/mrl2d/src/flowacoustics/acoustics2d.py reproduction/src/flowacoustics/acoustics2d.py
python reproduction/scripts/prepare_case4_spectra.py --root reproduction --output reproduction/regenerated
python -m pytest -c source/case4-spectra/pytest.ini reproduction/tests -q
```

Use a fresh `reproduction` directory. The calculation and plotting require Python 3.12, NumPy, SciPy and Matplotlib; tests also require pytest, Node.js, and `/usr/bin/google-chrome` for the isolated browser test. Original project constraints were NumPy >=1.26,<3, SciPy >=1.13,<2, Matplotlib >=3.8,<4 and pytest >=8,<10. The browser test uses structural fixtures only; native-data browser and download acceptance is a separate release check.

`regenerated/manifest.json` records output hashes and the exact input hashes. Compare it and the numerical tables against `site/data/case4-spectra/`. Plot byte identity also depends on the rendering environment; numerical comparison is the portable criterion.

The website is `site/case4-spectra.html`, linked from the original homepage. The download pack contains three vector-PDF pages, three PNG previews, all 2,352 receiver-frequency rows, a twelve-row peak table, a twelve-row directivity table and JSON provenance. Historical source archives and all pre-existing scientific outputs remain unchanged.
