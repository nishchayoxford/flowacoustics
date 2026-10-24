# Four-case 2D turbine: full data

All four native cases completed 10 s with 26,751 fluid cells. SST URANS and
2D WALE each have on-body and stationary permeable recording alternatives.
WALE here is not resolved 3D LES. Body and envelope estimates are never summed.

Each case supplies the exact original NPZ and four CSVs: source loading,
12 receiver histories, 12 spectra and sparse 12-direction directivity.
The complete chart bundle is ../results.json. The direction control changes
only the displayed receiver/spectrum traces, not these downloads.
Source history includes startup (0.002 to 10 s); acoustic analysis retains
2.0020000000000002 to 10 s at 0.002 s cadence, with an 8 s synthesis period.
Selected frequencies are 0.625 to 25 Hz. Acoustic pressure is Pa; loading is N/m.
Spectra are coherent Hann-window bin RMS, not PSD. This window-conditioned
band synthesis is not full-band sound or a causal arrival record.

The native movies show gauge pressure and speed, not acoustic waves. Each
contains 500 frames at 50 fps, 1920 x 1080, lasting 10 s; source times are
0.02 to 10 s. Shared body/permeable movies reflect byte-identical field data
within each flow model, not a substitute calculation. The mesh image uses
executed connectivity; source contours use native recorded panels at 5 s.
The full native field-history CSV is ../../assets/mrl2d/field-history.csv.

verification.json records independent implementation-consistency checks:
9,408 complex coefficients, 192,000 pressure values, 16 CSVs and 108 chart
traces. It is not physics validation. Stationarity and acoustic bandwidth
remain unverified; no mesh/time convergence or experimental validation is
claimed. The unused original comparisons field is omitted because its
physical_interpretation_allowed flags denote matched support only.
The public directivity wording now matches unconnected sparse markers.
All numeric arrays, bins, windows and original numerical files are unchanged.

public-manifest.json enumerates every public data/media payload except itself,
with byte counts and SHA-256. Original private analysis JSON is not included.
This handover adds no license grant.
