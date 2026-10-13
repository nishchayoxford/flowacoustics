# Reading the outputs and their limits

## Current and historical evidence

The current case is a **459,264-cell**, eight-rank, fixed-step WALE demonstration ending at **10 s exactly**. Its preserved directory name is `cylinder_circular_60s`; it does not contain a 60 s calculation. It has 16,000 force rows and 160,000 native pressure values, with full fields reconstructed at 2, 4, 6, 8 and 10 s. Original large volume fields, mesh connectivity and raw/converted sample archives are not shipped here.

The 138,240-cell case in `cfd/cylinder_circular/` is **historical**, ending at 5.999717956 s with four ranks and adaptive stepping. Its restart offsets, counts and approximately 1.45 Hz coarse-bin candidate must not be transferred to the new data.

## What the short record establishes

The native solve, reconstruction and table exports completed. That is not proof of mesh/time/domain independence, resolved LES adequacy, near-wall resolution, stationarity or experimental acoustic accuracy. The 2–10 s primary lift spectrum peaks in its lowest positive bin (0.5 Hz), so the shedding peak is **unresolved**, not a measured stable tone or prescribed Strouhal value. Main analysis status remains `partial`.

Native Curle uses a fixed-step derivative stencil. The old adaptive-step mismatch is avoided by the current fixed cadence; matching the stencil's assumption **does not prove time convergence** or make startup derivatives reliable. Receiver support and derivative startup exclusions still matter. Never synthesize a missing received tail.

## Auxiliary force files are not the acoustic pressure

In the inspected v2206/libAcoustics coupling, the FW-H object's inherited force/moment accumulators are unused and remain zero: its acoustic calculation samples pressure separately. Curle's inherited load files lag the independent load by one solver step because the writer precedes its load update. Neither is the authoritative force history. Use the independent OpenFOAM `forces` object, exported as **`forces.csv`**, and use `observers.csv` for actual acoustic pressures. Native startup buffer zeros are not measurements of silence.

Upstream references, relative to `third_party/libAcoustics-v2212/upstream/Sources/lib/analogies/`: `AcousticAnalogy/AcousticAnalogy.H` (forces inheritance), `AcousticAnalogy/AcousticAnalogy.C` (writer order), `CurleAnalogy/CurleAnalogy.C` (load update), and `FfowcsWilliamsHawkings/FfowcsWilliamsHawkings.C` (independent acoustic correction).

## Physical boundaries

- The fixed impermeable closed body has no moving-surface thickness source. Analytical monopoles remain separate lessons.
- Native Curle uses on-body **total** traction at unretarded source time; native FW-H uses retarded **pressure-only** loading with outward normals. Do not fit away the sign/clock/content differences and never add the two predictions.
- The main independent compact model uses total exported force; the signature report's offline model uses pressure force only, converted to force on fluid by sign reversal.
- No exterior turbulent-volume quadrupole integral is evaluated. Illustrative quadrupole plots do not establish that the omitted contribution is zero.
- Wall pressure is hydrodynamic pressure, not automatically radiated sound. At low water frequencies even a receiver far from the body geometrically can be in the acoustic near field.

## Sample viewing and figure formats

OpenFOAM `.raw` sampled files are coordinate/value text, not images or full CFD meshes. `scripts/view_cylinder_exports.py` supports CSV/VTP conversion and PVD time collections. VTPs represent sample points; they do not invent volume connectivity. Kinematic wall pressure stays in m²/s² unless explicitly converted to Pa using density once. Reconstructed native volumes require a separate OpenFOAM reader.

The converter's archive-building route needs its original case tree and identity manifests and intentionally cannot run from this lightweight checkout alone. For independent trees use its explicit generic conversion API with a selected case name; inspect the API and CLI help before selecting a case.

The completed production archive contains **580 sampled raw files**, converted to **580 CSVs, 580 VTPs and five chronological PVD collections**, covering 4,447,080 rows. Every CSV, VTP and collection timestep was read back with the installed readers; coordinate/component values match the original raw-text binary64 parse exactly. A separate regeneration reproduced all 1,170 bundle files byte-for-byte. The large converted archive stays local, rather than being embedded in this lightweight repository. Its four flow collections cover 0.5–10 s; wall pressure covers 0.02–10 s.

The actual reconstructed volumes were read in ParaView at all five retained times: 459,264 cells / 473,514 points internally and 8,064 cylinder-wall faces / 8,066 points. All five fields are finite. This reader exposes float32 arrays, so compatibility is not a claim of bitwise binary64 field equality. At 10 s, wall yPlus spans approximately 1.015–16.091; this does not establish wall-resolved LES adequacy. Internal zero yPlus is not a wall statistic.

In ParaView, open a converted vorticity PVD, Apply, use Points/Point Gaussian and choose vorticity component Z; view down Z and use the time controls. Do not apply unmasked Delaunay across the cylinder. Flow contours are sample-based display interpolation with solid/support masks, not invented native connectivity.

The signature gallery supplies PNG, SVG, PDF, CSV and sanitized JSON. **I05's spectrogram PDF is mixed vector/raster**: four QuadMesh layers (fields/colorbars) at 160 dpi; text/axes remain vector. The other eight signature PDFs are vector. Rasterization changes rendering, not scientific arrays.
