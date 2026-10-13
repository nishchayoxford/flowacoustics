# Reading the cylinder outputs

## Accuracy

The accepted 138,240-cell WALE calculation is a real three-dimensional demonstration, not a validated LES or noise benchmark. It ran to 5.999717956 s. Mesh quality, solver convergence and software tests do not establish mesh/time/domain independence, near-wall resolution, statistical development or experimental accuracy. The native Curle derivative also retains a fixed-step formula despite adaptive timesteps; that implementation error has not been quantified.

## Both acoustic calculations executed

OpenFOAM **pimpleFoam v2206** loaded **libAcoustics.so** and executed CurleAnalogy and FfowcsWilliamsHawkings/Farassat1A on the closed cylinder surface. The public `site/downloads/observers.csv` contains 57,340 pressure values: 5,734 times, five observers and two methods. Curle has 28,670 nonzero values; FW-H has 28,658, with twelve startup/restart buffer zeros. Native raw pressures and the CSV agree exactly after restoring the continuation clock by +0.1 s. That shift is already included in the public CSV.

### Why are FW-H forces and moments zero?

`AcousticAnalogy` inherits OpenFOAM `forces`. In the executed v2206 implementation, `forces::execute()` calculates loads while `forces::write()` only writes accumulators. FW-H overrides `execute()` with a no-op. Its acoustic `correct()` independently samples surface pressure and calculates observer pressure, without updating the inherited force accumulators. Thus the auxiliary FW-H `force.dat` and `moment.dat` remain zero. They are not evidence of zero physical force or zero acoustic pressure.

Curle's inherited force/moment files are also misleading: the base writer runs before its current force update, so they lag by exactly one solver timestep. Its pressure calculation happens after that update. **Use the independent OpenFOAM `forces` object or the public `forces.csv`, not either acoustic object's auxiliary load files.** The website uses the correct independent export. Original native records and library code have not been silently rewritten.

Source references in the included upstream tree:
- `AcousticAnalogy/AcousticAnalogy.H:23–26`: forces inheritance.
- `AcousticAnalogy/AcousticAnalogy.C:234–283`: inherited force writer before acoustic correction/output.
- `FfowcsWilliamsHawkings/FfowcsWilliamsHawkings.C:276–380`: no-op execute and independent surface calculation.
- `CurleAnalogy/CurleAnalogy.C:143–164`: current force and acoustic calculation.

Paths above are relative to `third_party/libAcoustics-v2212/upstream/Sources/lib/analogies/`. The directory label is not the solver version.

## Which source lessons belong to this CFD?

- **Monopole:** the fixed rigid impermeable cylinder has zero moving-surface thickness source. The pulsating-sphere illustration is an independent analytical example.
- **Dipole/loading:** actual CFD traction drives Curle, and actual wall pressure drives the executed FW-H surface-loading calculation. Their force signs, viscous content and propagation clocks differ; compare them, never add them.
- **Quadrupole:** no exterior turbulent volume integral was evaluated. The ideal four-lobe plot is not a CFD quadrupole decomposition, and an omitted term is not demonstrated to be zero.

The 13 interactive figures and four analytical reference charts are separate from the 35 computed charts.

## Circular vorticity display

The website now shows the full 0.24 m-radius circular domain, with equal geometric scales, at the same recorded time of 5.6 s. Midspan is z=0; the second plane is z=0.035 m, beyond the cap, so it has no solid disk. Values are native spanwise vorticity, displayed as explicitly sample-based linear contours. Solid-intersecting triangles are masked before interpolation; unsampled support remains blank. PNG and vector PDF are available on the page. This display correction did not rerun CFD or change pressure/force histories.

## Raw sample conversion

OpenFOAM `.raw` files here are ASCII coordinate/value tables, not images. `scripts/view_cylinder_exports.py` provides lossless CSV/VTP conversion and PVD time-series collections. CSVs retain units in their headers. VTPs are sample-point datasets, not native volume meshes; wall face-centre pressure is explicitly retained as sampled data. Kinematic wall pressure remains m²/s², not Pa.

The complete local archive contains 715 raw exports (334 circular and 381 historical rectangular), converted to 715 CSVs, 715 VTPs and 10 PVDs. Every value and all 715 PVD time entries were checked using installed ParaView/VTK with zero numerical difference. The large original/converted archive is not bundled into this public repository.

The converter's `build` command is archive-specific: it requires the complete original local case tree and its 403-pin run manifest, which are not the public summary JSON. It will intentionally reject this lightweight public checkout alone. Its generic `convert_tree(cfd, output, allowed_root=...)` API supports independently supplied raw sample trees. Tests use synthetic isolated files. Converter tests additionally require ParaView (`/usr/bin/pvpython` on the tested Linux setup) and Poppler's `pdfimages`/`pdftotext`.

In ParaView open a converted `vorticity_midspan.pvd`, click Apply, choose Points/Point Gaussian, color by vorticity component Z, and view down Z. Use time controls to inspect the saved sequence; do not apply unmasked Delaunay across the body.
