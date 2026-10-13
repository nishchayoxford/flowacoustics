# Refined circular-cylinder inputs — 10 s, not 60 s

The preserved `cylinder_circular_60s` identifier predates the shortened execution. These are the accepted **fresh 0–10 s** input dictionaries: **459,264 cells**, **eight MPI ranks**, fixed **Δt = 0.000625 s**, no restart or adaptive stepping. The native run used OpenCFD OpenFOAM v2206 and unmodified libAcoustics.

Only `0/` initial conditions, two constant-property dictionaries and `system/` dictionaries are shipped. Generated `constant/polyMesh`, positive-time fields, processor directories, `postProcessing` and `acousticData` are intentionally absent. The mesh is generated from the included `blockMeshDict`; a new `checkMesh -allTopology -allGeometry` must pass before any new solve.

Forces and both native acoustic methods sample every solver step. Wall pressure samples every 0.02 s; flow samples every 0.5 s; full fields are retained every 2 s in binary **uncompressed** form. The completed original run reconstructed 2, 4, 6, 8 and 10 s; these volumes remain local.

See [reproduction instructions](../../docs/REPRODUCE.md). New mesh/solver work requires new resource permission; the historical `prepared.json` is machine-bound and is not included. Do not invoke `cylinder_longrun.py` as if it were a preflight-free launcher or assume its default timestep is the executed value.
