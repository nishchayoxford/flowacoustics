# Source and dependency licensing

- `acoustics/mrl2d/mrl2dRecorder.C` and `productionGuard.H` retain their original
  `SPDX-License-Identifier: GPL-3.0-or-later` notices. The accompanying GPL text
  is in `GPL-3.0-or-later.txt` (copied from the installed standard license text).
- No project-wide license was present in the inspected original tree. No
  license grant for the Python source, tests, docs, initial inputs or frontend
  model is invented here. **No additional reuse permission is granted.** This
  private packaging operation does not settle rights for a public release.
- OpenFOAM/OpenCFD source and libraries are external prerequisites, not bundled.
  Preserve their own GPL notices when building an instrumented solver.
- NumPy, SciPy, Matplotlib, pandas, pytest, Ruff, Node.js, VTK and Gmsh are
  separate third-party dependencies. Their packages supply their own licenses;
  this source handover does not relicense them. Gmsh/VTK are optional native
  mesh/field tooling and are not pinned by the existing `uv.lock`.
- Downloaded scientific papers, compiled libraries, object files and vendored
  third-party installations are not included.

This notice records available licensing information, not publication permission.
