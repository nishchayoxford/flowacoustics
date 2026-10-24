# Public interface to the unchanged offline exporter

The exporter is `scripts/postprocess_mrl2d.py`; its implementation and completion
adapter are copied unchanged. This document is a packaging guide, not a new
native acceptance receipt. See the package README and NATIVE_REPLAY.md.

## Input

The JSON schema is `mrl2d-postprocess-input-v1`, with `cases` containing exactly
`mrl2d_urans_body`, `mrl2d_urans_permeable`, `mrl2d_les_body` and
`mrl2d_les_permeable`. Each entry provides:

- `id`, original archive `root`, `end_s: 10`, `cadence_s: 0.002`;
- `envelope_recorded`: the actual native recorder setting (false for body-only
  cases, true for the permeable cases);
- `source_sha256`: nonempty expected native rank-file digest map, with paths
  relative to that archive, retained from the authentic original run;
- `native_completion`: schema `mrl2d-direct-native-completion-v2` and trusted
  expected `input_sha256` / `run_sha256` of the original `mrl2d-input.json` /
  `mrl2d-run.json`. Do not initialize trust by blessing current edited files.

No example pretends to have these unavailable archives or supplies dummy native
hashes. The initial-input examples are not completion evidence. Native execution
reports can contain private absolute paths; they are deliberately excluded from
this source handover. Reduced public metadata and original execution identity
are separate artifacts, never a rewritten original report.

## Gates and output

Production requires authentic input/run pins, all required native stages, exact
executed dictionary compatibility, complete 160000-step solver/post-solve/AMI
coverage, source history and field identity, and actual VTK reads of 500 planes.
There are 5000 source frames. Solver binary64 roundoff tolerance is 1e-9 s;
Float32 VTK TimeValue tolerance is separately 5e-7 s. Do not conflate the two.
The direct-native adapter declares reconstruction **not_required**, not completed.
All validation precedes creation of the new result directory.

Write only a fresh child of `.cache/mrl2d/postprocessing/` in this copy. Native
case roots, site files and existing results are never output destinations.
Traversal, symlinked output ancestors and existing targets are rejected. The
same-source guard tests are exercised from this package, not a parent checkout.

A complete numerical export contains a schema-v2 results bundle, manifest and
per-case NPZ, analysis JSON, loading, receiver, spectrum and directivity CSVs.
NPZ complex coefficients and support metadata remain primary numerical records.
Loading is N/m; acoustic pressure is Pa. The frontend model is supplied only for
schema verification; the final webpage and assets are separate parent deliverables.
The publication decision requires the final native/independent verification and
an integrated site/data bundle; this source package does not make it.

The resource budget preserves every temporal sample and chunks only panels.
Do not replace it with temporal segmentation or material-panel demeaning.
No native acoustic reduction, native rendering or field conversion was run in
this package preparation. The default unit test skips absent historical native
integration archives explicitly while retaining all acoustic numerical controls.
