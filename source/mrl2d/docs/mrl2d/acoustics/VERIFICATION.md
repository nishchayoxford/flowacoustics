# Copied-package verification — software, not new native evidence

Three final, disjoint suites ran from the isolated copy using its explicit
pytest config, source pins and import-root guards. JUnit cases were parsed and
deduplicated; the first two suites reproduce the complete 169-test selection
from public-source-001 without deleting tests or changing their expectations.

| Suite | Passed | Skipped | Failed/errors |
|---|---:|---:|---:|
| Safe default (`pytest.ini`) | 69 | 5 | 0 |
| Explicit contract nodes (`tests/unit-contract-nodes.txt`) | 93 | 2 | 0 |
| Portability and production-policy preparation | 11 | 0 | 0 |
| **Distinct total** | **173** | **7** | **0** |

All seven skips are genuine absence-based archive integration checks: five
historical diagnostic/plane checks, native mesh-motion evidence and a bounded
run report. Missing native evidence was not replaced with fixtures. All acoustic
core tests are unchanged. No numerical tolerance or historical pin was relaxed.

The default covers signed monopole/dipole fields, moving-disk harmonics,
independent moving-loading quadrature, series bounds, units, frame conditioning,
Parseval, malformed data and chunked reduction. The contract suite includes
source readback/motion, mesh-log/AMI gates and one full **synthetic** completion
positive control. Its labelled synthetic 160000-step logs are not native CFD.

The portability tests exercise missing/nonexistent environment locators,
validation before mesh log creation, source failure preventing a harmless
sentinel command, and successful shell fixtures with spaces/metacharacters and
one-thread controls. They compile only the relevant helper/prefix for fixture
execution; the original autouse native/owner-lock prohibition remains intact.
No actual OpenFOAM bashrc, native command or supervisor is launched by them.

The production-policy emitter had an observed missing-feature RED followed by
GREEN. Its real CLI emits exactly the four-literal candidate policy; it rejects
existing/linked destinations and missing/unsupported actions. Additional
verification-only coverage was not mislabelled RED. Historical locator and
source-failure RED receipts and the approval timeout remain in the private audit.

A broader original completion-file run reached its 240-second timeout during
public-source-001 work. That incomplete result is preserved and was not retried
or counted. Other native/lifecycle and full synthetic-pipeline tests are not
represented as verified here. This does not invalidate the scoped unit result.

Ruff **lint passes for all source/scripts/tests**. The full format check retains
one advisory: the pre-existing pin-metadata dictionary in protected
`tests/conftest.py`. That entire file was left unchanged on resumption. A scoped
format check excluding it passes for the other **30 files**; the full nonzero
receipt is retained, not described as a full formatting pass.

All ten module imports resolve to the copy and satisfy their declared hashes.
All three original CLI help routes and the new emitter help route completed.
The 48 retained initial/control files reproduce byte-for-byte with depth 0.01,
dt 0.0000625, ranks 2 and end 10; their original executed files also match.
Across all runtime modules, AST and source-segment comparisons prove every
definition and all bytes outside `native_command` and `_native_stage` unchanged.
The exact candidate mesh mathematics are identical to retained production-policy
source; its different public whole-file hash is declared, not repinned as the
historical original. No mesh connectivity was regenerated.

Environment: Python 3.12.3; NumPy 2.5.3; SciPy 1.18.1; Matplotlib 3.11.2;
pytest 9.1.1; Ruff 0.16.10. Those dependency versions match the supplied lock.
Lock installation was not replayed. Each test subprocess has numerical thread
variables set to one and a Linux 4 GiB address-space cap. Suite wall caps were
120, 120 and 60 seconds respectively.

Local packaging checks are complete. Remaining boundaries are explicit: no
native replay or Gmsh binary identity, no full completion-suite pass, no
additional reuse permission, and no publication authorization or site acceptance
from this source task. Those are not claimed by ordinary software verification.
