# libAcoustics — pinned selective source export

- Upstream: https://github.com/unicfdlab/libAcoustics
- Exact commit: `b9cd1d62ad8c5b445240a506a945231d12454b73` (v2212 label).
- Commit tree: `e2970b2c07b30ee67f90892a9559cdf08dfa8b4b`.
- `upstream/` contains 39 unmodified selected upstream files, not a complete clone. It includes the library C++/headers/Make files, original README/scripts and selected analytical test source. It does not include upstream tutorial geometry, meshes or complete test harnesses.
- Selected source files were originally checked against the pinned Git blob identities. The native library used for this demonstration was built from unchanged numerical source on OpenFOAM v2206.
- Original copyright and licence notices remain intact. Several numerical-source headers specify GPL version 3 or later; the GPLv3 text is supplied in `COPYING.GPL-3`. Files without individual notices are not assigned an invented licence by this project. No blanket MIT licence is applied to third-party code.
- Preserve the software citation request in `upstream/README.md`: Epikhin et al., DOI https://doi.org/10.1016/j.procs.2015.11.018.
- No compiled library is distributed. See `../../docs/REPRODUCE.md` for the separate-copy build procedure and dependencies.
