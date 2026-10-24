# Frozen native-history / 2D acoustics API v1

This is the sampler handoff. New implementation lives only in
`src/flowacoustics/acoustics2d.py`. No retired 3D modules are dependencies.

## Native arrays (SI; stable panel identity)

`ContourHistory(time, xy, normal, ds, p, U, surface_velocity, *, kind,
blade_id=None, rho0=1000., c0=1480., provenance=None)`:

- `time`: float64 `[Nt]`, physical seconds, strictly increasing uniform cadence;
  recommended 0.002 s. Do not silently interpolate adaptive output.
- `xy`: float64 `[Nt,Np,2]` moving or `[Np,2]` fixed, metres, global x/y.
- `normal`: same shape options as xy, unit normals **out of each solid into
  fluid** for body; **out of enclosed region** for permeable. OpenFOAM wall
  fluid-out `Sf` therefore needs negating. No front/back faces.
- `ds`: float64 `[Np]`, positive segment lengths in metres; rigid panels only.
  Native side-face area divided by the declared numerical extrusion depth
  **once**. The acoustic code never divides by depth or multiplies by span.
- `p`: float64 `[Nt,Np]`, gauge pressure relative to the declared ambient p0,
  **Pa** (native incompressible kinematic pressure times rho0 exactly once).
  Preserve material-panel mean loads on moving bodies. Never send abs(p),
  integrated forces, or already demeaned per-panel rotating loads.
- `U`: float64 `[Nt,Np,2]`, actual fluid velocity m/s, global inertial frame.
- `surface_velocity`: same `[Nt,Np,2]` or fixed `[Np,2]`, actual instantaneous
  global panel velocity m/s; zero for stationary permeable contour. Do not
  confuse this with U or the carrier's velocity; body orbit AND absolute
  half-rate pitch must be included.
- `kind`: exactly `"body"` or `"permeable"`.
- `blade_id`: optional int `[Np]`; body sampler should supply 0,1,2 for its
  three complete blade curves. Permeable can omit it. `Nblades` is metadata,
  not an acoustic gain: all sampled panels are integrated exactly once.
- `rho0`, `c0`: positive SI scalars. This v1 consumes incompressible constant
  density histories; no invented density fluctuations. Viscous traction and
  exterior volume quadrupoles are omitted and disclosed.
- `provenance`: optional JSON-serializable dict. Native sampler must retain
  case identity, native file paths/hash, pressure conversion, depth removed,
  normal convention, cadence, complete panel coverage, interpolation scheme,
  source type (`native_cfd` vs analytical verification), and model label.

Coordinates/normals can be stored full-time even if constant. Every time slice
must cover all panels in the same ordering. Stream/file packaging can be chosen
by the sampler; pass these arrays to Python without lossy rounding. A compact
NPZ containing the named arrays plus JSON metadata is an uncomplicated adapter;
this API does not prescribe a new OpenFOAM SDK or callback.

## Entry points

- `common_receivers(radius=1.) -> [12,2]`: angles 0,30,...330 degrees from +x.
- `source_terms(history) -> (Q[Nt,Np], L[Nt,Np,2])`.
- `contour_spectrum(history, receivers=None, *, fmax=25., fmin=0.,
  window="hann", method="auto", series_terms=6) -> Spectrum2D`.
  Positive DFT bins only, below Nyquist, on the complete supplied uniform
  record. Caller selects startup exclusion by slicing **all** time-dependent
  arrays together before construction. No automatic stationarity assertion.
- `Spectrum2D.frequency` Hz, `Spectrum2D.coefficient` complex `[Nf,No]` Pa,
  Fourier-series coefficients (a real bin-centred cosine of amplitude A has
  coefficient A/2), `Spectrum2D.receivers` m, `Spectrum2D.metadata` dict.
- `Spectrum2D.synthesize(time=None)` returns real `[Nt,No]` Pa, retaining ONLY
  the requested positive band; `Spectrum2D.band_rms` derives the periodic
  band RMS as `sqrt(2*sum(abs(coefficient)**2,axis=0))`.

## Fixed mathematical convention

NumPy-compatible transform `a_hat(omega)=integral a(t) exp(-i omega t) dt`;
reconstruction uses `exp(+i omega t)`. Coefficients use time relative to the
first retained native sample, recorded in metadata. Outgoing 2D kernel is
`G=-i H_0^(2)(kR)/4`, `(Laplacian+k^2)G=-delta_2`, `k=omega/c0`.
For moving panels the integrand is evaluated at `xy(t)` and includes
`i omega Q(t) G(x,xy(t)) + L(t) dot grad_y G(x,xy(t))`, integrated in both
time and contour length. It is NOT a stationary force formula.

DC is not evaluated. A constant Eulerian source contribution is removed before
windowing; material-frame mean rotating loads must not be removed. Window
normalization, effective temporal support and the exact mean-conditioning
rule are recorded by the implementation. Finite-record results are explicitly
periodic/window-conditioned; no causal first-arrival or history-flushing claim.
No automatic BPF or rigid-rotor period is assigned to articulated motion.

## Interpretation

This predicts an ideal span-invariant **2D free-space surface-only analogy**,
not finite-turbine absolute acoustics. Zero acoustic propagation mean flow is
assumed despite the CFD's 1 m/s inflow. A chosen 25–50 Hz synthesis is a
band-limited pressure, not the full sound waveform. Body and permeable are
alternative estimates; do not sum them or identify their difference with a
measured quadrupole. Analytical tests are not CFD; native coupling remains
pending until the parent executes and checks actual source histories.

Implementation details and checked equations are in [METHOD.md](METHOD.md),
with copied-package unit evidence in [VERIFICATION.md](VERIFICATION.md).
The package README gives the current four-case status; this numerical API
cannot itself establish native completion or final independent acceptance.

## Implemented algorithm choices

- `fixed`: source FFT then exact Hankel surface integration. Rejects moving xy.
- `direct`: exact sampled time-space Hankel integration at moving panel positions.
  Useful for small-array/reference checks, not required for every production bin.
- `series`: exact logarithmic Bessel expansion separated into time moments/FFTs,
  truncated to `series_terms` (1–12). Requires every requested `kR <= 1`.
  Metadata includes a conservative **absolute truncation-only Pa bound** for
  every bin/receiver. It is not a CFD error bar or floating-point roundoff bound.
- `auto`: fixed FFT for fixed xy; bounded series for low-kR moving sources;
  direct otherwise. For the intended 1 m receivers and <=50 Hz water band,
  the series is efficient without dropping time samples or evaluating unused
  high-frequency bins. See the actual full-size analytical benchmark.

Only `boxcar` and periodic `hann` windows are accepted. A positive requested band
must contain a DFT bin and lie strictly below Nyquist. The code applies no extra
amplitude calibration, span factor, source-period choice or blade-count multiplier.
`kind` labels the boundary but uses the same general permeable Q/L expressions:
body simplification follows from the recorded `U_n=v_n`, not from overwriting U.

`metadata` retains the supplied provenance, sample/panel counts, source support,
nominal period, requested/actual band, integer bins, band-edge roundoff tolerance,
window gains, rho/c0, omitted terms and signal interpretation. Its
`native_coupling_verified=False` is deliberate: the numerical API cannot verify
native-file identity, curve closure or callback coverage. The parent must record
those checks in a separate coupled-run report before publishing native results.

## Minimal adapter call (real arrays only)

```python
from flowacoustics.acoustics2d import ContourHistory, contour_spectrum

# `record` below is the parent's real native-reader output, not generated here.
# Apply the same explicit startup slice to every time-dependent array beforehand.
history = ContourHistory(
    time=record["time"], xy=record["xy"], normal=record["normal"],
    ds=record["ds"], p=record["p"], U=record["U"],
    surface_velocity=record["surface_velocity"],
    kind=kind, blade_id=record.get("blade_id"),
    rho0=1000., c0=1480., provenance=verified_source_provenance,
)
spectrum = contour_spectrum(history, fmax=25., window="hann")
pressure_band_Pa = spectrum.synthesize()
pressure_periodic_band_rms_Pa = spectrum.band_rms
```

This is an adapter example, not a claim that a particular native file format or
CFD run exists. Preserve the complex coefficients, time origin and metadata with
any receiver traces; label plotted RMS as the selected, window-conditioned band.

