# Verified 2D free-space moving-contour surface analogy

## Scope and status

`acoustics2d.py` implements a **two-dimensional, span-invariant**, surface-only
acoustic analogy. It does not call a 3D Farassat kernel, integrate an open 3D
strip, select a coherence span, or invent end-cap pressures. An actual CFD
adapter must supply the native histories described in [API.md](API.md).
All local verification inputs and benchmark inputs are **analytical, not CFD**.
No CFD solver was invoked by this implementation task; native coupling must
be accepted separately by the parent after obtaining real source histories.

The general moving/permeable FW-H distributional equation and source definitions
are taken from Lopes et al., equations (1), (4), (5), not from a stationary-force
approximation.[6] The restriction to two spatial coordinates below is our own
explicit reduction; the free-space Green function, not a span correction, changes.
The pressure units remain Pa: sources are infinite and invariant in z, not
finite turbines with a claimed absolute sound level.

## 1. Distributional equation, signs and units

Let the closed contour C(t) bound an excluded region; n points **from that region
out into the retained fluid**. For separate bodies, use each body's own exterior
normal. On a solid wall OpenFOAM's fluid-domain outward normal is the opposite
normal. Let δ_C be the line delta normalized by
`integral a(y) δ_C(y,t) d²y = integral_C(t) a ds`. Thus no unstated `|grad f|`
remains in a level-set delta. Let H be 1 outside the excluded region.

With quiescent-medium wave operator `D = c0^(-2) ∂tt - Δxy`, the dimensional
surface analogy is

```text
D p' = ∂t[Q δ_C] - ∂yi[L_i δ_C] + ∂yi∂yj[T_ij H].             (1)
Q   = rho0 v_n + rho (u_n - v_n),                            (2)
L_i = (p-p0)n_i - tau_ij n_j + rho u_i (u_n-v_n).             (3)
```

Equations (1)–(3) follow the general permeable source definitions; the exterior
volume term is part of the full analogy and is not automatically negligible.[6]
Here p' equals the acoustic pressure in the homogeneous linear observer region;
using incompressible CFD source fields is a hybrid acoustic-analogy approximation,
not a claim that those fields solve a compressible acoustic wave equation.

This API fixes `rho=rho0` and omits viscous traction, so the actual inputs give

```text
Q = rho0 (U dot n)
L = p_Pa n + rho0 U [(U-v) dot n].                           (4)
```

This simplification does **not** eliminate moving-body thickness: when no normal
penetration is satisfied, `U dot n = v dot n`, therefore
`Q=rho0 v_n` and `L=p_Pa n`. For a fixed permeable contour, `v=0`, hence
`Q=rho0 U_n`, `L=p_Pa n+rho0 U U_n`. The full inertial vector U, not only its
normal component, is needed in momentum flux. Actual measured U is retained;
the code does not silently replace imperfect native wall flux by prescribed v.
A real-CFD handoff must audit that normal-slip residual.

| Quantity | SI unit |
|---|---|
| Q | kg m^-2 s^-1 |
| L | Pa = kg m^-1 s^-2 |
| ds | m (no extrusion depth) |
| G_omega | dimensionless |
| grad_y G_omega | m^-1 |
| omega Q G ds and L·grad_y G ds | Pa |

Native kinematic pressure is converted to Pa once in the sampler. Side-face area
is divided by numerical extrusion depth once in the sampler. Neither operation
occurs again in this module. Nblades is not a multiplier; integrate the actual
complete panel set exactly once. Panel material means must be preserved: a
constant load attached to a rotating blade is not a stationary Eulerian source.

## 2. Outgoing Green function and exact moving-kernel integration

We use NumPy's Fourier sign:

```text
hat(a)(omega) = integral a(t) exp(-i omega t) dt,
a(t) = (1/2pi) integral hat(a)(omega) exp(+i omega t) d omega.
```

Consequently for positive omega and `k=omega/c0`,

```text
G_omega(x,y) = -i/4 H_0^(2)(k R),  R=|x-y|,
(Delta+k²)G_omega = -delta_2,
grad_y G_omega = -i k/4 H_1^(2)(kR) (x-y)/R.                (5)
```

Beylkin's explicitly stated `+i H_0^(1)/4` outgoing Green function corresponds
to the opposite time sign; (5) is its conjugate, not the same Hankel branch with
a changed sign in only one term.[2] It is full cylindrical propagation, retaining
reactive as well as radiating content. No `kr >> 1` approximation is made.

The time-domain wave Green function has an interior light-cone tail:
`G_ret(R,t)=H(t-R/c0)/(2pi sqrt(t²-(R/c0)²))` for the operator D above. Its
Fourier transform is (5), understood with outgoing limiting absorption. Its
normalization can also be checked by integrating the 3D Green function along
the invariant coordinate; that calculation defines a 2D Green function and
is **not** integration of CFD loads over an arbitrary finite span.

Fourier transform the **entire Eulerian distributions** in (1), not just a
material panel's time history. Integration by parts in space and time gives
our implemented surface term

```text
hat(p_S)(x,omega)
 = integral dt exp(-i omega t) integral_C(t)
       [ i omega Q(y,t) G_omega(x,y)
         + L(y,t) dot grad_y G_omega(x,y) ] ds.             (6)
```

There is a **plus** source-gradient loading term: the minus divergence in (1)
combined with integration by parts produces `+L·grad_y G`. This is independently
checked by a signed acoustic dipole and normal reversal, not only by RMS.

For a material parameter a, (6) is integrated using `y=y(a,t)` and its moving
normal. Rigid motion has constant ds(a). Differentiating `Q delta(x-y(t))` would
produce both a derivative of Q and the `-Q v·grad delta` motion term. Keeping the
entire moving distribution under the temporal transform accounts for both;
freezing G at `y(a,0)` loses that second term. Similarly all load orientation,
translation, acceleration and attitude effects reside in the time-dependent
spatial distribution. No extra guessed Doppler factor belongs in (6). Doppler
factors arise when time-domain delta integrals are explicitly collapsed; doing
that on top of (6) would double-count the change of variables. A stretching
boundary would need time-dependent ds, which this rigid-panel API does not support.

## 3. Finite records: precisely what is calculated

The input is uniformly sampled at Nt physical times, without a repeated endpoint.
The nominal period is `T=Nt*dt`, whereas the last sample is `t0+(Nt-1)*dt`.
Reject adaptive timestamps rather than secretly resampling them. Only positive
DFT bins strictly below Nyquist in the explicitly requested band are evaluated.
Band edges are selected in integer-bin coordinates with a recorded roundoff-only
clock tolerance capped at 1e-7 bin.

For each requested omega define the **Eulerian kernel-projected** time series
`S_omega(t)=sum_panels ds [i omega Q G + L·grad G]`. The returned coefficient is

```text
c(omega) = sum_j w_j [S_omega(t_j)-mean_j S_omega] exp(-i omega (t_j-t0)) / sum_j w_j.
```

`boxcar` sets `w=1`; `hann` uses the periodic Hann `0.5-0.5*cos(2pi*j/Nt)`.
The normalization is coherent gain, not a PSD normalization. A bin-centred
real cosine of peak pressure A has coefficient A/2. A Hann spectrum includes
window sidebands, so selected-bin RMS is **not** automatically the unwindowed
physical RMS or a stationary noise PSD. Coherent and RMS window gains, requested
and actual band, sample support, cadence, origin, and source provenance survive
in metadata.

This conditioning removes constant *Eulerian* projected fields before tapering.
For fixed geometry it commutes with subtracting time means of Q and L; for moving
geometry subtracting each material panel's mean is wrong. The mean subtraction
is intentionally part of the finite-window estimator, not a claim that arbitrary
windowing commutes with the original time-domain wave equation. In particular,
these are **periodic/window-conditioned, band-limited signals**, not causal
retarded arrivals from an initially silent source. Do not infer a first-arrival
time, flush a fictitious tail, or label the synthesis the full-band waveform.
`band_rms=sqrt(2 sum_positive_bins |c|²)` is exactly the RMS of this periodic band
synthesis. Startup exclusion is a caller's explicit slice, not proof of stationarity.

## 4. Efficient moving evaluation with a bounded Bessel expansion

Fixed geometry uses one FFT of Q and L, then Hankel integration per selected bin.
`method="direct"` evaluates (6) with SciPy Hankel functions at **every native
moving position**, one observer and frequency at a time. It is the retained
reference algorithm, not the default full-sized low-frequency workhorse.

For `method="series"`, expand both J0 and Y0 using their exact defining power
series, including the logarithmic Y0 terms.[4][5] Writing `H_j=sum_(l=1)^j 1/l`,
`H_0=0`, `a_j=(-1)^j/(4^j(j!)²)`, and
`C_j(k)=(H_j-log(k/2)-gamma)/(2pi)-i/4`,

```text
G(k,R) = sum_(j=0)^infinity a_j k^(2j) R^(2j) [C_j(k)-log(R)/(2pi)],
grad_y G_j = -(x-y)/R² * a_j k^(2j) R^(2j)
             * [2j(C_j-log(R)/(2pi))-1/(2pi)].              (7)
```

Logs are evaluated numerically with R in metres and k in inverse metres; their
sum is the dimensionless `log(kR/2)`, so no physical scale depends on the split.
For each observer, form four time-only moments per j:

```text
A_Qj = sum ds Q R^(2j)              B_Qj = sum ds Q R^(2j) log R
A_Lj = sum ds D R^(2j) 2j           B_Lj = sum ds D R^(2j) (2j log R+1)
D = -L dot (x-y) / R².
```

Apply the same mean/window/FFT operator to those moments once. Reassemble:

```text
c = sum_j a_j k^(2j) { i omega [C_j FFT(A_Qj)-FFT(B_Qj)/(2pi)]
                                  + C_j FFT(A_Lj)-FFT(B_Lj)/(2pi) }.       (8)
```

This is **not** an acoustically compact force model and not an unbounded
multipole truncation. R is the actual distance at every sample, including its
full orbital and pitching motion. No time samples or panels are skipped. The
cost is approximately `O(No*Nt*Np*J + No*J*Nt log Nt + No*J*Nf)` instead of a
Hankel evaluation for every `No*Nt*Np*Nf` combination. There is no need to evaluate
the unused Nyquist band. Six terms are the default; require `max(kR)<=1` for the
fast path. Auto uses fixed FFT for static positions, this series in the bounded
low-argument regime, otherwise the exact sampled Hankel quadrature.

### Explicit absolute truncation bound

The following bound is our derivation from the cited convergent series, not a
claim extracted from a paper. Let J>=1 be the first omitted index,
`zmax=k Rmax <= 1`, `zmin=k Rmin > 0`,

```text
t_J = (zmax²/4)^J/(J!)²,
b_J = 1/4 + (max(|log(zmin/2)+gamma|, |log(zmax/2)+gamma|)+H_J)/(2pi).
```

Then a conservative bound on the Green and gradient tails is

```text
e_G <= 2 t_J b_J,
e_grad <= 2 t_J/Rmin [2J b_J + 1/(2pi)].                   (9)
```

To see this, the absolute radial-power ratio is at most `1/[4(j+1)²]`;
`b_(j+1)/b_j <= 1+2/[pi(j+1)]` because `b_j>=1/4`. The gradient's extra factor
has ratio at most `(j+1)/j` times that b ratio. For every j>=1, both resulting
ratios are less than 1/2, so each positive majorant tail is at most twice its
first term. This is uniform over **all recorded distances**, not a fit to a
handful of observer directions.

Let `A_Q=max_t sum ds |Q|` and `A_L=max_t sum ds |L|`. Nonnegative normalized
window weights have norm one, and the mean-removal operator has norm at most two.
The returned per-bin/per-observer bound is therefore

```text
|delta c| <= 2 [omega A_Q e_G + A_L e_grad] Pa.             (10)
```

It bounds **only series truncation relative to the same sampled discrete integral**.
It does not bound floating-point summation error, time/panel quadrature error,
CFD discretization/sampling error, omitted physics, or cancellation-relative
error. Tests deliberately use J=1 to resolve nonzero truncation error and check
it is below (10); J=4/6 comparisons approach floating-point noise.

## 5. Independent physical and numerical controls

The tests do not compare one call to itself:

- **Monopole:** a cylindrical wave `P=A G(k,r)` with fluid velocity from independent
  linear Euler `U=-grad P/(i omega rho0)` is sampled on an enclosing circle.
  Recover its signed exterior coefficient at all twelve receivers. Different
  contour radii represent the same linear exterior solution.
- **Dipole:** `P=-i k H_1^(2)(kr) cos(theta)/4`; the velocity uses an independently
  differentiated radial/azimuthal field. Check phase, axes, anti-symmetry,
  reversed normals, interior cancellation, ds linearity and no extra rho factor.
- **Moving thickness:** a radius-b disk orbits with radius a and angular rate Ω,
  `U=v`, `p=0`. This is a prescribed thickness-only source, **not a physically
  solved body flow**. The divergence theorem gives
  `integral_C rho0 v_n G ds = rho0 d/dt integral_disk G dA`.
  Since `integral_disk G dA=2pi b J1(kb)/k * G(x,center(t))`, the exact positive
  harmonic is
  `c_n=-(nΩ)² rho0 (2pi b J1(kb)/k) [-i H_n^(2)(kR) J_n(ka)/4] exp(-in theta_x)`.
  The angular translation identity is Graf's addition theorem.[7]
  This oracle catches omitted moving-distribution terms and the wrong time sign.
- **Body versus permeable:** a tiny rigidly translating circular body has the
  linear vibrating-cylinder solution `P=A H1(kr) cos(theta)`,
  `A=-i omega rho0 V/(k H1'(kb))`. The body actually translates by
  `(V/omega) sin(omega t)` and retains its true surface velocity. Its leading-order
  prediction and a stationary outer contour agree with the same exact exterior
  dipole. Higher-order displacement corrections are explicitly outside the
  linear oracle; this is not a claim of nonlinear body/envelope equality.
- **Independent moving loading quadrature:** orbit and absolute half-rate pitch,
  constant material angular loads, separate Gauss–Legendre nodes in time and
  angle, Bessel J/Y (not the production Hankel call), and a fourth-order spatial
  finite-difference Green gradient. This tests the moving pressure-load term as
  well as thickness, retaining material mean loads.
- **Steady/flux controls:** static pressure and steady through-flow give no
  nonzero-frequency output even under Hann; closed-circle net uniform mass flux
  is zero. Periodic synthesis obeys Parseval. Clock translation and decimal bin
  edges are tested without silently changing the source timestamps.
- **Omission control:** a linear monopole p/U supplied to the full quadratic
  surface expression produces an O(amplitude²) second harmonic. Its amplitude
  scales by four when the source doubles. The supplied first-order analytical
  field has no consistent second-order volume solution; ignoring that caveat
  and calling this a full nonlinear validation would be wrong.

## 6. Limits that cannot be calibrated away

A finite permeable surface can include interior quadrupole effects in its source
histories, but exterior volume effects remain; wake crossing can generate
spurious signals when that exterior cancellation is omitted.[1][6] Do not drop a
closing side, invent its pressure, fit a coherence correction, or identify
body-minus-envelope as an isolated quadrupole measurement. The two predictions
overlap and must **not** be summed.

Viscous traction would contribute `(-tau·n)·grad_y G`; it is not computed because
the native contract supplies p/U rather than resolved/modelled traction. No
viscous error bound is asserted. Constant density discards compressible-density
source corrections. Acoustic propagation assumes zero mean flow even though
CFD inflow is 1 m/s. There are no fluid boundaries, refraction, cavitation,
free-surface effects, finite-span effects, or experimental validation.

Panel midpoint arrays cannot prove topological closure, outward orientation,
face ownership or complete coverage; these are **native sampler acceptance gates**
requiring the original curve connectivity and panel identities. The code validates
shapes, finiteness, uniform time, positive ds, unit normals and nonsingular sample
positions, but intentionally does not pretend that `sum(n ds)=0` proves closure.
Native wall flux, pressure units, actual orbit+pitch geometry, side-face extrusion
removal, complete envelope sampling and provenance must be checked before
publishing real-CFD acoustic plots. The `native_coupling_verified` field is false:
this numerical function cannot authenticate a caller's metadata.

Two-dimensional WALE remains a **2D WALE / LES-model calculation**, not resolved
three-dimensional turbulent LES. This acoustic verification cannot change that
source-physics limitation. No automatic blade-passing frequency is inferred from
blade count for the articulated turbine.

## Source access notes

The original method audit inspected the Lopes equations (1), (4), (5), the
Ricciardi HTML, Beylkin's accessible introduction (not the paywalled full
article), and relevant official NIST equations. Those historical access records
and downloaded papers are not distributed in this source package. Citations
below are retained; this packaging task did not repeat literature retrieval.
Mathematical derivations (6)–(10) are the implementation's explicit reduction
and bound, checked by independent analytical tests, not falsely attributed to
those papers. Current copied-package verification is in VERIFICATION.md;
external native acceptance remains a separate gate.

## Sources

[1] https://ar5iv.labs.arxiv.org/html/2111.03551
[2] https://www.sciencedirect.com/science/article/abs/pii/S1063520324000101
[4] https://dlmf.nist.gov/10.8
[5] https://dlmf.nist.gov/10.2
[6] https://ntrs.nasa.gov/api/citations/20170005478/downloads/20170005478.pdf
[7] https://dlmf.nist.gov/10.23
