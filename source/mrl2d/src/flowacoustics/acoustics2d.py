"""Span-invariant 2D surface acoustics; never a finite-span 3D correction.

See docs/mrl2d/acoustics/{API,METHOD}.md for conventions and limitations.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ContourHistory:
    """Uniform native histories, SI units, normals out of the enclosed region."""

    time: np.ndarray
    xy: np.ndarray
    normal: np.ndarray
    ds: np.ndarray
    p: np.ndarray
    U: np.ndarray
    surface_velocity: np.ndarray
    kind: str
    blade_id: np.ndarray | None = None
    rho0: float = 1000.0
    c0: float = 1480.0
    provenance: dict | None = None

    def __post_init__(self):
        for name in ("time", "xy", "normal", "ds", "p", "U", "surface_velocity"):
            raw = getattr(self, name)
            if np.iscomplexobj(raw):
                raise ValueError(f"{name} must contain real histories")
            value = np.asarray(raw, dtype=float)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.time.ndim != 1 or len(self.time) < 4:
            raise ValueError("time must be 1D with at least four samples")
        delta = np.diff(self.time)
        if np.any(delta <= 0):
            raise ValueError("time must be strictly increasing")
        dt = (self.time[-1] - self.time[0]) / (len(self.time) - 1)
        tolerance = max(1e-9 * dt, 8 * np.finfo(float).eps * np.max(abs(self.time)))
        if (
            np.max(abs(delta - dt)) > tolerance
            or np.max(abs(self.time - self.time[0] - np.arange(len(self.time)) * dt)) > tolerance
        ):
            raise ValueError("time must be uniform; no implicit resampling")
        if self.ds.ndim != 1 or len(self.ds) < 3 or np.any(self.ds <= 0):
            raise ValueError("ds must have at least three positive lengths")
        nt, panels = len(self.time), len(self.ds)
        for name in ("xy", "normal", "surface_velocity"):
            if getattr(self, name).shape not in {(panels, 2), (nt, panels, 2)}:
                raise ValueError(f"{name} shape must be [Np,2] or [Nt,Np,2]")
        if self.p.shape != (nt, panels) or self.U.shape != (nt, panels, 2):
            raise ValueError("p/U shape must be [Nt,Np] / [Nt,Np,2]")
        if not np.allclose(np.linalg.norm(self.normal, axis=-1), 1.0, atol=1e-7, rtol=0):
            raise ValueError("normal must contain unit normals")
        if self.kind not in {"body", "permeable"}:
            raise ValueError("kind must be body or permeable")
        if any(not np.isfinite(x) or x <= 0 for x in (self.rho0, self.c0)):
            raise ValueError("rho0 and c0 must be positive finite scalars")
        if self.blade_id is not None:
            self.blade_id = np.asarray(self.blade_id)
            if (
                self.blade_id.shape != (panels,)
                or not np.issubdtype(self.blade_id.dtype, np.integer)
                or np.any(self.blade_id < 0)
            ):
                raise ValueError("blade_id must be nonnegative integers [Np]")


def source_terms(history):
    """Return Q [kg/(m² s)] and L [Pa], including relative advective flux."""
    h = history
    un = np.sum(h.U * h.normal, axis=-1)
    vn = np.sum(h.surface_velocity * h.normal, axis=-1)
    q = h.rho0 * un
    loading = h.p[..., None] * h.normal + h.rho0 * h.U * (un - vn)[..., None]
    return q, loading


def common_receivers(radius=1.0):
    """Twelve x-y receivers, counterclockwise from +x; radius in metres."""
    angles = np.arange(12) * np.pi / 6
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


@dataclass
class Spectrum2D:
    frequency: np.ndarray
    coefficient: np.ndarray
    receivers: np.ndarray
    time: np.ndarray
    metadata: dict

    def synthesize(self, time=None):
        """Periodic band-only synthesis; NOT causal retarded arrivals."""
        time = self.time if time is None else np.asarray(time, dtype=float)
        phase = np.exp(2j * np.pi * np.outer(time - self.time[0], self.frequency))
        return 2 * np.real(phase @ self.coefficient)

    @property
    def band_rms(self):
        """RMS of periodic selected-bin synthesis, not full-band sound."""
        return np.sqrt(2 * np.sum(abs(self.coefficient) ** 2, axis=0))


def _transform(values, weights, bins):
    centered = values - np.mean(values, axis=0)
    return np.fft.rfft(centered * weights.reshape((-1,) + (1,) * (values.ndim - 1)), axis=0)[
        bins
    ] / np.sum(weights)


def _series_spectrum(h, receivers, q, loading, weights, bins, frequency, terms):
    """Separate exact logarithmic Bessel series into time-only FFT moments."""
    from math import factorial

    omega = 2 * np.pi * frequency
    k = omega / h.c0
    result = np.zeros((len(bins), len(receivers)), complex)
    bounds = np.zeros(result.shape)
    qabs = np.max(np.sum(abs(q) * h.ds, axis=-1))
    labs = np.max(np.sum(np.linalg.norm(loading, axis=-1) * h.ds, axis=-1))
    for o, observer in enumerate(receivers):
        rv = observer - h.xy
        r2 = np.sum(rv * rv, axis=-1)
        rmin, rmax = np.sqrt(np.min(r2)), np.sqrt(np.max(r2))
        if np.max(k) * rmax > 1.0:
            raise ValueError("series requires max(k*R) <= 1; use direct")
        log_r = 0.5 * np.log(r2)
        radial_loading = -np.sum(loading * rv, axis=-1) / r2
        power = np.ones(r2.shape)
        harmonic = 0.0
        for j in range(terms):
            if j:
                harmonic += 1.0 / j
            qw = q * power * h.ds
            lw = radial_loading * power * h.ds
            moments = np.column_stack(
                (
                    np.sum(qw, axis=-1),
                    np.sum(qw * log_r, axis=-1),
                    np.sum(2 * j * lw, axis=-1),
                    np.sum(lw * (2 * j * log_r + 1), axis=-1),
                )
            )
            transformed = _transform(moments, weights, bins)
            c = (harmonic - np.log(k / 2) - np.euler_gamma) / (2 * np.pi) - 0.25j
            factor = (-k * k / 4) ** j / factorial(j) ** 2
            result[:, o] += factor * (
                1j * omega * (c * transformed[:, 0] - transformed[:, 1] / (2 * np.pi))
                + c * transformed[:, 2]
                - transformed[:, 3] / (2 * np.pi)
            )
            power *= r2
        # Analytic absolute tail bound, not an observed agreement heuristic.
        # For kR<=1 and J>=1, both tail ratios are <1/2; see METHOD.md.
        harmonic += 1.0 / terms
        zmax, zmin = k * rmax, k * rmin
        logbound = np.maximum(
            abs(np.log(zmin / 2) + np.euler_gamma), abs(np.log(zmax / 2) + np.euler_gamma)
        )
        b = 0.25 + (logbound + harmonic) / (2 * np.pi)
        first = (zmax * zmax / 4) ** terms / factorial(terms) ** 2
        eg = 2 * first * b
        edg = 2 * first / rmin * (2 * terms * b + 1 / (2 * np.pi))
        bounds[:, o] = 2 * (omega * qabs * eg + labs * edg)
    return result, bounds


def contour_spectrum(
    history, receivers=None, *, fmax=25.0, fmin=0.0, window="hann", method="auto", series_terms=6
):
    """Evaluate positive-frequency surface pressure coefficients, in Pa."""
    from scipy.special import hankel2

    h = history
    receivers = common_receivers() if receivers is None else np.asarray(receivers, dtype=float)
    n = len(h.time)
    dt = (h.time[-1] - h.time[0]) / (n - 1)
    if (
        receivers.ndim != 2
        or receivers.shape[1] != 2
        or not len(receivers)
        or not np.all(np.isfinite(receivers))
    ):
        raise ValueError("receivers must be finite [No,2]")
    if not np.isfinite(fmin) or not np.isfinite(fmax) or not 0 <= fmin <= fmax or fmax <= 0:
        raise ValueError("invalid frequency interval")
    if fmax >= 0.5 / dt:
        raise ValueError("fmax must be strictly below Nyquist")
    if window not in {"boxcar", "hann"}:
        raise ValueError("window must be boxcar or hann")
    if not isinstance(series_terms, int) or not 1 <= series_terms <= 12:
        raise ValueError("series_terms must be an integer in [1,12]")
    for observer in receivers:
        if np.any(np.linalg.norm(observer - h.xy, axis=-1) <= 1e-12):
            raise ValueError("receiver must not coincide with a source panel")
    frequencies = np.fft.rfftfreq(n, dt)
    index = np.arange(len(frequencies))
    df = 1 / (n * dt)
    # Snap only clock/FP-scale edge uncertainty, never a physically off-edge tone.
    edge_tol = min(
        1e-7,
        max(
            64 * np.finfo(float).eps * max(1.0, fmax / df),
            16 * np.finfo(float).eps * max(1.0, np.max(abs(h.time)) / dt),
        ),
    )
    bins = np.flatnonzero(
        (index > 0)
        & (index >= fmin / df - edge_tol)
        & (index <= fmax / df + edge_tol)
        & (index < n / 2)
    )
    frequency = frequencies[bins]
    if not len(bins):
        raise ValueError("no positive DFT bins in requested interval")
    weights = np.ones(n) if window == "boxcar" else np.hanning(n + 1)[:-1]
    q, loading = source_terms(h)
    moving = h.xy.ndim == 3 and not np.array_equal(h.xy, np.broadcast_to(h.xy[0], h.xy.shape))
    if method == "auto":
        max_r = np.max(np.linalg.norm(h.xy, axis=-1)) + np.max(np.linalg.norm(receivers, axis=-1))
        method = (
            ("series" if 2 * np.pi * fmax * max_r / h.c0 <= 1 else "direct") if moving else "fixed"
        )
    if method not in {"direct", "fixed", "series"}:
        raise ValueError("unsupported method")
    if method == "fixed" and moving:
        raise ValueError("fixed method cannot freeze a moving contour")
    coefficient = np.empty((len(bins), len(receivers)), dtype=complex)
    if method == "fixed":
        qhat, lhat = _transform(q, weights, bins), _transform(loading, weights, bins)
        xy = h.xy if h.xy.ndim == 2 else h.xy[0]
        rv = receivers[:, None, :] - xy
        radius = np.linalg.norm(rv, axis=-1)
        for b, f in enumerate(frequency):
            omega = 2 * np.pi * f
            k = omega / h.c0
            g = -0.25j * hankel2(0, k * radius)
            grad = (-0.25j * k * hankel2(1, k * radius))[..., None] * rv / radius[..., None]
            coefficient[b] = np.sum(
                (1j * omega * qhat[b] * g + np.sum(lhat[b] * grad, axis=-1)) * h.ds, axis=-1
            )
    elif method == "series":
        coefficient, bounds = _series_spectrum(
            h, receivers, q, loading, weights, bins, frequency, series_terms
        )
    else:
        # One observer at a time bounds memory by O(Nt*Np), not Nt*Np*No*Nf.
        for o, observer in enumerate(receivers):
            rv = observer - h.xy
            radius = np.linalg.norm(rv, axis=-1)
            for b, f in enumerate(frequency):
                omega = 2 * np.pi * f
                k = omega / h.c0
                g = -0.25j * hankel2(0, k * radius)
                grad = (-0.25j * k * hankel2(1, k * radius))[..., None] * rv / radius[..., None]
                projected = np.sum(
                    (1j * omega * q * g + np.sum(loading * grad, axis=-1)) * h.ds, axis=-1
                )
                projected -= np.mean(projected)
                phase = np.exp(-1j * omega * (h.time - h.time[0]))
                coefficient[b, o] = np.sum(weights * phase * projected) / np.sum(weights)
    metadata = dict(
        signal_semantics="periodic/window-conditioned band-limited",
        source_type=(h.provenance or {}).get("source_type", "unspecified"),
        method=method,
        window=window,
        dt=float(dt),
        time_origin=float(h.time[0]),
        mean_conditioning="Eulerian kernel-projected time mean removed before window",
        coefficient_units="Pa",
        convention="forward exp(-i omega t), inverse +i",
        propagation="2D free-space, zero acoustic mean flow",
        omitted_terms=["viscous traction", "exterior volume quadrupoles"],
        pressure_interpretation="ideal span-invariant; not finite-turbine sound",
    )
    metadata.update(
        provenance=dict(h.provenance or {}),
        sample_count=n,
        panel_count=len(h.ds),
        last_sample_time=float(h.time[-1]),
        period_s=float(n * dt),
        requested_band_hz=[float(fmin), float(fmax)],
        actual_band_hz=[float(frequency[0]), float(frequency[-1])],
        frequency_bin_indices=bins.tolist(),
        edge_tolerance_bins=float(edge_tol),
        window_coherent_gain=float(np.mean(weights)),
        window_rms_gain=float(np.sqrt(np.mean(weights**2))),
        density_model="constant incompressible input",
        rho0=h.rho0,
        c0=h.c0,
        native_coupling_verified=False,
    )
    if method == "series":
        metadata["series_terms"] = series_terms
        metadata["series_truncation_bound_pa"] = bounds.tolist()
    return Spectrum2D(frequency, coefficient, receivers, h.time, metadata)
