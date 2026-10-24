"""Independent analytic controls; no fixtures are CFD observations."""

import importlib

import numpy as np
import pytest
from numpy.testing import assert_allclose


def acoustic_module():
    try:
        return importlib.import_module("flowacoustics.acoustics2d")
    except ModuleNotFoundError:
        raise AssertionError("2D acoustic implementation is missing") from None


def circle(np_=64, radius=0.15):
    theta = np.arange(np_) * 2 * np.pi / np_
    normal = np.column_stack((np.cos(theta), np.sin(theta)))
    return radius * normal, normal, np.full(np_, 2 * np.pi * radius / np_)


def history(kind="permeable", nt=64, np_=64, rho0=1000.0, **overrides):
    a = acoustic_module()
    xy, normal, ds = circle(np_)
    args = dict(
        time=np.arange(nt) / nt,
        xy=xy,
        normal=normal,
        ds=ds,
        p=np.zeros((nt, np_)),
        U=np.zeros((nt, np_, 2)),
        surface_velocity=np.zeros((np_, 2)),
        kind=kind,
        rho0=rho0,
    )
    args.update(overrides)
    return a.ContourHistory(**args)


def test_source_terms_keep_moving_thickness_and_flux_loading_in_si():
    a = acoustic_module()
    xy, normal, ds = circle(8)
    v = np.broadcast_to([0.3, -0.2], (8, 2))
    h = history(
        np_=8,
        p=np.full((64, 8), 7.0),
        U=np.broadcast_to([1.0, 0.5], (64, 8, 2)),
        surface_velocity=v,
    )
    q, loading = a.source_terms(h)
    un = normal @ np.array([1.0, 0.5])
    vn = normal @ np.array([0.3, -0.2])
    assert_allclose(q, np.broadcast_to(1000 * un, (64, 8)))
    expected = 7 * normal + 1000 * np.array([1.0, 0.5]) * (un - vn)[:, None]
    assert_allclose(loading, np.broadcast_to(expected, (64, 8, 2)))
    wall = history(
        kind="body",
        np_=8,
        p=np.full((64, 8), 7.0),
        U=np.broadcast_to(v, (64, 8, 2)),
        surface_velocity=v,
    )
    q, loading = a.source_terms(wall)
    assert_allclose(q, np.broadcast_to(1000 * vn, (64, 8)))
    assert_allclose(loading, np.broadcast_to(7 * normal, (64, 8, 2)))


def monopole_history(radius=0.15, frequency=4.0, **kwargs):
    from scipy.special import hankel2

    nt, np_ = 256, 96
    t = np.arange(nt) / nt
    xy, normal, ds = circle(np_, radius)
    omega = 2 * np.pi * frequency
    k = omega / 1480.0
    phase = np.exp(1j * omega * t)
    phat = -1j / 4 * hankel2(0, k * radius)
    uhat = -k / (4 * omega * 1000) * hankel2(1, k * radius)
    return history(
        nt=nt,
        np_=np_,
        xy=xy,
        ds=ds,
        p=np.real(phase[:, None] * np.full(np_, phat)),
        U=np.real(phase[:, None, None] * uhat * normal),
        **kwargs,
    )


@pytest.mark.parametrize("contour_radius", [0.08, 0.15, 0.35])
def test_fixed_contour_recovers_signed_analytic_monopole_and_band_waveform(contour_radius):
    from scipy.special import hankel2

    a = acoustic_module()
    h = monopole_history(radius=contour_radius)
    s = a.contour_spectrum(h, fmin=4.0, fmax=4.0, window="boxcar", method="fixed")
    assert_allclose(s.frequency, [4.0])
    assert_allclose(s.receivers.shape, (12, 2))
    assert_allclose(np.linalg.norm(s.receivers, axis=-1), 1.0)
    expected = -1j / 4 * hankel2(0, 2 * np.pi * 4 / 1480.0)
    assert_allclose(s.coefficient[0], expected / 2, rtol=2e-12, atol=2e-12)
    assert_allclose(
        s.synthesize(),
        np.broadcast_to(np.real(expected * np.exp(2j * np.pi * 4 * h.time))[:, None], (256, 12)),
        rtol=1e-11,
        atol=1e-12,
    )
    assert_allclose(s.band_rms, abs(expected) / np.sqrt(2), rtol=2e-12)
    assert s.metadata["signal_semantics"] == "periodic/window-conditioned band-limited"
    assert s.metadata["source_type"] == "unspecified"


def orbiting_circle(nt=256, np_=32):
    t = np.arange(nt) / nt
    w = 2 * np.pi * 2
    base, normal, ds = circle(np_, 0.025)
    center = 0.12 * np.column_stack((np.cos(w * t), np.sin(w * t)))
    v = 0.12 * w * np.column_stack((-np.sin(w * t), np.cos(w * t)))
    return history(
        nt=nt,
        np_=np_,
        kind="body",
        xy=center[:, None] + base,
        normal=normal,
        ds=ds,
        U=np.broadcast_to(v[:, None], (nt, np_, 2)),
        surface_velocity=np.broadcast_to(v[:, None], (nt, np_, 2)),
    )


def test_moving_kernel_recovers_exact_orbiting_disk_thickness_harmonics():
    # Independent area-divergence + Graf theorem oracle, not a stationary force.
    from scipy.special import hankel2, jv

    a = acoustic_module()
    h = orbiting_circle()
    receivers = a.common_receivers()
    s = a.contour_spectrum(h, receivers, fmin=2.0, fmax=6.0, window="boxcar", method="direct")
    theta = np.arctan2(receivers[:, 1], receivers[:, 0])
    for harmonic in (1, 2, 3):
        omega = 2 * np.pi * 2 * harmonic
        k = omega / h.c0
        disk = 2 * np.pi * 0.025 * jv(1, k * 0.025) / k
        gc = -0.25j * hankel2(harmonic, k) * jv(harmonic, k * 0.12)
        expected = -(omega**2) * h.rho0 * disk * gc * np.exp(-1j * harmonic * theta)
        assert_allclose(
            s.coefficient[np.flatnonzero(s.frequency == 2 * harmonic)[0]],
            expected,
            rtol=2e-10,
            atol=2e-10,
        )


def test_log_bessel_moments_match_direct_moving_loading_with_absolute_bound():
    from dataclasses import replace

    a = acoustic_module()
    h = orbiting_circle(nt=96, np_=24)
    # A material-frame steady loading must survive rotation/translation.
    p = np.broadcast_to(20 + 5 * h.normal[:, 0], h.p.shape)
    h = replace(h, p=p)
    obs = a.common_receivers()[::3]
    direct = a.contour_spectrum(h, obs, fmax=12.0, window="hann", method="direct")
    fast = a.contour_spectrum(h, obs, fmax=12.0, window="hann", method="series", series_terms=4)
    assert_allclose(fast.coefficient, direct.coefficient, atol=3e-11, rtol=2e-11)
    bound = np.asarray(fast.metadata["series_truncation_bound_pa"])
    assert bound.shape == fast.coefficient.shape
    assert np.all(bound >= 0)
    assert np.max(bound) < 1e-8
    # Truncation only: permit floating-point summation noise separately.
    assert np.all(abs(fast.coefficient - direct.coefficient) <= bound + 3e-11)
    coarse = a.contour_spectrum(h, obs, fmax=12.0, window="hann", method="series", series_terms=1)
    bound = np.asarray(coarse.metadata["series_truncation_bound_pa"])
    assert np.max(abs(coarse.coefficient - direct.coefficient)) > 1e-5
    assert np.all(abs(coarse.coefficient - direct.coefficient) <= bound + 3e-11)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("time", np.arange(64) ** 1.01 / 64, "uniform"),
        ("time", np.zeros(64), "increasing"),
        ("time", np.arange(3), "at least"),
        ("p", np.full((64, 64), np.nan), "finite"),
        ("p", np.zeros((64, 3)), "shape"),
        ("ds", np.zeros(64), "positive"),
        ("normal", np.ones((64, 2)), "unit"),
        ("U", np.zeros((64, 64, 3)), "shape"),
        ("kind", "open_strip", "kind"),
        ("rho0", -1000.0, "positive"),
        ("c0", np.inf, "positive"),
        ("blade_id", np.zeros(3), "blade_id"),
    ],
)
def test_history_rejects_invalid_native_arrays(field, value, match):
    with pytest.raises(ValueError, match=match):
        history(**{field: value})


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"fmax": 32.0}, "Nyquist"),
        ({"fmax": -1.0}, "frequency"),
        ({"fmin": 4.0, "fmax": 3.0}, "frequency"),
        ({"window": "invented"}, "window"),
        ({"fmin": 0.2, "fmax": 0.3}, "no positive"),
        ({"series_terms": 0}, "series_terms"),
        ({"receivers": [[np.nan, 1.0]]}, "receivers"),
        ({"receivers": [[1.0, 2.0, 3.0]]}, "receivers"),
    ],
)
def test_spectrum_rejects_ambiguous_or_unsupported_parameters(kw, match):
    with pytest.raises(ValueError, match=match):
        acoustic_module().contour_spectrum(history(), **kw)


def test_observer_on_panel_is_rejected_before_special_functions():
    h = history()
    with pytest.raises(ValueError, match="coincide"):
        acoustic_module().contour_spectrum(h, h.xy[:1])


def dipole_field(points, omega, strength=1.0):
    """Independent exact cylindrical dipole pressure and linear Euler velocity."""
    from scipy.special import h2vp, hankel2

    r = np.linalg.norm(points, axis=-1)
    n = points / r[..., None]
    k = omega / 1480.0
    radial = -0.25j * k * strength * hankel2(1, k * r)
    dradial = -0.25j * k * k * strength * h2vp(1, k * r)
    p = radial * n[..., 0]
    grad = dradial[..., None] * n[..., 0, None] * n
    grad += radial[..., None] / r[..., None] * (np.array([1.0, 0.0]) - n[..., 0, None] * n)
    return p, -grad / (1j * omega * 1000.0)


def test_signed_dipole_orientation_interior_cancellation_and_units():
    from dataclasses import replace

    a = acoustic_module()
    xy, n, ds = circle(96, 0.25)
    t = np.arange(256) / 256
    omega = 2 * np.pi * 5
    p, u = dipole_field(xy, omega)
    phase = np.exp(1j * omega * t)
    h = history(
        nt=256,
        np_=96,
        xy=xy,
        normal=n,
        ds=ds,
        p=np.real(phase[:, None] * p),
        U=np.real(phase[:, None, None] * u),
    )
    obs = np.vstack((a.common_receivers(), [[0.0, 0.0]]))
    s = a.contour_spectrum(h, obs, fmin=5.0, fmax=5.0, window="boxcar")
    exact, _ = dipole_field(obs[:-1], omega)
    assert_allclose(s.coefficient[0, :-1], exact / 2, atol=1e-13, rtol=1e-12)
    assert abs(s.coefficient[0, -1]) < 1e-12
    reversed_ = a.contour_spectrum(replace(h, normal=-n), obs, fmin=5.0, fmax=5.0, window="boxcar")
    assert_allclose(reversed_.coefficient, -s.coefficient, atol=1e-13)
    doubled = a.contour_spectrum(replace(h, ds=2 * ds), obs, fmin=5.0, fmax=5.0, window="boxcar")
    assert_allclose(doubled.coefficient, 2 * s.coefficient, atol=1e-13)
    # Density appears once in Q/advective L, never applied again to pressure.
    pressure_only = replace(h, U=np.zeros_like(h.U))
    first = a.contour_spectrum(pressure_only, fmin=5.0, fmax=5.0, window="boxcar")
    other = a.contour_spectrum(
        replace(pressure_only, rho0=7.0), fmin=5.0, fmax=5.0, window="boxcar"
    )
    assert_allclose(first.coefficient, other.coefficient, atol=0)


@pytest.mark.parametrize("method", ["fixed", "direct", "series"])
@pytest.mark.parametrize("window", ["boxcar", "hann"])
def test_steady_eulerian_fields_are_zero_not_window_leakage(method, window):
    h = history(p=np.full((64, 64), 101325.0), U=np.broadcast_to([1.0, 0.0], (64, 64, 2)))
    result = acoustic_module().contour_spectrum(h, fmax=10.0, method=method, window=window)
    assert_allclose(result.coefficient, 0, atol=2e-10)
    assert_allclose(np.sum(acoustic_module().source_terms(h)[0] * h.ds, axis=-1), 0, atol=1e-12)


def test_linear_vibrating_body_and_stationary_envelope_match_same_exterior_dipole():
    from scipy.special import h2vp, hankel2

    a = acoustic_module()
    nt, panels, radius, speed = 256, 96, 0.06, 1e-5
    t = np.arange(nt) / nt
    omega = 2 * np.pi * 3
    k = omega / 1480.0
    xy, normal, ds = circle(panels, radius)
    # Prescribed rigid translation, amplitude speed/omega. Linear acoustic p.
    amplitude = -1j * omega * 1000 * speed / (k * h2vp(1, k * radius))
    pressure = amplitude * hankel2(1, k * radius) * normal[:, 0]
    displacement = (speed / omega * np.sin(omega * t))[:, None, None] * np.array([1.0, 0.0])
    vel = (speed * np.cos(omega * t))[:, None, None] * np.array([1.0, 0.0])
    body = history(
        kind="body",
        nt=nt,
        np_=panels,
        xy=xy + displacement,
        normal=normal,
        ds=ds,
        U=np.broadcast_to(vel, (nt, panels, 2)),
        surface_velocity=np.broadcast_to(vel, (nt, panels, 2)),
        p=np.real(np.exp(1j * omega * t[:, None]) * pressure),
    )
    envelope_xy, envelope_n, envelope_ds = circle(panels, 0.3)
    strength = amplitude / (-0.25j * k)
    ep, eu = dipole_field(envelope_xy, omega, strength)
    envelope = history(
        nt=nt,
        np_=panels,
        xy=envelope_xy,
        normal=envelope_n,
        ds=envelope_ds,
        p=np.real(np.exp(1j * omega * t[:, None]) * ep),
        U=np.real(np.exp(1j * omega * t[:, None, None]) * eu),
    )
    bs = a.contour_spectrum(body, fmin=3.0, fmax=3.0, window="boxcar")
    es = a.contour_spectrum(envelope, fmin=3.0, fmax=3.0, window="boxcar")
    expected, _ = dipole_field(bs.receivers, omega, strength)
    assert_allclose(es.coefficient[0], expected / 2, atol=1e-15, rtol=1e-11)
    # Moving geometry has higher-order displacement effects absent in linear oracle.
    assert_allclose(bs.coefficient[0], expected / 2, atol=1e-13, rtol=2e-8)
    assert_allclose(bs.coefficient, es.coefficient, atol=1e-13, rtol=2e-8)


def articulated_circle(t, theta):
    """Analytic two-rate orbit + material rotation, no imported motion code."""
    orbit, pitch = -4 * np.pi, -2 * np.pi
    angle = theta[None, :] + pitch * t[:, None]
    n = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    center = 0.12 * np.column_stack((np.cos(orbit * t), np.sin(orbit * t)))
    vc = 0.12 * orbit * np.column_stack((-np.sin(orbit * t), np.cos(orbit * t)))
    xy = center[:, None, :] + 0.025 * n
    velocity = vc[:, None, :] + pitch * 0.025 * np.stack((-n[..., 1], n[..., 0]), axis=-1)
    pressure = np.broadcast_to(20 + 5 * np.cos(theta)[None, :], angle.shape)
    return xy, n, velocity, pressure


def test_moving_loading_against_independent_space_time_gauss_quadrature():
    # Separate quadrature nodes, Bessel J/Y implementation, finite-difference gradient.
    from numpy.polynomial.legendre import leggauss
    from scipy.special import jv, yv

    a = acoustic_module()
    nt, panels = 128, 64
    t = np.arange(nt) / nt
    theta = np.arange(panels) * 2 * np.pi / panels
    xy, normal, v, p = articulated_circle(t, theta)
    h = history(
        nt=nt,
        np_=panels,
        kind="body",
        xy=xy,
        normal=normal,
        ds=np.full(panels, 2 * np.pi * 0.025 / panels),
        p=p,
        U=v,
        surface_velocity=v,
    )
    obs = np.array([[0.8, 0.6]])
    result = a.contour_spectrum(h, obs, fmin=3.0, fmax=3.0, window="boxcar", method="series")
    nodes, weights = leggauss(100)
    tg, thg = (nodes + 1) / 2, np.pi * (nodes + 1)
    xy, normal, v, p = articulated_circle(tg, thg)
    omega = 6 * np.pi
    k = omega / 1480.0

    def green(points):
        r = np.linalg.norm(obs[0] - points, axis=-1)
        return -yv(0, k * r) / 4 - 1j * jv(0, k * r) / 4

    grad = np.empty(xy.shape, complex)
    epsilon = 2e-5
    for axis in (0, 1):
        d = np.eye(2)[axis] * epsilon
        grad[..., axis] = (
            green(xy - 2 * d) - 8 * green(xy - d) + 8 * green(xy + d) - green(xy + 2 * d)
        ) / (12 * epsilon)
    q = 1000 * np.sum(v * normal, axis=-1)
    loading = p[..., None] * normal
    integrand = 1j * omega * q * green(xy) + np.sum(loading * grad, axis=-1)
    reference = np.sum(
        integrand
        * (weights / 2 * np.exp(-1j * omega * tg))[:, None]
        * (np.pi * 0.025 * weights)[None, :]
    )
    assert_allclose(result.coefficient[0, 0], reference, atol=1e-9, rtol=1e-8)
    assert abs(reference) > 1e-4
    with pytest.raises(ValueError, match="cannot freeze"):
        a.contour_spectrum(h, obs, fmax=3.0, method="fixed")


def test_hann_conditioning_parseval_and_clock_shift_preserve_source_relative_phase():
    from dataclasses import replace

    a = acoustic_module()
    h = monopole_history(frequency=7.0)
    s = a.contour_spectrum(h, fmax=20.0, window="hann")
    assert_allclose(
        np.sqrt(np.mean(s.synthesize() ** 2, axis=0)), s.band_rms, rtol=1e-13, atol=1e-13
    )
    shifted = a.contour_spectrum(replace(h, time=h.time + 1000.0), fmax=20.0, window="hann")
    assert_allclose(shifted.coefficient, s.coefficient, atol=1e-13)
    assert_allclose(shifted.synthesize(), s.synthesize(), atol=1e-12)
    assert shifted.metadata["time_origin"] == 1000.0


def test_omitted_quadratic_volume_is_visible_in_small_signal_permeable_fixture():
    # Linear wave p,U is only a first-order exact fluid solution. Feeding full
    # rho*u*u_n generates a second harmonic; no exterior volume cancels it here.
    from dataclasses import replace

    a = acoustic_module()
    h = monopole_history(frequency=4.0)
    h = replace(h, p=100 * h.p, U=100 * h.U)
    first = a.contour_spectrum(h, fmin=8.0, fmax=8.0, window="boxcar")
    double = a.contour_spectrum(
        replace(h, p=2 * h.p, U=2 * h.U), fmin=8.0, fmax=8.0, window="boxcar"
    )
    assert np.max(abs(first.coefficient)) > 1e-10
    assert_allclose(double.coefficient, 4 * first.coefficient, atol=1e-14, rtol=1e-8)


def test_metadata_retains_native_identity_and_processing_support_without_claiming_validation():
    from dataclasses import replace

    a = acoustic_module()
    # Metadata fixture only, deliberately NOT labelled native_cfd.
    h = replace(
        monopole_history(),
        provenance={
            "source_type": "analytical_verification",
            "case_id": "test-only",
            "depth_removed_m": 0.02,
        },
    )
    s = a.contour_spectrum(h, fmax=12.0, window="hann")
    assert s.metadata["provenance"] == h.provenance
    assert s.metadata["sample_count"] == 256
    assert s.metadata["last_sample_time"] == h.time[-1]
    assert s.metadata["period_s"] == 1.0
    assert s.metadata["window_coherent_gain"] == 0.5
    assert s.metadata["window_rms_gain"] == pytest.approx(np.sqrt(3 / 8))
    assert s.metadata["requested_band_hz"] == [0.0, 12.0]
    assert s.metadata["density_model"] == "constant incompressible input"
    assert s.metadata["native_coupling_verified"] is False


def test_decimal_clock_band_edges_select_same_integer_bins():
    from dataclasses import replace

    a = acoustic_module()
    base = history()
    t = np.array([float(f"{900 + 0.002 * i:.3f}") for i in range(64)])
    h = replace(base, time=t)
    s = a.contour_spectrum(h, fmin=15.625, fmax=15.625, window="boxcar")
    assert len(s.frequency) == 1
    assert s.frequency[0] == pytest.approx(15.625, rel=1e-10)


@pytest.mark.parametrize("terms", [1, 3, 6])
def test_series_absolute_bound_near_upper_validity_limit(terms):
    from dataclasses import replace

    a = acoustic_module()
    h = orbiting_circle(nt=64, np_=16)
    rng = np.random.default_rng(714)
    h = replace(h, c0=100.0, p=50 * rng.standard_normal(h.p.shape))
    obs = np.array([[1.0, 0.0], [-0.8, 0.6]])
    direct = a.contour_spectrum(h, obs, fmin=1, fmax=12, window="hann", method="direct")
    fast = a.contour_spectrum(
        h, obs, fmin=1, fmax=12, window="hann", method="series", series_terms=terms
    )
    error = abs(direct.coefficient - fast.coefficient)
    bound = np.asarray(fast.metadata["series_truncation_bound_pa"])
    assert np.all(error <= bound + 2e-11)
    if terms == 6:
        assert np.max(error) < 1e-7
    with pytest.raises(ValueError, match="max"):
        a.contour_spectrum(h, obs, fmax=20, method="series")
