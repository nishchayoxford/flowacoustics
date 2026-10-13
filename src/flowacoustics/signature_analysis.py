"""Auditable, short-record cylinder signatures; no solver or source writes.

The two native analogies are alternative unvalidated loading predictions, never
summed. Offline compact terms use force ON FLUID = minus exported body force.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

from .analysis import integrated_power, pressure_spectrogram, welch_psd
from .cylinder_acoustics import uniform_record

BIN_EDGE_EPS_MULTIPLIER = 32
BIN_EDGE_MAX_SNAP = 1e-9


def validate_status(run, *, historical_smoke=False):
    """Gate before opening scientific arrays; filenames do not imply duration."""
    version = run.get("schema_version")
    if version == 1:
        if not historical_smoke or run.get("status") != "completed_demo":
            raise ValueError("Schema 1 requires explicit historical smoke permission")
        label = "historical_smoke_only"
    elif version == 2:
        if run.get("status") != "completed_native_reconstructed":
            raise ValueError("Require completed_native_reconstructed; no running results")
        for field in ("exports", "reconstruction"):
            if run.get(field, {}).get("status") != "completed":
                raise ValueError(f"Require completed {field}")
        label = "native_completed_short_record"
    else:
        raise ValueError("Unsupported run schema")
    if run.get("evidence_kind") != "native_execution":
        raise ValueError("Require evidence_kind native_execution")
    return label


def digest(path):
    """Stream large reconstructed fields without retaining their arrays."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_run(run_path, *, project, required_paths=(), historical_smoke=False):
    """Verify all retained identities, never replace a missing or mismatched pin."""
    project = Path(project).resolve()
    raw = Path(run_path).read_bytes()
    run = json.loads(raw)
    label = validate_status(run, historical_smoke=historical_smoke)
    maps = (
        [run.get("sha256")]
        if run["schema_version"] == 1
        else [
            run.get("input_sha256"),
            run["exports"].get("sha256"),
            run["reconstruction"].get("sha256"),
        ]
    )
    pins = {}
    for mapping in maps:
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Missing/empty expected identity map")
        for name, expected in mapping.items():
            p = Path(name)
            if (
                ".." in p.parts
                or not isinstance(expected, str)
                or not re.fullmatch("[0-9a-f]{64}", expected)
            ):
                raise ValueError("Malformed expected identity")
            p = (p if p.is_absolute() else project / p).resolve()
            if str(p) in pins and pins[str(p)] != expected:
                raise ValueError("Contradictory expected identities")
            pins[str(p)] = expected
    for p in required_paths:
        if str(Path(p).resolve()) not in pins:
            raise ValueError(f"missing expected identity: {p}")
    for name, expected in sorted(pins.items()):
        if not Path(name).is_file() or digest(name) != expected:
            raise ValueError(f"Pinned identity mismatch: {name}")
    return dict(
        run=run,
        label=label,
        verified_count=len(pins),
        pins=pins,
        run_sha256=hashlib.sha256(raw).hexdigest(),
    )


def cadence_contract(time_s, nominal_dt_s=None):
    """Validate one physical sample lattice, without changing clocks or values.

    The interval AND accumulated lattice residual must fit 1 ppm of one
    interval plus 32-eps absolute-clock roundoff. This is not permission to
    replace an adaptive/drifting clock by a nominal one. Keep the observed
    cadence and jitter even when a declared cadence supplies the FFT scale.
    """
    t = np.asarray(time_s, dtype=float)
    if t.ndim != 1 or len(t) < 4 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Finite increasing timestamps required for cadence")
    intervals = np.diff(t)
    observed = float(np.mean(intervals))
    dt = observed if nominal_dt_s is None else float(nominal_dt_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive finite nominal cadence required")
    tolerance = dt * 1e-6 + 32 * np.finfo(float).eps * max(1, float(np.max(abs(t))))
    jitter = float(np.max(abs(intervals - dt)))
    residual = float(np.max(abs((t - t[0]) - np.arange(len(t)) * dt)))
    if jitter > tolerance or residual > tolerance:
        raise ValueError("Physical timestamps do not support the canonical cadence")
    return dict(
        dt_s=dt,
        nominal_dt_s=None if nominal_dt_s is None else dt,
        observed_mean_dt_s=observed,
        observed_min_dt_s=float(intervals.min()),
        observed_max_dt_s=float(intervals.max()),
        max_interval_jitter_s=jitter,
        max_lattice_residual_s=residual,
        validation_tolerance_s=tolerance,
        sample_support_s=[float(t[0]), float(t[-1])],
        sample_count=len(t),
        clock_replaced=False,
    )


def spectrum(
    time_s,
    values,
    nperseg,
    *,
    resample_dt_s=None,
    cadence_dt_s=None,
    native_to_uniform=False,
    retarded_interpolation=False,
):
    """Hann Welch/STFT on a validated cadence; explicit interpolation only."""
    t, x, _, resampled = uniform_record(time_s, values, resample_dt_s=resample_dt_s)
    nominal = resample_dt_s if cadence_dt_s is None else cadence_dt_s
    cadence = cadence_contract(t, nominal)
    dt = cadence["dt_s"]
    f, psd = welch_psd(x, 1 / dt, nperseg=nperseg)
    _, centers, columns = pressure_spectrogram(x, 1 / dt, nperseg=nperseg)
    hop = nperseg - nperseg // 2
    segments = 1 + (len(t) - nperseg) // hop
    result = dict(
        time_s=t,
        values=x,
        dt_s=dt,
        cadence=cadence,
        resampled=bool(native_to_uniform or retarded_interpolation or resampled),
        native_to_uniform=bool(native_to_uniform),
        retarded_interpolation=bool(retarded_interpolation),
        local_spectral=bool(resampled),
        frequency_Hz=f,
        psd=psd,
        integrated_power=integrated_power(f, psd),
        spectrogram_time_s=t[0] + centers,
        spectrogram_psd=columns,
        segments=segments,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        segment_detrend="constant",
        unused_tail_samples=len(t) - (nperseg + (segments - 1) * hop),
        frequency_resolution_Hz=float(f[1] - f[0]),
        frequency_bin_contract=dict(
            frequency_origin_Hz=float(f[0]),
            bin_count=len(f),
            bin_spacing_Hz=float(f[1] - f[0]),
            edge_snap_eps_multiplier=BIN_EDGE_EPS_MULTIPLIER,
            edge_snap_max_bins=BIN_EDGE_MAX_SNAP,
            edge_coordinate="q=(edge_Hz-frequency_origin_Hz)/bin_spacing_Hz",
            edge_snap_rule="nearest integer only within min(multiplier*eps*max(1,abs(q)),max_bins)",
            interior_rule="half_open_right_owns_boundary",
            final_upper_edge_inclusive=True,
        ),
    )
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nperseg) / nperseg)
    energy = []
    for i in range(0, len(x) - nperseg + 1, hop):
        y = x[i : i + nperseg] - np.mean(x[i : i + nperseg])
        energy.append(float(np.sum((w * y) ** 2) / np.sum(w * w)))
    result.update(
        density=result["psd"],
        columns=result["spectrogram_psd"],
        centers_s=result["spectrogram_time_s"],
        weighted_power=float(np.mean(energy)),
        bin_spacing_Hz=1 / (nperseg * dt),
        segment_duration_s=nperseg * dt,
        hop_s=hop * dt,
        sample_count=len(x),
    )
    if not np.allclose(
        result["density"],
        result["columns"].mean(axis=1),
        rtol=1e-12,
        atol=64 * np.finfo(float).eps * result["density"].max(),
    ):
        raise ValueError("Welch versus spectrogram column mean mismatch")
    if not np.isclose(result["integrated_power"], result["weighted_power"], rtol=1e-12, atol=0):
        raise ValueError("Window-weighted Parseval mismatch")
    return result


def level(power, reference=1e-6):
    """10 log10(power/reference²); no density floor, exact silence = -inf."""
    p = np.asarray(power, dtype=float)
    if not np.isfinite(p).all() or np.any(p < 0) or reference <= 0:
        raise ValueError("Nonnegative finite power and positive reference required")
    with np.errstate(divide="ignore"):
        return 10 * np.log10(p / reference**2)


def band_powers(frequency, density, edges):
    """Centre-selected FFT bins; half-open interior, inclusive final boundary.

    An edge q=(edge-f[0])/df snaps to its nearest integer only within
    min(32*eps*max(1, abs(q)), 1e-9) bins. This is a roundoff allowance,
    not a frequency-resolution tolerance. Each shared edge is snapped once.
    """
    from .analysis import integrated_power

    f, p, e = map(lambda a: np.asarray(a, dtype=float), (frequency, density, edges))
    integrated_power(f, p)  # validates density/grid, not reused inclusive band selector
    if e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all() or np.any(np.diff(e) <= 0) or e[0] < 0:
        raise ValueError("Bands outside frequency support or unordered")
    df = float(f[1] - f[0])
    indices = np.arange(len(f))
    grid_q = (f - f[0]) / df
    grid_tol = np.minimum(
        BIN_EDGE_EPS_MULTIPLIER * np.finfo(float).eps * np.maximum(1, indices), BIN_EDGE_MAX_SNAP
    )
    if np.any(abs(grid_q - indices) > grid_tol):
        raise ValueError("Frequency grid is not a roundoff-consistent FFT-bin lattice")
    edge_q = (e - f[0]) / df
    edge_tol = np.minimum(
        BIN_EDGE_EPS_MULTIPLIER * np.finfo(float).eps * np.maximum(1, abs(edge_q)),
        BIN_EDGE_MAX_SNAP,
    )
    nearest = np.rint(edge_q)
    edge_q = np.where(abs(edge_q - nearest) <= edge_tol, nearest, edge_q)
    if edge_q[-1] > indices[-1]:
        raise ValueError("Bands outside frequency support or unordered")
    result = []
    for i, (lo, hi) in enumerate(zip(e[:-1], e[1:])):
        left, right = edge_q[i : i + 2]
        mask = (indices >= left) & ((indices <= right) if i == len(e) - 2 else (indices < right))
        result.append(
            dict(
                low_Hz=float(lo),
                high_Hz=float(hi),
                final_inclusive=i == len(e) - 2,
                bin_count=int(mask.sum()),
                bin_indices=indices[mask].tolist(),
                edge_bin_coordinates=edge_q[i : i + 2].tolist(),
                edge_snap_tolerance_bins=edge_tol[i : i + 2].tolist(),
                bin_centers_Hz=f[mask].tolist(),
                bin_edges_Hz=[[float(v - df / 2), float(v + df / 2)] for v in f[mask]],
                effective_bandwidth_Hz=float(mask.sum() * df),
                power=float(p[mask].sum() * df),
            )
        )
    return result


def block_statistics(time_s, values, *, start, end, duration=2.0):
    """Complete half-open time blocks only; leftover support explicitly counted."""
    t, x = np.asarray(time_s), np.asarray(values)
    if (
        t.ndim != 1
        or x.shape != t.shape
        or not np.isfinite(t).all()
        or not np.isfinite(x).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Finite increasing paired block data required")
    if not np.isfinite(start + end + duration) or duration <= 0 or end <= start:
        raise ValueError("Invalid block window")
    blocks = []
    for i in range(int(np.floor((end - start) / duration + 1e-12))):
        lo, hi = start + i * duration, start + (i + 1) * duration
        mask = (t >= lo) & (t < hi)
        if mask.sum() < 4:
            continue
        y = x[mask]
        blocks.append(
            dict(
                start_s=lo,
                end_s=hi,
                samples=int(mask.sum()),
                first_sample_s=float(t[mask][0]),
                last_sample_s=float(t[mask][-1]),
                mean=float(y.mean()),
                ac_rms=float(np.std(y)),
            )
        )
    selected = (t >= start - 1e-12) & (t <= end + 1e-12)
    return dict(
        blocks=blocks,
        block_duration_s=duration,
        tail_samples=int(selected.sum()) - sum(b["samples"] for b in blocks),
        stationarity="not_established",
        uncertainty="descriptive_block_spread_not_CI",
        support_status="short_support" if len(blocks) < 4 else "descriptive_only",
    )


def compact_terms(time_s, force_N, arrival_s, observer_m, sound_speed):
    """Interior-derivative-supported retarded terms; exported force is on body."""
    from .cylinder_acoustics import compact_curle

    t = np.asarray(time_s)
    tau = np.asarray(arrival_s) - np.linalg.norm(observer_m) / sound_speed
    if np.any(tau < t[1] - 1e-12) or np.any(tau > t[-2] + 1e-12):
        raise ValueError("Outside interior derivative support; no extrapolation")
    return compact_curle(t, force_N, arrival_s, observer_m, sound_speed, force_convention="on_body")


def covariance_directivity(vector_history, angles_rad):
    """Population AC covariance: RMS(theta) = sqrt(n.T Cov(v) n)."""
    v, a = np.asarray(vector_history, dtype=float), np.asarray(angles_rad, dtype=float)
    if (
        v.ndim != 2
        or v.shape[1] != 3
        or len(v) < 4
        or not np.isfinite(v).all()
        or not np.isfinite(a).all()
    ):
        raise ValueError("Finite three-component history required")
    centered = v - v.mean(axis=0)
    covariance = centered.T @ centered / len(v)
    directions = np.column_stack((np.cos(a), np.sin(a), np.zeros_like(a)))
    power = np.einsum("ni,ij,nj->n", directions, covariance, directions)
    # Positive-semidefinite roundoff only, not a pressure/PSD display floor.
    return np.sqrt(np.maximum(power, 0)), covariance


def coupling(time_s, projection_N, pressure_Pa, nperseg):
    """Descriptive MSC/phase; short support cannot display misleading unit MSC."""
    from .cylinder_acoustics import coherence_phase

    result = coherence_phase(time_s, projection_N, pressure_Pa, nperseg=nperseg)
    if result["segments"] < 4:
        result["coherence"][:] = np.nan
    phase = np.where(result["phase_valid"], result["phase_rad"], np.nan)
    # Break wrapped phase lines rather than joining across a 2-pi discontinuity.
    jumps = np.flatnonzero(np.abs(np.diff(phase)) > np.pi) + 1
    phase[jumps] = np.nan
    result["display_phase_rad"] = phase
    return result


METHODS = ("libacoustics_curle", "libacoustics_fwh")
RING = ("cross_plus", "cross_minus", "downstream", "oblique")
FORCE_COLUMNS = ["time_s"] + [
    prefix + c + "_N" for prefix in ("force_", "pressure_force_", "viscous_force_") for c in "xyz"
]


def checked_bytes(path, pins):
    """Parse exactly the bytes compared with the retained expected digest."""
    p = Path(path).resolve()
    raw = p.read_bytes()
    if str(p) not in pins or hashlib.sha256(raw).hexdigest() != pins[str(p)]:
        raise ValueError(f"missing expected identity or identity mismatch: {p}")
    return raw


def load_records(
    *,
    run_path,
    forces_path,
    observers_path,
    case,
    project,
    start_time_s,
    end_time_s,
    resample_dt_s=None,
    historical_smoke=False,
):
    """Read original v1/v2 directly, preserving all metadata and native clocks."""
    from .cylinder_acoustics import uniform_record

    verification = verify_run(
        run_path,
        project=project,
        required_paths=[forces_path, observers_path],
        historical_smoke=historical_smoke,
    )
    run, pins = verification["run"], verification["pins"]
    if run["schema_version"] == 2:
        coverage, solver = run.get("time_coverage", {}), run.get("solver_verification", {})
        endpoint = coverage.get("end_s")
        if (
            not isinstance(endpoint, (int, float))
            or not np.isfinite(endpoint)
            or endpoint <= 0
            or endpoint != coverage.get("requested_end_s")
            or endpoint != solver.get("end_s")
            or solver.get("solver_reached_target") is not True
        ):
            raise ValueError("Inconsistent or missing native endpoint evidence")
    reader = csv.DictReader(io.StringIO(checked_bytes(forces_path, pins).decode()))
    if reader.fieldnames != FORCE_COLUMNS:
        raise ValueError("Require exact ten force columns")
    force = np.asarray([[float(row[c]) for c in FORCE_COLUMNS] for row in reader])
    if (
        force.ndim != 2
        or len(force) < 4
        or not np.isfinite(force).all()
        or np.any(np.diff(force[:, 0]) <= 0)
    ):
        raise ValueError("Force values/time invalid")
    if not np.allclose(force[:, 1:4], force[:, 4:7] + force[:, 7:10], rtol=2e-9, atol=1e-12):
        raise ValueError("Total force does not equal pressure plus viscous force")
    coords = run.get("observer_coordinates_m")
    if coords is None:
        coords = {row["name"]: row["position_m"] for row in run["observers"]}
    if set(coords) != set(RING) | {"cross_double"}:
        raise ValueError("Require four 1m observers and cross_double")
    center = np.asarray(run["geometry"]["centre_m"], dtype=float)
    for name, xyz in coords.items():
        a = np.asarray(xyz, dtype=float) - center
        if (
            a.shape != (3,)
            or not np.isfinite(a).all()
            or abs(a[2]) > 1e-9
            or not np.isclose(np.linalg.norm(a), 2 if name == "cross_double" else 1)
        ):
            raise ValueError("Observers must be equal-radius coplanar ring plus 2m repeat")
    if not np.allclose(
        np.asarray(coords["cross_double"]) - center, 2 * (np.asarray(coords["cross_plus"]) - center)
    ):
        raise ValueError("1m and 2m observers must share a ray")
    if run.get("force_convention") != "on_body" or run.get("force_content") != "total_traction":
        raise ValueError("Explicit on-body total-traction convention required")
    clock = run["clock"]
    if clock.get("forces_offset_s") != 0 or any(
        clock.get(m, {}).get("offset_s") != 0 for m in METHODS
    ):
        raise ValueError("Require explicit zero-offset absolute clocks; no implicit clock repair")
    if [clock[m].get("kind") for m in METHODS] != ["source_unretarded", "observer"]:
        raise ValueError("Native method clock semantics missing")
    reader = csv.DictReader(io.StringIO(checked_bytes(observers_path, pins).decode()))
    if reader.fieldnames != ["time_s", "observer", "method", "pressure_Pa"]:
        raise ValueError("Unexpected observer columns")
    channels = {(o, m): [] for o in coords for m in METHODS}
    rows = 0
    for row in reader:
        key = row["observer"], row["method"]
        if key not in channels:
            raise ValueError("Unexpected observer/method")
        channels[key].append([float(row["time_s"]), float(row["pressure_Pa"])])
        rows += 1
    channels = {key: np.asarray(value) for key, value in channels.items()}
    for a in channels.values():
        if a.shape != (len(force), 2) or not np.isfinite(a).all() or np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError("Incomplete, nonfinite or unordered observer channel")
        if np.max(abs(a[:, 0] - force[:, 0])) > 1e-8:
            raise ValueError("Observer/force absolute clocks disagree")
    counts = dict(force_rows=len(force), observer_rows=rows, observer_channels=len(channels))
    declared = run.get("tables", run.get("exports", {}).get("tables", {}))
    if declared.get("force_samples") != len(force) or declared.get("observer_rows") != rows:
        raise ValueError("Declared table counts disagree")
    t = force[:, 0]
    if np.max(np.diff(t)) > 1.5 * np.min(np.diff(t)):
        raise ValueError("Source clock gap; no interpolation across missing records")
    native_cadence = None
    if not run["sampling"]["adaptive"]:
        native_cadence = cadence_contract(t, run["sampling"]["fixed_dt_s"])
    lo = max(t[0], *(a[0, 0] for a in channels.values()))
    hi = min(t[-1], *(a[-1, 0] for a in channels.values()))
    if (
        not np.isfinite(start_time_s + end_time_s)
        or not lo <= start_time_s < end_time_s <= hi + 1e-9
    ):
        raise ValueError(f"Requested window outside support [{lo}, {hi}]; no extrapolation")
    exclusions = run.get("acoustic_conventions", {}).get(
        "suggested_warmup_exclusions_s", [[0, 0.01]]
    )
    for left, right in exclusions:
        if start_time_s < right and end_time_s > left:
            raise ValueError("Analysis window intersects native derivative/restart warmup")
    mask = (t >= start_time_s) & (t <= min(end_time_s, hi))
    grid, selected, dt, resampled = uniform_record(
        t[mask], force[mask, 1:], resample_dt_s=resample_dt_s
    )
    nominal = resample_dt_s if resampled else (native_cadence or {}).get("dt_s")
    cadence = cadence_contract(grid, nominal)
    dt = cadence["dt_s"]
    pressure = {key: np.interp(grid, a[:, 0], a[:, 1]) for key, a in channels.items()}
    # Equal native knots are selection, not interpolation; an admitted small
    # receiver-clock difference still requires an honest per-channel flag.
    pressure_resampled = {
        key: bool(
            resampled
            or not np.array_equal(
                a[np.minimum(np.searchsorted(a[:, 0], grid), len(a) - 1), 0], grid
            )
        )
        for key, a in channels.items()
    }
    return dict(
        run=run,
        verification=verification,
        counts=counts,
        coordinates=coords,
        force_native=force,
        channels_native=channels,
        time_s=grid,
        force=selected,
        pressure=pressure,
        pressure_resampled=pressure_resampled,
        dt_s=dt,
        resampled=resampled,
        cadence=cadence,
        native_cadence=native_cadence,
        exclusions=exclusions,
        requested_window_s=[start_time_s, end_time_s],
        actual_window_s=[float(grid[0]), float(grid[-1])],
        native_selected_samples=int(mask.sum()),
        case=Path(case),
        input_paths=[str(Path(p).resolve()) for p in (run_path, forces_path, observers_path)],
    )


def analyze_records(data, *, nperseg):
    """Nine-figure reductions, retaining full numerical bandwidth and all counts."""
    t, force, run = data["time_s"], data["force"], data["run"]
    start, end = data["requested_window_s"]
    medium, geometry = run["medium"], run["geometry"]
    c, dt = medium["sound_speed_m_s"], data["dt_s"]

    def native_spectrum(times, values, length, channel=None):
        upstream = (
            data.get("pressure_resampled", {}).get(channel, data["resampled"])
            if channel is not None
            else data["resampled"]
        )
        return spectrum(times, values, length, cadence_dt_s=dt, native_to_uniform=upstream)

    normalization = (
        0.5
        * medium["density_kg_m3"]
        * medium["velocity_m_s"] ** 2
        * geometry["diameter_m"]
        * geometry["span_m"]
    )
    forces = {
        key: native_spectrum(t, force[:, idx], nperseg) for key, idx in [("drag", 0), ("lift", 1)]
    }
    spectra = {
        key: native_spectrum(t, p, nperseg, channel=key) for key, p in data["pressure"].items()
    }
    f, density = forces["lift"]["frequency_Hz"], forces["lift"]["density"]
    search = (f > 0) & (f <= min(20.0, f[-1]))
    k = np.flatnonzero(search)[np.argmax(density[search])]
    peak = dict(
        candidate_Hz=float(f[k]),
        bin_spacing_Hz=float(f[1] - f[0]),
        peak_search_Hz=[float(f[1]), float(min(20.0, f[-1]))],
        record_candidate_cycles=float((t[-1] - t[0]) * f[k]),
        segment_candidate_cycles=float(nperseg * dt * f[k]),
        stationary_tone_established=False,
        interpretation="largest non-DC bin below 20 Hz; not a resolved stationary tone",
    )
    windows = []
    for lo in sorted(set([start] + [v for v in [4.0, 6.0] if v > start])):
        mask = t >= lo
        if mask.sum() < nperseg:
            windows.append(
                dict(
                    requested_start_s=lo,
                    status="insufficient_support",
                    segments=0,
                    samples=int(mask.sum()),
                )
            )
        else:
            s = native_spectrum(t[mask], force[mask, 1], nperseg)
            windows.append(
                dict(
                    requested_start_s=lo,
                    status="descriptive_only",
                    segments=s["segments"],
                    samples=int(mask.sum()),
                    spectrum=s,
                )
            )
    segment_sensitivities = []
    for n in sorted(set([max(4, nperseg // 2), nperseg])):
        segment_sensitivities.append(native_spectrum(t, force[:, 1], n))
    blocks = {
        key: block_statistics(t, values, start=start, end=end)
        for key, values in [
            ("lift", force[:, 1] / normalization),
            ("fwh", data["pressure"][("cross_plus", METHODS[1])]),
        ]
    }
    edges = [0, 1, 3, 10, 20]
    if f[-1] < 20 or f[1] > 1:
        raise ValueError("Insufficient frequency support for fixed 0–1–3–10–20 Hz bands")
    band_rows = []
    band_details = {}
    pressure_summary = []
    for (observer, method), s in spectra.items():
        bands = band_powers(f, s["density"], edges)
        pressure_summary.append(
            dict(
                observer=observer,
                method=method,
                samples=len(t),
                global_ac_variance_Pa2=float(np.var(data["pressure"][(observer, method)])),
                whole_nyquist_welch_power_Pa2=s["integrated_power"],
                window_weighted_power_Pa2=s["weighted_power"],
                analysed_band_power_Pa2=sum(b["power"] for b in bands),
                whole_nyquist_high_Hz=float(f[-1]),
                analysed_band_high_Hz=20.0,
            )
        )
        band_details[observer + "/" + method] = bands
        for i, b in enumerate(bands):
            band_rows.append(
                dict(
                    observer=observer,
                    method=method,
                    block=-1,
                    start_s=float(t[0]),
                    end_s=float(t[-1]),
                    samples=len(t),
                    band=i,
                    power_Pa2=b["power"],
                    level_dB=float(level(b["power"])),
                    spectral_bin_count=b["bin_count"],
                    segments=s["segments"],
                )
            )
        for j, b in enumerate(blocks["lift"]["blocks"]):
            mask = (t >= b["start_s"]) & (t < b["end_s"])
            if mask.sum() < nperseg:
                continue
            bs = native_spectrum(
                t[mask],
                data["pressure"][(observer, method)][mask],
                nperseg,
                channel=(observer, method),
            )
            for i, bp in enumerate(band_powers(bs["frequency_Hz"], bs["density"], edges)):
                band_rows.append(
                    dict(
                        observer=observer,
                        method=method,
                        block=j,
                        start_s=b["start_s"],
                        end_s=b["end_s"],
                        samples=int(mask.sum()),
                        band=i,
                        power_Pa2=bp["power"],
                        level_dB=float(level(bp["power"])),
                        spectral_bin_count=bp["bin_count"],
                        segments=bs["segments"],
                    )
                )
    direction = np.asarray(data["coordinates"]["cross_plus"], dtype=float) - np.asarray(
        geometry["centre_m"]
    )
    direction /= np.linalg.norm(direction)
    coherence = {}
    # Compare total on-fluid force; preserve raw opposite-sign/unretarded Curle.
    for m, idx in [(METHODS[0], 0), (METHODS[1], 3)]:
        coherence[m] = coupling(
            t, -force[:, idx : idx + 3] @ direction, data["pressure"][("cross_plus", m)], nperseg
        )
    kr = 2 * np.pi * f / c
    reference_phase = np.angle((1 + 1j * kr) * np.exp(-1j * kr))
    native = data["force_native"]
    ts = native[:, 0]
    compact_mask = (t - 1 / c >= ts[1]) & (t - 1 / c <= ts[-2])
    ct = t[compact_mask]
    near = []
    far = []
    for direction in np.eye(3):
        comp = compact_terms(ts, native[:, 4:7], ct, direction, c)
        near.append(comp["near_Pa"])
        far.append(comp["far_Pa"])
    near, far = np.column_stack(near), np.column_stack(far)
    total = near + far
    cross_direction = np.asarray(data["coordinates"]["cross_plus"]) - np.asarray(
        geometry["centre_m"]
    )
    terms = dict(
        time_s=ct,
        source_time_s=ct - 1 / c,
        near_Pa=near @ cross_direction,
        far_Pa=far @ cross_direction,
        pressure_Pa=total @ cross_direction,
    )
    # These histories bypass the uniform force table: their sole upstream
    # interpolation is native source/derivative -> retarded emission time.
    terms["near_spectrum"] = spectrum(
        ct, terms["near_Pa"], nperseg, cadence_dt_s=dt, retarded_interpolation=True
    )
    terms["far_spectrum"] = spectrum(
        ct, terms["far_Pa"], nperseg, cadence_dt_s=dt, retarded_interpolation=True
    )
    angles = np.deg2rad(np.arange(0, 361, 2))
    direct, cov = covariance_directivity(total, angles)
    radiating, fcov = covariance_directivity(far, angles)
    oracle = np.array([np.std(total @ np.array([np.cos(a), np.sin(a), 0])) for a in angles])
    directivity = dict(
        angles_rad=angles,
        total_rms_Pa=direct,
        radiating_rms_Pa=radiating,
        covariance_Pa2=cov,
        far_covariance_Pa2=fcov,
        covariance_max_abs_error_Pa=float(np.max(abs(oracle - direct))),
        samples=len(ct),
        emission_window_s=[float(ct[0] - 1 / c), float(ct[-1] - 1 / c)],
    )
    ratios = []
    for m in METHODS:
        a = data["channels_native"][("cross_plus", m)]
        b = data["channels_native"][("cross_double", m)]
        delay1, delay2 = (0, 0) if m == METHODS[0] else (1 / c, 2 / c)
        # A centre-delay alignment is a compact diagnostic; distributed face delays differ.
        tau = t[(t + delay1 <= a[-1, 0]) & (t + delay2 <= b[-1, 0])]
        first = np.interp(tau + delay1, a[:, 0], a[:, 1])
        second = np.interp(tau + delay2, b[:, 0], b[:, 1])
        ratios.append(
            dict(
                method=m,
                ratio_2m_over_1m=float(np.std(second) / np.std(first)),
                matched_samples=len(tau),
                emission_start_s=float(tau[0]),
                emission_end_s=float(tau[-1]),
                alignment="unretarded source clock"
                if m == METHODS[0]
                else "centre r/c matched emission; face-delay spread retained",
            )
        )
    tau = t[(t >= ts[1]) & (t <= ts[-2]) & (t + 2 / c <= min(a[-1, 0], b[-1, 0]))]
    p1 = compact_terms(ts, native[:, 4:7], tau + 1 / c, cross_direction, c)["pressure_Pa"]
    p2 = compact_terms(ts, native[:, 4:7], tau + 2 / c, 2 * cross_direction, c)["pressure_Pa"]
    ratios.append(
        dict(
            method="offline_compact",
            ratio_2m_over_1m=float(np.std(p2) / np.std(p1)),
            matched_samples=len(tau),
            emission_start_s=float(tau[0]),
            emission_end_s=float(tau[-1]),
            alignment="exact common compact emissions",
        )
    )
    return dict(
        force_spectra=forces,
        pressure_spectra=spectra,
        normalization_N=normalization,
        blocks=blocks,
        sensitivity_windows=windows,
        segment_sensitivities=segment_sensitivities,
        peak=peak,
        band_edges_Hz=edges,
        band_details=band_details,
        band_rows=band_rows,
        pressure_summary=pressure_summary,
        coupling=coherence,
        reference_phase_rad=reference_phase,
        compact=terms,
        directivity=directivity,
        ratios=ratios,
        stationarity="not_established",
    )


def load_geometry(data):
    """Native ASCII connectivity and latest supported raw midspan snapshot.

    Schema 2 does not currently pre-pin raw flow planes. Such planes retain an
    explicit analysis-capture identity (not falsely claimed as an execution pin).
    Native connectivity MUST be pre-pinned. No native helper with writes is called.
    """
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[2]
    helper = root / "scripts/view_cylinder_exports.py"
    spec = importlib.util.spec_from_file_location("_signature_export_viewer", helper)
    viewer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = viewer
    spec.loader.exec_module(viewer)
    mesh = data["case"] / "constant/polyMesh"
    pins = data["verification"]["pins"]
    points = np.asarray(
        [
            list(map(float, m.split()))
            for m in re.findall(
                r"^\(([^()]+)\)$", checked_bytes(mesh / "points", pins).decode(), re.M
            )
        ]
    )
    faces = np.asarray(
        [
            list(map(int, m.split()))
            for m in re.findall(
                r"^4\(([^()]+)\)$", checked_bytes(mesh / "faces", pins).decode(), re.M
            )
        ]
    )
    boundary = checked_bytes(mesh / "boundary", pins).decode()
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or faces.ndim != 2
        or faces.shape[1] != 4
    ):
        raise ValueError("Finite native ASCII points and quadrilateral connectivity required")
    if faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Native connectivity outside point support")
    section = faces[np.all(np.isclose(points[faces, 2], 0, atol=1e-10, rtol=0), axis=1)]
    edges = sorted(
        {
            tuple(sorted((int(a), int(b))))
            for face in section
            for a, b in zip(face, np.roll(face, -1))
        }
    )
    if not edges:
        raise ValueError("No exact native z=0 mesh section")
    wall = re.search(r"\bcylinder\s*\{([^}]+)\}", boundary).group(1)
    first = int(re.search(r"startFace\s+(\d+)", wall).group(1))
    count = int(re.search(r"nFaces\s+(\d+)", wall).group(1))
    wall_faces = faces[first : first + count]
    if len(wall_faces) != count or not count:
        raise ValueError("Incomplete native body connectivity")
    candidates = [
        p
        for p in (data["case"] / "postProcessing/flowSlice").glob("*/vorticity_midspan.raw")
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", p.parent.name)
        and float(p.parent.name) <= data["requested_window_s"][1]
    ]
    if not candidates:
        raise ValueError("No supported native midspan snapshot")
    raw_path = max(candidates, key=lambda p: float(p.parent.name))
    raw = raw_path.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    identity = "preexisting_execution_pin"
    if str(raw_path.resolve()) in pins:
        if source_hash != pins[str(raw_path.resolve())]:
            raise ValueError("Flow snapshot identity mismatch")
    elif data["run"]["schema_version"] == 2:
        identity = "analysis_capture_not_preexisting_execution_pin"
    else:
        raise ValueError("Missing historical flow identity")
    lines = raw.decode().splitlines()
    match = re.fullmatch(r"#\s+vorticity\s+POINT_DATA\s+(\d+)", lines[0])
    if not match or lines[1].split() != [
        "#",
        "x",
        "y",
        "z",
        "vorticity_x",
        "vorticity_y",
        "vorticity_z",
    ]:
        raise ValueError("Unexpected native vorticity raw header")
    values = np.loadtxt(io.StringIO(raw.decode()), ndmin=2)
    if values.shape != (int(match.group(1)), 6) or not np.isfinite(values).all():
        raise ValueError("Invalid raw snapshot values/count")
    g = data["run"]["geometry"]
    tri, audit = viewer.circular_triangulation(
        values[:, :3],
        outer_radius=g["outer_radius_m"],
        body_radius=g["diameter_m"] / 2,
        half_span=g["span_m"] / 2,
    )
    sources = {
        str((mesh / p).resolve()): dict(
            sha256=pins[str((mesh / p).resolve())], identity="preexisting_execution_pin"
        )
        for p in ["points", "faces", "boundary"]
    }
    sources[str(raw_path.resolve())] = dict(sha256=source_hash, identity=identity)
    sources[str(helper)] = dict(sha256=digest(helper), identity="analysis_dependency")
    return dict(
        mesh_segments_m=points[np.asarray(edges)],
        wall_faces_m=points[wall_faces],
        snapshot_values=values,
        triangulation=tri,
        audit=audit,
        time_s=float(raw_path.parent.name),
        sources=sources,
        native_points=len(points),
        native_faces=len(faces),
        section_faces=len(section),
    )


def preflight_output(output, allowed_root):
    """No output writes or input reads before destination containment is checked."""
    output, root = Path(output), Path(allowed_root)
    for path in (output, root):
        if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Unsafe output symlink or parent traversal")
    output, root = output.absolute(), root.absolute()
    if output == root or not output.is_relative_to(root) or output.exists():
        raise ValueError("Output must be a new child of the allowed root")
    return output


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {
            ("/".join(k) if isinstance(k, tuple) else str(k)): _jsonable(v)
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "-Infinity" if value == -np.inf else None
    return value


def _json(path, value):
    Path(path).write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _table(folder, name, headers, rows):
    path = folder / (name + ".csv")
    count = 0
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [format(v, ".17g") if isinstance(v, (float, np.floating)) else v for v in row]
            )
            count += 1
    return dict(
        path=folder.name + "/" + path.name, rows=count, columns=headers, sha256=digest(path)
    )


def _smeta(s):
    return {
        key: s[key]
        for key in [
            "dt_s",
            "resampled",
            "native_to_uniform",
            "retarded_interpolation",
            "local_spectral",
            "cadence",
            "frequency_bin_contract",
            "segments",
            "window",
            "nperseg",
            "noverlap",
            "segment_detrend",
            "unused_tail_samples",
            "bin_spacing_Hz",
            "sample_count",
            "segment_duration_s",
            "hop_s",
            "integrated_power",
            "weighted_power",
        ]
    }


STYLES = {
    METHODS[0]: dict(color="#167d8d", linestyle="-", label="Native Curle"),
    METHODS[1]: dict(color="#b35b36", linestyle="--", label="Native FW-H"),
    "offline_compact": dict(color="#30343b", linestyle="-.", label="Offline compact"),
}
PSD_LABEL = "PSD level [dB re (1 µPa)²/Hz]"
LIMITS = [
    "Short single realization; stationarity not established; no formal confidence interval.",
    "Unvalidated loading-analogy predictions; no measured sensor or detection-range claim.",
    "Native Curle and FW-H are alternatives, not additive contributions.",
    "No mesh/time convergence or full volume-quadrupole acoustic validation.",
]
EQUATIONS = {
    "psd": (
        "Sxx[k] = mean_j(g[k]|DFT(w*(x_j-mean(x_j)))|²/(fs*sum(w²))); "
        "periodic Hann; one-sided endpoints g=1, interior g=2; no zero padding"
    ),
    "blocks": "AC_RMS_B=sqrt(mean_B((x-mean_B(x))²)); complete nonoverlapping 2 s half-open blocks",
    "band": (
        "P_B=sum_{k in B} Spp[k] df; half-open interior bands, final upper edge inclusive; "
        "L_B=10log10(P_B/(1e-6 Pa)²)"
    ),
    "compact": (
        "F_fluid=-F_pressure_on_body; tau=t-r/c; p_near=n.F_fluid(tau)/(4*pi*r²); "
        "p_far=n.dF_fluid/dt(tau)/(4*pi*c*r); "
        "derivative on native timestamps before interpolation"
    ),
    "coherence": "S_Fp=conj(FFT(F_fluid))*FFT(p); MSC=|S_Fp|²/(S_FF*S_pp); phase=arg(S_Fp)",
    "directivity": (
        "RMS(theta)=sqrt(n(theta).Cov(p_vector).n(theta)); "
        "population AC covariance, common retarded support, z=0 and r=1 m"
    ),
}


def _render_report(data, result, geometry, stage):
    """Render scientific arrays once; all figures share deterministic style."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import SymLogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    run = data["run"]
    t = data["time_s"]
    f = result["force_spectra"]["lift"]["frequency_Hz"]
    end = data["requested_window_s"][1]

    label = (
        "TEST FIXTURE ONLY"
        if run.get("test_fixture_only")
        else (
            "HISTORICAL SMOKE ONLY"
            if data["verification"]["label"] == "historical_smoke_only"
            else "10 s native record · descriptive only"
        )
    )
    sources = {
        p: dict(
            sha256=digest(p),
            identity="run_manifest" if p == data["input_paths"][0] else "preexisting_execution_pin",
        )
        for p in data["input_paths"]
    }
    root = Path(__file__).resolve().parents[2]
    for p in [
        Path(__file__),
        root / "src/flowacoustics/analysis.py",
        root / "src/flowacoustics/cylinder_acoustics.py",
        root / "scripts/build_cylinder_signature.py",
    ]:
        if p.is_file():
            sources[str(p)] = dict(sha256=digest(p), identity="analysis_dependency")
    products = []

    def begin(identifier, title, rows=2, cols=2, polar=False):
        folder = stage / identifier
        folder.mkdir()
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(10.8, 7.4) if rows == 2 else (10.8, 5.8),
            squeeze=False,
            subplot_kw={"projection": "polar"} if polar else {},
        )
        fig.subplots_adjust(
            left=0.095,
            right=0.95,
            bottom=0.22 if polar else 0.17,
            top=0.73 if polar else 0.81,
            wspace=0.30,
            hspace=0.58,
        )
        fig.suptitle(title, fontsize=14, y=0.99)
        fig.text(0.5, 0.935, label, ha="center", fontsize=9, color="#52565c")
        for ax in axes.flat:
            ax.grid(True, alpha=0.18, linewidth=0.5)
            if not polar:
                ax.spines[["top", "right"]].set_visible(False)
        return folder, fig, list(axes.flat), []

    def method_legend(fig, include_offline=False):
        keys = [*METHODS] + (["offline_compact"] if include_offline else [])
        fig.legend(
            handles=[Line2D([], [], **STYLES[m]) for m in keys],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.902),
            ncol=len(keys),
            frameon=False,
        )

    def finish(folder, fig, axes, tables, caption, equations, layout, extra=None):
        caption_artist = fig.text(
            0.5, 0.075, caption, ha="center", va="center", fontsize=9, linespacing=1.6
        )
        fig.text(
            0.5,
            0.023,
            "Unvalidated loading-analogy prediction · short record; no stationarity claim",
            ha="center",
            fontsize=8,
            color="#5a5f64",
        )
        for ax in fig.axes:
            for collection in ax.collections:
                # I05 retains its disclosed field/colorbar raster fallback: Poppler
                # still showed seams with face edges, zero width and antialias off.
                if folder.name != "I05_spectrograms":
                    collection.set_rasterized(False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        caption_box = caption_artist.get_window_extent(renderer)
        legends = [*fig.legends] + [ax.get_legend() for ax in axes if ax.get_legend() is not None]
        overlaps = sum(
            legend.get_window_extent(renderer).overlaps(caption_box) for legend in legends
        )
        title_overlaps = sum(
            legend.get_window_extent(renderer).overlaps(ax.title.get_window_extent(renderer))
            for legend in legends
            for ax in axes
            if ax.get_title()
        )
        layout_audit = dict(legend_caption_overlaps=overlaps, legend_title_overlaps=title_overlaps)
        if overlaps or title_overlaps:
            raise ValueError("Legend/title/caption overlap detected")
        outputs = {}
        for ext in ["png", "svg", "pdf"]:
            p = folder / ("figure." + ext)
            metadata = (
                {"Date": None}
                if ext == "svg"
                else {"CreationDate": None, "ModDate": None}
                if ext == "pdf"
                else {"Software": "FlowAcoustics signature generator"}
            )
            fig.savefig(p, dpi=160, metadata=metadata)
            outputs[ext] = dict(path=folder.name + "/" + p.name, sha256=digest(p))
        record = dict(
            figure_id=folder.name,
            title=fig._suptitle.get_text(),
            caption=caption,
            evidence_label=label,
            origin="native histories and explicitly labelled offline reductions",
            sources=sources,
            source_metadata="run_metadata.json",
            stationarity="not_established",
            window=dict(
                requested_s=data["requested_window_s"],
                actual_samples_s=data["actual_window_s"],
                samples=len(t),
                native_selected_samples=data["native_selected_samples"],
                resampled=data["resampled"],
                resample_rule="linear interpolation; downsampling rejected",
                exclusions_s=data["exclusions"],
            ),
            estimator={
                k: v
                for k, v in _smeta(result["force_spectra"]["lift"]).items()
                if k not in ("integrated_power", "weighted_power")
            },
            equations=equations,
            limitations=LIMITS,
            numeric_tables=tables,
            outputs=outputs,
            layout=layout,
            layout_audit=layout_audit,
            axes=[
                dict(
                    title=ax.get_title(),
                    left_title=ax.get_title(loc="left"),
                    xlabel=ax.get_xlabel(),
                    ylabel=ax.get_ylabel(),
                    xlim=list(ax.get_xlim()),
                    ylim=list(ax.get_ylim()),
                    xscale=ax.get_xscale(),
                    yscale=ax.get_yscale(),
                )
                for ax in axes
            ],
        )
        if extra:
            record.update(extra)
        _json(folder / "manifest.json", record)
        products.append(
            dict(id=folder.name, manifest=folder.name + "/manifest.json", caption=caption)
        )
        plt.close(fig)

    def psd_table(folder, name, s, unit):
        entry = _table(folder, name, ["frequency_Hz", unit], zip(s["frequency_Hz"], s["density"]))
        entry.update(
            estimator=_smeta(s),
            sample_window_s=[float(s["time_s"][0]), float(s["time_s"][-1])],
            full_frequency_support_Hz=[0, float(s["frequency_Hz"][-1])],
        )
        return entry

    def positive_plot(ax, s, **style):
        valid = (s["frequency_Hz"] > 0) & (s["density"] > 0)
        ax.plot(s["frequency_Hz"][valid], s["density"][valid], **style)

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "svg.hashsalt": "flowacoustics-signature-v1",
        "pdf.fonttype": 42,
        "path.simplify": False,
        "axes.formatter.useoffset": False,
    }
    with plt.rc_context(style):
        # I01: actual connectivity, finite body, conservative circular field, true receivers.
        folder, fig, axes, tables = begin("I01_geometry", "Geometry, flow and native receivers")
        ax = axes[0]
        segments = geometry["mesh_segments_m"]
        R = run["geometry"]["outer_radius_m"]
        ax.add_collection(LineCollection(segments[:, :, :2], colors="#465863", linewidths=0.22))
        ax.set(
            xlim=(-R, R),
            ylim=(-R, R),
            aspect="equal",
            xlabel="x [m]",
            ylabel="y [m]",
            title="(a) Native midspan connectivity",
        )
        axes[1].remove()
        axes[1] = fig.add_subplot(222, projection="3d")
        ax = axes[1]
        ax.add_collection3d(
            Poly3DCollection(
                geometry["wall_faces_m"], facecolor="#d9dfe2", edgecolor="#607381", linewidths=0.2
            )
        )
        D = run["geometry"]["diameter_m"]
        L = run["geometry"]["span_m"]
        ax.set(
            xlim=(-D / 2, D / 2),
            ylim=(-D / 2, D / 2),
            zlim=(-L / 2, L / 2),
            xlabel="x [m]",
            ylabel="y [m]",
            zlabel="z [m]",
            title="(b) Native finite body / caps",
        )
        ax.set_box_aspect((D, D, L))
        ax.view_init(22, -48)
        ax.set_xticks([-D / 2, D / 2])
        ax.set_yticks([])  # Same diameter as x; avoid projected corner tick collisions.
        ax.set_zticks([-L / 2, 0, L / 2])
        ax.tick_params(labelsize=7, pad=0)
        # Tall physical aspect needs independent clearance from the z spine/title.
        ax.tick_params(axis="z", pad=8)
        ax.zaxis.labelpad = 14
        ax = axes[2]
        values = geometry["snapshot_values"]
        vmax = float(np.max(abs(values[:, 5])))
        if vmax == 0:
            raise ValueError("All-zero vorticity snapshot cannot support signed contour levels")
        norm = SymLogNorm(1, vmin=-vmax, vmax=vmax)
        levels = norm.inverse(np.linspace(0, 1, 49))
        levels[[0, -1]] = [-vmax, vmax]
        art = ax.tricontourf(
            geometry["triangulation"], values[:, 5], levels=levels, norm=norm, cmap="RdBu_r"
        )
        ax.add_patch(plt.Circle((0, 0), R, fill=False, color="#444444", lw=0.7))
        ax.add_patch(plt.Circle((0, 0), D / 2, color="#444444"))
        ax.set(
            xlim=(-R, R),
            ylim=(-R, R),
            aspect="equal",
            xlabel="x [m]",
            ylabel="y [m]",
            title=f"(c) Midspan ωz · t = {geometry['time_s']:g} s",
        )
        cb = fig.colorbar(art, ax=ax, fraction=0.035, pad=0.035)
        cb.set_label("ωz [s⁻¹]")
        cb.solids.set_rasterized(False)
        ax = axes[3]
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#b7bdc2", ls=":"))
        marker = ["o", "s", "^", "D", "v"]
        for i, name in enumerate([*RING, "cross_double"]):
            x, y, z = data["coordinates"][name]
            ax.plot(x, y, marker=marker[i], ls="none", mfc="white", mec="#30343b", ms=6, label=name)
        ax.set(
            xlim=(-1.25, 1.3),
            ylim=(-1.25, 2.25),
            aspect="equal",
            xlabel="x [m]",
            ylabel="y [m]",
            title="(d) Four 1 m sites + 2 m repeat",
        )
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
        tables.extend(
            [
                _table(
                    folder,
                    "mesh_segments",
                    ["x1_m", "y1_m", "z1_m", "x2_m", "y2_m", "z2_m"],
                    segments.reshape(-1, 6),
                ),
                _table(
                    folder,
                    "body_faces",
                    [f"{c}{i}_m" for i in range(4) for c in "xyz"],
                    geometry["wall_faces_m"].reshape(-1, 12),
                ),
                _table(
                    folder,
                    "snapshot",
                    ["x_m", "y_m", "z_m", "omega_x_s-1", "omega_y_s-1", "omega_z_s-1"],
                    values,
                ),
                _table(
                    folder,
                    "triangles",
                    ["point0", "point1", "point2", "masked"],
                    np.column_stack(
                        (
                            geometry["triangulation"].triangles,
                            geometry["triangulation"].mask.astype(int),
                        )
                    ),
                ),
                _table(
                    folder,
                    "observers",
                    ["name", "x_m", "y_m", "z_m"],
                    [[name, *data["coordinates"][name]] for name in [*RING, "cross_double"]],
                ),
            ]
        )
        finish(
            folder,
            fig,
            axes,
            tables,
            "Incompressible LES → surface loading → free-space pressure analogy.\n"
            "Stationary thickness is zero; exterior volume quadrupoles are not evaluated.",
            [
                "Native points/faces connectivity; piecewise-linear masked sample contours; "
                "signed vorticity_z [s^-1]"
            ],
            [2, 2],
            dict(
                sources={**sources, **geometry["sources"]},
                geometry_audit=geometry["audit"],
                snapshot_time_s=geometry["time_s"],
                native_points=geometry["native_points"],
                native_faces=geometry["native_faces"],
                section_faces=geometry["section_faces"],
            ),
        )
        # I02: complete native source trace, fixed blocks in the declared later interval.
        folder, fig, axes, tables = begin(
            "I02_record_support", "Source development and nonoverlapping blocks"
        )
        native = data["force_native"]
        q = result["normalization_N"]
        shade_keys = [Patch(color="#167d8d", alpha=0.07, label="Analysis interval")]
        if data["exclusions"]:
            shade_keys.append(Patch(color="#b35b36", alpha=0.3, label="Excluded interval"))
        if "startup_plot_interval_s" in run.get("time_coverage", {}):
            shade_keys.append(Patch(color="#777777", alpha=0.1, label="Startup context"))
        fig.legend(
            handles=shade_keys,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.902),
            ncol=len(shade_keys),
            frameon=False,
        )
        for i, key in enumerate(["Cd", "Cl"]):
            ax = axes[i]
            ax.plot(native[:, 0], native[:, i + 1] / q, color="#30343b", lw=0.75)
            ax.axvspan(*data["requested_window_s"], color="#167d8d", alpha=0.07)
            for lo, hi in data["exclusions"]:
                ax.axvspan(lo, hi, color="#b35b36", alpha=0.3)
            if "startup_plot_interval_s" in run.get("time_coverage", {}):
                ax.axvspan(
                    *run["time_coverage"]["startup_plot_interval_s"], color="#777777", alpha=0.1
                )
            ax.set(
                xlabel="Native CFD time [s]",
                ylabel=key + " [–]",
                title=f"({chr(97 + i)}) Entire {key} history",
            )
        for ax, key, unit in [
            (axes[2], "lift", "Cl [–]"),
            (axes[3], "fwh", "Cross-plus FW-H [Pa]"),
        ]:
            b = result["blocks"][key]["blocks"]
            centers = [(v["start_s"] + v["end_s"]) / 2 for v in b]
            ax.plot(
                centers,
                [v["mean"] for v in b],
                "o",
                mfc="white",
                color="#30343b",
                label="Block mean",
            )
            ax.plot(
                centers,
                [v["ac_rms"] for v in b],
                "s",
                mfc="white",
                color="#b35b36",
                label="Block AC RMS",
            )
            letter = "c" if key == "lift" else "d"
            ax.set(
                xlabel="Block centre [s]",
                ylabel=unit,
                title=f"({letter}) 2 s blocks · N = {len(b)}",
                xlim=data["requested_window_s"],
            )
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2, frameon=False)
            tables.append(
                _table(
                    folder,
                    key + "_blocks",
                    list(b[0]) if b else ["no_complete_blocks"],
                    [list(v.values()) for v in b] or [["insufficient support"]],
                )
            )
        tables.append(
            _table(
                folder,
                "entire_Cd_Cl",
                ["time_s", "Cd", "Cl"],
                np.column_stack((native[:, 0], native[:, 1:3] / q)),
            )
        )
        tables.append(
            _table(
                folder,
                "analysis_Cl_pressure",
                ["time_s", "Cl", "cross_plus_fwh_Pa"],
                zip(t, data["force"][:, 1] / q, data["pressure"][("cross_plus", METHODS[1])]),
            )
        )
        s = result["force_spectra"]["lift"]
        caption = (
            f"Actual samples {t[0]:.4f}–{t[-1]:.4f} s; fs = {1 / data['dt_s']:.3f} Hz; "
            f"Hann {s['segment_duration_s']:.3f} s, 50% overlap.\n"
            "2 s block spread is descriptive, not physical uncertainty; "
            "incomplete tail is excluded from blocks."
        )
        finish(
            folder,
            fig,
            axes,
            tables,
            caption,
            [EQUATIONS["blocks"], "Cd=Fx/(0.5*rho*U²*D*L); Cl=Fy/(0.5*rho*U²*D*L)"],
            [2, 2],
            dict(
                block_support=result["blocks"],
                history_window_s=[float(native[0, 0]), float(native[-1, 0])],
            ),
        )
        # I03: force evidence, estimator sensitivity, later-window sensitivity, local candidate.
        folder, fig, axes, tables = begin(
            "I03_force_spectra", "Loading spectra and estimator sensitivity"
        )
        for name, color in [("lift", "#30343b"), ("drag", "#667c92")]:
            s = result["force_spectra"][name]
            positive_plot(axes[0], s, color=color, label=name.capitalize())
            tables.append(psd_table(folder, name, s, "force_PSD_N2_Hz"))
        axes[0].legend(frameon=False)
        for s in result["segment_sensitivities"]:
            positive_plot(axes[1], s, label=f"{s['segment_duration_s']:.3f} s; K={s['segments']}")
            tables.append(psd_table(folder, "segment_" + str(s["nperseg"]), s, "lift_PSD_N2_Hz"))
        axes[1].legend(frameon=False)
        for row in result["sensitivity_windows"]:
            if "spectrum" in row:
                s = row["spectrum"]
                positive_plot(
                    axes[2], s, label=f"{row['requested_start_s']:g}–{end:g} s; K={s['segments']}"
                )
                tables.append(
                    psd_table(
                        folder,
                        "start_" + str(row["requested_start_s"]).replace(".", "p"),
                        s,
                        "lift_PSD_N2_Hz",
                    )
                )
        axes[2].legend(frameon=False)
        s = result["force_spectra"]["lift"]
        positive_plot(axes[3], s, color="#30343b")
        peak = result["peak"]
        axes[3].axvline(peak["candidate_Hz"], color="#b35b36", ls=":", lw=0.8)
        for i, ax in enumerate(axes):
            ax.set(
                xlabel="Frequency [Hz]",
                ylabel="Force PSD [N²/Hz]",
                xscale="log",
                yscale="log",
                xlim=(f[1], 20),
            )
            ax.set_title(
                [
                    "(a) Native total loading",
                    "(b) Same window, segment choice",
                    "(c) Same estimator, later window",
                    "(d) Lift candidate; not a fitted tone",
                ][i]
            )
        axes[3].set(xscale="linear", xlim=(0, min(8, 20)))
        for ax in axes:
            visible_density = []
            lo, hi = ax.get_xlim()
            for line in ax.lines:
                x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())
                if len(np.unique(x)) <= 1:  # Exclude the candidate-bin guide.
                    continue
                keep = (x >= lo) & (x <= hi) & np.isfinite(y) & (y > 0)
                visible_density.extend(y[keep])
            if visible_density:
                low, high = np.log10([min(visible_density), max(visible_density)])
                padding = 0.08 * max(high - low, 1.0)
                ax.set_ylim(10 ** (low - padding), 10 ** (high + padding))
        D, U = run["geometry"]["diameter_m"], run["medium"]["velocity_m_s"]
        secondary = axes[0].secondary_xaxis(
            "top", functions=(lambda f: f * D / U, lambda st: st * U / D)
        )
        secondary.set_xlabel("St = fD/U [–]")
        caption = (
            f"Candidate lift bin {peak['candidate_Hz']:.3f} Hz; "
            f"Δf = {peak['bin_spacing_Hz']:.3f} Hz; "
            f"{peak['segment_candidate_cycles']:.2f} candidate cycles/segment.\n"
            "No prescribed Strouhal number or artificial harmonics; "
            "later-window/segment changes are estimator sensitivities."
        )
        finish(
            folder,
            fig,
            axes,
            tables,
            caption,
            [EQUATIONS["psd"]],
            [2, 2],
            dict(
                peak=peak,
                sensitivity_windows=[
                    {k: v for k, v in row.items() if k != "spectrum"}
                    for row in result["sensitivity_windows"]
                ],
            ),
        )
        # I04: fixed paper-style 2x2 shared native receiver PSD axes.
        folder, fig, axes, tables = begin(
            "I04_receiver_spectra", "Native receiver spectral signature · r = 1 m"
        )
        method_legend(fig)
        for i, (ax, o) in enumerate(zip(axes, RING)):
            for m in METHODS:
                s = result["pressure_spectra"][(o, m)]
                p = s["density"]
                good = (f > 0) & (p > 0)
                ax.plot(f[good], level(p[good]), **STYLES[m])
                tables.append(psd_table(folder, o + "_" + m, s, "PSD_Pa2_Hz"))
            ax.set(
                xscale="log",
                xlim=(f[1], 20),
                xlabel="Frequency [Hz]",
                ylabel=PSD_LABEL,
                title=f"({chr(97 + i)}) {o}",
            )
        common = [min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes)]
        for ax in axes:
            ax.set_ylim(common)
        finish(
            folder,
            fig,
            axes,
            tables,
            "Same sample window and estimator at all four sites; "
            "full Nyquist arrays retained in CSV.\n"
            "0–20 Hz is a display/integration choice, not demonstrated CFD acoustic bandwidth.",
            [
                EQUATIONS["psd"],
                "PSD level=10log10(Spp/((1e-6 Pa)²/Hz)); zero density omitted, no display floor",
            ],
            [2, 2],
            dict(
                zero_density_omitted=True,
                display_floor=None,
                display_frequency_Hz=[float(f[1]), 20],
            ),
        )
        # I05: aligned STFTs, separate dimensional keys.
        folder, fig, axes, tables = begin(
            "I05_spectrograms", "Time-frequency support and evolution", 2, 1
        )
        for i, (ax, s, unit, title) in enumerate(
            zip(
                axes,
                [
                    result["force_spectra"]["lift"],
                    result["pressure_spectra"][("cross_plus", METHODS[1])],
                ],
                ["N²/Hz", "Pa²/Hz"],
                ["Total lift loading", "Cross-plus native FW-H"],
            )
        ):
            z = s["columns"]
            freq = s["frequency_Hz"]
            keep = (freq > 0) & (freq <= 20)
            # Positive linear density, no log floor or cross-unit common color key.
            art = ax.pcolormesh(
                s["centers_s"],
                freq[keep],
                z[keep],
                shading="nearest",
                cmap="viridis",
                rasterized=True,
                edgecolors="face",
                linewidth=0,
                antialiased=False,
            )
            art.set_antialiased(False)
            cb = fig.colorbar(art, ax=ax, pad=0.025, fraction=0.027)
            cb.set_label("PSD [" + unit + "]")
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")
            cb.solids.set_linewidth(0)
            cb.solids.set_antialiased(False)
            ax.set(
                xlim=(t[0], t[-1]),
                ylim=(0, 20),
                xlabel="Absolute segment-centre time [s]",
                ylabel="Frequency [Hz]",
                title=f"({chr(97 + i)}) {title}",
            )
            tables.append(
                _table(
                    folder,
                    "lift_stft" if i == 0 else "fwh_stft",
                    [
                        "centre_time_s",
                        "frequency_Hz",
                        "density_" + ("N2_Hz" if i == 0 else "Pa2_Hz"),
                    ],
                    (
                        (ti, fi, z[k, j])
                        for j, ti in enumerate(s["centers_s"])
                        for k, fi in enumerate(freq)
                    ),
                )
            )
        finish(
            folder,
            fig,
            axes,
            tables,
            f"Hann window {s['segment_duration_s']:.3f} s; hop {s['hop_s']:.3f} s; "
            "centres are not record endpoints.\nBlank margins have no segment centres; "
            "each column is a density FFT, and its mean reproduces Welch.",
            [EQUATIONS["psd"]],
            [2, 1],
            dict(
                color_scale="linear_nonnegative_density",
                display_floor=None,
                spectrogram_columns=s["segments"],
                rendering=dict(
                    pdf="mixed vector text/axes with rasterized QuadMesh fields and colorbars",
                    raster_dpi=160,
                    reason=(
                        "Poppler cell seams persisted with face edges/zero linewidth/antialias off"
                    ),
                    scientific_arrays_changed=False,
                ),
            ),
        )
        # I06: aggregate bands and actual nonoverlap block dots / descriptive range.
        folder, fig, axes, tables = begin(
            "I06_band_variability", "Band levels and descriptive block spread", 1, 2
        )
        fig.subplots_adjust(top=0.73, bottom=0.21)
        band_markers = [["o", "s", "^", "D"], ["P", "X", "v", "*"]]
        rows = result["band_rows"]
        summary_rows = result["pressure_summary"]
        tables.append(
            _table(
                folder,
                "pressure_power_estimands",
                list(summary_rows[0]),
                [list(row.values()) for row in summary_rows],
            )
        )
        tables.append(
            _table(folder, "all_band_levels", list(rows[0]), [list(row.values()) for row in rows])
        )
        bandlabels = ["0–1", "1–3", "3–10", "10–20"]
        for mi, m in enumerate(METHODS):
            for oi, o in enumerate(RING):
                rr = [
                    r for r in rows if r["observer"] == o and r["method"] == m and r["block"] == -1
                ]
                axes[0].plot(
                    np.arange(4) + (oi - 1.5) * 0.11 + (mi - 0.5) * 0.04,
                    [r["level_dB"] for r in rr],
                    ls="none",
                    marker=band_markers[mi][oi],
                    mfc="white",
                    color=STYLES[m]["color"],
                    ms=5,
                    label=STYLES[m]["label"] + " · " + o,
                )
            for bi in range(4):
                vals = [
                    r["level_dB"]
                    for r in rows
                    if r["observer"] == "cross_plus"
                    and r["method"] == m
                    and r["block"] >= 0
                    and r["band"] == bi
                ]
                x = bi + (mi - 0.5) * 0.14
                if vals:
                    axes[1].plot(
                        [x] * len(vals),
                        vals,
                        marker=band_markers[mi][0],
                        ls="none",
                        mfc="white",
                        color=STYLES[m]["color"],
                        ms=5,
                    )
                    axes[1].vlines(x, min(vals), max(vals), color=STYLES[m]["color"], lw=1.1)
        for ax in axes:
            ax.set(
                xticks=range(4),
                xticklabels=bandlabels,
                xlabel="Band [Hz]",
                ylabel="AC band pressure level [dB re 1 µPa]",
            )
        axes[0].set_title("(a) Four 1 m sites; entire analysis window")
        axes[1].set_title(
            f"(b) Cross-plus · {len(result['blocks']['lift']['blocks'])} complete 2 s blocks"
        )
        fig.legend(
            handles=[axes[0].lines[i] for oi in range(4) for i in (oi, oi + 4)],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.902),
            ncol=4,
            frameon=False,
        )
        finish(
            folder,
            fig,
            axes,
            tables,
            "Dots/ranges describe this realization; "
            "N is small and is not a 95% confidence interval.\n"
            "Linear bin powers add; global AC variance, whole-Nyquist Welch power "
            "and 0–20 Hz power are distinct.",
            [EQUATIONS["band"], EQUATIONS["blocks"]],
            [1, 2],
            dict(band_definitions=result["band_details"], block_support=result["blocks"]["lift"]),
        )
        # I07: preserve native clocks/signs; no unsupported short-record MSC/phase curves.
        folder, fig, axes, tables = begin(
            "I07_coherence_phase", "Loading / pressure coherence and phase", 2, 1
        )
        coupling_styles = {
            m: dict(
                **STYLES[m],
                marker=["o", "s"][i],
                markersize=3.5,
                markerfacecolor="none",
                markevery=(2 * i, 4),
                linewidth=1.3 if i == 0 else 1.0,
            )
            for i, m in enumerate(METHODS)
        }
        fig.legend(
            handles=[Line2D([], [], **coupling_styles[m]) for m in METHODS],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.902),
            ncol=2,
            frameon=False,
        )
        for m, s in result["coupling"].items():
            axes[0].plot(f, s["coherence"], **coupling_styles[m])
            axes[1].plot(f, s["display_phase_rad"], **coupling_styles[m])
            tables.append(
                _table(
                    folder,
                    m,
                    ["frequency_Hz", "MSC", "phase_rad", "phase_valid", "display_phase_rad"],
                    zip(
                        f,
                        s["coherence"],
                        s["phase_rad"],
                        s["phase_valid"].astype(int),
                        s["display_phase_rad"],
                    ),
                )
            )
        axes[0].set(
            ylim=(0, 1),
            ylabel="Magnitude-squared coherence [–]",
            title="(a) Descriptive gate: K ≥ 4",
        )
        axes[1].plot(
            f,
            result["reference_phase_rad"],
            color="#30343b",
            ls=":",
            label="Analytical compact phase",
        )
        axes[1].set(
            ylim=(-1.08 * np.pi, 1.08 * np.pi),
            yticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            yticklabels=["−π", "−π/2", "0", "π/2", "π"],
            ylabel="Signed phase [rad]",
            title="(b) Phase only on supported coherent bins",
        )
        axes[1].legend(
            handles=[Line2D([], [], color="#30343b", ls=":", label="Analytical compact reference")],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.33),
            frameon=False,
        )
        for ax in axes:
            ax.set(xlim=(f[1], 20), xscale="log", xlabel="Frequency [Hz]")
        tables.append(
            _table(
                folder,
                "analytical_reference",
                ["frequency_Hz", "phase_rad"],
                zip(f, result["reference_phase_rad"]),
            )
        )
        statuses = "; ".join(
            STYLES[m]["label"] + ": " + s["status"] for m, s in result["coupling"].items()
        )
        finish(
            folder,
            fig,
            axes,
            tables,
            "Force on fluid; raw Curle remains opposite-sign/unretarded. "
            "FW-H uses pressure-only force.\n" + statuses,
            [
                EQUATIONS["coherence"],
                "H=(r^-2+i*omega/(c*r))*exp(-i*omega*r/c)/(4*pi); reference phase=atan(kr)-kr",
            ],
            [2, 1],
            dict(
                phase_gates=dict(
                    min_segments=4,
                    min_MSC=0.5,
                    relative_both_PSD_support=1e-10,
                    significance_test=False,
                ),
                coupling_status={m: s["status"] for m, s in result["coupling"].items()},
            ),
        )
        # I08: only OFFLINE terms labelled reactive/radiating, and actual same-ray ratios.
        folder, fig, axes, tables = begin(
            "I08_near_radiating", "Compact reactive / radiating terms and range diagnostic"
        )
        comp = result["compact"]
        ct = comp["time_s"]
        for i, (term, title) in enumerate(
            [("near_Pa", "Reactive 1/r² term"), ("far_Pa", "Radiating 1/r term")]
        ):
            p = comp[term]
            axes[i].plot(ct, p - p.mean(), color="#30343b", lw=0.85)
            axes[i].set(
                xlabel="Observer time [s]",
                ylabel="Offline AC pressure [Pa]",
                title=f"({chr(97 + i)}) {title} · r = 1 m",
            )
        for key, ls, title in [
            ("near_spectrum", "-", "Reactive"),
            ("far_spectrum", "--", "Radiating"),
        ]:
            s = comp[key]
            positive_plot(axes[2], s, color="#30343b", ls=ls, label=title)
            tables.append(psd_table(folder, key, s, "PSD_Pa2_Hz"))
        axes[2].set(
            xscale="log",
            yscale="log",
            xlim=(f[1], 20),
            xlabel="Frequency [Hz]",
            ylabel="Offline PSD [Pa²/Hz]",
            title="(c) Separate terms, not native decomposition",
        )
        axes[2].legend(frameon=False)
        ratios = result["ratios"]
        for i, row in enumerate(ratios):
            axes[3].plot(
                i, row["ratio_2m_over_1m"], "o", mfc="white", color=STYLES[row["method"]]["color"]
            )
        axes[3].axhline(0.25, color="#999999", ls=":", label="Reactive reference 0.25")
        axes[3].axhline(0.5, color="#555555", ls="--", label="Radiating reference 0.50")
        axes[3].set(
            xticks=range(3),
            xticklabels=["Curle", "FW-H", "Compact"],
            ylabel="AC RMS(2 m) / AC RMS(1 m) [–]",
            title="(d) Actual matched-support range ratios",
        )
        axes[3].legend(loc="center", frameon=False)
        tables.append(
            _table(
                folder,
                "compact_terms",
                ["observer_time_s", "emission_time_s", "near_Pa", "far_Pa", "total_Pa"],
                zip(
                    ct, comp["source_time_s"], comp["near_Pa"], comp["far_Pa"], comp["pressure_Pa"]
                ),
            )
        )
        tables.append(
            _table(folder, "range_ratios", list(ratios[0]), [list(row.values()) for row in ratios])
        )
        kr = 2 * np.pi * result["peak"]["candidate_Hz"] / run["medium"]["sound_speed_m_s"]
        a = np.hypot(D / 2, L / 2)
        finish(
            folder,
            fig,
            axes,
            tables,
            f"Offline pressure-loading compact model; a/r = {a:.4f}, "
            f"candidate kr = {kr:.4f} at 1 m.\nNear + far cross covariance need not vanish; "
            "centre-delay FW-H alignment is approximate.",
            [
                EQUATIONS["compact"],
                "range ratio = std(p_2m)/std(p_1m) on common supported emission times",
            ],
            [2, 2],
            dict(
                ratios=ratios,
                compact_samples=len(ct),
                compact_window_s=[float(ct[0]), float(ct[-1])],
                kr_candidate=kr,
                source_size_over_r=a,
            ),
        )
        # I09: native markers NEVER connected; dense curves are a labelled covariance model.
        folder, fig, axes, tables = begin(
            "I09_directivity",
            "Sparse native support versus an offline angular model",
            1,
            2,
            polar=True,
        )
        marker_rows = []
        native_styles = {
            m: dict(
                marker="o" if i == 0 else "s",
                ls="none",
                mfc="none",
                ms=7 if i == 0 else 5,
                color=STYLES[m]["color"],
                label=STYLES[m]["label"],
                zorder=3,
            )
            for i, m in enumerate(METHODS)
        }
        for m in METHODS:
            for o in RING:
                xyz = np.asarray(data["coordinates"][o]) - np.asarray(run["geometry"]["centre_m"])
                theta = np.arctan2(xyz[1], xyz[0])
                rms = float(np.std(data["pressure"][(o, m)]))
                axes[0].plot([theta], [rms], **native_styles[m])
                marker_rows.append([o, m, theta, rms])
        direct = result["directivity"]
        a = direct["angles_rad"]
        axes[0].plot(a, direct["total_rms_Pa"], color="#30343b", ls="-.", label="Compact total")
        axes[0].plot(
            a, direct["radiating_rms_Pa"], color="#30343b", ls="--", label="Compact radiating only"
        )
        axes[1].plot(
            a, direct["radiating_rms_Pa"], color="#30343b", ls="--", label="Compact radiating only"
        )
        fig.legend(
            handles=[
                *[Line2D([], [], **native_styles[m]) for m in METHODS],
                Line2D([], [], color="#30343b", ls="-.", label="Compact total"),
                Line2D([], [], color="#30343b", ls="--", label="Compact radiating only"),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.902),
            ncol=4,
            frameon=False,
        )
        axes[0].set_title("(a) All methods · shared absolute Pa scale", pad=14)
        axes[1].set_title("(b) Offline radiating only · separate Pa scale", pad=14)
        maxr = max(max(row[3] for row in marker_rows), float(direct["total_rms_Pa"].max()))
        axes[0].set_ylim(0, maxr * 1.10)
        farmax = float(direct["radiating_rms_Pa"].max())
        if farmax > 0:
            axes[1].set_ylim(0, farmax * 1.10)
        for ax in axes:
            ax.set_ylabel("AC RMS pressure [Pa]", labelpad=28)
            ax.set_rlabel_position(35)
        # Keep the separate-scale labels inside the clear radiating lobe.
        axes[1].set_rlabel_position(105)
        tables.append(
            _table(
                folder,
                "native_markers",
                ["observer", "method", "angle_rad", "AC_RMS_Pa"],
                marker_rows,
            )
        )
        tables.append(
            _table(
                folder,
                "offline_dense",
                ["angle_rad", "total_RMS_Pa", "radiating_RMS_Pa"],
                zip(a, direct["total_rms_Pa"], direct["radiating_rms_Pa"]),
            )
        )
        tables.append(
            _table(
                folder,
                "covariance",
                ["component_i", "component_j", "total_Pa2", "radiating_Pa2"],
                (
                    (i, j, direct["covariance_Pa2"][i, j], direct["far_covariance_Pa2"][i, j])
                    for i in range(3)
                    for j in range(3)
                ),
            )
        )
        finish(
            folder,
            fig,
            axes,
            tables,
            "Absolute linear Pa RMS; separate radiating scale, not normalized; "
            "native markers unconnected.\n"
            "Offline curves at 2° sampling are force-model consequences, "
            "not additional CFD receivers.",
            [EQUATIONS["compact"], EQUATIONS["directivity"]],
            [1, 2],
            dict(
                native_angular_support=4,
                native_connecting_lines=False,
                dense_angles=len(a),
                covariance_check_max_abs_Pa=direct["covariance_max_abs_error_Pa"],
                emission_window_s=direct["emission_window_s"],
            ),
        )
    return products


def build_report(*, output, allowed_root=None, nperseg=3200, **load_args):
    """Stage a complete report and atomically install only into an absent target."""
    import ctypes
    import os
    import shutil
    import tempfile

    import matplotlib
    import scipy

    allowed_root = (
        allowed_root or Path(__file__).resolve().parents[2] / "results/cylinder/signature"
    )
    output = preflight_output(output, allowed_root)
    data = load_records(**load_args)
    result = analyze_records(data, nperseg=nperseg)
    geometry = load_geometry(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".signature-stage-", dir=output.parent))
    try:
        _json(stage / "run_metadata.json", data["run"])
        products = _render_report(data, result, geometry, stage)
        summary = dict(
            schema_version=1,
            evidence_label="test_fixture_only"
            if data["run"].get("test_fixture_only")
            else data["verification"]["label"],
            figures=products,
            figure_count=len(products),
            counts=data["counts"],
            requested_window_s=data["requested_window_s"],
            actual_window_s=data["actual_window_s"],
            analysis_samples=len(data["time_s"]),
            native_selected_samples=data["native_selected_samples"],
            estimator=_smeta(result["force_spectra"]["lift"]),
            peak=result["peak"],
            blocks=result["blocks"],
            ratios=result["ratios"],
            stationarity="not_established",
            run_sha256=data["verification"]["run_sha256"],
            verified_expected_identities=data["verification"]["verified_count"],
            versions=dict(
                numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__
            ),
            numerical_checks=dict(
                covariance_max_abs_error_Pa=result["directivity"]["covariance_max_abs_error_Pa"],
                weighted_parseval_max_abs_error=max(
                    abs(s["integrated_power"] - s["weighted_power"])
                    for s in [
                        *result["force_spectra"].values(),
                        *result["pressure_spectra"].values(),
                    ]
                ),
            ),
            limitations=LIMITS,
            source_identity_scope=(
                "All existing run pins verified; any unpinned v2 raw plane "
                "explicitly labelled analysis capture in I01"
            ),
        )
        # Recheck the exact target inputs, not a newly blessed set of files.
        if digest(load_args["run_path"]) != data["verification"]["run_sha256"]:
            raise ValueError("Run manifest changed during analysis")
        for name, expected in data["verification"]["pins"].items():
            if digest(name) != expected:
                raise ValueError("Pinned source changed during analysis: " + name)
        for name, record in geometry["sources"].items():
            if digest(name) != record["sha256"]:
                raise ValueError("Geometry source changed during analysis: " + name)
        summary["output_sha256"] = {
            p.relative_to(stage).as_posix(): digest(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        _json(stage / "manifest.json", summary)
        preflight_output(output, allowed_root)
        # Linux renameat2(RENAME_NOREPLACE): no check-then-rename overwrite race.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(output), 1) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
        return summary
    finally:
        if stage.exists():
            shutil.rmtree(stage)
