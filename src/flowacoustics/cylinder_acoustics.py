"""Stationary compact loading acoustics. SI throughout; see CYLINDER_ACOUSTICS.md."""

from __future__ import annotations

import numpy as np
from scipy import signal

from .analysis import integrated_power, pressure_spectrogram, welch_psd


def uniform_record(time_s, values, *, resample_dt_s=None):
    """Validate regular sampling or explicitly interpolate; never silently FFT adaptive time."""
    t, a = _time(time_s), _real(values, "record")
    if a.ndim not in (1, 2) or a.shape[0] != len(t):
        raise ValueError("Record and time lengths must match")
    intervals = np.diff(t)
    dt = float(np.mean(intervals))
    if resample_dt_s is None:
        if not np.allclose(intervals, dt, rtol=1e-6, atol=1e-12):
            raise ValueError("Nonuniform timestamps require explicit resample_dt_s")
        return t, a, dt, False
    dt = _positive(resample_dt_s, "resampling interval")
    if dt > intervals.min() * (1 + 1e-6):
        raise ValueError("Refusing downsampling without an anti-alias filter")
    grid = t[0] + np.arange(int(np.floor((t[-1] - t[0]) / dt)) + 1) * dt
    grid = grid[grid <= t[-1]]
    if a.ndim == 1:
        a = np.interp(grid, t, a)
    else:
        a = np.column_stack([np.interp(grid, t, column) for column in a.T])
    return grid, a, dt, True


def spectral_bundle(time_s, values, *, nperseg, resample_dt_s=None):
    """One-sided Hann densities in input-unit²/Hz; full spectral arrays, no dB floor."""
    t, a, dt, resampled = uniform_record(time_s, values, resample_dt_s=resample_dt_s)
    f, psd = welch_psd(a, 1 / dt, nperseg=nperseg)
    _, centers, spectrogram = pressure_spectrogram(a, 1 / dt, nperseg=nperseg)
    step = nperseg - nperseg // 2
    segments = 1 + (len(t) - nperseg) // step
    return {
        "time_s": t,
        "values": a,
        "dt_s": dt,
        "resampled": resampled,
        "frequency_Hz": f,
        "psd": psd,
        "integrated_power": integrated_power(f, psd),
        "spectrogram_time_s": t[0] + centers,
        "spectrogram_psd": spectrogram,
        "segments": segments,
        "window": "hann",
        "nperseg": nperseg,
        "noverlap": nperseg // 2,
        "segment_detrend": "constant",
        "unused_tail_samples": len(t) - (nperseg + (segments - 1) * step),
        "frequency_resolution_Hz": float(f[1] - f[0]),
    }


def coherence_phase(time_s, force_projection, pressure, *, nperseg, resample_dt_s=None):
    """CSD = conj(force FFT)*pressure FFT; positive phase means pressure leads.

    Four overlapping segments minimum, coherence >= 0.5, and both spectra above
    1e-10 of their maxima. Descriptive gate, NOT a statistical significance test.
    """
    x, y = _real(force_projection, "force projection"), _real(pressure, "pressure")
    if x.ndim != 1 or y.shape != x.shape:
        raise ValueError("Coherence requires paired one-dimensional records")
    t, xy, dt, resampled = uniform_record(
        time_s, np.column_stack((x, y)), resample_dt_s=resample_dt_s
    )
    f, px = welch_psd(xy[:, 0], 1 / dt, nperseg=nperseg)
    _, py = welch_psd(xy[:, 1], 1 / dt, nperseg=nperseg)
    _, cross = signal.csd(
        xy[:, 0],
        xy[:, 1],
        fs=1 / dt,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    denominator = px * py
    coherence = np.divide(
        abs(cross) ** 2, denominator, out=np.zeros_like(px), where=denominator > 0
    )
    coherence = np.clip(coherence, 0, 1)
    segments = 1 + (len(t) - nperseg) // (nperseg - nperseg // 2)
    valid = (
        (segments >= 4)
        & (coherence >= 0.5)
        & (f > 0)
        & (px > px.max() * 1e-10)
        & (py > py.max() * 1e-10)
    )
    return {
        "frequency_Hz": f,
        "coherence": coherence,
        "phase_rad": np.angle(cross),
        "phase_valid": valid,
        "segments": segments,
        "resampled": resampled,
        "status": "descriptive" if segments >= 4 else "insufficient_segments",
    }


def _real(values, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    a = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    return a


def _time(values, minimum=4):
    t = _real(values, "time")
    if t.ndim != 1 or len(t) < minimum or np.any(np.diff(t) <= 0):
        raise ValueError(f"Time must be strictly increasing with at least {minimum} samples")
    return t


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def compact_curle(
    time_s, force_N, observer_time_s, observer_m, sound_speed_m_s, *, force_convention
):
    """Retarded point-force pressure; force-on-body is minus force-on-fluid.

    p = n·[F_fluid(t-r/c)/r² + dF_fluid(t-r/c)/dt/(c r)]/(4 pi).
    Differentiate on source timestamps, then linearly interpolate each term.
    """
    t = _time(time_s)
    force = _real(force_N, "force")
    arrival = _time(observer_time_s, minimum=1)
    vector = _real(observer_m, "observer")
    if force.shape != (len(t), 3) or vector.shape != (3,):
        raise ValueError("Force must have shape (samples,3); observer must have shape (3,)")
    sound_speed_m_s = _positive(sound_speed_m_s, "sound speed")
    if force_convention not in ("on_body", "on_fluid"):
        raise ValueError("force_convention must be on_body or on_fluid")
    r = _positive(np.linalg.norm(vector), "source-observer distance")
    n = vector / r
    tau = arrival - r / sound_speed_m_s
    # Boundary round-off only; never extrapolate an unsupported history.
    tolerance = 32 * np.finfo(float).eps * max(1, abs(t[0]), abs(t[-1]))
    if np.any(tau < t[0] - tolerance) or np.any(tau > t[-1] + tolerance):
        raise ValueError("Retarded source time outside recorded support; no zero extension")
    tau = np.clip(tau, t[0], t[-1])
    sign = -1 if force_convention == "on_body" else 1
    projected = sign * (force @ n)
    derivative = np.gradient(projected, t, edge_order=2)
    near = np.interp(tau, t, projected) / (4 * np.pi * r**2)
    far = np.interp(tau, t, derivative) / (4 * np.pi * sound_speed_m_s * r)
    return {"source_time_s": tau, "near_Pa": near, "far_Pa": far, "pressure_Pa": near + far}


def _trace(name, x, y):
    return {"name": name, "x": np.asarray(x).tolist(), "y": np.asarray(y).tolist()}


def _chart(
    identifier,
    title,
    origin,
    traces,
    samples,
    *,
    description,
    x_label="Time (s)",
    y_label="Pressure (Pa)",
    kind="line",
    **extra,
):
    return {
        "id": identifier,
        "title": title,
        "origin": origin,
        "kind": kind,
        "description": description,
        "x_label": x_label,
        "y_label": y_label,
        "traces": traces,
        "samples": samples,
        **extra,
    }


def _samples(t, total, **extra):
    return {
        "included": len(t),
        "excluded": total - len(t),
        "start_s": float(t[0]),
        "end_s": float(t[-1]),
        **extra,
    }


def build_dataset(
    run,
    forces,
    observer_rows,
    *,
    nperseg=256,
    resample_dt_s=None,
    start_time_s=None,
    end_time_s=None,
):
    """Return the site-contract dataset and a long-form aligned pressure table.

    Metadata schema is intentionally explicit (see documentation). Unknown clocks,
    force conventions or missing medium values are errors, never guessed defaults.
    """
    if run["schema_version"] != 1:
        raise ValueError("Unsupported run schema_version")
    if run["force_convention"] not in ("on_body", "on_fluid"):
        raise ValueError("Explicit force_convention must be on_body or on_fluid")
    if run["force_content"] not in ("total_traction", "pressure_only"):
        raise ValueError("Explicit force_content must be total_traction or pressure_only")
    names = [o["name"] for o in run["observers"]]
    if len(set(names)) != len(names):
        raise ValueError("duplicate observer names")
    medium, geometry = run["medium"], run["geometry"]
    rho = _positive(medium["density_kg_m3"], "density")
    c = _positive(medium["sound_speed_m_s"], "sound speed")
    u = _positive(medium["velocity_m_s"], "inflow velocity")
    nu = _positive(medium["kinematic_viscosity_m2_s"], "viscosity")
    d = _positive(geometry["diameter_m"], "diameter")
    span = _positive(geometry["span_m"], "span")
    raw = _real(forces, "forces")
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError("Force table must contain time and three components in N")
    full_t = _time(raw[:, 0] + run["clock"]["forces_offset_s"])
    selected = np.ones(len(full_t), dtype=bool)
    if start_time_s is not None:
        selected &= full_t >= start_time_s
    if end_time_s is not None:
        selected &= full_t <= end_time_s
    t = _time(full_t[selected])
    force = raw[selected, 1:]
    origin = "analytical" if run.get("evidence_kind") == "analytical_fixture" else "computed"
    data = {
        "schema_version": 1,
        "status": "analytical_fixture" if origin == "analytical" else "partial",
        "summary": "Educational analysis; no claim of CFD convergence or research validation.",
        "metadata": {
            "medium": medium,
            "geometry": geometry,
            "native_status": run["status"],
            "evidence_kind": run.get("evidence_kind", "unverified"),
            "reynolds_number": u * d / nu,
            "mach_number": u / c,
            "pressure_reference_Pa": 1e-6,
            "force_convention": run["force_convention"],
            "force_content": run["force_content"],
            "clock": run["clock"],
            "transient_policy": "No automatic transient removal; explicit time window only",
            "resample_dt_s": resample_dt_s,
            "zero_floor_rule": "No numeric floor: omit zero on log axes; null is unsupported",
        },
        "charts": [],
        "checks": [],
        "downloads": [],
    }
    charts, checks = data["charts"], data["checks"]
    samples = _samples(t, len(full_t))
    charts.append(
        _chart(
            "forces",
            "Recorded cylinder forces",
            origin,
            [_trace(axis, t, force[:, i]) for i, axis in enumerate(("Fx", "Fy", "Fz"))],
            samples,
            description="Full selected samples in N; mean is retained.",
            y_label="Force (N)",
        )
    )
    denominator = 0.5 * rho * u**2 * d * span
    body_sign = 1 if run["force_convention"] == "on_body" else -1
    charts.append(
        _chart(
            "force_coefficients",
            "Drag, lift and axial force coefficients",
            origin,
            [
                _trace(name, t, body_sign * force[:, i] / denominator)
                for i, name in enumerate(("Cd", "Cl", "Cz"))
            ],
            samples,
            description="Force-on-body coefficients; reference area D × physical span.",
            y_label="Force / (0.5 ρ U² D span)",
        )
    )
    spectra = [
        spectral_bundle(t, force[:, i], nperseg=nperseg, resample_dt_s=resample_dt_s)
        for i in range(3)
    ]
    spec = spectra[1]
    charts.append(
        _chart(
            "force_psd",
            "Force spectra and observed lift peak",
            origin,
            [
                _trace(name, s["frequency_Hz"], s["psd"])
                for name, s in zip(("Fx", "Fy", "Fz"), spectra)
            ],
            {
                **samples,
                "welch_unused_tail_samples": spec["unused_tail_samples"],
                "welch_used_samples": len(spec["time_s"]) - spec["unused_tail_samples"],
            },
            description="One-sided Hann Welch density; segment mean removed; 50% overlap.",
            x_label="Frequency (Hz)",
            y_label="Force PSD (N²/Hz)",
            y_scale="log",
        )
    )
    peak = int(np.argmax(spec["psd"][1:]) + 1)
    freq = float(spec["frequency_Hz"][peak])
    cycles = freq * (t[-1] - t[0])
    resolved = bool(spec["psd"][peak] > 0 and 2 <= peak < len(spec["psd"]) - 1 and cycles >= 4)
    data["metadata"]["shedding"] = {
        "status": "candidate_lift_peak" if resolved else "unresolved",
        "frequency_Hz": freq if resolved else None,
        "strouhal": freq * d / u if resolved else None,
        "frequency_resolution_Hz": spec["frequency_resolution_Hz"],
        "strouhal_bin_width": spec["frequency_resolution_Hz"] * d / u,
        "cycles_in_window": float(cycles),
        "interpretation": (
            "Measured spectral candidate, not a prescribed St or proof of developed shedding"
        ),
    }
    a = float(np.hypot(d / 2, span / 2))
    k = 2 * np.pi * freq / c if resolved else None
    data["metadata"]["compactness"] = {
        "source_radius_bound_m": a,
        "k_a_at_lift_peak": k * a if k is not None else None,
        "k_r_at_lift_peak": {
            obs["name"]: float(k * np.linalg.norm(obs["position_m"])) if k is not None else None
            for obs in run["observers"]
        },
        "interpretation": "Small k*a is not equivalent to large k*r",
    }
    data["metadata"]["spectral_estimator"] = {
        key: spec[key]
        for key in (
            "window",
            "nperseg",
            "noverlap",
            "segments",
            "segment_detrend",
            "unused_tail_samples",
            "resampled",
        )
    }
    checks.append(
        {
            "name": "shedding",
            "status": "warning",
            "detail": data["metadata"]["shedding"]["interpretation"],
        }
    )
    aligned = []
    if observer_rows:
        aligned, records = _observer_charts(
            data, run, t, force, observer_rows, origin, nperseg, resample_dt_s
        )
        _spatial_charts(data, records, c, origin)
    else:
        checks.append(
            {
                "name": "native_observers",
                "status": "unavailable",
                "detail": "No observer records supplied",
            }
        )
    charts.extend(analytical_lessons(c))
    data["metadata"]["chart_count"] = len(charts)
    data["metadata"]["aligned_row_count"] = len(aligned)
    return data, aligned


def _observer_charts(data, run, t, force, rows, origin, nperseg, resample_dt_s):
    charts, checks = data["charts"], data["checks"]
    c = run["medium"]["sound_speed_m_s"]
    a = np.hypot(run["geometry"]["diameter_m"] / 2, run["geometry"]["span_m"] / 2)
    positions = {o["name"]: _real(o["position_m"], "observer position") for o in run["observers"]}
    groups = {}
    for row in rows:
        name, method = row["observer"], row["method"]
        if name not in positions or method not in ("libacoustics_curle", "libacoustics_fwh"):
            raise ValueError("Unknown observer or native method")
        clock = run["clock"][method]
        expected = "source_unretarded" if method == "libacoustics_curle" else "observer"
        if clock["kind"] != expected:
            raise ValueError(f"{method} clock kind must be {expected}")
        groups.setdefault(name, {}).setdefault(method, []).append(
            [float(row["time_s"]) + clock["offset_s"], float(row["pressure_Pa"])]
        )
    aligned, records = [], {}
    data["metadata"]["comparisons"] = {}
    for name, methods in groups.items():
        position = positions[name]
        if position.shape != (3,):
            raise ValueError("Observer positions require three components")
        r = _positive(np.linalg.norm(position), "observer radius")
        if r <= a:
            raise ValueError("Observer must lie outside the cylinder bounding sphere")
        arrays = {m: _real(values, "observer history") for m, values in methods.items()}
        for values in arrays.values():
            _time(values[:, 0])
        # Exclude source derivative end stencils and incomplete retarded surface support.
        low = max(t[1] + (r + a) / c, *(v[0, 0] for v in arrays.values()))
        high = min(t[-2], *(v[-1, 0] for v in arrays.values()))
        first = next(iter(arrays.values()))
        grid = first[(first[:, 0] >= low) & (first[:, 0] <= high), 0]
        if len(grid) < 4:
            checks.append(
                {
                    "name": f"observer_{name}",
                    "status": "unavailable",
                    "detail": "Fewer than four samples in common retarded support",
                }
            )
            continue
        if resample_dt_s is not None:
            dt = _positive(resample_dt_s, "resampling interval")
            if any(dt > np.diff(v[:, 0]).min() * (1 + 1e-6) for v in arrays.values()):
                raise ValueError("Refusing observer downsampling without anti-alias filtering")
            grid = low + np.arange(int(np.floor((high - low) / dt)) + 1) * dt
            grid = grid[grid <= high]
        else:
            for values in arrays.values():
                other = values[(values[:, 0] >= low) & (values[:, 0] <= high), 0]
                if len(other) != len(grid) or not np.allclose(other, grid, rtol=0, atol=1e-10):
                    raise ValueError("Observer clocks differ: explicit resample_dt_s required")
        native = {
            method: np.interp(grid, values[:, 0], values[:, 1]) for method, values in arrays.items()
        }
        prediction = compact_curle(
            t, force, grid, position, c, force_convention=run["force_convention"]
        )
        waveforms = {**native, "compact_curle": prediction["pressure_Pa"]}
        sign = -1 if run["force_convention"] == "on_body" else 1
        projected = np.interp(grid, t, sign * (force @ (position / r)))
        samples = {
            "included": len(grid),
            "excluded": len(first) - np.count_nonzero((first[:, 0] >= low) & (first[:, 0] <= high)),
            "raw_samples_per_method": {m: len(v) for m, v in arrays.items()},
            "start_s": float(grid[0]),
            "end_s": float(grid[-1]),
            "resampled": resample_dt_s is not None,
            "exclusion_reason": (
                "common observer support, retarded surface startup, derivative edges"
            ),
        }
        description = (
            "Same physical timestamps; signed native outputs retained. Compact model uses "
            "force-on-fluid, retarded at t-r/c. No gain, phase or delay fitting."
        )
        charts.append(
            _chart(
                f"pressure_{name}",
                f"Native and compact pressure: {name}",
                origin,
                [_trace(m, grid, p) for m, p in waveforms.items()],
                samples,
                description=description,
            )
        )
        charts.append(
            _chart(
                f"near_far_{name}",
                f"Compact loading terms: {name}",
                origin,
                [_trace(m, grid, prediction[m]) for m in ("near_Pa", "far_Pa")],
                samples,
                description="Independent compact model only, not native decomposition.",
            )
        )
        stats = {}
        for method, pressure in waveforms.items():
            rms = float(np.sqrt(np.mean((pressure - pressure.mean()) ** 2)))
            stats[method] = {
                "mean_Pa": float(pressure.mean()),
                "ac_rms_Pa": rms,
                "spl_dB_re_1uPa": float(20 * np.log10(rms / 1e-6)) if rms else None,
            }
        data["metadata"]["comparisons"][name] = {
            "position_m": position.tolist(),
            "radius_m": r,
            "samples": samples,
            "gain_fitted": False,
            "delay_fitted": False,
            "statistics": stats,
            "source_extent_over_r": a / r,
            "signed_rmse_Pa": {
                m: float(np.sqrt(np.mean((p - prediction["pressure_Pa"]) ** 2)))
                for m, p in native.items()
            },
        }
        for i, time in enumerate(grid):
            for method, pressure in native.items():
                aligned.append(
                    {
                        "time_s": float(time),
                        "source_time_s": float(prediction["source_time_s"][i]),
                        "observer": name,
                        "method": method,
                        "native_pressure_Pa": float(pressure[i]),
                        "compact_pressure_Pa": float(prediction["pressure_Pa"][i]),
                        "compact_near_Pa": float(prediction["near_Pa"][i]),
                        "compact_far_Pa": float(prediction["far_Pa"][i]),
                        "force_on_fluid_projection_N": float(projected[i]),
                    }
                )
        records[name] = {
            "time_s": grid,
            "waveforms": waveforms,
            "position_m": position,
            "radius_m": r,
            "samples": samples,
        }
        if len(grid) < nperseg:
            checks.append(
                {
                    "name": f"spectra_{name}",
                    "status": "unavailable",
                    "detail": f"Need nperseg={nperseg}; only {len(grid)} aligned samples",
                }
            )
            continue
        specs = {m: spectral_bundle(grid, p, nperseg=nperseg) for m, p in waveforms.items()}
        unused = next(iter(specs.values()))["unused_tail_samples"]
        samples = {
            **samples,
            "welch_unused_tail_samples": unused,
            "welch_used_samples": len(grid) - unused,
        }
        charts.append(
            _chart(
                f"pressure_psd_{name}",
                f"Pressure PSD: {name}",
                origin,
                [_trace(m, s["frequency_Hz"], s["psd"]) for m, s in specs.items()],
                samples,
                description="Hann Welch, same time support and estimator; Pa²/Hz.",
                x_label="Frequency (Hz)",
                y_label="Pressure PSD (Pa²/Hz)",
                y_scale="log",
            )
        )
        preferred = "libacoustics_fwh" if "libacoustics_fwh" in specs else next(iter(native))
        spec = specs[preferred]
        charts.append(
            _chart(
                f"spectrogram_{name}",
                f"{preferred} spectrogram: {name}",
                origin,
                [],
                samples,
                kind="heatmap",
                description="Linear one-sided pressure density; z rows are frequencies. "
                "Shared zero-based color scale across observer spectrograms (Pa²/Hz).",
                x_label="Observer time (s)",
                y_label="Frequency (Hz)",
                x=spec["spectrogram_time_s"].tolist(),
                y=spec["frequency_Hz"].tolist(),
                z=spec["spectrogram_psd"].tolist(),
                color_label="PSD (Pa²/Hz)",
                diverging=False,
            )
        )
        crosses = {
            m: coherence_phase(grid, projected, p, nperseg=nperseg) for m, p in waveforms.items()
        }
        cross_description = (
            "Force-on-fluid projected toward observer at the same physical clock. "
            "CSD=conj(F)*p. Four overlapping segments minimum (not independent); "
            "phase masked below coherence 0.5 or relative spectral power 1e-10. "
            "Descriptive diagnostic, not a significance or source-causality test."
        )
        charts.append(
            _chart(
                f"coherence_{name}",
                f"Force/pressure coherence: {name}",
                origin,
                [
                    _trace(
                        m,
                        s["frequency_Hz"],
                        s["coherence"] if s["segments"] >= 4 else [None] * len(s["coherence"]),
                    )
                    for m, s in crosses.items()
                ],
                samples,
                description=cross_description,
                x_label="Frequency (Hz)",
                y_label="Coherence (0–1)",
                y_domain=[0, 1],
            )
        )
        charts.append(
            _chart(
                f"phase_{name}",
                f"Force-to-pressure phase: {name}",
                origin,
                [
                    _trace(
                        m,
                        s["frequency_Hz"],
                        [
                            float(p) if ok else None
                            for p, ok in zip(s["phase_rad"], s["phase_valid"])
                        ],
                    )
                    for m, s in crosses.items()
                ],
                samples,
                description=cross_description,
                x_label="Frequency (Hz)",
                y_label="Phase (rad)",
                phase_wrap_period=2 * np.pi,
            )
        )
        checks.append(
            {
                "name": f"coherence_{name}",
                "status": "warning",
                "detail": (
                    f"{spec['segments']} overlapping Hann segments; {crosses[preferred]['status']}"
                ),
            }
        )
    spectrograms = [chart for chart in charts if chart["id"].startswith("spectrogram_")]
    if spectrograms:
        # Display metadata only: retain every linear density, including exact zeros.
        common_limit = max(float(np.max(chart["z"])) for chart in spectrograms)
        for chart in spectrograms:
            chart["color_limit"] = common_limit
    checks.extend(
        [
            {
                "name": "native_curle_convention",
                "status": "warning",
                "detail": (
                    "Inspected libAcoustics Curle uses +forceEff and no propagation delay; "
                    "body-force sign differs from physical fluid loading. Retained without fitting."
                ),
            },
            {
                "name": "surface_only_fwh",
                "status": "warning",
                "detail": (
                    "Stationary no-slip wall thickness is zero; surface-only FW-H does not "
                    "evaluate the quadrupole volume integral. No complete decomposition claimed."
                ),
            },
        ]
    )
    return aligned, records


def point_source_phasor(kind, observer_m, frequency_Hz, sound_speed_m_s, density_kg_m3, strength):
    """Exact stationary point-source pressure phasor, exp(+i omega t).

    Strength: monopole volume velocity (m³/s), dipole force-on-fluid vector (N),
    quadrupole moment tensor (N m). Quadrupole pressure is Q_ij ∂i∂j G.
    These mathematical sources are not a cylinder-source decomposition.
    """
    xyz = _real(observer_m, "observer")
    if xyz.shape != (3,):
        raise ValueError("Observer requires three components")
    r = _positive(np.linalg.norm(xyz), "radius")
    omega = 2 * np.pi * _positive(frequency_Hz, "frequency")
    c = _positive(sound_speed_m_s, "sound speed")
    rho = _positive(density_kg_m3, "density")
    s = _real(strength, "source strength")
    k, n = omega / c, xyz / r
    green = np.exp(-1j * k * r) / (4 * np.pi * r)
    if kind == "monopole" and s.shape == ():
        return 1j * omega * rho * s * green
    if kind == "dipole" and s.shape == (3,):
        return (s @ n) * green * (1 / r + 1j * k)
    if kind == "quadrupole" and s.shape == (3, 3):
        return green * (
            (n @ s @ n) * (-(k**2) + 3j * k / r + 3 / r**2) + np.trace(s) * (-1j * k / r - 1 / r**2)
        )
    raise ValueError("Unknown point source or invalid strength shape")


def analytical_lessons(sound_speed_m_s):
    c = _positive(sound_speed_m_s, "sound speed")
    theta = np.linspace(0, 2 * np.pi, 361)
    degrees = np.degrees(theta)
    patterns = {
        "monopole": np.ones_like(theta),
        "x-directed dipole": np.cos(theta),
        "trace-free Qxx=-Qyy quadrupole": np.cos(2 * theta),
    }
    samples = {"included": len(theta), "excluded": 0, "type": "analytical angular samples"}
    description = (
        "Ideal sources, not computed cylinder contributions. Individually normalized far-field "
        "patterns; strengths have different units. Quadrupole shown is one tensor choice, "
        "not the directivity of every quadrupole."
    )
    charts = [
        _chart(
            "lesson_directivity",
            "Ideal source RMS directivity",
            "analytical",
            [_trace(name, degrees, abs(p)) for name, p in patterns.items()],
            samples,
            description=description,
            kind="polar",
            x_label="Azimuth (degrees)",
            y_label="Individually normalized pressure RMS",
        ),
        _chart(
            "lesson_signed_directivity",
            "Ideal signed far-field patterns",
            "analytical",
            [_trace(name, degrees, p) for name, p in patterns.items()],
            samples,
            description=description,
            x_label="Azimuth (degrees)",
            y_label="Signed angular factor",
        ),
    ]
    frequency = 2.0
    omega = 2 * np.pi * frequency
    radius = np.geomspace(0.1, 1e5, 301)
    near = 1 / (4 * np.pi * radius**2 * np.sqrt(2))
    far = omega / (4 * np.pi * c * radius * np.sqrt(2))
    charts.append(
        _chart(
            "lesson_near_far",
            "A 1 N harmonic point force: near and radiating terms",
            "analytical",
            [
                _trace("1/r² near RMS", radius, near),
                _trace("1/r far RMS", radius, far),
                _trace("total RMS (quadrature)", radius, np.hypot(near, far)),
            ],
            {
                "included": len(radius),
                "excluded": 0,
                "frequency_Hz": frequency,
                "crossover_radius_m": c / omega,
            },
            description="Analytical dipole on axis, not computed cylinder pressure. "
            "Near and far terms are in temporal quadrature for a sinusoid; "
            "RMS totals do not add linearly. Equality at kr=1.",
            x_label="Radius (m)",
            y_label="RMS pressure (Pa)",
            x_scale="log",
            y_scale="log",
        )
    )
    time = np.linspace(0, 2, 1001)
    radius = c / (8 * frequency)
    source = np.sin(omega * time)
    delayed = np.sin(omega * (time - radius / c))
    charts.append(
        _chart(
            "lesson_retardation",
            "Retardation is a delay, not an amplitude fit",
            "analytical",
            [
                _trace("source sin(ωt)", time, source),
                _trace("at observer: sin[ω(t-r/c)]", time, delayed),
            ],
            {"included": len(time), "excluded": 0, "radius_m": radius, "delay_s": radius / c},
            description="Infinite-duration harmonic teaching source, not computed cylinder data. "
            "Actual finite records are never zero-extended to create arrival history.",
            y_label="Normalized waveform",
        )
    )
    return charts


def _ac_rms(values):
    return float(np.sqrt(np.mean((values - np.mean(values)) ** 2)))


def _spatial_charts(data, records, c, origin):
    rings = {}
    for name, rec in records.items():
        if abs(rec["position_m"][2]) < 1e-10:
            rings.setdefault(round(rec["radius_m"], 9), []).append(name)
    for index, (radius, names) in enumerate(sorted(rings.items())):
        if len(names) < 2:
            continue
        names.sort(key=lambda name: np.arctan2(*records[name]["position_m"][1::-1]))
        low = max(records[name]["time_s"][0] for name in names)
        high = min(records[name]["time_s"][-1] for name in names)
        first = records[names[0]]["time_s"]
        grid = first[(first >= low) & (first <= high)]
        if len(grid) < 4:
            continue
        methods = sorted(set.intersection(*(set(records[name]["waveforms"]) for name in names)))
        angles = [
            float(np.degrees(np.arctan2(*records[name]["position_m"][1::-1])) % 360)
            for name in names
        ]
        traces = []
        for method in methods:
            rms = [
                _ac_rms(
                    np.interp(grid, records[name]["time_s"], records[name]["waveforms"][method])
                )
                for name in names
            ]
            traces.append(_trace(method, angles, rms))
        data["charts"].append(
            _chart(
                f"directivity_{index}",
                f"Actual observer RMS at r={radius:g} m",
                origin,
                traces,
                _samples(grid, len(first), observer_names=names),
                kind="polar",
                x_label="Azimuth from +x toward +y (degrees)",
                y_label="AC RMS pressure (Pa)",
                description="Only supplied z=0 observers at equal radius and common time support. "
                "Discrete samples: no swept native directivity or interpolated angular evidence.",
                connect_points=False,
            )
        )
    pairs = []
    names = sorted(records, key=lambda name: records[name]["radius_m"])
    for i, near_name in enumerate(names):
        near = records[near_name]
        for far_name in names[i + 1 :]:
            far = records[far_name]
            ra, rb = near["radius_m"], far["radius_m"]
            if rb <= ra or not np.allclose(
                near["position_m"] / ra, far["position_m"] / rb, rtol=0, atol=1e-8
            ):
                continue
            for method in sorted(set(near["waveforms"]) & set(far["waveforms"])):
                unretarded = method == "libacoustics_curle"
                da, db = (0.0, 0.0) if unretarded else (ra / c, rb / c)
                ta, tb = near["time_s"] - da, far["time_s"] - db
                tau = ta[(ta >= tb[0]) & (ta <= tb[-1])]
                if len(tau) < 4:
                    continue
                pa = np.interp(tau, ta, near["waveforms"][method])
                pb = np.interp(tau, tb, far["waveforms"][method])
                rms_a, rms_b = _ac_rms(pa), _ac_rms(pb)
                ratio = rms_b / rms_a if rms_a > 0 else None
                pairs.append(
                    {
                        "near_observer": near_name,
                        "far_observer": far_name,
                        "method": method,
                        "radius_ratio_far_over_near": rb / ra,
                        "rms_ratio_far_over_near": ratio,
                        "reference_far_1_over_r": ra / rb,
                        "reference_near_1_over_r2": (ra / rb) ** 2,
                        "arrival_delay_difference_s": db - da,
                        "alignment": "source_unretarded" if unretarded else "common_source_time",
                        "samples": _samples(tau, len(ta)),
                    }
                )
    data["metadata"]["distance_pairs"] = pairs
    if pairs:
        methods = sorted({p["method"] for p in pairs})
        radius_ratios = sorted({p["radius_ratio_far_over_near"] for p in pairs})
        references = []
        for label, exponent in (("Reactive 1/r²", 2), ("Radiating 1/r", 1)):
            values = [ratio**-exponent for ratio in radius_ratios]
            suffix = f": {values[0]:g}" if len(values) == 1 else ""
            references.append(_trace(f"{label} reference{suffix}", radius_ratios, values))
        receiver_pairs = {(p["near_observer"], p["far_observer"]) for p in pairs}
        pair_note = (
            "These models compare one actual receiver pair, not independent experiments. "
            if len(receiver_pairs) == 1
            else f"Models compare {len(receiver_pairs)} actual same-ray receiver pairs. "
        )
        traces = [
            _trace(
                method,
                [p["radius_ratio_far_over_near"] for p in pairs if p["method"] == method],
                [p["rms_ratio_far_over_near"] for p in pairs if p["method"] == method],
            )
            for method in methods
        ]
        data["charts"].append(
            _chart(
                "distance_ratios",
                "Distance ratios on actual observer rays",
                origin,
                traces,
                {
                    "included_pairs": len(pairs),
                    "excluded": "non-collinear or unsupported pairs",
                    "per_pair": [p["samples"] for p in pairs],
                },
                kind="scatter",
                description="AC RMS at matched emission times t-r/c, "
                "with interpolation only inside support. "
                "Native Curle has no delay, so uses its original source clock. "
                + pair_note
                + "Reactive 1/r² and radiating 1/r references are shown; neither is fitted. "
                "The table reports signed residuals from the reactive reference. "
                "Do not apply a radiating 20log10(r) correction to reactive total pressure.",
                x_label="Far/near radius ratio",
                y_label="Far/near AC RMS pressure ratio",
                reference_traces=references,
                y_domain=[
                    0,
                    1.1
                    * max(
                        [v for ref in references for v in ref["y"]]
                        + [
                            p["rms_ratio_far_over_near"]
                            for p in pairs
                            if p["rms_ratio_far_over_near"] is not None
                        ]
                    ),
                ],
                comparison_rows=[
                    {
                        "label": f"{p['method']}: {p['near_observer']} → {p['far_observer']}",
                        "ratio": p["rms_ratio_far_over_near"],
                        "delta_reactive": (
                            p["rms_ratio_far_over_near"] - p["reference_near_1_over_r2"]
                            if p["rms_ratio_far_over_near"] is not None
                            else None
                        ),
                    }
                    for p in pairs
                ],
            )
        )
    else:
        data["checks"].append(
            {
                "name": "distance_ratios",
                "status": "unavailable",
                "detail": "No same-ray observer pair with overlapping source-time support",
            }
        )
