"""Presentation-only reduction; full numerical analysis remains a separate download."""

from __future__ import annotations

import copy


def make_display_dataset(source, *, max_time_points=800, max_frequency_hz=20):
    """Select original extrema for display, never recompute statistics from display points."""
    if max_time_points < 4:
        raise ValueError("At least four time points required")
    data = copy.deepcopy(source)
    estimator = data.get("metadata", {}).get("spectral_estimator")
    if isinstance(estimator, dict):
        estimator["scope"] = "force_spectrum_only"
    for chart in data["charts"]:
        if (
            chart.get("id") == "force_psd"
            and data.get("metadata", {}).get("shedding", {}).get("status") == "unresolved"
        ):
            chart["title"] = "Force spectra - lift candidate unresolved"
        if chart["origin"] == "computed" and chart["x_label"] == "Frequency (Hz)":
            for trace in chart["traces"]:
                selected = [i for i, f in enumerate(trace["x"]) if 0 <= f <= max_frequency_hz]
                trace["x"] = [trace["x"][i] for i in selected]
                trace["y"] = [trace["y"][i] for i in selected]
            chart["display_frequency_window_Hz"] = [0, max_frequency_hz]
        if chart["kind"] == "heatmap" and chart.get("y_label") == "Frequency (Hz)":
            selected = [i for i, f in enumerate(chart["y"]) if 0 <= f <= max_frequency_hz]
            chart["y"] = [chart["y"][i] for i in selected]
            chart["z"] = [chart["z"][i] for i in selected]
            chart["display_frequency_window_Hz"] = [0, max_frequency_hz]
        if chart["kind"] == "line" and "time" in chart["x_label"].lower():
            for trace in chart["traces"]:
                x, y = trace["x"], trace["y"]
                n = len(x)
                if n <= max_time_points:
                    continue
                buckets = (max_time_points - 2) // 2
                selected = {0, n - 1}
                for bucket in range(buckets):
                    indices = range(bucket * n // buckets, (bucket + 1) * n // buckets)
                    valid = [i for i in indices if y[i] is not None]
                    if valid:
                        selected.add(min(valid, key=lambda i: y[i]))
                        selected.add(max(valid, key=lambda i: y[i]))
                selected = sorted(selected)
                trace["x"], trace["y"] = [x[i] for i in selected], [y[i] for i in selected]
            chart["display_sample_policy"] = (
                f"At most {max_time_points} original points per trace: endpoints and "
                "local minima/maxima. Display only, not resampling or an analysis filter. "
                "Statistics use the full analysis record; full arrays are downloadable."
            )
    return data
