"""Offline Case 4 spectral products from immutable coherent Hann coefficients.

No CFD or acoustic source reduction is executed here. The density estimate is
conditional on the existing single-record propagated coefficients and band.
"""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

CASE_ID = "mrl2d_les_permeable"
SOURCE_PATHS = (
    "site/data/mrl2d/mrl2d_les_permeable.npz",
    "site/data/results.json",
    "site/data/mrl2d/public-manifest.json",
    "src/flowacoustics/acoustics2d.py",
)


def coefficient_spectra(coefficients, time):
    """Return positive-bin RMS, one-sided Hann PSD, and sampling metadata.

    Coefficients must already be divided by sum(periodic Hann). DC and Nyquist
    are excluded upstream. Do not window a synthesized pressure history again.
    """
    time = np.asarray(time, dtype=float)
    coefficients = np.asarray(coefficients, dtype=complex)
    if time.ndim != 1 or time.size < 3 or not np.all(np.isfinite(time)):
        raise ValueError("time must be a finite one-dimensional sample record of length >= 3")
    n = len(time)
    dt = float((time[-1] - time[0]) / (n - 1))
    if dt <= 0 or not np.allclose(np.diff(time), dt, rtol=1e-11, atol=1e-14):
        raise ValueError("time must be increasing and uniformly sampled")
    if coefficients.ndim != 2 or not coefficients.size or not np.all(np.isfinite(coefficients)):
        raise ValueError("coefficients must be a finite nonempty [frequency, observer] matrix")
    weights = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    enbw = float(np.sum(weights**2) / (dt * np.sum(weights) ** 2))
    rms = np.sqrt(2.0) * np.abs(coefficients)
    psd = 2 * np.abs(coefficients) ** 2 / enbw
    return (
        rms,
        psd,
        {
            "sample_count": n,
            "sample_interval_s": dt,
            "frequency_spacing_hz": 1 / (n * dt),
            "enbw_hz": enbw,
            "sample_window_s": [float(time[0]), float(time[-1])],
            "window": "periodic Hann",
        },
    )


def peak_table(frequency, angles, bin_rms, psd):
    """Largest sampled bin; exact ties select the first (lowest) frequency.

    A zero spectrum has no identified peak. Near-ties are not rounded into ties.
    """
    rows = []
    for j, angle in enumerate(angles):
        values = np.asarray(bin_rms)[:, j]
        maximum = float(np.max(values))
        index = int(np.argmax(values)) if maximum > 0 else None
        rows.append(
            {
                "angle_deg": float(angle),
                "index": index,
                "frequency_hz": None if index is None else float(frequency[index]),
                "bin_rms_pa": maximum,
                "psd_pa2_per_hz": 0.0 if index is None else float(psd[index, j]),
                "is_band_edge": index in (0, len(frequency) - 1),
                "tied_maximum_count": int(np.count_nonzero(values == maximum)) if maximum else 0,
            }
        )
    return rows


def band_levels(band_rms):
    """Band level dB re 1 microPa; exactly zero has no finite dB level."""
    values = np.asarray(band_rms, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("band RMS must be nonnegative finite")
    # Difference of logs avoids underflow/overflow from dividing tiny/huge RMS.
    return [None if value == 0 else float(20 * (np.log10(value) + 6)) for value in values]


def build_summary(root):
    """Authenticate immutable public buffers, then derive the fixed JSON schema."""
    root = Path(root)
    buffers = {path: (root / path).read_bytes() for path in SOURCE_PATHS}
    digests = {path: hashlib.sha256(data).hexdigest() for path, data in buffers.items()}
    manifest = json.loads(buffers[SOURCE_PATHS[2]])
    for path in SOURCE_PATHS[:2]:
        entry = manifest["files"][path.removeprefix("site/")]
        if digests[path] != entry["sha256"] or len(buffers[path]) != entry["bytes"]:
            raise ValueError(f"SHA256 mismatch or byte-count mismatch: {path}")
    original = json.loads(buffers[SOURCE_PATHS[1]])
    case = next(case for case in original["cases"] if case["id"] == CASE_ID)
    if case["flow_model"] != "WALE" or case["acoustic_surface"] != "permeable":
        raise ValueError("Case 4 identity mismatch")
    with np.load(io.BytesIO(buffers[SOURCE_PATHS[0]]), allow_pickle=False) as native:
        time = native["time_s"]
        frequency = native["frequency_hz"]
        coefficients = native["coefficients_pa"]
        receivers = native["receivers_m"]
        band_rms = native["band_rms_pa"]
        if (
            time.shape != (4000,)
            or frequency.shape != (196,)
            or coefficients.shape != (196, 12)
            or receivers.shape != (12, 2)
            or band_rms.shape != (12,)
        ):
            raise ValueError("Unexpected Case 4 native shape")
        if not all(
            np.all(np.isfinite(a)) for a in (time, frequency, coefficients, receivers, band_rms)
        ):
            raise ValueError("Nonfinite native values")
        if time[[0, -1]].tolist() != case["metadata"]["analysis_window_s"]:
            raise ValueError("Original sample window mismatch")
        if frequency[[0, -1]].tolist() != case["metadata"]["acoustic_band_hz"]:
            raise ValueError("Original retained band mismatch")
        angles = np.mod(np.rad2deg(np.arctan2(receivers[:, 1], receivers[:, 0])), 360)
        if not np.all(np.diff(angles) > 0):
            raise ValueError("Native receivers must be ordered by angle")
        radii = np.linalg.norm(receivers, axis=1)
        if not np.allclose(radii, radii[0], rtol=1e-13, atol=0):
            raise ValueError("Receivers must share a radius")
        rms, psd, estimator = coefficient_spectra(coefficients, time)
        df = estimator["frequency_spacing_hz"]
        if (
            np.any(frequency <= 0)
            or np.any(frequency >= 0.5 / estimator["sample_interval_s"])
            or not np.allclose(np.diff(frequency), df, rtol=1e-11, atol=0)
            or not np.allclose(frequency / df, np.rint(frequency / df), atol=1e-10, rtol=0)
        ):
            raise ValueError("Expected consecutive positive bins excluding Nyquist")
        coherent_square = np.sum(rms**2, axis=0)
        if not np.allclose(band_rms**2, coherent_square, rtol=1e-12, atol=0):
            raise ValueError("Original band RMS disagrees with coherent coefficients")
        estimator.update(
            {
                "band_hz": frequency[[0, -1]].tolist(),
                "psd_definition": (
                    "One-sided propagated single-record Hann density: "
                    "PSD = 2|c|²/ENBW = bin_rms²/ENBW, Pa²/Hz; "
                    "ENBW = fs sum(w²)/sum(w)². No second window, Welch averaging, "
                    "stationarity or broadband/physical validation is implied. "
                    "Selected-band sum(PSD * df) = coherent band RMS² * df/ENBW "
                    "(2/3 for periodic Hann); this integral is not coherent synthesis mean square."
                ),
                "bin_rms_definition": "bin_rms = sqrt(2)|c|, Pa; c has coherent Hann gain removed.",
                "band_level_definition": (
                    "20 log10(original band_rms_pa / 1e-6), dB re 1 µPa; zero is null. "
                    "Original coherent selected-band RMS is preserved, "
                    "not replaced by PSD-integral RMS."
                ),
                "psd_band_integral_pa2": (np.sum(psd, axis=0) * df).tolist(),
                "coherent_band_mean_square_pa2": coherent_square.tolist(),
            }
        )
        return {
            "schema": "mrl2d-case4-spectral-v1",
            "case_id": CASE_ID,
            "frequency_hz": frequency.tolist(),
            "angles_deg": angles.tolist(),
            "bin_rms_pa": rms.tolist(),
            "psd_pa2_per_hz": psd.tolist(),
            "band_rms_pa": band_rms.tolist(),
            "band_level_db_re_1upa": band_levels(band_rms),
            "peaks": peak_table(frequency, angles, rms, psd),
            "estimator": estimator,
            "provenance": {
                "sources": [{"path": path, "sha256": digests[path]} for path in SOURCE_PATHS],
                "receiver_radius_m": float(radii[0]),
                "case_description": "Single-phase strict-2D WALE; stationary permeable contour.",
                "operation": (
                    "Offline algebraic spectral extension only; no CFD or source reduction."
                ),
            },
        }


def export_tables(summary, output):
    """Write lossless JSON and observer-major long-form numerical CSVs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    )
    with (output / "spectra.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["angle_deg", "frequency_hz", "bin_rms_pa", "psd_pa2_per_hz"])
        for j, angle in enumerate(summary["angles_deg"]):
            for k, frequency in enumerate(summary["frequency_hz"]):
                writer.writerow(
                    [angle, frequency, summary["bin_rms_pa"][k][j], summary["psd_pa2_per_hz"][k][j]]
                )
    with (output / "peaks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["peaks"][0]))
        writer.writeheader()
        writer.writerows(summary["peaks"])
    with (output / "directivity.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["angle_deg", "band_rms_pa", "band_level_db_re_1upa"])
        writer.writerows(
            zip(
                summary["angles_deg"],
                summary["band_rms_pa"],
                summary["band_level_db_re_1upa"],
                strict=True,
            )
        )
    return ["summary.json", "spectra.csv", "peaks.csv", "directivity.csv"]


def _figure_header(summary, heading):
    estimator = summary["estimator"]
    start, end = estimator["sample_window_s"]
    low, high = estimator["band_hz"]
    radius = summary["provenance"]["receiver_radius_m"]
    case = (
        "Case 4 · single-phase strict-2D WALE / stationary permeable contour"
        if summary["case_id"] == CASE_ID
        else summary["case_id"]
    )
    return (
        f"{heading}\n{case}\n"
        f"Samples {start:g}–{end:g} s · periodic Hann · retained band {low:g}–{high:g} Hz "
        f"· receiver radius {radius:g} m"
    )


def directivity_figure(summary):
    """Two polar panels with actual radial Pa/dB values, not interpolated curves."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure, axes = plt.subplots(1, 2, figsize=(14, 8), subplot_kw={"projection": "polar"})
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.20, top=0.73, wspace=0.36)
    figure.suptitle(
        _figure_header(summary, "Selected-band directivity · 12 sampled observers"),
        fontsize=14,
        y=0.97,
        linespacing=1.7,
    )
    theta = np.deg2rad(summary["angles_deg"])
    for axis, key, unit, title in zip(
        axes,
        ["band_rms_pa", "band_level_db_re_1upa"],
        ["Pa", "dB"],
        ["Original coherent band RMS (Pa)", "Band level (dB re 1 µPa)"],
        strict=True,
    ):
        values = np.array([np.nan if x is None else x for x in summary[key]])
        axis.scatter(theta, values, s=49, c="#176b68", edgecolors="white", linewidths=0.6, zorder=3)
        axis.set_thetagrids(np.arange(0, 360, 30))
        axis.set_theta_zero_location("E")
        axis.set_theta_direction(1)
        axis.set_title(title, fontsize=13, pad=30)
        axis.tick_params(labelsize=10)
        axis.set_rlabel_position(15)
        axis.grid(color="#ccd4d2", linewidth=0.7)
        finite = values[np.isfinite(values)]
        if len(finite):
            lower = 0 if unit == "Pa" else 2 * np.floor(finite.min() / 2) - 2
            upper = finite.max() * 1.15 if unit == "Pa" else 2 * np.ceil(finite.max() / 2) + 2
            if upper == lower:
                upper = lower + 1
            ticks = np.asarray(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]).tick_values(lower, upper))
            ticks = ticks[(ticks > lower) & (ticks <= upper)]
            axis.set_ylim(lower, upper)
            axis.set_yticks(ticks, labels=[f"{value:g} {unit}" for value in ticks])
            index = int(np.nanargmax(values))
            note = (
                f"Maximum sampled direction: {summary['angles_deg'][index]:g}°\n"
                f"{values[index]:.6g} {unit}"
            )
        else:
            axis.set_ylim(0, 1)
            axis.set_yticks([])
            note = "Maximum undefined: all band levels are null (zero RMS)"
        axis.text(0.5, -0.16, note, transform=axis.transAxes, ha="center", fontsize=11)
    figure.text(
        0.5,
        0.035,
        "Unconnected markers: no angular interpolation. "
        "dB = 20 log₁₀(original band RMS / 1 µPa).\n"
        "Finite window-conditioned selected-band result; not a stationary-noise "
        "or absolute finite-turbine validation.",
        ha="center",
        fontsize=11,
        linespacing=1.6,
    )
    return figure


def spectra_figure(summary, metric):
    """Readable 4-by-3 sampled spectra with a common scale and all peak labels."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    is_psd = metric == "psd_pa2_per_hz"
    unit = "Pa²/Hz" if is_psd else "Pa"
    title = "One-sided Hann PSD" if is_psd else "Coherent bin RMS"
    data = np.asarray(summary[metric])
    frequency = np.asarray(summary["frequency_hz"])
    figure, grid = plt.subplots(4, 3, figsize=(15, 14), sharex=True, sharey=True)
    figure.subplots_adjust(
        left=0.075, right=0.975, bottom=0.16, top=0.835, hspace=0.65, wspace=0.22
    )
    figure.suptitle(
        _figure_header(summary, f"{title} · all 12 observers"),
        fontsize=15,
        y=0.975,
        linespacing=1.6,
    )
    upper = float(data.max()) * 1.12 if data.max() > 0 else 1.0
    for j, axis in enumerate(grid.flat):
        axis.plot(frequency, data[:, j], color="#176b68", linewidth=0.9, marker=".", markersize=2.1)
        peak = summary["peaks"][j]
        if peak["index"] is None:
            annotation = "No peak (zero spectrum)"
        else:
            axis.scatter(
                [peak["frequency_hz"]],
                [peak[metric]],
                s=33,
                color="#bd531f",
                edgecolors="white",
                linewidths=0.5,
                zorder=4,
            )
            suffix = " · band edge" if peak["is_band_edge"] else ""
            if peak["tied_maximum_count"] > 1:
                suffix += f" · {peak['tied_maximum_count']} tied bins"
            annotation = f"peak {peak['frequency_hz']:.5g} Hz{suffix}\n{peak[metric]:.5g} {unit}"
        axis.set_title(f"{summary['angles_deg'][j]:g}° · {annotation}", fontsize=11, pad=10)
        axis.set_xlim(float(frequency[0]), float(frequency[-1]))
        axis.set_ylim(0, upper)
        axis.set_xticks([5, 10, 15, 20, 25])
        axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        axis.tick_params(labelsize=10, labelbottom=True, labelleft=True)
        axis.grid(alpha=0.22, color="#6d7a78", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    figure.supxlabel("Sampled frequency (Hz)", fontsize=12, y=0.108)
    figure.supylabel(f"{title} ({unit})", fontsize=12, x=0.017)
    estimator = summary["estimator"]
    definition = "PSD = 2|c|² / ENBW (Pa²/Hz); " if is_psd else "Bin RMS = √2 |c| (Pa); "
    definition += (
        f"ENBW = {estimator['enbw_hz']:g} Hz; "
        f"bin spacing = {estimator['frequency_spacing_hz']:g} Hz. "
        "Orange marker: largest retained sampled bin."
    )
    figure.text(0.5, 0.069, definition, ha="center", fontsize=11)
    figure.text(
        0.5,
        0.035,
        "Single-record, propagated, window-conditioned estimate; no second Hann, "
        "Welch average or stationarity claim.\n"
        "Selected-band PSD integral = coherent band mean square × df/ENBW "
        "(2/3 here). Peaks do not prove a blade-passing mechanism.",
        ha="center",
        fontsize=10.5,
        linespacing=1.7,
    )
    return figure


def export_bundle(root, output):
    """Create the additive dataset and deterministic PNG/three-page vector PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    root, output = Path(root), Path(output)
    summary = build_summary(root)
    names = export_tables(summary, output)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    ):
        figures = [
            ("directivity", directivity_figure(summary)),
            ("bin-rms", spectra_figure(summary, "bin_rms_pa")),
            ("psd", spectra_figure(summary, "psd_pa2_per_hz")),
        ]
        with PdfPages(
            output / "plots.pdf",
            metadata={
                "Title": "Case 4 spectral extension: directivity, bin RMS, Hann PSD",
                "Author": "FlowAcoustics",
                "CreationDate": None,
                "ModDate": None,
            },
        ) as pdf:
            for name, figure in figures:
                figure.savefig(output / f"{name}.png", dpi=180)
                pdf.savefig(figure)
                plt.close(figure)
                names.append(f"{name}.png")
        names.append("plots.pdf")
    manifest = {
        "schema": "mrl2d-case4-spectral-manifest-v1",
        "case_id": CASE_ID,
        "manifest_self_excluded": True,
        "counts": {
            "observers": len(summary["angles_deg"]),
            "frequencies": len(summary["frequency_hz"]),
            "spectra_rows": len(summary["angles_deg"]) * len(summary["frequency_hz"]),
            "peaks_rows": len(summary["peaks"]),
            "directivity_rows": len(summary["band_rms_pa"]),
            "pdf_pages": len(figures),
        },
        "pdf_page_order": [
            "Pa/dB directivity",
            "12 coherent bin-RMS spectra",
            "12 Hann PSD spectra",
        ],
        "sources": summary["provenance"]["sources"],
        "outputs": {
            name: {
                "sha256": hashlib.sha256((output / name).read_bytes()).hexdigest(),
                "bytes": (output / name).stat().st_size,
            }
            for name in sorted(names)
        },
    }
    for source in manifest["sources"]:
        if hashlib.sha256((root / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise RuntimeError(f"Immutable source changed during export: {source['path']}")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output", type=Path, help="Additive destination (default: site/data/case4-spectra)"
    )
    args = parser.parse_args()
    output = args.output or args.root / "site/data/case4-spectra"
    manifest = export_bundle(args.root, output)
    print(json.dumps({"output": str(output), "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
