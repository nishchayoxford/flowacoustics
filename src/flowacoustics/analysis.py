"""Pressure-domain analysis. Input pressure is Pa; frequency is Hz.

Equations, references and cylinder analysis: docs/CYLINDER_ACOUSTICS.md.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

P_REF_PA = 1e-6


def pressure_record(x: np.ndarray, fs: float) -> np.ndarray:
    """Validate a real, finite pressure record and positive sampling frequency."""
    raw = np.asarray(x)
    if np.iscomplexobj(raw):
        raise ValueError("Pressure records must be real")
    a = np.asarray(raw, dtype=float)
    if a.ndim != 1 or len(a) < 4 or not np.all(np.isfinite(a)):
        raise ValueError("Pressure record must be finite, one-dimensional and at least 4 samples")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sample rate must be finite and positive")
    return a


def welch_psd(
    x: np.ndarray, fs: float, nperseg: int = 2048, window: str = "hann"
) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided density in Pa²/Hz, 50% overlap, segment-mean removal."""
    a = pressure_record(x, fs)
    if not isinstance(nperseg, (int, np.integer)) or not 4 <= nperseg <= len(a):
        raise ValueError("Segment length must be an integer between 4 and record length")
    return signal.welch(
        a,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )


def integrated_power(
    f: np.ndarray, density: np.ndarray, band_hz: tuple[float, float] | None = None
) -> float:
    """Sum FFT-bin powers (Pa²); band selects bin centres inclusively.

    Rectangle-bin integration preserves discrete Parseval scaling, including DC
    and Nyquist. Welch segment detrending/window weighting differs from global variance.
    """
    f, density = np.asarray(f, dtype=float), np.asarray(density, dtype=float)
    if (
        f.ndim != 1
        or len(f) < 2
        or density.shape != f.shape
        or not np.all(np.isfinite(f))
        or not np.all(np.isfinite(density))
        or np.any(density < 0)
        or f[0] < 0
        or np.any(np.diff(f) <= 0)
        or not np.allclose(np.diff(f), f[1] - f[0], rtol=1e-8, atol=1e-12)
    ):
        raise ValueError(
            "PSD requires finite nonnegative density on uniform increasing frequencies"
        )
    mask = np.ones(len(f), dtype=bool)
    if band_hz is not None:
        low, high = band_hz
        if not np.isfinite(low + high) or not 0 <= low < high <= f[-1]:
            raise ValueError("Band must lie inside PSD frequency support")
        mask = (f >= low) & (f <= high)
    return float(np.sum(density[mask]) * (f[1] - f[0]))


def rms_to_db(rms_pa: float | np.ndarray) -> np.ndarray:
    """Pressure level in dB re 1 µPa; exact silence is -infinity."""
    a = np.asarray(rms_pa, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("RMS pressure must be finite and nonnegative")
    with np.errstate(divide="ignore"):
        return 20 * np.log10(a / P_REF_PA)


def pressure_from_db(level_db: float) -> float:
    """Convert chosen finite dB re 1 µPa RMS into Pa."""
    if not np.isfinite(level_db):
        raise ValueError("Level must be finite")
    return float(P_REF_PA * 10 ** (level_db / 20))


def psd_to_db(density_pa2_hz: np.ndarray) -> np.ndarray:
    """dB re 1 µPa²/Hz, not a band SPL."""
    a = np.asarray(density_pa2_hz, dtype=float)
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("PSD must be finite and nonnegative")
    with np.errstate(divide="ignore"):
        return 10 * np.log10(a / P_REF_PA**2)


def asd_to_db(asd_pa_sqrthz: np.ndarray) -> np.ndarray:
    """dB re 1 µPa/√Hz. Equals PSD dB when ASD = sqrt(PSD)."""
    return rms_to_db(asd_pa_sqrthz)


def fft_amplitude(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Hann coherent peak amplitude, Pa (not RMS or PSD)."""
    a = pressure_record(x, fs)
    w = signal.get_window("hann", len(a), fftbins=True)
    amplitude = np.abs(np.fft.rfft((a - np.mean(a)) * w)) / np.sum(w)
    amplitude[1 : -1 if len(a) % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(len(a), 1 / fs), amplitude


def tonal_peaks(f: np.ndarray, amplitude: np.ndarray, n_peaks: int = 5) -> np.ndarray:
    """Largest local spectral peaks; not a source-mechanism classifier."""
    f, a = np.asarray(f), np.asarray(amplitude)
    if (
        f.ndim != 1
        or f.shape != a.shape
        or len(f) < 3
        or n_peaks < 1
        or not np.all(np.isfinite(f))
        or not np.all(np.isfinite(a))
        or np.any(a < 0)
        or np.any(np.diff(f) <= 0)
    ):
        raise ValueError("Invalid spectral peak inputs")
    indices, _ = signal.find_peaks(a)
    order = indices[np.argsort(a[indices])[::-1][:n_peaks]]
    return f[order]


def pressure_spectrogram(
    x: np.ndarray, fs: float, nperseg: int = 512
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sided time-frequency pressure density in Pa²/Hz."""
    a = pressure_record(x, fs)
    if not isinstance(nperseg, (int, np.integer)) or not 4 <= nperseg <= len(a):
        raise ValueError("Invalid spectrogram segment length")
    return signal.spectrogram(
        a,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )


def pressure_statistics(x: np.ndarray, fs: float) -> dict[str, float]:
    """AC statistics; mean retained separately to distinguish static pressure."""
    a = pressure_record(x, fs)
    ac = a - np.mean(a)
    rms = float(np.sqrt(np.mean(ac**2)))
    return {
        "mean_pa": float(np.mean(a)),
        "rms_pa": rms,
        "peak_pa": float(np.max(np.abs(ac))),
        "crest_factor": float(np.max(np.abs(ac)) / rms) if rms else 0.0,
    }


def spl(x: np.ndarray, fs: float) -> float:
    """AC RMS sound pressure level, dB re 1 µPa."""
    return float(rms_to_db(pressure_statistics(x, fs)["rms_pa"]))
