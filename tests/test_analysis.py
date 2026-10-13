"""Reference checks for pressure analysis, added one behaviour at a time."""

import numpy as np
import pytest

from flowacoustics import analysis


def test_welch_recovers_sinusoid_frequency_and_variance():
    fs = 4096.0
    t = np.arange(8192) / fs
    x = 0.02 * np.sin(2 * np.pi * 256 * t)
    f, density = analysis.welch_psd(x, fs, nperseg=4096)
    assert f[np.argmax(density)] == pytest.approx(256)
    assert analysis.integrated_power(f, density) == pytest.approx(np.var(x), rel=1e-10)


def test_underwater_pressure_level_roundtrip():
    assert analysis.rms_to_db(1e-6) == pytest.approx(0)
    assert analysis.rms_to_db(1e-5) == pytest.approx(20)
    assert analysis.pressure_from_db(120) == pytest.approx(1)
    assert analysis.rms_to_db(0) == -np.inf
    with pytest.raises(ValueError):
        analysis.rms_to_db(-1)


def test_psd_and_asd_decibels_have_equivalent_reference():
    density = np.array([1e-12, 1e-10, 0])
    np.testing.assert_allclose(analysis.psd_to_db(density), [0, 20, -np.inf])
    np.testing.assert_allclose(analysis.asd_to_db(np.sqrt(density)), [0, 20, -np.inf])
    with pytest.raises(ValueError):
        analysis.psd_to_db(np.array([-1]))


def test_fft_coherent_amplitude_and_peak_recovery():
    fs = 4096
    t = np.arange(4096) / fs
    x = 0.03 * np.sin(2 * np.pi * 256 * t)
    f, amplitude = analysis.fft_amplitude(x, fs)
    assert f[np.argmax(amplitude)] == pytest.approx(256)
    assert amplitude.max() == pytest.approx(0.03, rel=1e-12)
    assert analysis.tonal_peaks(f, amplitude, n_peaks=1)[0] == pytest.approx(256)


def test_spectrogram_scaling_and_ac_statistics():
    fs = 4096
    t = np.arange(8192) / fs
    x = 0.02 * np.sin(2 * np.pi * 256 * t) + 7
    f, times, density = analysis.pressure_spectrogram(x, fs, nperseg=1024)
    assert density.shape == (len(f), len(times))
    np.testing.assert_allclose(np.sum(density, axis=0) * (f[1] - f[0]), 0.02**2 / 2)
    stats = analysis.pressure_statistics(x, fs)
    assert stats["rms_pa"] == pytest.approx(0.02 / np.sqrt(2))
    assert stats["mean_pa"] == pytest.approx(7)
    assert stats["crest_factor"] == pytest.approx(np.sqrt(2))
    assert analysis.spl(x, fs) == pytest.approx(analysis.rms_to_db(0.02 / np.sqrt(2)))


def test_boxcar_psd_integrates_white_noise_variance_and_band_energy():
    x = np.random.default_rng(7).normal(size=32768)
    f, p = analysis.welch_psd(x, 4096, nperseg=len(x), window="boxcar")
    assert analysis.integrated_power(f, p) == pytest.approx(np.var(x), rel=1e-12)
    assert 0 < analysis.integrated_power(f, p, (100, 500)) < np.var(x)
    f, p = analysis.welch_psd(x, 4096, nperseg=1024)
    assert analysis.integrated_power(f, p) == pytest.approx(np.var(x), rel=0.02)


@pytest.mark.parametrize(
    "x,fs",
    [
        (np.array([1, 2]), 10),
        (np.ones(4), 0),
        (np.array([1, 2, 3, np.nan]), 10),
        (np.ones(4, dtype=complex), 10),
    ],
)
def test_pressure_records_reject_invalid_input(x, fs):
    with pytest.raises(ValueError):
        analysis.welch_psd(x, fs, nperseg=4)


def test_invalid_psd_grid_and_band_are_rejected():
    with pytest.raises(ValueError):
        analysis.integrated_power(np.array([0, 1, 3]), np.ones(3))
    with pytest.raises(ValueError):
        analysis.integrated_power(np.arange(4), np.ones(4), (0, 9))
