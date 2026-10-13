"""F1/F2 regressions: synthetic signals only, never native CFD evidence."""
from functools import lru_cache

import numpy as np
import pytest

from flowacoustics import signature_analysis as api

DT = 0.000625
N = 3200
EDGES = [0, 1, 3, 10, 20]
EXPECTED_BINS = [list(range(0, 2)), list(range(2, 6)), list(range(6, 20)), list(range(20, 41))]


def clock(encoding):
    indices = np.arange(3200, 16001)
    if encoding == "multiply":
        return indices * DT
    return np.array([float(format(i / 1600, '.9f')) for i in indices])


@lru_cache(maxsize=1)
def dft_matrix():
    # Explicit complex DFT, not numpy/scipy FFT or any production helper.
    j = np.arange(N)
    return np.exp((-2j * np.pi / N) * np.outer(np.arange(41), j))


def fixed_dft_band_oracle(values):
    """Fixed physical cadence, independent periodic-Hann definition and weights."""
    w = np.sin(np.pi * np.arange(N) / N) ** 2
    transform = dft_matrix() @ ((values - np.mean(values)) * w)
    density = np.abs(transform) ** 2 * DT / np.dot(w, w)
    density[1:] *= 2  # bins 1..40 are interior, not Nyquist
    return [float(np.sum(density[indices]) / (N * DT)) for indices in EXPECTED_BINS]


@pytest.mark.parametrize('encoding', ['multiply', 'decimal_csv'])
@pytest.mark.parametrize('tone', [1, 3, 10, 20])
@pytest.mark.parametrize('lo', [2, 4, 6, 8])
def test_stationary_edge_tones_membership_and_power_match_explicit_dft(encoding, tone, lo):
    t = clock(encoding)
    t = t[(t >= lo) & (t < lo + 2)]
    assert len(t) == N
    x = np.sin(2 * np.pi * tone * t)
    s = api.spectrum(t, x, N)
    bands = api.band_powers(s['frequency_Hz'], s['density'], EDGES)
    actual_bins = [np.rint(np.array(b['bin_centers_Hz']) / 0.5).astype(int).tolist() for b in bands]
    expected_power = fixed_dft_band_oracle(x)
    # Check powers first: catches the actual false variability, not only a label.
    np.testing.assert_allclose([b['power'] for b in bands], expected_power, rtol=3e-13, atol=1e-26)
    assert actual_bins == EXPECTED_BINS


@pytest.mark.parametrize('encoding', ['multiply', 'decimal_csv'])
def test_declared_cadence_is_shared_without_replacing_physical_clocks(encoding):
    t = clock(encoding)
    original = t.copy()
    for lo in [2, 4, 6, 8]:
        tb = t[(t >= lo) & (t < lo + 2)]
        x = np.sin(2 * np.pi * tb)
        s = api.spectrum(tb, x, N, cadence_dt_s=DT)
        assert s['dt_s'] == DT
        np.testing.assert_array_equal(s['frequency_Hz'], np.arange(N // 2 + 1) * 0.5)
        np.testing.assert_array_equal(s['time_s'], tb)
        np.testing.assert_array_equal(s['values'], x)
        assert not s['resampled']
        audit = s['cadence']
        assert audit['nominal_dt_s'] == DT
        assert audit['observed_mean_dt_s'] == float(np.mean(np.diff(tb)))
        assert audit['max_interval_jitter_s'] == float(np.max(abs(np.diff(tb) - DT)))
        assert audit['max_lattice_residual_s'] <= audit['validation_tolerance_s']
        assert audit['sample_support_s'] == [float(tb[0]), float(tb[-1])]
    np.testing.assert_array_equal(t, original)


@pytest.mark.parametrize('fault', ['wrong_nominal', 'drift', 'missing', 'jitter'])
def test_canonical_cadence_rejects_unsupported_physical_lattice(fault):
    t = clock('multiply')
    nominal = DT
    if fault == 'wrong_nominal':
        nominal *= 1.01
    elif fault == 'drift':
        t = t[0] + (t - t[0]) * (1 + 5e-7)
    elif fault == 'missing':
        t = np.delete(t, 200)
    else:
        t[100] += DT * 0.01
    with pytest.raises(ValueError, match='cadence|Nonuniform'):
        api.spectrum(t, np.sin(t), N, cadence_dt_s=nominal)


def test_loaded_nominal_cadence_reaches_all_primary_and_sensitivity_spectra(tmp_path):
    from test_cylinder_signature import table_fixture

    args = table_fixture(tmp_path)
    data = api.load_records(**args, start_time_s=2, end_time_s=10)
    result = api.analyze_records(data, nperseg=200)
    specs = [*result['force_spectra'].values(), *result['pressure_spectra'].values(),
             *result['segment_sensitivities'],
             *(row['spectrum'] for row in result['sensitivity_windows']),
             result['compact']['near_spectrum'], result['compact']['far_spectrum']]
    assert data['dt_s'] == 0.01
    for s in specs:
        assert s['dt_s'] == 0.01
        assert s['cadence']['nominal_dt_s'] == 0.01
        np.testing.assert_array_equal(s['frequency_Hz'], np.arange(s['nperseg'] // 2 + 1) / (s['nperseg'] * 0.01))


def spectrum_objects(result):
    return {
        **{'force/' + k: (s, False) for k, s in result['force_spectra'].items()},
        **{'pressure/' + '/'.join(k): (s, False) for k, s in result['pressure_spectra'].items()},
        **{'segment/' + str(s['nperseg']): (s, False) for s in result['segment_sensitivities']},
        **{'start/' + str(row['requested_start_s']): (row['spectrum'], False)
           for row in result['sensitivity_windows'] if 'spectrum' in row},
        **{key: (result['compact'][key], True) for key in ['near_spectrum', 'far_spectrum']},
    }


def assert_interpolation_flags(meta, native, retarded, local=False):
    assert meta['resampled'] is (native or retarded or local)
    assert meta['native_to_uniform'] is native
    assert meta['retarded_interpolation'] is retarded
    assert meta['local_spectral'] is local


@pytest.mark.parametrize('resample_dt', [0.009, None])
def test_all_spectrum_objects_and_serialized_metadata_preserve_actual_interpolation(tmp_path, resample_dt):
    import json
    from test_cylinder_signature import table_fixture, geometry_fixture

    args = table_fixture(tmp_path)
    geometry_fixture(args)
    data = api.load_records(**args, start_time_s=2, end_time_s=10, resample_dt_s=resample_dt)
    result = api.analyze_records(data, nperseg=200)
    specs = spectrum_objects(result)
    assert len(specs) == 19  # 2 force + 10 pressure + 2 segment + 3 start + 2 compact
    # First reproduce the existing false aggregate; new stage fields checked next.
    for name, (s, compact) in specs.items():
        assert s['resampled'] is (resample_dt is not None or compact), name
    for s, compact in specs.values():
        # Compact histories use original native force, not the interpolated force grid.
        native = resample_dt is not None and not compact
        assert_interpolation_flags(s, native, compact)
        serialized = json.loads(json.dumps(api._jsonable(api._smeta(s))))
        assert_interpolation_flags(serialized, native, compact)
        assert serialized['cadence'] == s['cadence']

    out = tmp_path / 'reports/new'
    root = api.build_report(**args, output=out, allowed_root=tmp_path / 'reports',
                            start_time_s=2, end_time_s=10, nperseg=200,
                            resample_dt_s=resample_dt)
    stored_root = json.loads((out / 'manifest.json').read_text())
    assert_interpolation_flags(stored_root['estimator'], resample_dt is not None, False)
    assert stored_root['estimator'] == root['estimator']
    count = 0
    for entry in root['figures']:
        m = json.loads((out / entry['manifest']).read_text())
        assert_interpolation_flags(m['estimator'], resample_dt is not None, False)
        for table in m['numeric_tables']:
            if table['columns'][0] != 'frequency_Hz' or 'PSD' not in table['columns'][1]:
                continue
            count += 1
            compact = m['figure_id'] == 'I08_near_radiating'
            assert_interpolation_flags(table['estimator'], resample_dt is not None and not compact, compact)
            assert table['estimator']['cadence']['nominal_dt_s'] == (resample_dt or 0.01)
    assert count == 17  # four-site pressure tables (eight); 2m spectra remain objects only


def test_local_spectral_interpolation_is_not_native_or_retarded():
    t = np.cumsum(np.tile([0.001, 0.0011], 200))
    s = api.spectrum(t, np.sin(2 * np.pi * 10 * t), 64, resample_dt_s=0.0009)
    assert_interpolation_flags(s, False, False, True)
    assert_interpolation_flags(api._smeta(s), False, False, True)


def test_native_interpolation_is_tracked_per_receiver_clock(tmp_path):
    import csv
    import json
    from test_cylinder_signature import table_fixture

    args = table_fixture(tmp_path)
    path = args['observers_path']
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row['observer'] == 'cross_plus' and row['method'] == 'libacoustics_fwh':
            row['time_s'] = repr(float(row['time_s']) + 2e-9)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run = json.loads(args['run_path'].read_text())
    for pins in [run['input_sha256'], run['exports']['sha256'], run['reconstruction']['sha256']]:
        pins[str(path)] = api.digest(path)  # synthetic fixture only
    args['run_path'].write_text(json.dumps(run))
    data = api.load_records(**args, start_time_s=2, end_time_s=9.99)
    result = api.analyze_records(data, nperseg=200)
    shifted = result['pressure_spectra'][('cross_plus', 'libacoustics_fwh')]
    assert shifted['resampled'] is True
    for key, s in result['pressure_spectra'].items():
        assert_interpolation_flags(s, key == ('cross_plus', 'libacoustics_fwh'), False)
        assert_interpolation_flags(api._smeta(s), key == ('cross_plus', 'libacoustics_fwh'), False)
    assert_interpolation_flags(result['force_spectra']['lift'], False, False)


def test_bin_contract_and_snapping_evidence_survive_metadata_serialization():
    import json

    t = clock('decimal_csv')
    s = api.spectrum(t, np.sin(2 * np.pi * t), N, cadence_dt_s=DT)
    meta = json.loads(json.dumps(api._jsonable(api._smeta(s))))
    contract = meta['frequency_bin_contract']
    assert contract['frequency_origin_Hz'] == 0
    assert contract['bin_count'] == N // 2 + 1
    assert contract['bin_spacing_Hz'] == 0.5
    assert contract['edge_snap_eps_multiplier'] == 32
    assert contract['edge_snap_max_bins'] == 1e-9
    assert contract['interior_rule'] == 'half_open_right_owns_boundary'
    assert contract['final_upper_edge_inclusive'] is True
    bands = api.band_powers(s['frequency_Hz'], s['density'], EDGES)
    assert [b['bin_indices'] for b in bands] == EXPECTED_BINS
    assert [b['edge_bin_coordinates'] for b in bands] == [[0, 2], [2, 6], [6, 20], [20, 40]]


@pytest.mark.parametrize('edge_index', [1, 2, 3, 4])
@pytest.mark.parametrize('offset_bins', [-0.25, -1e-6, -1e-10, 1e-10, 1e-6, 0.25])
def test_genuinely_off_edge_fractional_bins_are_not_snapped(edge_index, offset_bins):
    # Verification-only negative controls, not claimed as initially failing RED.
    f = np.arange(1601) * 0.5
    density = 1 + np.arange(len(f)) / 1600
    edges = np.array(EDGES, dtype=float)
    edges[edge_index] += offset_bins * 0.5
    bands = api.band_powers(f, density, edges)
    all_indices = []
    for i, b in enumerate(bands):
        mask = (f >= edges[i]) & ((f <= edges[i + 1]) if i == 3 else (f < edges[i + 1]))
        assert b['bin_centers_Hz'] == f[mask].tolist()
        assert b['power'] == float(density[mask].sum() * 0.5)
        all_indices.extend(b['bin_centers_Hz'])
    assert len(all_indices) == len(set(all_indices))


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('edge_index', [1, 2, 3, 4])
def test_roundoff_snapping_is_bounded_on_both_sides(edge_index, direction):
    f = np.arange(1601) * 0.5
    edges = np.array(EDGES, dtype=float)
    q = edges[edge_index] / 0.5
    # Inside 32-eps allowance, and outside it (128 eps), in bin coordinates.
    for multiplier in [8, 128]:
        shifted = edges.copy()
        shifted[edge_index] += direction * multiplier * np.finfo(float).eps * q * 0.5
        bands = api.band_powers(f, np.ones_like(f), shifted)
        if multiplier == 8:
            assert [b['bin_count'] for b in bands] == [2, 4, 14, 21]
        else:
            expected = []
            for i in range(4):
                keep = (f >= shifted[i]) & ((f <= shifted[i + 1]) if i == 3 else (f < shifted[i + 1]))
                expected.append(int(keep.sum()))
            assert [b['bin_count'] for b in bands] == expected


def test_empty_silent_bands_and_final_nyquist_keep_endpoint_weights():
    t = np.arange(N) * DT
    s = api.spectrum(t, (-1.0) ** np.arange(N), N, cadence_dt_s=DT)
    b = api.band_powers(s['frequency_Hz'], s['density'], [0, 0.1, 0.2, 799.75, 800])
    assert b[1]['bin_count'] == 0 and b[1]['power'] == 0
    # Independent Nyquist direct DFT: its one-sided weight is 1, not 2.
    w = np.sin(np.pi * np.arange(N) / N) ** 2
    x = (-1.0) ** np.arange(N)
    transformed = sum((x - x.mean()) * w * (-1.0) ** np.arange(N))
    expected = transformed ** 2 / (N * sum(w * w))
    assert b[-1]['bin_count'] == 1
    assert b[-1]['power'] == pytest.approx(expected, rel=1e-13)
    assert sum(v['power'] for v in b) == pytest.approx(s['integrated_power'], rel=1e-13)
    rounded = api.band_powers(s['frequency_Hz'], s['density'], [0, np.nextafter(800., np.inf)])
    assert rounded[0]['bin_count'] == N // 2 + 1
    with pytest.raises(ValueError, match='support'):
        api.band_powers(s['frequency_Hz'], s['density'], [0, 800 + 1e-9])
    silent = api.band_powers(s['frequency_Hz'], np.zeros_like(s['density']), EDGES)
    assert all(v['power'] == 0 for v in silent)
    assert np.isneginf(api.level([v['power'] for v in silent])).all()


@pytest.mark.parametrize('encoding', ['multiply', 'decimal_csv'])
@pytest.mark.parametrize('tone', [1, 3, 10, 20])
def test_canonical_four_block_oracle_has_no_spurious_level_spread(encoding, tone):
    powers, expected = [], []
    t = clock(encoding)
    for lo in [2, 4, 6, 8]:
        tb = t[(t >= lo) & (t < lo + 2)]
        x = np.sin(2 * np.pi * tone * tb)
        s = api.spectrum(tb, x, N, cadence_dt_s=DT)
        bands = api.band_powers(s['frequency_Hz'], s['density'], EDGES)
        assert [b['bin_count'] for b in bands] == [2, 4, 14, 21]
        powers.append([b['power'] for b in bands])
        expected.append(fixed_dft_band_oracle(x))
    np.testing.assert_allclose(powers, expected, rtol=3e-13, atol=1e-26)
    active = np.max(expected, axis=0) > 1e-10
    # Only energy-bearing bands: cancellation-scale leakage is not a level floor.
    assert np.max(np.ptp(10 * np.log10(np.asarray(powers)[:, active]), axis=0)) < 1e-10


@pytest.mark.parametrize('direction', [-1, 1])
def test_absolute_snap_cap_remains_subnanobin_for_large_indices(direction):
    # At q=500000, 32*eps*q exceeds the explicit 1e-9-bin cap.
    f = np.arange(500002, dtype=float)
    edges = [499999., 500000. + direction * 2e-9, 500001.]
    bands = api.band_powers(f, np.ones_like(f), edges)
    assert [b['bin_count'] for b in bands] == ([1, 2] if direction < 0 else [2, 1])
    assert all(max(b['edge_snap_tolerance_bins']) <= 1e-9 for b in bands)


@pytest.mark.parametrize('direction', [-1, 1])
def test_genuinely_off_edge_frequency_grid_does_not_move_into_nominal_bands(direction):
    f = np.arange(1601) * 0.5 * (1 + direction * 1e-8)
    bands = api.band_powers(f, np.ones_like(f), EDGES)
    for i, b in enumerate(bands):
        keep = (f >= EDGES[i]) & ((f <= EDGES[i + 1]) if i == 3 else (f < EDGES[i + 1]))
        assert b['bin_indices'] == np.flatnonzero(keep).tolist()


def test_nonlattice_frequency_perturbation_is_rejected_not_silently_relabelled():
    f = np.arange(1601) * 0.5
    f[2] += 1e-10
    with pytest.raises(ValueError, match='lattice'):
        api.band_powers(f, np.ones_like(f), EDGES)
