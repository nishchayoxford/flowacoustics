/* Read-only presentation of the original single-phase Case 4 reduction. */
import {escapeHTML as e, formatNumber as number} from './mrl2d-plots.mjs';

const label = (x, y, value, anchor = 'middle', cls = '') =>
  `<text x="${x}" y="${y}" text-anchor="${anchor}" class="${cls}">${e(value)}</text>`;

/* Renderers accept only admitted data; no synthesis, smoothing or re-windowing. */
export function renderPolar(data, metric = 'db') {
  if (!['db', 'pa'].includes(metric)) throw new Error('Unknown polar metric.');
  const db = metric === 'db';
  const values = db ? data.band_level_db_re_1upa : data.band_rms_pa;
  const finite = values.filter(Number.isFinite);
  const units = db ? 'dB re 1 µPa' : 'Pa';
  const title = db ? 'Selected-band level (dB re 1 µPa; band RMS)' : 'Band RMS (Pa)';
  if (!finite.length) return '<p>No finite dB levels: all observers have zero RMS. No floor is substituted.</p>';
  const min = db ? Math.floor(Math.min(...finite) / 10) * 10 - 10 : 0;
  const max = db ? Math.max(min + 20, Math.ceil(Math.max(...finite) / 10) * 10) : Math.max(...finite) || 1;
  const step = db ? 10 : max / 4;
  const cx = 210, cy = 182, radius = 133;
  const R = value => (value - min) / (max - min) * radius;
  const ticks = Array.from({length: Math.round((max - min) / step)}, (_, i) => min + (i + 1) * step);
  const markers = values.flatMap((v, j) => {
    if (!Number.isFinite(v)) return [];
    const a = data.angles_deg[j] * Math.PI / 180;
    return [{x: cx + R(v) * Math.cos(a), y: cy - R(v) * Math.sin(a)}];
  });
  // Pick each radial label's direction clear of the actual discrete receivers.
  // Conservative 12px-text bounds include the label halo and marker stroke.
  const clear = (angle, value) => {
    const a = angle * Math.PI / 180, x = cx + R(value) * Math.cos(a);
    const y = cy - R(value) * Math.sin(a) - 5, half = number(value).length * 4 + 3;
    return markers.every(m => m.x + 7 < x - half || m.x - 7 > x + half
      || m.y + 7 < y - 15 || m.y - 7 > y + 3);
  };
  let content = '';
  for (const value of ticks) {
    const r = R(value);
    const labelAngle = ([45, 135, 225, 315, 15, 75, 105, 165, 195, 255, 285, 345].find(a => clear(a, value)) ?? 45) * Math.PI / 180;
    content += `<circle class="grid" data-radial-tick="${value}" cx="${cx}" cy="${cy}" r="${r}"/>`;
    content += label(cx + r * Math.cos(labelAngle), cy - r * Math.sin(labelAngle) - 5, number(value), 'middle', 'radial-label');
  }
  for (const angle of [0, 90, 180, 270]) {
    const a = angle * Math.PI / 180;
    content += `<path class="grid" d="M${cx},${cy} L${cx + radius * Math.cos(a)},${cy - radius * Math.sin(a)}"/>`;
    content += label(cx + (radius + 26) * Math.cos(a), cy - (radius + 26) * Math.sin(a) + 4, `${angle}°`);
  }
  values.forEach((v, j) => {
    if (!Number.isFinite(v)) return;
    const angle = data.angles_deg[j], a = angle * Math.PI / 180;
    content += `<circle data-polar-angle="${angle}" data-value="${v}" data-radius="${R(v)}" cx="${cx + R(v) * Math.cos(a)}" cy="${cy - R(v) * Math.sin(a)}" r="4" class="receiver-marker"><title>${number(angle)}°: ${number(v)} ${units}</title></circle>`;
  });
  return `<svg viewBox="0 0 420 366" role="img" aria-label="${e(title)}" data-polar-metric="${metric}" data-radial-min="${min}" data-radial-max="${max}"><title>${e(title)}</title><desc>Unconnected receiver samples. Radial ticks in ${units}; centre ${number(min)} ${units}. 0° is +x; angles increase counterclockwise.</desc>${content}</svg><p class="figure-note">Centre: ${number(min)} ${units}. Rings show ${units}, not normalized levels. No interpolation between receivers.${finite.length < 12 ? ' Observers with zero RMS have no finite dB level and are omitted; no floor is substituted.' : ''}</p>`;
}
export function renderSpectra(data, metric = 'rms') {
  if (!['rms', 'psd'].includes(metric)) throw new Error('Unknown spectral metric.');
  const rms = metric === 'rms', matrix = rms ? data.bin_rms_pa : data.psd_pa2_per_hz;
  const units = rms ? 'Pa' : 'Pa²/Hz', title = rms ? 'Bin RMS (Pa)' : 'PSD (Pa²/Hz)';
  const max = Math.max(...matrix.flat()) * 1.05 || 1;
  const X = value => 75 + (value - .625) / (25 - .625) * 270;
  const Y = value => 181 - value / max * 161;
  return data.angles_deg.map((angle, j) => {
    let grid = '';
    for (const fraction of [0, .5, 1]) {
      const v = max * fraction;
      grid += `<path class="grid" d="M75,${Y(v)} H345"/>` + label(67, Y(v) + 4, number(v), 'end');
    }
    for (const f of [.625, 5, 10, 15, 20, 25]) {
      grid += `<path class="grid" d="M${X(f)},20 V181"/>` + label(X(f), 201, number(f), f === .625 ? 'start' : f === 25 ? 'end' : 'middle');
    }
    const values = matrix.map(row => row[j]);
    const path = data.frequency_hz.map((f, i) => `${i ? 'L' : 'M'}${X(f)},${Y(values[i])}`).join(' ');
    const p = data.peaks[j], value = rms ? p.bin_rms_pa : p.psd_pa2_per_hz;
    const peak = p.index === null ? '' : `<circle data-peak-marker="${angle}" data-frequency-hz="${p.frequency_hz}" data-amplitude="${value}" cx="${X(p.frequency_hz)}" cy="${Y(value)}" r="4" class="peak-marker"><title>Largest sampled bin: ${number(p.frequency_hz)} Hz, ${number(value)} ${units}</title></circle>`;
    const qualification = p.index === null ? 'No nonzero peak' : `Peak: ${number(p.frequency_hz)} Hz · ${number(value)} ${units}${p.is_band_edge ? ' · band-edge candidate' : ''}${p.tied_maximum_count > 1 ? ` · ${p.tied_maximum_count} tied maxima` : ''}`;
    return `<article class="observer-card" data-observer-card="${angle}" data-metric="${metric}"><h3>Observer ${number(angle)}°</h3><div class="axis-heading">${title}</div><svg viewBox="0 0 360 215" role="img" aria-label="${title} at ${number(angle)} degrees" data-y-max="${max}"><title>Observer ${number(angle)}° — ${title}</title><desc>196 retained bins; common axes across all 12 observers. Open rust marker: largest sampled bin.</desc>${grid}<path data-spectrum-path="${angle}" d="${path}" class="spectrum-line"/>${peak}</svg><div class="axis-bottom">Frequency (Hz)</div><p class="peak-caption">${qualification}</p></article>`;
  }).join('');
}

export function renderPeakTable(data) {
  const headers = ['Observer', 'Peak frequency (Hz)', 'Bin RMS (Pa)', 'PSD (Pa²/Hz)', 'Band RMS (Pa)', 'Band level (dB re 1 µPa)', 'Peak qualification'];
  const rows = data.peaks.map((p, j) => {
    const note = p.index === null ? 'No nonzero peak' : [p.is_band_edge ? 'Band-edge candidate' : 'Interior sampled bin', p.tied_maximum_count > 1 ? `${p.tied_maximum_count} tied maxima` : 'Unique maximum'].join('; ');
    return `<tr data-peak-row="${p.angle_deg}"><th scope="row">${number(p.angle_deg)}°</th><td>${p.frequency_hz === null ? '—' : number(p.frequency_hz)}</td><td>${number(p.bin_rms_pa)}</td><td>${number(p.psd_pa2_per_hz)}</td><td>${number(data.band_rms_pa[j])}</td><td>${data.band_level_db_re_1upa[j] === null ? '−∞ (zero RMS)' : number(data.band_level_db_re_1upa[j])}</td><td>${note}</td></tr>`;
  }).join('');
  return `<table><caption>Largest retained bin at each of the 12 observers; band quantities use all 196 bins.</caption><thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table>`;
}

export function validateSummary(raw) {
  if (!raw || typeof raw !== 'object') throw new Error('Unavailable spectral summary.');
  if (raw.schema !== 'mrl2d-case4-spectral-v1') throw new Error('Unsupported schema.');
  if (raw.case_id !== 'mrl2d_les_permeable') throw new Error('Unexpected case identity.');
  const require = (ok, field) => { if (!ok) throw new Error(`Invalid spectral summary: ${field}.`); };
  const finite = v => typeof v === 'number' && Number.isFinite(v);
  const nonnegative = v => finite(v) && v >= 0;
  const close = (a, b) => finite(a) && finite(b) && Math.abs(a - b) <= 1e-9 * Math.max(Math.abs(b), 1e-25);
  const array = (v, n, check, field) => require(Array.isArray(v) && v.length === n && v.every(check), field);
  array(raw.frequency_hz, 196, finite, 'frequencies');
  array(raw.angles_deg, 12, finite, 'angles');
  require(raw.angles_deg.every((a, j) => close(a, j * 30)), 'observer angles');
  const s = raw.estimator;
  require(s && s.sample_count === 4000 && s.window === 'periodic Hann', 'estimator');
  require(close(s.sample_interval_s, .002) && close(s.frequency_spacing_hz, .125)
    && close(s.enbw_hz, .1875), 'cadence, bin spacing or ENBW');
  array(s.sample_window_s, 2, finite, 'sample window');
  require(close(s.sample_window_s[0], 2.002) && close(s.sample_window_s[1], 10), 'sample support');
  array(s.band_hz, 2, finite, 'selected band');
  require(close(s.band_hz[0], .625) && close(s.band_hz[1], 25), 'band endpoints');
  require(raw.frequency_hz.every((f, i) => close(f, .625 + i * s.frequency_spacing_hz)), 'frequency grid');
  for (const key of ['psd_definition', 'bin_rms_definition', 'band_level_definition']) {
    require(typeof s[key] === 'string' && s[key].trim().length > 0, key);
  }
  for (const key of ['bin_rms_pa', 'psd_pa2_per_hz']) {
    array(raw[key], 196, row => Array.isArray(row) && row.length === 12 && row.every(nonnegative), key);
  }
  array(raw.band_rms_pa, 12, nonnegative, 'band RMS');
  array(raw.band_level_db_re_1upa, 12, v => v === null || finite(v), 'band levels');
  array(s.psd_band_integral_pa2, 12, nonnegative, 'PSD integrals');
  array(s.coherent_band_mean_square_pa2, 12, nonnegative, 'coherent mean square');
  array(raw.peaks, 12, p => p && typeof p === 'object', 'peaks');
  for (let j = 0; j < 12; j++) {
    const bins = raw.bin_rms_pa.map(row => row[j]);
    const density = raw.psd_pa2_per_hz.map(row => row[j]);
    require(bins.every((v, i) => close(density[i], v * v / s.enbw_hz)), 'PSD normalization');
    const meanSquare = bins.reduce((sum, v) => sum + v * v, 0);
    require(close(raw.band_rms_pa[j] ** 2, meanSquare), 'band RMS consistency');
    require(close(s.coherent_band_mean_square_pa2[j], meanSquare), 'coherent integral');
    require(close(s.psd_band_integral_pa2[j], density.reduce((sum, v) => sum + v * s.frequency_spacing_hz, 0)), 'PSD integral');
    const band = raw.band_rms_pa[j], db = raw.band_level_db_re_1upa[j];
    require(band === 0 ? db === null : close(db, 20 * Math.log10(band / 1e-6)), 'dB reference');
    const p = raw.peaks[j], maximum = Math.max(...bins);
    require(close(p.angle_deg, raw.angles_deg[j]), 'peak observer');
    if (maximum === 0) {
      require(p.index === null && p.frequency_hz === null && p.bin_rms_pa === 0
        && p.psd_pa2_per_hz === 0 && p.tied_maximum_count === 0 && p.is_band_edge === false, 'zero peak');
    } else {
      require(Number.isInteger(p.index) && p.index >= 0 && p.index < 196 && bins[p.index] === maximum, 'peak bin');
      require(close(p.frequency_hz, raw.frequency_hz[p.index]) && close(p.bin_rms_pa, maximum)
        && close(p.psd_pa2_per_hz, density[p.index]), 'peak values');
      require(p.is_band_edge === (p.index === 0 || p.index === 195)
        && p.tied_maximum_count === bins.filter(v => v === maximum).length, 'peak qualification');
    }
  }
  let hashes = 0, originalSource = false;
  const sourcePath = value => /(?:^|\/)mrl2d_les_permeable\.npz$/.test(value);
  const safe = value => !/(^\/|\.\.|\\|^[a-z]+:|<|>)/i.test(value);
  const provenance = (value, key = '', depth = 0) => {
    require(depth < 10 && safe(key), 'provenance nesting/path');
    if (sourcePath(key)) originalSource = true;
    if (typeof value === 'string') {
      require(safe(value), 'private or unsafe provenance');
      if (sourcePath(value)) originalSource = true;
      if (/sha256/i.test(key)) require(/^[a-f0-9]{64}$/.test(value), 'source SHA256');
      if (/^[a-f0-9]{64}$/.test(value)) hashes++;
    } else if (Array.isArray(value)) value.forEach(v => provenance(v, key, depth + 1));
    else if (value && typeof value === 'object') Object.entries(value).forEach(([k, v]) => provenance(v, k, depth + 1));
    else require(value === null || typeof value === 'boolean' || finite(value), 'provenance value');
  };
  require(raw.provenance && typeof raw.provenance === 'object', 'provenance');
  provenance(raw.provenance);
  require(hashes > 0 && originalSource, 'original Case 4 source path and hashes');
  return raw;
}

export function mountPage(root = document) {
  const get = id => root.getElementById(id);
  const status = get('spectra-status'), results = get('spectra-results');
  const reload = get('reload-spectra');
  const controls = [...root.querySelectorAll('#metric-controls button')];
  let data = null, metric = 'rms';
  const setStatus = (state, message) => {
    status.dataset.state = state;
    status.textContent = message;
  };
  const renderMetric = () => {
    if (!data) return;
    get('observer-spectra').innerHTML = renderSpectra(data, metric);
    controls.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.metric === metric)));
    get('metric-note').textContent = metric === 'rms'
      ? 'Showing coherent bin RMS (Pa). Common axes across all 12 observers.'
      : 'Showing one-sided PSD (Pa²/Hz). ENBW 0.1875 Hz; common axes across all 12 observers.';
  };
  const load = async () => {
    if (reload.disabled) return;
    data = null;
    results.hidden = true;
    for (const id of ['polar-db', 'polar-pa', 'observer-spectra', 'peak-table']) get(id).replaceChildren();
    controls.forEach(button => { button.disabled = true; });
    reload.disabled = true;
    setStatus('loading', 'Loading the original Case 4 spectral summary…');
    try {
      const response = await fetch('./data/case4-spectra/summary.json', {
        cache: 'no-store', signal: AbortSignal.timeout(20000),
      });
      if (response.status !== 200) {
        setStatus('unavailable', `Spectral summary unavailable (HTTP ${response.status}). No replacement results are shown. Retry with Reload data.`);
        return;
      }
      data = validateSummary(await response.json());
      get('polar-db').innerHTML = renderPolar(data, 'db');
      get('polar-pa').innerHTML = renderPolar(data, 'pa');
      get('peak-table').innerHTML = renderPeakTable(data);
      renderMetric();
      controls.forEach(button => { button.disabled = false; });
      results.hidden = false;
      setStatus('ready', 'Original Case 4 loaded · 12 observers · 196 retained bins · 0.625–25 Hz.');
    } catch {
      data = null;
      results.hidden = true;
      for (const id of ['polar-db', 'polar-pa', 'observer-spectra', 'peak-table']) get(id).replaceChildren();
      setStatus('error', 'Could not load a valid Case 4 spectral summary. The request failed or its data/metadata did not match the fixed-case contract. No replacement results are shown. Retry with Reload data.');
    } finally {
      reload.disabled = false;
    }
  };
  controls.forEach(button => button.addEventListener('click', () => {
    metric = button.dataset.metric;
    renderMetric();
  }));
  reload.addEventListener('click', load);
  return load();
}

if (typeof document !== 'undefined' && document.getElementById('spectra-status')) mountPage();
