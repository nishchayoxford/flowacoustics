/* Independent, read-only pressure-loading adapter. No solver or fixture fallback. */
import {renderChart, escapeHTML, formatNumber} from './mrl2d-plots.mjs';

export function buildView(data, {window='primary',receiver=0,component='x'} = {}, options = {}) {
  validateComparison(data,options);
  requireValue(Object.hasOwn(WINDOWS,window) && Number.isInteger(receiver) && receiver>=0 && receiver<12 && ['x','y'].includes(component),'selection');
  const w=data.windows[window],a=w.cases.on.analysis,angle=data.receivers.angle_deg[receiver];
  const testOnly=data.mode==='fixture',origin=testOnly?'TEST ONLY — analytical fixture, not native CFD':'Native pressure-loading comparison';
  const chart=(id,title,x_label,y_label,traces,kind='line')=>({id,title,x_label,y_label,traces,kind,origin,window,x_scale:'linear',y_scale:'linear',description:origin});
  const traces=fn=>['on','off'].map((id,i)=>({name:i?'OFF · counterfactual control':'ON · phase change enabled',dash:i===1,...fn(w.cases[id],data.sources[id])}));
  return {testOnly,origin,selection:{window,receiver,component,angle},analysis:a,
    metrics:{on:{rms:w.cases.on.band_rms_pa[receiver],level:w.cases.on.band_level_db_re_1uPa[receiver]},off:{rms:w.cases.off.band_rms_pa[receiver],level:w.cases.off.band_level_db_re_1uPa[receiver]},deltaRMS:w.comparison.band_rms_on_minus_off_pa[receiver],deltaLevel:w.comparison.band_level_on_minus_off_db[receiver],deltaStatus:w.comparison.band_level_delta_status[receiver]},
    charts:{
      history:chart('loading-history',`Selected-band pressure at ${angle}°`,'Periodic synthesis time · s','Pressure-loading contribution · Pa',traces(c=>({x:c.time_s,y:c.pressure_pa.map(r=>r[receiver])}))),
      spectrum:chart('loading-spectrum',`Coherent bin RMS at ${angle}°`,'Frequency · Hz','Coherent bin RMS · Pa',traces(c=>({x:c.frequency_hz,y:c.bin_rms_pa.map(r=>r[receiver])}))),
      directions:chart('loading-directions','Twelve unconnected directional samples','Angle counterclockwise from +x · degrees','Selected-band RMS · Pa',traces(c=>({x:data.receivers.angle_deg,y:c.band_rms_pa})),'polar'),
      force:{...chart('loading-force',`Full source pressure force · F${component}`,'Full source time · s','Pressure force on solid · N/m',traces((c,s)=>({x:s.source_time_s,y:s.pressure_force_n_per_m.map(r=>r[component==='x'?0:1])}))),window:[data.sources.on.source_time_s[0],data.sources.on.source_time_s.at(-1)]}
    }
  };
}

const e=escapeHTML, f=formatNumber;
export const COMPARISON_PATH='./data/cavitation-loading/comparison.json';
const levelText = value => value===null?'Undefined at exactly zero pressure':`${f(value)} dB re 1 µPa`;
const deltaText = (value,status) => status==='finite'?`${f(value)} dB`:({both_zero:'Undefined ratio: both ON and OFF are zero',off_zero:'Formally +infinity: OFF is zero',on_zero:'Formally −infinity: ON is zero'}[status]);
export function renderShell() {
  return `<div class="loading-controls">
    <label>1 · Receiver direction<select data-loading-receiver disabled>${Array.from({length:12},(_,i)=>`<option value="${i}">${i*30}° from +x</option>`).join('')}</select></label>
    <label>2 · Source analysis window<select data-loading-window disabled><option value="primary">Primary · 2–10 s</option><option value="late">Late · 6–10 s</option></select></label>
    <button data-loading-reload type="button" disabled>Reload loading results</button>
  </div><p data-loading-status class="bundle-status" role="status" data-state="pending">Native loading results are withheld pending verified adapter and completed matched data. No substitute values are shown.</p><div data-loading-results></div>`;
}
export function renderResults(data,v,width=720) {
  const {analysis:a,selection:s,metrics:m,charts}=v;
  const fig=(key,caption)=>`<figure class="computed-chart" data-loading-chart="${key}"><h3>${e(charts[key].title)}</h3>${renderChart(charts[key],width)}<figcaption>${caption}</figcaption></figure>`;
  const cell=id=>`<article><h3>${id.toUpperCase()} · selected band</h3><p class="loading-value">${f(m[id].rms)} Pa <span>RMS</span></p><p>${levelText(m[id].level)}</p></article>`;
  const cases=data.windows[s.window].cases,c=data.windows[s.window].comparison;
  const rows=data.receivers.angle_deg.map((angle,i)=>`<tr${i===s.receiver?' aria-current="true"':''}><th scope="row">${angle}°</th>${['on','off'].map(id=>`<td>${f(cases[id].band_rms_pa[i])}</td><td>${levelText(cases[id].band_level_db_re_1uPa[i])}</td>`).join('')}<td>${f(c.band_rms_on_minus_off_pa[i])}</td><td>${deltaText(c.band_level_on_minus_off_db[i],c.band_level_delta_status[i])}</td></tr>`).join('');
  const links=v.testOnly?'<p>TEST ONLY: downloads are withheld. The analytical fixture exists only in the isolated test fetch; it is not published data.</p>':`<ul class="loading-downloads">${['comparison.json','comparison.schema.json','manifest.json','native-evidence.json','clock-audit.json','on-loading.csv','off-loading.csv',`${s.window}-on-receivers.csv`,`${s.window}-off-receivers.csv`,`${s.window}-on-spectrum.csv`,`${s.window}-off-spectrum.csv`,`${s.window}-on.npz`,`${s.window}-off.npz`,`${s.window}-delta.npz`,`${s.window}-delta-receivers.csv`,`${s.window}-band-comparison.csv`].map(name=>`<li><a download href="./data/cavitation-loading/${name}">${name}</a></li>`).join('')}</ul>`;
  return `<p class="loading-origin${v.testOnly?' loading-test-only':''}">${e(v.origin)}</p>
    <p data-loading-selection><strong>${s.angle}° receiver · 1 m · ${s.window==='primary'?'Primary 2–10 s':'Late 6–10 s'}</strong><br>Selected band ${f(a.actual_band_hz[0])}–${f(a.actual_band_hz[1])} Hz. Actual source samples ${f(a.sample_window_s[0])}–${f(a.sample_window_s[1])} s (${a.sample_count} samples); Δf = ${f(a.frequency_spacing_hz)} Hz.</p>
    <p class="loading-window-note">Try another direction: the pressure-loading contribution depends on where the receiver sits. Then choose the late window: its shorter record uses a <strong>changed band, 1.25–25 Hz rather than 0.625–25 Hz</strong>. This is window/estimator sensitivity, not a same-band comparison, convergence test or uncertainty interval.</p>
    <div class="loading-metrics">${cell('on')}${cell('off')}<article><h3>ON minus OFF</h3><p class="loading-value">${f(m.deltaRMS)} Pa <span>RMS magnitude difference</span></p><p>${deltaText(m.deltaLevel,m.deltaStatus)}</p></article></div>
    <p class="figure-note">The difference is ON RMS minus OFF RMS, not RMS of the difference waveform and not an isolated cavitation source. Levels use 1 µPa; exact zero is undefined in dB. No floors are added; tiny numerical values need caution.</p>
    ${fig('history','Paired selected-bin periodic synthesis, conditioned by the source window; not causal arrivals, full-band recovered sound or a recording to listen to. Every supplied sample is retained.')}
    <div class="loading-paired-plots">${fig('spectrum','Periodic Hann, coherent gain 0.5. Coherent bin RMS in Pa, not PSD; the linear scale keeps exact zeros. Read which frequencies contribute, not a resolved-CFD bandwidth claim.')}${fig('directions','Twelve receiver directions, two unconnected sets of samples. No angular interpolation or far-field claim. The receiver selector changes the history and spectrum above; this plot always retains all twelve directions.')}</div>
    <details data-loading-details="source"><summary>Source force, method and exact data</summary>
      <p><strong>Full source record: 0–10 s, actual samples ${f(data.sources.on.source_time_s[0])}–${f(data.sources.on.source_time_s.at(-1))} s.</strong> All 5000 untapered samples remain here when the analysis window changes. This is force on the solid; integrated acoustic source loading has the opposite sign.</p>
      <label class="loading-force-control">Source force component<select data-loading-component><option value="x"${s.component==='x'?' selected':''}>Fx · N/m</option><option value="y"${s.component==='y'?' selected':''}>Fy · N/m</option></select></label>
      ${fig('force','Full source force per unit span, including startup; neither receiver pressure nor a sound signal. No density multiplication or additional depth division.')}
      <dl class="loading-method"><dt>Source definition</dt><dd>Q = 0; L = (absolute p − 3500 Pa)n, pressure loading only. Actual recorded U and motion stay unchanged; no density gain.</dd><dt>Propagation</dt><dd>Homogeneous water, c₀ = 1480 m/s; zero acoustic mean flow. No vapour refraction or scattering.</dd><dt>Conditioning</dt><dd>${e(a.mean_conditioning)}. ${e(a.signal_semantics)}.</dd><dt>Estimator</dt><dd>${a.sample_count} samples, nominal ${f(a.nominal_period_s)} s; ${f(a.cadence_s)} s cadence; Hann coherent gain ${f(a.window_coherent_gain)}, RMS gain ${f(a.window_rms_gain)}, ENBW ${f(a.hann_enbw_hz)} Hz. Bins ${a.bin_indices[0]}–${a.bin_indices.at(-1)}.</dd><dt>Series</dt><dd>${a.series_terms} terms; exported bounds concern truncation only, not CFD accuracy or total numerical error.</dd></dl>
      ${v.testOnly?'':'<p data-loading-clock>Raw source times are preserved exactly, including binary64 accumulation. The FFT uses an endpoint-fit index clock, treating fixed-step roundoff as nearly uniform sampling—not resampling. Analysis cadence, period, Δf and ENBW use the full-record fit; actual frequency bins use each window fit. See the clock audit for timing bounds; this is not CFD validation.</p>'}
      <ul>${data.limitations.map(t=>`<li>${e(t)}</li>`).join('')}</ul>
      <h3>All receiver values · ${e(s.window)} window</h3><div class="loading-table-wrap" tabindex="0" role="region" aria-label="Receiver values; scroll horizontally"><table><thead><tr><th>Direction</th><th>ON RMS · Pa</th><th>ON level</th><th>OFF RMS · Pa</th><th>OFF level</th><th>ON − OFF · Pa</th><th>Level difference</th></tr></thead><tbody>${rows}</tbody></table></div>
      <details data-loading-details="identity"><summary>Declared source identities and matching</summary><p>These digests identify the producer inputs. Browser shape checks do not independently establish CFD completion, coupling or physical accuracy.</p><pre>${e(JSON.stringify({status:data.status,matching:data.matching,identities:data.identities},null,2))}</pre></details>
      <h3>Full arrays and provenance</h3><p>Displayed numbers are rounded; JSON, CSV and NPZ retain the exported values. Pressure histories and spectra are supplied arrays, not regenerated by this page.</p>${links}
    </details>`;
}

export async function loadComparison({baseURL=globalThis.location?.href,fetchImpl=globalThis.fetch,TEST_ONLY_allowAnalyticalFixture=false}={}) {
  try {
    const base=new URL(baseURL),url=new URL(COMPARISON_PATH,base);
    requireValue(['http:','https:'].includes(base.protocol) && !base.username && !base.password && url.origin===base.origin,'same-origin HTTP results path');
    const response=await fetchImpl(url.href,{redirect:'error',cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(30000)});
    requireValue(response.url===url.href && !response.redirected,'same-origin response identity');
    if(response.status===404 || response.status===202) {
      await response.body?.cancel();
      return {state:response.status===404?'absent':'pending',message:response.status===404?'Native loading results are not supplied. Verified adapter and completed matched data are still required; no substitute values are shown.':'Native loading results are pending. No partial arrays or substitute values are shown.'};
    }
    if(!response.ok) {await response.body?.cancel();throw Error(`HTTP ${response.status}`);}
    const data=validateComparison(await response.json(),{TEST_ONLY_allowAnalyticalFixture});
    return {state:data.mode==='fixture'?'test-only':'ready',data,message:data.mode==='fixture'?'TEST ONLY — analytical fixture, not native CFD. Actual native results remain withheld.':'Verified-status native pressure-loading bundle loaded. This is not physical validation.'};
  } catch(error) {
    return {state:'error',message:`Loading results unavailable or rejected. No values shown. ${error instanceof Error?error.message:'Invalid response'}`};
  }
}

const requireValue = (condition, name) => { if (!condition) throw new Error(`Loading comparison rejected: ${name}`); };

export function validateComparison(data, {TEST_ONLY_allowAnalyticalFixture = false} = {}) {
  requireValue(data?.schema === 'mrl2d-cavitation-loading-comparison-v1', 'schema');
  const native = data.status === 'completed_native_pressure_loading_comparison' && data.mode === 'production' && data.source_type === 'native_cfd';
  const fixture = TEST_ONLY_allowAnalyticalFixture === true && data.status === 'analytical_fixture_not_native' && data.mode === 'fixture' && data.source_type === 'analytical_fixture';
  requireValue(native || fixture, 'native admission required; analytical fixture refused by default');
  requireValue([data.native_coupling_verified, data.model?.native_coupling_verified, data.matching?.native_coupling_verified, data.matching?.native_completion_verified].every(v => v === native), 'native verification gates');
  requireValue(data.matching?.matched_array_support === true && data.matching?.matched_input_identities === true, 'matching admission');
  validateShape(data, native);
  return data;
}

const finite = Number.isFinite;
const nonnegative = v => finite(v) && v >= 0;
const positiveInteger = v => Number.isSafeInteger(v) && v > 0;
const text = v => typeof v === 'string' && v.length > 0 && v.length <= 4096;
const close = (a, b, tolerance = 1e-12) => finite(a) && finite(b) && Math.abs(a-b) <= tolerance;
function keys(value, names, label) {
  const expected = names.split(' ');
  requireValue(value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === expected.length && expected.every(k => Object.hasOwn(value,k)), `${label} keys`);
}
function vector(value, n, label, predicate = finite) {
  requireValue(Array.isArray(value) && value.length === n && value.every(predicate), `${label} shape/finite values`);
}
function matrix(value, n, m, label, predicate = finite) {
  requireValue(Array.isArray(value) && value.length === n, `${label} rows`);
  value.forEach(row => vector(row,m,label,predicate));
}
function constants(value, expected, label) {
  keys(value,Object.keys(expected).join(' '),label);
  for (const [k,v] of Object.entries(expected)) requireValue(JSON.stringify(value[k]) === JSON.stringify(v), `${label}.${k}`);
}
function clock(times, n, start, label) {
  vector(times,n,label);
  times.forEach((t,i) => requireValue(close(t,start+(i+1)*.002), `${label} uniform complete clock`));
}
// Independent oracle: one ordinary binary64 addition at EVERY solver step.
// Never replace input times with this reference or use source-cadence leaps.
const NATIVE_CLOCK = Object.freeze((() => {
  const samples=[];let time=0;
  for(let step=1;step<=160000;step++) {
    time+=6.25e-5;
    if(step%32===0)samples.push(time);
  }
  return samples;
})());
function nativeClock(times,n,label) {
  vector(times,n,label);
  times.forEach((t,i)=>requireValue(t===NATIVE_CLOCK[5000-n+i],`${label} exact binary64 recurrence`));
}
// Small arithmetic-order allowance only; NOT a clock or cadence tolerance.
const roundoffEqual = (a,b) => finite(a) && finite(b) && Math.abs(a-b)<=8*Number.EPSILON*Math.max(Math.abs(a),Math.abs(b));
const endpointCadence = times => (times.at(-1)-times[0])/(times.length-1);
const MODEL = {
  Q_included:false, convective_loading_included:false,
  density_model:'not used in pressure-loading projection', mixture_density_used_in_acoustic_source:false,
  omitted_terms:['thickness','relative advective momentum flux','viscous traction','cavity-volume source','exterior volume quadrupoles'],
  pressure_conversion:'absolute Pa minus 3500 Pa once; no density gain',
  pressure_interpretation:'ideal span-invariant pressure-loading contribution; not total sound',
  projection:'explicit source arrays Q=0 and L=(p-pInf)n; actual U and motion unchanged',
  source_component:'solid_wall_pressure_loading_only', source_definition:'Q=0; L=(p_absolute-3500 Pa)*n', source_model:'pressure_loading_only'
};
const WINDOWS = {primary:{start:2,n:4000,f:196,period:8,df:.125,low:.625},late:{start:6,n:2000,f:96,period:4,df:.25,low:1.25}};
function validateShape(d, native) {
  keys(d,'schema status mode source_type native_coupling_verified model medium receivers matching limitations identities sources windows','bundle');
  constants(d.medium,{c0_m_s:1480,reference_pressure_pa:3500},'medium/units');
  constants(d.model,{...MODEL,native_coupling_verified:native},'model/units');
  requireValue(Array.isArray(d.limitations) && d.limitations.length > 0 && d.limitations.every(text),'limitations');
  keys(d.identities,'on off matched_inputs','identities');
  for (const name of ['on','off','matched_inputs']) {
    keys(d.identities[name],name==='matched_inputs'?'boundary_conditions initial_state mesh motion numerics solver':'time xy normal ds p U surface_velocity blade_id panel_ids',`identities.${name}`);
    requireValue(Object.values(d.identities[name]).every(v => typeof v==='string' && /^[0-9a-f]{64}$/.test(v)),'identity SHA256 tokens');
  }
  for(const k of ['xy','normal','ds','surface_velocity','blade_id','panel_ids']) requireValue(d.identities.on[k]===d.identities.off[k],`matched identity ${k}`);
  const m=d.matching;
  keys(m,'cadence_s clock_match_atol_s clock_max_abs_difference_s conservative_peak_bytes geometry_match matched_array_support matched_input_identities native_completion_verified native_coupling_verified source_frames source_panels','matching');
  requireValue(m.cadence_s===.002 && m.clock_match_atol_s===1e-12 && nonnegative(m.clock_max_abs_difference_s) && m.clock_max_abs_difference_s<=1e-12 && m.source_frames===5000 && positiveInteger(m.source_panels) && positiveInteger(m.conservative_peak_bytes) && m.geometry_match===(native?'exact native arrays, including every raw ds; stable sorted global IDs':'exact arrays; no reordering'),'matching support');
  keys(d.receivers,'xy_m angle_deg display','receivers');
  vector(d.receivers.angle_deg,12,'receiver angles');matrix(d.receivers.xy_m,12,2,'receivers');
  requireValue(d.receivers.display==='discrete_unconnected_samples','unconnected receivers');
  d.receivers.angle_deg.forEach((a,i) => requireValue(a===i*30 && close(d.receivers.xy_m[i][0],Math.cos(a*Math.PI/180)) && close(d.receivers.xy_m[i][1],Math.sin(a*Math.PI/180)),'1 m receiver geometry'));
  keys(d.sources,'on off','sources');
  for(const name of ['on','off']) {
    const s=d.sources[name];keys(s,'source_time_s pressure_force_n_per_m integrated_source_loading_n_per_m','source');
    if(native)nativeClock(s.source_time_s,5000,'full source');
    else clock(s.source_time_s,5000,0,'full source');
    matrix(s.pressure_force_n_per_m,5000,2,'pressure force');matrix(s.integrated_source_loading_n_per_m,5000,2,'source loading');
  }
  d.sources.on.source_time_s.forEach((t,i) => requireValue(close(t,d.sources.off.source_time_s[i]),'matched source clocks'));
  keys(d.windows,'primary late','windows');
  for(const [name,spec] of Object.entries(WINDOWS)) {
    const w=d.windows[name];keys(w,'description cases comparison','window');requireValue(text(w.description),'window description');keys(w.cases,'on off','cases');
    for(const id of ['on','off']) validateCase(w.cases[id],spec,d.sources[id],native);
    w.cases.on.time_s.forEach((t,i) => requireValue(close(t,w.cases.off.time_s[i]),'matched window clocks'));
    w.cases.on.frequency_hz.forEach((f,i) => requireValue((native?roundoffEqual:close)(f,w.cases.off.frequency_hz[i]),'matched window frequencies'));
    const c=w.comparison;
    keys(c,'alignment band_level_delta_status band_level_on_minus_off_db band_rms_on_minus_off_pa bin_rms_on_minus_off_pa coefficients_on_minus_off_imag_pa coefficients_on_minus_off_real_pa receiver_pressure_on_minus_off_pa','comparison');
    requireValue(c.alignment==='matching native sample/bin indices; original clocks retained','comparison alignment');
    vector(c.band_level_delta_status,12,'delta status',v=>['finite','both_zero','on_zero','off_zero'].includes(v));
    vector(c.band_level_on_minus_off_db,12,'delta levels',v=>v===null||finite(v));vector(c.band_rms_on_minus_off_pa,12,'RMS difference');
    matrix(c.receiver_pressure_on_minus_off_pa,spec.n,12,'pressure difference');
    for(const k of ['bin_rms_on_minus_off_pa','coefficients_on_minus_off_imag_pa','coefficients_on_minus_off_real_pa']) matrix(c[k],spec.f,12,k);
    for(let i=0;i<12;i++) {
      const on=w.cases.on.band_rms_pa[i],off=w.cases.off.band_rms_pa[i];
      const status=on===0?(off===0?'both_zero':'on_zero'):(off===0?'off_zero':'finite');
      requireValue(c.band_level_delta_status[i]===status,'level delta status');
      requireValue(status==='finite'?close(c.band_level_on_minus_off_db[i],20*(Math.log10(on)-Math.log10(off)),1e-8):c.band_level_on_minus_off_db[i]===null,'level difference');
      requireValue(close(c.band_rms_on_minus_off_pa[i],on-off,Math.max(Math.abs(on),Math.abs(off))*1e-10),'RMS difference');
      for(const id of ['on','off']) {const rms=w.cases[id].band_rms_pa[i],level=w.cases[id].band_level_db_re_1uPa[i];requireValue(rms===0?level===null:close(level,20*(Math.log10(rms)+6),1e-8),'band level/zero semantics');}
    }
  }
}
function validateCase(c,s,source,native) {
  keys(c,'analysis band_level_db_re_1uPa band_rms_pa bin_rms_pa coefficients_imag_pa coefficients_real_pa frequency_hz pressure_pa time_s','case');
  if(native)nativeClock(c.time_s,s.n,'window');
  else clock(c.time_s,s.n,s.start,'window');
  c.time_s.forEach((t,i) => requireValue(t===source.source_time_s[5000-s.n+i],'original window clock retained'));
  const period=native?s.n*endpointCadence(c.time_s):s.period;
  vector(c.frequency_hz,s.f,'frequency');c.frequency_hz.forEach((v,i)=>requireValue(native?roundoffEqual(v,(5+i)/period):close(v,(5+i)*s.df),'frequency bins'));
  matrix(c.pressure_pa,s.n,12,'pressure');
  for(const k of ['bin_rms_pa','coefficients_imag_pa','coefficients_real_pa']) matrix(c[k],s.f,12,k,k==='bin_rms_pa'?nonnegative:finite);
  vector(c.band_rms_pa,12,'band RMS',nonnegative);vector(c.band_level_db_re_1uPa,12,'band levels',v=>v===null||finite(v));
  const a=c.analysis;
  keys(a,'actual_band_hz algorithms bin_indices cadence_s chunk_count frequency_spacing_hz hann_enbw_hz mean_conditioning nominal_period_s nyquist_hz requested_band_hz sample_count sample_window_s series_terms series_truncation_bound_pa signal_semantics startup_excluded_s window window_coherent_gain window_rms_gain','analysis');
  const exact={cadence_s:.002,frequency_spacing_hz:s.df,hann_enbw_hz:1.5*s.df,nominal_period_s:s.period,nyquist_hz:250,sample_count:s.n,startup_excluded_s:s.start,window:'hann',window_coherent_gain:.5,mean_conditioning:'Eulerian kernel-projected time mean removed before window',signal_semantics:'periodic/window-conditioned band-limited, not causal arrivals'};
  if(native) {
    // Unchanged analysis_plan fits the FULL clock, before selecting the tail.
    // Its legacy "nominal" fields are numerical, not literal 8/4 s constants.
    // The actual spectrum above independently fits each selected window.
    const dt=endpointCadence(source.source_time_s),planPeriod=s.n*dt;
    const fitted={cadence_s:dt,frequency_spacing_hz:1/planPeriod,hann_enbw_hz:1.5/planPeriod,nominal_period_s:planPeriod,nyquist_hz:.5/dt};
    for(const [k,v] of Object.entries(fitted)) {
      requireValue(roundoffEqual(a[k],v),`analysis ${k} full-record fit`);
      delete exact[k];
    }
  }
  for(const [k,v] of Object.entries(exact)) requireValue(a[k]===v,`analysis ${k}`);
  requireValue((native?roundoffEqual:close)(a.window_rms_gain,Math.sqrt(.375)) && positiveInteger(a.chunk_count) && positiveInteger(a.series_terms),'analysis gains/order');
  requireValue(Array.isArray(a.algorithms) && a.algorithms.length>0 && a.algorithms.every(v=>v==='series'),'series algorithm');
  for(const k of ['actual_band_hz','requested_band_hz']) {
    vector(a[k],2,k);
    requireValue(native?(k==='actual_band_hz'?
      a[k][0]===c.frequency_hz[0] && a[k][1]===c.frequency_hz.at(-1):
      roundoffEqual(a[k][0],5/(s.n*endpointCadence(source.source_time_s))) && a[k][1]===25):
      close(a[k][0],s.low) && close(a[k][1],25),k);
  }
  vector(a.sample_window_s,2,'sample window');requireValue(a.sample_window_s[0]===c.time_s[0] && a.sample_window_s[1]===c.time_s.at(-1),'sample window endpoints');
  vector(a.bin_indices,s.f,'bin indices',Number.isSafeInteger);requireValue(a.bin_indices.every((v,i)=>v===5+i),'bin indices');
  matrix(a.series_truncation_bound_pa,s.f,12,'truncation bounds',nonnegative);
}

const mounted=new WeakMap();
export function mountComparison(root,options={}) {
  requireValue(root?.nodeType===1,'mount target');
  mounted.get(root)?.destroy();
  root.innerHTML=renderShell();
  const select=selector=>root.querySelector(selector),status=select('[data-loading-status]'),results=select('[data-loading-results]');
  const receiver=select('[data-loading-receiver]'),windowPicker=select('[data-loading-window]'),reloadButton=select('[data-loading-reload]');
  let data=null,view=null,component='x',disposed=false,generation=0,lastWidth=0;
  function draw() {
    if(disposed || !data)return;
    const opened=[...results.querySelectorAll('details[open]')].map(el=>el.dataset.loadingDetails);
    const componentFocused=root.ownerDocument.activeElement?.matches('[data-loading-component]');
    view=buildView(data,{window:windowPicker.value,receiver:Number(receiver.value),component},options);
    results.innerHTML=renderResults(data,view,Math.max(240,root.clientWidth-38));
    for(const el of results.querySelectorAll('details'))el.open=opened.includes(el.dataset.loadingDetails);
    // Size each plot to its own column, rather than scaling desktop labels down.
    for(const fig of results.querySelectorAll('[data-loading-chart]')) {
      const chart=view.charts[fig.dataset.loadingChart],style=getComputedStyle(fig);
      const width=Math.max(240,Math.round((fig.clientWidth || root.clientWidth)-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight)));
      const heading=fig.querySelector('h3').outerHTML,caption=fig.querySelector('figcaption').outerHTML;
      fig.innerHTML=heading+renderChart(chart,width)+caption;
    }
    if(componentFocused)select('[data-loading-component]')?.focus({preventScroll:true});
  }
  async function reload() {
    if(disposed || reloadButton.disabled && generation>0)return;
    const ticket=++generation;data=null;view=null;results.replaceChildren();
    receiver.disabled=windowPicker.disabled=reloadButton.disabled=true;
    status.dataset.state='pending';status.textContent='Checking the separate native loading bundle… No partial values are shown.';
    const outcome=await loadComparison(options);
    if(disposed || ticket!==generation)return;
    status.dataset.state=outcome.state;status.textContent=outcome.message;
    reloadButton.disabled=false;
    if(outcome.data) {data=outcome.data;receiver.disabled=windowPicker.disabled=false;draw();}
  }
  function change(event) {
    if(!data)return;
    if(event.target===receiver || event.target===windowPicker)draw();
    else if(event.target.matches('[data-loading-component]')) {component=event.target.value;draw();}
  }
  function toggle(event) {
    if(event.target.matches('[data-loading-details="source"]') && event.target.open) {
      // Hidden force charts had no layout width. Redraw once when opened.
      const fig=select('[data-loading-chart="force"]');if(!fig || !view)return;
      const style=getComputedStyle(fig),width=Math.max(240,Math.round(fig.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight)));
      fig.innerHTML=fig.querySelector('h3').outerHTML+renderChart(view.charts.force,width)+fig.querySelector('figcaption').outerHTML;
    }
  }
  root.addEventListener('change',change);root.addEventListener('toggle',toggle,true);reloadButton.addEventListener('click',reload);
  const observer=typeof ResizeObserver==='function'?new ResizeObserver(entries=>{const width=Math.round(entries[0].contentRect.width);if(width!==lastWidth){lastWidth=width;draw();}}):null;
  observer?.observe(root);
  const controller={reload,destroy(){disposed=true;generation++;observer?.disconnect();root.removeEventListener('change',change);root.removeEventListener('toggle',toggle,true);reloadButton.removeEventListener('click',reload);mounted.delete(root);}};
  mounted.set(root,controller);void reload();return controller;
}
if(typeof document!=='undefined') {
  // No query-string, storage or page-attribute switch can enable fixtures.
  for(const root of document.querySelectorAll('[data-loading-app]'))mountComparison(root);
}

