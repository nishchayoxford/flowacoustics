/* Read-only, DOM-free contract for the four-case teaching page. */
export const CASES = Object.freeze([
  {id:'mrl2d_urans_body', label:'URANS · on-body', flow_model:'SST URANS', acoustic_surface:'on-body'},
  {id:'mrl2d_urans_permeable', label:'URANS · permeable', flow_model:'SST URANS', acoustic_surface:'permeable'},
  {id:'mrl2d_les_body', label:'WALE · on-body', flow_model:'WALE', acoustic_surface:'on-body'},
  {id:'mrl2d_les_permeable', label:'WALE · permeable', flow_model:'WALE', acoustic_surface:'permeable'},
].map(Object.freeze));
export function adaptResults(raw) {
  if (raw == null) return {state:'pending', message:'Results not yet available. No computed data are displayed.',
    cases:CASES.map(c=>({...c,status:'not_run',summary:'Awaiting native results.',metadata:{},charts:[],flow:null,mesh:null,checks:[],downloads:[]})),
    limitations:[],provenance:{},downloads:[]};
  need(raw && raw.schema_version === 2 && raw.study === 'mrl2d', 'Expected the MRL2D version 2 bundle.');
  need(Array.isArray(raw.cases) && raw.cases.length === 4, 'Exactly four cases are required.');
  const cases = CASES.map(expected => {
    const found = raw.cases.filter(c=>c.id === expected.id);
    need(found.length === 1, 'Case IDs are missing, duplicated or unrecognized.');
    const c = found[0];
    need(c.flow_model === expected.flow_model && c.acoustic_surface === expected.acoustic_surface, 'Case identity does not match its model or surface.');
    need(['not_run','running','completed','failed'].includes(c.status), 'Unknown case status.');
    const metadata = {};
    need(object(c.metadata), 'Missing case metadata.');
    for (const key of META_KEYS) {
      const v = c.metadata[key];
      if (v == null) { metadata[key] = null; continue; }
      if (key === 'spanwise_model') metadata[key] = string(v);
      else if (key === 'native_End') { need(typeof v === 'boolean', 'Invalid native End flag.'); metadata[key] = v; }
      else if (key === 'analysis_window_s' || key === 'acoustic_band_hz') {
        need(Array.isArray(v) && v.length === 2 && v.every(finite) && v[0] >= 0 && v[1] > v[0], 'Invalid analysis support.'); metadata[key] = [...v];
      } else { need(finite(v) && (key === 'native_returncode' || v >= 0), 'Invalid numeric metadata.'); metadata[key] = v; }
    }
    if (metadata.cells != null) need(Number.isInteger(metadata.cells) && metadata.cells > 0 && metadata.cells <= 250000, 'Cell count exceeds the case contract.');
    if (metadata.ranks != null) need(Number.isInteger(metadata.ranks) && metadata.ranks > 0 && metadata.ranks <= 8, 'Invalid MPI rank count.');
    const mesh = c.mesh == null ? null : {image:safeAsset(c.mesh.image),caption:string(c.mesh.caption)};
    let flow = null;
    if (c.flow != null) {
      flow = {caption:string(c.flow.caption),image:c.flow.image ? safeAsset(c.flow.image) : null,video:c.flow.video ? safeAsset(c.flow.video) : null};
      need(flow.image || flow.video, 'Flow has no asset.');
      for (const k of ['start_s','end_s']) { need(c.flow[k] == null || finite(c.flow[k]), 'Invalid flow time.'); flow[k] = c.flow[k] ?? null; }
      need(flow.start_s == null || flow.end_s == null || flow.end_s >= flow.start_s, 'Invalid flow interval.');
    }
    need(Array.isArray(c.charts) && c.charts.length <= 5, 'At most five chart groups are supported.');
    const charts = c.charts.map(chart=>validateChart(chart, c.id));
    need(new Set(charts.map(x=>x.id)).size === charts.length, 'Duplicate chart IDs.');
    need(Array.isArray(c.checks), 'Missing checks.');
    const checks = c.checks.map(v=>({name:string(v.name),status:string(v.status),detail:string(v.detail)}));
    return {...expected,label:string(c.label),summary:string(c.summary),status:c.status,metadata,mesh,flow,charts,checks,downloads:downloads(c.downloads)};
  });
  need(new Set(cases.flatMap(c=>c.charts.map(x=>x.id))).size <= 5, 'More than five semantic figure groups.');
  need(Array.isArray(raw.limitations), 'Missing limitations.');
  need(object(raw.provenance), 'Missing provenance.');
  inspectPublic(raw.provenance);
  return {state:'ready',message:'Results bundle loaded.',title:string(raw.title),summary:string(raw.summary),cases,
    limitations:raw.limitations.map(string),provenance:structuredClone(raw.provenance),downloads:downloads(raw.downloads)};
}

export const META_KEYS = ['cells','ranks','requested_end_s','achieved_end_s','dt_s','source_cadence_s','density_kg_m3','sound_speed_m_s','spanwise_model','analysis_window_s','acoustic_band_hz','native_returncode','native_End'];
const finite = v=>typeof v === 'number' && Number.isFinite(v);
const object = v=>v !== null && typeof v === 'object' && !Array.isArray(v);
function need(ok, message) { if (!ok) throw new Error(message); }
function string(v) { need(typeof v === 'string' && v.length <= 20000, 'Invalid text field.'); return v; }
export function safeAsset(value) {
  need(typeof value === 'string' && /^(?:\.\/)?[A-Za-z0-9_-]+(?:\/[A-Za-z0-9_.-]+)*\.[A-Za-z0-9]+$/.test(value), 'Unsafe relative asset path.');
  need(!value.split('/').some(p=>p === '..' || p === ''), 'Unsafe relative asset path.');
  return value;
}
function downloads(value) {
  need(Array.isArray(value), 'Missing download list.');
  return value.map(d=>({label:string(d.label),href:safeAsset(d.href)}));
}
function inspectPublic(value, depth=0) {
  need(depth < 16, 'Provenance is too deeply nested.');
  if (typeof value === 'number') need(finite(value), 'Nonfinite provenance.');
  else if (typeof value === 'string') {
    need(!/(?:^|\s)(?:\/(?:home|Users|etc|tmp|root|mnt|var)\/|[A-Za-z]:[\\/])/.test(value), 'Private filesystem path in provenance.');
  } else if (value && typeof value === 'object') for (const [key,v] of Object.entries(value)) {
    need(!/(?:password|secret|token|api.?key|username|credential)/i.test(key), 'Private field in provenance.'); inspectPublic(v,depth+1);
  }
}
function validateChart(c, caseId) {
  need(object(c) && /^[A-Za-z][A-Za-z0-9_-]*$/.test(c.id), 'Invalid chart identifier.');
  need(c.case_id == null || c.case_id === caseId, 'Chart case ID mismatch.');
  need(c.origin === 'computed' && ['line','polar'].includes(c.kind), 'Chart must contain computed line or polar data.');
  need(c.window != null, 'Chart support window is required.'); inspectPublic(c.window);
  const chart = {id:c.id,title:string(c.title),description:string(c.description),kind:c.kind,x_label:string(c.x_label),y_label:string(c.y_label),origin:'computed',window:structuredClone(c.window)};
  for (const key of ['x_scale','y_scale']) { need(c[key] == null || ['linear','log'].includes(c[key]), 'Invalid plot scale.'); chart[key] = c[key] ?? 'linear'; }
  need(c.kind !== 'polar' || (chart.x_scale === 'linear' && chart.y_scale === 'linear'), 'Polar charts require linear magnitude and angle.');
  need(Array.isArray(c.traces) && c.traces.length > 0 && c.traces.length <= 32, 'Invalid trace collection.');
  chart.traces = c.traces.map(t=>{
    need(t.case_id == null || t.case_id === caseId, 'Trace case ID mismatch.');
    need(Array.isArray(t.x) && Array.isArray(t.y) && t.x.length === t.y.length && t.x.length > 0 && t.x.length <= 200000, 'Mismatched or empty trace arrays.');
    need([...t.x,...t.y].every(v=>v === null || finite(v)), 'Nonfinite or nonnumeric trace value.');
    need(c.kind !== 'polar' || t.y.every(v=>v === null || v >= 0), 'Polar radius must be nonnegative.');
    need(t.color == null || /^#[\da-fA-F]{6}$/.test(t.color), 'Invalid trace color.');
    need(t.dash == null || typeof t.dash === 'boolean' || (typeof t.dash === 'string' && /^(?:solid|dashed|[\d. ,]+)$/.test(t.dash)), 'Invalid trace dash.');
    return {name:string(t.name),x:[...t.x],y:[...t.y],color:t.color ?? null,dash:t.dash ?? null};
  });
  return chart;
}

/** Geometry only: clockwise orbit and half-speed absolute pitch. */
export function geometryAt(phaseDegrees) {
  need(finite(phaseDegrees), 'Invalid geometry phase.');
  const a = -phaseDegrees * Math.PI / 180, co = Math.cos(a), si = Math.sin(a);
  return [[0,.055,90],[Math.sqrt(3)*.055/2,-.055/2,30],[-Math.sqrt(3)*.055/2,-.055/2,-30]]
    .map(([x,y,angle])=>({x:co*x-si*y,y:si*x+co*y,angle:angle-phaseDegrees/2}));
}
const canonical = v=>JSON.stringify(v && typeof v === 'object' && !Array.isArray(v)
  ? Object.fromEntries(Object.keys(v).sort().map(k=>[k,JSON.parse(canonical(v[k]))])) : v);
export function comparisonFor(a,b,chartId) {
  const no = reason=>({allowed:false,reason});
  if (!a || !b || a.status !== 'completed' || b.status !== 'completed') return no('Both cases must be completed.');
  if ((a.flow_model === b.flow_model) === (a.acoustic_surface === b.acoustic_surface)) return no('Change one factor: model or surface, not both.');
  for (const key of ['analysis_window_s','acoustic_band_hz','source_cadence_s','density_kg_m3','sound_speed_m_s','spanwise_model'])
    if (a.metadata[key] == null || b.metadata[key] == null || canonical(a.metadata[key]) !== canonical(b.metadata[key])) return no(`Unmatched or missing ${key.replaceAll('_',' ')}.`);
  const ca = a.charts.find(c=>c.id === chartId), cb = b.charts.find(c=>c.id === chartId);
  if (!ca || !cb) return no('A matching figure is not available.');
  for (const key of ['kind','window','x_label','y_label','x_scale','y_scale'])
    if (canonical(ca[key]) !== canonical(cb[key])) return no('Figure units, estimator window or scales differ.');
  if (ca.traces.length !== cb.traces.length || ca.traces.some((t,i)=>t.name !== cb.traces[i].name || canonical(t.x) !== canonical(cb.traces[i].x))) return no('Receiver identities or sample support differ.');
  return {allowed:true,reason:'Matched window, band, cadence, units and sample support.',chart:{...ca,traces:[
    ...ca.traces.map(t=>({...t,name:`${a.flow_model} · ${a.acoustic_surface} / ${t.name}`})),
    ...cb.traces.map(t=>({...t,name:`${b.flow_model} · ${b.acoustic_surface} / ${t.name}`,dash:'7 4'})),
  ]}};
}
