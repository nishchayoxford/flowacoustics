// Separate flow-only presentation. No imports from the four-case acoustic adapter.
const requireValue = (ok, message) => { if (!ok) throw new Error(message); };
export function validateResults(raw) {
  if (raw === null) return null;
  requireValue(raw?.schema_version === 1 && raw.status === 'completed', 'A completed schema-1 result is required.');
  const c = raw.case, v = raw.verification;
  requireValue(c?.id === 'mrl2d_les_cavitation' && c.end_time_s === 10 && Number.isInteger(c.cells) && c.cells > 0 && c.cells <= 250000, 'Case identity, 10 s endpoint or cell budget is invalid.');
  requireValue(v && ['native_completed','fields_verified','baseline_preserved','vapour_detected'].every(k => v[k] === true) && v.physical_validation === false, 'Native verification is incomplete or physical-validation disclosure is invalid.');
  const text = x => typeof x === 'string' && x.trim().length > 0;
  const positive = x => Number.isFinite(x) && x > 0;
  const range = x => Array.isArray(x) && x.length === 2 && x.every(Number.isFinite) && x[0] >= 0 && x[0] < x[1] && x[1] <= c.end_time_s + 1e-5;
  for (const k of ['solver','flow_model','phase_change_model']) requireValue(text(c[k]), `Missing case ${k}.`);
  for (const k of ['farfield_absolute_pressure_pa','saturation_pressure_pa','liquid_density_kg_m3','vapour_density_kg_m3','output_cadence_s']) requireValue(positive(c[k]), `Invalid case ${k}.`);
  requireValue(Number.isFinite(c.temperature_c) && Number.isFinite(c.cavitation_number) && c.cavitation_number >= 0 && Number.isInteger(c.frame_count) && c.frame_count > 1 && range(c.source_time_range_s), 'Invalid case support.');
  requireValue(raw.reference?.id === 'mrl2d_les_body' && ['label','solver','pressure_conversion','comparison_note'].every(k=>text(raw.reference[k])), 'Reference metadata are incomplete.');
  const h = raw.history, t = h?.time_s;
  requireValue(Array.isArray(t) && t.length >= 2 && t.every((x,i)=>Number.isFinite(x) && x >= 0 && x <= c.end_time_s + 1e-5 && (!i || x > t[i-1])), 'History times must be finite and strictly increasing.');
  for (const k of ['vapour_area_m2','max_vapour_fraction','minimum_absolute_pressure_pa','maximum_speed_m_s']) {
    requireValue(Array.isArray(h[k]) && h[k].length === t.length && h[k].every(Number.isFinite), `Invalid ${k} history.`);
  }
  requireValue(h.vapour_area_m2.every(x=>x>=0) && h.vapour_area_m2.some(x=>x>0) && h.max_vapour_fraction.every(x=>x>=0 && x<=1) && h.max_vapour_fraction.some(x=>x>0) && h.maximum_speed_m_s.every(x=>x>=0), 'Invalid phase bounds, area or speed.');
  requireValue(text(raw.history_support), 'Missing native history-support definition.');
  for (const key of ['pressure','velocity','vapour']) {
    const m = raw.media?.[key];
    requireValue(m && safeRelative(m.src) && safeRelative(m.poster) && text(m.title) && text(m.description) && positive(m.duration_s) && positive(m.fps) && m.frame_count === c.frame_count && range(m.source_time_range_s) && m.source_time_range_s.every((x,i)=>x===c.source_time_range_s[i]), 'Invalid movie metadata or unsynchronized support.');
    if (key !== 'pressure') requireValue(m.duration_s === raw.media.pressure.duration_s && m.fps === raw.media.pressure.fps, 'Movie clocks must be synchronized.');
  }
  requireValue(Array.isArray(raw.downloads) && raw.downloads.every(d=>d && text(d.label) && safeRelative(d.href) && d.href.replace(/^\.\//,'') !== 'data/cavitation/results.json' && Number.isSafeInteger(d.bytes) && d.bytes > 0 && typeof d.sha256 === 'string' && /^[a-f0-9]{64}$/i.test(d.sha256)), 'Invalid download manifest.');
  requireValue(Array.isArray(raw.limitations) && raw.limitations.length > 0 && raw.limitations.every(text), 'Missing limitations.');
  requireValue(Array.isArray(raw.sources) && raw.sources.length > 0 && raw.sources.every(s=>s && text(s.label) && safeSource(s.href)), 'Invalid public sources.');
  return raw;
}
function safeRelative(value) {
  return typeof value === 'string' && /^(?:\.\/)?[a-zA-Z0-9_-][a-zA-Z0-9_./-]*$/.test(value) && !value.replace(/^\.\//,'').split('/').some(p=>!p || p==='.' || p==='..');
}
function safeSource(value) {
  if (typeof value !== 'string' || /[\s\\]/.test(value)) return false;
  try {
    const u = new URL(value);
    return u.protocol === 'https:' && !u.username && !u.password && !u.port && /^[a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,}$/i.test(u.hostname) && !/\.(local|localhost|internal|test|invalid|onion)$/i.test(u.hostname);
  } catch { return false; }
}

const escapeHTML = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const display = value => Array.isArray(value) ? value.join(' – ') : String(value);
const signed = value => `${value < 0 ? '−' : ''}${Number(Math.abs(value).toPrecision(10))}`;

export function renderHistory(bundle, kind, width) {
  requireValue(['area','pressure'].includes(kind), 'Unknown history quantity.');
  const h=bundle.history, t=h.time_s, y=kind==='area'?h.vapour_area_m2:h.minimum_absolute_pressure_pa;
  const reference=kind==='pressure'?bundle.case.saturation_pressure_pa:null;
  const w=Math.max(300,Math.round(width)), left=80, right=w-16, top=42, bottom=240;
  let low=kind==='area'?0:y.reduce((a,b)=>Math.min(a,b),reference), high=y.reduce((a,b)=>Math.max(a,b),reference??0);
  if(low===high){low-=Math.max(1,Math.abs(low)*0.05);high+=Math.max(1,Math.abs(high)*0.05);}
  const X=x=>left+(x-t[0])/(t.at(-1)-t[0])*(right-left), Y=x=>bottom-(x-low)/(high-low)*(bottom-top);
  const n=x=>Number(x.toFixed(3)), tick=x=>x!==0 && (Math.abs(x)<0.001 || Math.abs(x)>=100000)?x.toExponential(1).replace('-','−'):signed(Number(x.toPrecision(4)));
  const title=kind==='area'?'Vapour area':'Minimum absolute pressure', unit=kind==='area'?'m² per unit span':'Absolute pressure (Pa)', e=escapeHTML;
  const path=t.map((x,i)=>`${i?'L':'M'}${n(X(x))},${n(Y(y[i]))}`).join(' ');
  return `<svg data-history="${kind}" viewBox="0 0 ${w} 292" role="img" aria-labelledby="${kind}-title ${kind}-desc"><title id="${kind}-title">${title}</title><desc id="${kind}-desc">${t.length} supplied samples over ${e(display([t[0],t.at(-1)]))} s. ${unit}.${reference===null?'':' Dashed rust line: constant saturation pressure '+reference+' Pa.'} Connecting segments are a visual guide, not additional samples. ${e(bundle.history_support)}</desc><metadata>${e(JSON.stringify({time_s:t,values:y,reference_pa:reference}))}</metadata><text class="unit" x="16" y="19">${unit}</text>${[low,(low+high)/2,high].map(v=>`<path class="grid" d="M${left},${n(Y(v))} H${right}"/><text x="${left-8}" y="${n(Y(v))+5}" text-anchor="end">${e(tick(v))}</text>`).join('')}<path class="axis" d="M${left},${top} V${bottom} H${right}"/><path data-trace="history" d="${path}"/>${reference===null?'':`<path data-reference="pSat" d="M${left},${n(Y(reference))} L${right},${n(Y(reference))}"/>`}${[t[0],(t[0]+t.at(-1))/2,t.at(-1)].map((v,i)=>`<text x="${n(X(v))}" y="262" text-anchor="${i===0?'start':i===2?'end':'middle'}">${e(tick(v))}</text>`).join('')}<text x="${(left+right)/2}" y="286" text-anchor="middle">Source time (s)</text></svg>`;
}

export async function verifyDownload(bytes, manifest) {
  requireValue(bytes.byteLength === manifest.bytes, 'Download byte count differs.');
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
  requireValue(hash === manifest.sha256.toLowerCase(), 'Download checksum differs.');
  return true;
}

export function mount(doc = document) {
  const $ = id => doc.getElementById(id);
  let bundle = null, selected = 'pressure', mediaAttempt = 0, generation = 0, controller;
  const pendingMessage = 'No completed native cavitation results are supplied. The 10 s target is not a completion claim. No substitute animation or values are shown.';
  function clearResults() {
    $('recorded-flow')?.pause();
    bundle = null;
    $('view-picker').querySelectorAll('button').forEach(b=>b.disabled=true);
    $('conditions').replaceChildren();
    $('player').innerHTML = '<p class="empty-state">Native movies are pending. No replacement animation is shown.</p>';
    $('history-plots').innerHTML = '<p class="empty-state">No verified histories are available.</p>';
    $('history-support').textContent = '';
    $('source-metadata').textContent = 'Source metadata await a verified completed bundle.';
    $('downloads').textContent = 'No verified downloads are supplied.';
    $('download-status').textContent = '';
    $('sources').replaceChildren();
    $('interpretation').textContent = 'Select a quantity after a verified completed bundle is supplied. No values or movies are substituted.';
  }
  function renderReady() {
    const c = bundle.case, e = escapeHTML;
    $('result-status').dataset.state = 'ready';
    $('result-status').textContent = `Recorded native completion: ${c.end_time_s} s. All four supplied native verification flags are true. Physical validation: false. Completion is not physical accuracy.`;
    $('conditions').innerHTML = `<dl class="condition-grid"><div><dt>Farfield absolute pressure</dt><dd>${e(c.farfield_absolute_pressure_pa)} Pa</dd></div><div><dt>Saturation pressure · pSat</dt><dd>${e(c.saturation_pressure_pa)} Pa</dd></div><div><dt>Isothermal temperature</dt><dd>${e(c.temperature_c)} °C</dd></div></dl>`;
    $('view-picker').querySelectorAll('button').forEach(b=>b.disabled=false);
    const labels={id:'Case identity',solver:'Solver',flow_model:'Flow model',cells:'Fluid cells',end_time_s:'Recorded endpoint (s)',temperature_c:'Temperature (°C)',farfield_absolute_pressure_pa:'Farfield absolute pressure (Pa)',saturation_pressure_pa:'Saturation pressure (Pa)',liquid_density_kg_m3:'Liquid density (kg/m³)',vapour_density_kg_m3:'Vapour density (kg/m³)',cavitation_number:'Cavitation number (dimensionless)',phase_change_model:'Mass-transfer model',frame_count:'Recorded movie frames',source_time_range_s:'Source-time interval (s)',output_cadence_s:'Declared field output cadence (s)'};
    $('source-metadata').innerHTML = `<p><strong>Physical validation: false.</strong> The bundle reports native completion, verified fields, baseline preservation and detected vapour. These flags are producer declarations, not independent experimental validation.</p><dl class="cav-metadata">${Object.entries(labels).map(([k,label])=>`<div><dt>${e(label)}</dt><dd>${e(display(c[k]))}</dd></div>`).join('')}</dl><h3>Single-phase reference</h3><p>${e(bundle.reference.label)} · ${e(bundle.reference.id)} · ${e(bundle.reference.solver)}</p><p>${e(bundle.reference.pressure_conversion)}</p><p>${e(bundle.reference.comparison_note)}</p><h3>Recorded limitations</h3><ul>${bundle.limitations.map(x=>`<li>${e(x)}</li>`).join('')}</ul>`;
    $('history-support').textContent = bundle.history_support;
    $('downloads').innerHTML = bundle.downloads.length ? '<h3>Exact files</h3><p>Each click checks the supplied byte count and SHA256 before saving. These checks establish file identity, not physical accuracy.</p><ul class="download-list">'+bundle.downloads.map((d,i)=>`<li><a href="${e(d.href)}" data-download="${i}" download>${e(d.label)} · verify &amp; download</a><div>${e(d.bytes)} bytes · SHA256 <code>${e(d.sha256)}</code></div></li>`).join('')+'</ul>' : '<p>No downloads supplied in this bundle.</p>';
    $('sources').innerHTML = '<h3>Model &amp; thermodynamic references</h3><ul>'+bundle.sources.map(s=>`<li><a href="${e(s.href)}" target="_blank" rel="noopener">${e(s.label)} ↗</a></li>`).join('')+'</ul>';
    renderPlots();
    showMedia();
  }
  function renderPlots() {
    if(!bundle)return;
    $('history-plots').innerHTML = '<figure class="history-figure"><h3>Vapour area</h3><div id="area-plot"></div><figcaption>Solid teal: supplied vapour-volume integral / numerical depth (m² per unit span).</figcaption></figure><figure class="history-figure"><h3>Minimum absolute pressure</h3><div id="pressure-plot"></div><figcaption>Solid teal: supplied minimum pressure (Pa). <span class="reference-key">Dashed rust: constant pSat = '+escapeHTML(bundle.case.saturation_pressure_pa)+' Pa.</span></figcaption></figure>';
    for(const kind of ['area','pressure'])$(kind+'-plot').innerHTML=renderHistory(bundle,kind,$(kind+'-plot').clientWidth);
  }
  let resizeTimer;
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(renderPlots,100);});
  function showMedia(preserveTime = 0, retry = false) {
    if (!bundle) return;
    const old = $('recorded-flow');old?.pause();
    const m = bundle.media[selected], c = bundle.case, e = escapeHTML;
    $('view-picker').querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed', String(b.dataset.view===selected)));
    const explanations = {
      pressure:`Pressure departure p − p∞ (Pa), on shared baseline/cavitating colour scales. Farfield is ${c.farfield_absolute_pressure_pa} Pa absolute. Saturation is ${c.saturation_pressure_pa} Pa absolute, or ${signed(c.saturation_pressure_pa-c.farfield_absolute_pressure_pa)} Pa relative to that same farfield datum. This is flow pressure, not acoustic pressure.`,
      velocity:'Speed magnitude |U| (m/s), on shared baseline/cavitating colour scales. Inspect how the flow changes around the blades. Different solvers and numerics also affect the comparison.',
      vapour:'Vapour volume fraction αv (dimensionless, 0–1): 0 is liquid, 1 is vapour. This is the cavitating field alone. A phase mixture is not a resolved picture of individual bubbles.'
    };
    $('interpretation').textContent = explanations[selected];
    $('player').innerHTML = `<figure><span class="tag">FLOW, NOT ACOUSTIC WAVES</span><h3>${e(m.title)}</h3><video id="recorded-flow" controls playsinline preload="metadata" poster="${e(m.poster)}" aria-label="${e(m.title)}"></video><div class="player-actions"><button id="play" type="button">Play movie</button><button id="fullscreen" type="button">Fullscreen</button><button id="retry-media" type="button" hidden>Retry movie</button><a href="${e(m.src)}" target="_blank" rel="noopener">Open full-size movie ↗</a></div><p id="media-status" role="status">Loading recorded movie. Playback starts only when requested.</p><figcaption>${e(m.description)}<br>Source-time interval: ${e(display(m.source_time_range_s))} s · ${e(m.frame_count)} frames · ${e(m.fps)} fps · movie duration ${e(m.duration_s)} s. Playback time is not simulation time.</figcaption></figure>`;
    const video = $('recorded-flow'), isCurrent = ()=>$('recorded-flow')===video;
    const update = ()=>{if(isCurrent())$('play').textContent=video.paused?'Play movie':'Pause movie';};
    const failed = ()=>{if(!isCurrent())return;video.pause();$('media-status').textContent='Movie unavailable or unsupported: download or decoding failed. No substitute field is shown. Retry movie to recover.';$('retry-media').hidden=false;$('play').disabled=true;};
    video.addEventListener('error',failed);
    video.addEventListener('loadedmetadata',()=>{if(isCurrent() && preserveTime>0)video.currentTime=Math.min(preserveTime,Math.max(0,video.duration-0.001));});
    video.addEventListener('loadeddata',()=>{if(isCurrent())$('media-status').textContent='Movie ready. Play, pause or seek with the native timeline. No autoplay.';});
    for(const name of ['play','pause','ended'])video.addEventListener(name,update);
    // Chromium may cache failed decoded bytes even with no-store. Retry only
    // adds an application-owned cache key to the already-validated local URL.
    video.src=retry ? `${m.src}?retry=${++mediaAttempt}` : m.src;
    $('play').addEventListener('click',async()=>{
      if(!video.paused){video.pause();return;}
      try { await video.play(); }
      catch(error){if(!isCurrent())return;if(error.name==='AbortError')update();else failed();}
    });
    $('retry-media').addEventListener('click',()=>showMedia(0,true));
    $('fullscreen').addEventListener('click',async()=>{
      try { await video.requestFullscreen(); }
      catch { if(isCurrent())$('media-status').textContent='Fullscreen is unavailable here. Use Open full-size movie instead.'; }
    });
  }
  async function reload() {
    const current=++generation;
    controller?.abort();controller=new AbortController();
    $('reload').disabled = true;
    clearResults();
    $('result-status').dataset.state = 'loading';
    $('result-status').textContent = 'Checking the result bundle; no results are displayed yet.';
    try {
      const response = await fetch('./data/cavitation/results.json', {cache:'no-store',redirect:'error',signal:AbortSignal.any([controller.signal,AbortSignal.timeout(12000)])});
      if(current!==generation){await response.body?.cancel();return;}
      if (response.status === 404) {
        await response.arrayBuffer();
        if(current!==generation)return;
        $('result-status').dataset.state = 'pending';
        $('result-status').textContent = pendingMessage;
      } else {
        requireValue(response.ok, 'Request failed.');
        const raw=await response.json();
        if(current!==generation)return;
        bundle = validateResults(raw);
        requireValue(bundle, 'Empty result bundle.');
        renderReady();
      }
    } catch {
      if(current!==generation)return;
      clearResults();
      $('result-status').dataset.state = 'unavailable';
      $('result-status').textContent = 'Results unavailable: request failed or bundle is invalid. No computed values are displayed. Use Reload results to retry.';
    } finally { if(current===generation)$('reload').disabled = false; }
  }
  $('downloads').addEventListener('click',async event=>{
    const link=event.target.closest('a[data-download]');if(!link)return;
    event.preventDefault();if(!bundle || link.getAttribute('aria-disabled')==='true')return;
    const original=bundle, d=bundle.downloads[Number(link.dataset.download)];
    link.setAttribute('aria-disabled','true');$('download-status').textContent='Reading exact file and checking SHA256…';
    try {
      const response=await fetch(d.href,{redirect:'error',cache:'no-store',signal:AbortSignal.timeout(60000)});
      requireValue(response.ok,'Download unavailable.');
      const bytes=await response.arrayBuffer();await verifyDownload(bytes,d);
      if(bundle!==original)return;
      const url=URL.createObjectURL(new Blob([bytes])), save=doc.createElement('a');
      save.href=url;save.download=d.href.split('/').at(-1);save.click();setTimeout(()=>URL.revokeObjectURL(url),30000);
      $('download-status').textContent='Byte count and SHA256 verified. Exact supplied file prepared for download.';
    } catch {
      if(bundle===original)$('download-status').textContent='Download withheld: unavailable file, byte count or SHA256 mismatch. No substitute file was created. Retry the download link.';
    } finally { if(bundle===original)link.removeAttribute('aria-disabled'); }
  });
  $('view-picker').addEventListener('click',event=>{
    const button=event.target.closest('button[data-view]');if(!button || button.disabled || !bundle)return;
    const time=$('recorded-flow')?.currentTime || 0;selected=button.dataset.view;showMedia(time);
  });
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  motion.addEventListener('change',()=>{if(motion.matches)$('recorded-flow')?.pause();});
  doc.addEventListener('visibilitychange',()=>{if(doc.hidden)$('recorded-flow')?.pause();});
  $('reload').addEventListener('click', reload);
  reload();
}
if (typeof document !== 'undefined') mount();
