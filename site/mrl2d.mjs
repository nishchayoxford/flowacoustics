import {CASES, geometryAt, comparisonFor, META_KEYS, safeAsset} from './mrl2d-model.mjs';
import {adaptPresentation as adaptResults, receiverChart} from './mrl2d-presentation.mjs';
import {escapeHTML as e, formatNumber, renderChart} from './mrl2d-plots.mjs';
const $=id=>document.getElementById(id);
let bundle=adaptResults(null), selected=CASES[0].id;
const statuses={not_run:'Pending',running:'Computing',completed:'Completed',failed:'Failed'};
const current=()=>bundle.cases.find(c=>c.id===selected);
function geometry() {
  const phase=Number($('phase').value),permeable=current().acoustic_surface==='permeable';
  $('phase-value').textContent=`${phase}°`;
  const blades=geometryAt(phase);
  const X=x=>180+x*820,Y=y=>150-y*820;
  $('geometry-view').innerHTML=`<svg viewBox="0 0 460 300" role="img" aria-label="Prescribed three-blade dual motion at ${phase} degrees; geometry only"><title>Orbit plus half-speed absolute pitch</title><desc>Equal physical x and y scales. A dashed enclosing contour is illustrative, not an executed sampling contour. The farfield is outside this local view.</desc><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#52636a"/></marker></defs><path d="M15,150 H65" stroke="#52636a" marker-end="url(#arrow)"/><text x="15" y="132" font-size="18" fill="#52636a">U∞</text><circle cx="180" cy="150" r="${.055*820}" fill="none" stroke="#c8d0cb" stroke-dasharray="3 4"/><path d="M340,210 h40 M340,210 v-40" stroke="#52636a"/><text x="390" y="215" font-size="18" fill="#52636a">x</text><text x="335" y="160" font-size="18" fill="#52636a">y</text><rect x="${X(-.14)}" y="${Y(.14)}" width="${.44*820}" height="${.28*820}" fill="none" stroke="${permeable?'#ad4c2e':'#c8d0cb'}" stroke-width="${permeable?2.5:1}" stroke-dasharray="7 5"/>${blades.map((b,i)=>`<g transform="translate(${X(b.x)} ${Y(b.y)}) rotate(${-b.angle})"><ellipse rx="${.035*820}" ry="${.0035*820}" fill="#e6eeea" stroke="${permeable?'#52636a':'#006d70'}" stroke-width="2.5"/><circle r="2" fill="#20353d"/></g><text x="${X(b.x)+13}" y="${Y(b.y)+21}" font-size="18" fill="#52636a">${i+1}</text>`).join('')}<text x="${X(-.14)}" y="286" fill="#52636a" font-size="18">Example stationary contour</text></svg>`;
}
function render() {
  const c=current();
  $('case-picker').querySelectorAll('[data-case]').forEach(b=>{
    b.setAttribute('aria-pressed',String(b.dataset.case===selected));
    b.querySelector('small').textContent=statuses[bundle.cases.find(c=>c.id===b.dataset.case).status];
  });
  $('bundle-status').dataset.state=bundle.state;
  $('bundle-status').textContent=bundle.message;
  $('case-summary').textContent=`${c.flow_model==='WALE'?'2D WALE / LES-model':c.flow_model} · ${c.acoustic_surface} · ${statuses[c.status]}. ${c.summary}`;
  $('body-explanation').classList.toggle('selected',c.acoustic_surface==='on-body');
  $('permeable-explanation').classList.toggle('selected',c.acoustic_surface==='permeable');
  $('surface-selection').textContent=c.acoustic_surface==='on-body'?'Selected: moving on-body contours (solid teal in the geometry).':'Selected: stationary permeable contour (dashed rust in the geometry). Its illustrated position is not native sampling evidence.';
  geometry();
  media();
  figures();
  evidence();
}

function figures() {
  const c=current(),mode=$('compare-mode').value;
  const peer=bundle.cases.find(p=>p.id!==c.id && (mode==='surface' ? p.flow_model===c.flow_model : p.acoustic_surface===c.acoustic_surface));
  const groups={source:[],receiver:[]},notes=[];
  for(const supplied of c.charts) {
    let chart=supplied,comparison='';
    if(mode!=='none') {
      const result=comparisonFor(c,peer,chart.id);
      if(result.allowed){chart=result.chart;comparison=`Overlay: ${c.flow_model} / ${c.acoustic_surface} and ${peer.flow_model} / ${peer.acoustic_surface}.`;}
      else comparison=`Overlay withheld: ${result.reason} Selected case only.`;
      notes.push(`${chart.title}: ${comparison}`);
    }
    const target=/^(source|loading|force)(?:-|_|$)/.test(chart.id)?'source':'receiver';
    groups[target].push(`<figure class="computed-chart" data-case="${e(c.id)}" data-chart="${e(chart.id)}"><figcaption><span class="tag">COMPUTED · ${e(statuses[c.status])}</span><h3>${e(chart.title)}</h3><p class="chart-description">${e(chart.description)}</p>${comparison?`<p class="figure-note">${e(comparison)}</p>`:''}</figcaption><div class="plot" data-plot="${e(chart.id)}"></div><details><summary>Figure support &amp; units</summary><p class="chart-support">Window / estimator: ${e(JSON.stringify(chart.window))}</p><p class="chart-support">${e(chart.x_label)} (${e(chart.x_scale)}); ${e(chart.y_label)} (${e(chart.y_scale)}). Values are supplied by the results producer; this page does not recompute a spectrum.</p></details></figure>`);
  }
  $('source-charts').innerHTML=groups.source.join('') || '<p class="figure-note">No computed source-history figure is available for this case.</p>';
  $('receiver-charts').innerHTML=groups.receiver.join('') || '<p class="empty-state">Receiver histories, spectra and directivity are pending native source histories and verified acoustic reduction.</p>';
  for(const target of document.querySelectorAll('[data-plot]')) {
    const supplied=c.charts.find(chart=>chart.id===target.dataset.plot);
    const result=mode==='none'?null:comparisonFor(c,peer,supplied.id);
    target.innerHTML=renderChart(receiverChart(result?.allowed?result.chart:supplied,Number($('receiver-direction').value)),target.clientWidth);
  }
  $('comparison-status').textContent=notes.length?notes.join(' '):c.charts.length?'Selected case only. Choose a matched overlay to compare one factor.':'Comparison is pending completed, matched records.';
  $('comparison-table').innerHTML=`<table class="comparison-table"><caption>Four cases · recorded coverage, not requested targets</caption><thead><tr><th scope="col">Model / surface</th><th scope="col">State</th><th scope="col">End (s)</th><th scope="col">Cells</th></tr></thead><tbody>${bundle.cases.map(row=>`<tr><th scope="row">${e(row.flow_model)}<br>${e(row.acoustic_surface)}</th><td>${e(statuses[row.status])}</td><td>${e(formatNumber(row.metadata.achieved_end_s))}</td><td>${e(row.metadata.cells??'Not recorded')}</td></tr>`).join('')}</tbody></table>`;
}
const metaLabels={cells:'Fluid cells',ranks:'MPI ranks',requested_end_s:'Requested end (s)',achieved_end_s:'Recorded end (s)',dt_s:'Time step (s)',source_cadence_s:'Source cadence (s)',density_kg_m3:'Density (kg/m³)',sound_speed_m_s:'Sound speed (m/s)',spanwise_model:'Spanwise model',analysis_window_s:'Analysis window (s)',acoustic_band_hz:'Acoustic band (Hz)',native_returncode:'Native return code',native_End:'Native End marker'};
const value=v=>v==null?'Not recorded':Array.isArray(v)?v.join(' – '):String(v);
const reducedMotion=window.matchMedia('(prefers-reduced-motion: reduce)');
function imageMarkup(src,caption) {
  return `<a href="${e(src)}" target="_blank" rel="noopener"><img src="${e(src)}" alt="${e(caption)}"></a><figcaption>${e(caption)}<br><a href="${e(src)}" target="_blank" rel="noopener">Open full-resolution image ↗</a></figcaption>`;
}
function media() {
  const c=current();
  document.querySelector('#flow-media video')?.pause();
  $('mesh-media').innerHTML=c.mesh?`<figure>${imageMarkup(c.mesh.image,c.mesh.caption)}</figure>`:'<p>No native mesh image is available.</p>';
  $('flow-coverage').textContent=c.status==='completed'?`Recorded completion: ${value(c.metadata.achieved_end_s)} s; ${Number(c.metadata.cells).toLocaleString('en-US')} fluid cells in this case. The four-case table shows the recorded endpoints; completion is not physical validation.`:'No completed-run coverage is available for this case.';
  const p=c.presentation;
  $('source-contours').innerHTML=p?`<figure>${imageMarkup(p.source_contours.image,p.source_contours.caption)}</figure>`:'<p class="empty-state">Native recording-contour evidence is unavailable for this case. The geometry sketch is not native evidence.</p>';
  $('field-history-media').innerHTML=p?`<figure>${imageMarkup(p.field_history.image,p.field_history.caption)}</figure><p><a href="${e(p.field_history.download)}" download>Download complete field-history CSV</a></p>`:'<p>No native field history is available for this case.</p>';
  if(!c.flow) $('flow-media').innerHTML='<p class="empty-state">Native flow is pending. No substitute animation is shown.</p>';
  else {
    const f=c.flow;
    const image=f.image?imageMarkup(f.image,f.caption):'';
    if(f.video) $('flow-media').innerHTML=`<figure><video id="recorded-flow" src="${e(f.video)}" ${f.image&&!/\.gif$/i.test(f.image)?`poster="${e(f.image)}"`:''} controls playsinline preload="metadata" aria-label="${e(f.caption)}"></video><button id="flow-play" type="button">Play recorded flow</button><figcaption>${e(f.caption)}<br>Recorded field interval: ${e(value(f.start_s))} to ${e(value(f.end_s))} s. Playback never starts automatically.</figcaption></figure>`;
    else if(/\.gif$/i.test(f.image)) $('flow-media').innerHTML='<p class="figure-note">Supplied GIF animation is paused until requested. Use Hide animation to stop it.</p><button id="flow-gif" type="button" aria-pressed="false">Show recorded animation</button><figure id="gif-slot"></figure>';
    else $('flow-media').innerHTML=`<figure>${image}</figure>`;
    const video=$('recorded-flow');
    if(video) {
      const update=()=>{if($('recorded-flow')===video && $('flow-play'))$('flow-play').textContent=video.paused?'Play recorded flow':'Pause recorded flow';};
      for(const name of ['play','pause','ended'])video.addEventListener(name,update);
      video.addEventListener('error',()=>{$('flow-media').innerHTML='<p class="empty-state">The recorded flow video is unavailable or unsupported. No replacement animation is shown.</p>';});
      $('flow-play').addEventListener('click',async()=>{if(!video.paused)video.pause();else try{await video.play();}catch(error){if(error.name==='AbortError' && video.paused)update();else if($('recorded-flow')===video && $('flow-play'))$('flow-play').textContent='Playback unavailable';}});
    }
    $('flow-gif')?.addEventListener('click',()=>{
      const button=$('flow-gif'),show=button.getAttribute('aria-pressed')!=='true';
      button.setAttribute('aria-pressed',String(show));button.textContent=show?'Hide animation':'Show recorded animation';
      $('gif-slot').innerHTML=show?image:'';bindImageErrors();
    });
  }
  bindImageErrors();
}
function bindImageErrors() {
  document.querySelectorAll('#flow-media img,#mesh-media img,#source-contours img,#field-history-media img').forEach(img=>img.addEventListener('error',()=>{
    const note=document.createElement('p');note.className='empty-state';note.textContent='Recorded image unavailable. No substitute field is shown.';img.replaceWith(note);
  },{once:true}));
}
reducedMotion.addEventListener('change',()=>{
  if(!reducedMotion.matches)return;
  $('recorded-flow')?.pause();
  if($('flow-gif')?.getAttribute('aria-pressed')==='true')$('flow-gif').click();
});
document.addEventListener('visibilitychange',()=>{if(document.hidden)$('recorded-flow')?.pause();});
let downloadGeneration=0;
function evidence() {
  const c=current();
  $('coverage').innerHTML=['achieved_end_s','cells','native_returncode','native_End'].map(k=>`<div><dt>${e(metaLabels[k])}</dt><dd>${e(value(c.metadata[k]))}</dd></div>`).join('');
  $('case-evidence').innerHTML=`<p><strong>${e(c.flow_model)} · ${e(c.acoustic_surface)}</strong> / ${e(c.id)}</p><dl class="metadata">${META_KEYS.map(k=>`<div><dt>${e(metaLabels[k])}</dt><dd>${e(value(c.metadata[k]))}</dd></div>`).join('')}</dl><h3>Recorded checks</h3>${c.checks.length?`<ul class="checks">${c.checks.map(x=>`<li><strong>${e(x.name)} — ${e(x.status)}</strong><br>${e(x.detail)}</li>`).join('')}</ul>`:'<p>No recorded checks.</p>'}<h3>Limitations</h3>${bundle.limitations.length?`<ul>${bundle.limitations.map(x=>`<li>${e(x)}</li>`).join('')}</ul>`:'<p>No additional result-specific limitations have been supplied.</p>'}<h3>Public provenance</h3><pre>${e(JSON.stringify(bundle.provenance,null,2))}</pre>`;
  const links=[...bundle.downloads,...c.downloads];
  const generation=++downloadGeneration;
  $('downloads').innerHTML=links.length?links.map(d=>`<a role="link" tabindex="0" aria-disabled="true" data-download="${e(d.href)}">${e(d.label)} · checking</a>`).join(''):'<p>No verified downloads are available yet.</p>';
  $('download-status').textContent='';
  [...$('downloads').children].forEach(async(a,i)=>{
    if(!a.dataset.download)return;
    try {
      const response=await fetch(safeAsset(a.dataset.download),{method:'HEAD',redirect:'error',signal:AbortSignal.timeout(8000)});
      if(!response.ok || /text\/html/i.test(response.headers.get('content-type')||''))throw Error('File unavailable');
      await response.body?.cancel();
      if(generation!==downloadGeneration)return;
      a.textContent=links[i].label;a.setAttribute('aria-disabled','false');a.href=a.dataset.download;a.setAttribute('download','');
    }catch {if(generation===downloadGeneration)a.textContent=`${links[i].label} · unavailable`;}
  });
}
async function download(a) {
  if(a.getAttribute('aria-disabled')!=='false')return;
  $('download-status').textContent='Reading download…';
  try {
    const path=safeAsset(a.dataset.download),response=await fetch(path,{redirect:'error',signal:AbortSignal.timeout(30000)});
    if(!response.ok || /text\/html/i.test(response.headers.get('content-type')||''))throw Error('Download unavailable');
    const blob=await response.blob();
    if(!blob.size)throw Error('Download is empty');
    const href=URL.createObjectURL(blob),link=document.createElement('a');
    link.href=href;link.download=path.split('/').pop();link.click();
    setTimeout(()=>URL.revokeObjectURL(href),1000);
    $('download-status').textContent='Download prepared from the published file.';
  }catch { $('download-status').textContent='Download unavailable. No substitute file was created.'; }
}
async function reload() {
  $('reload-results').disabled=true;
  try {
    const response=await fetch('./data/results.json',{cache:'no-store',signal:AbortSignal.timeout(12000)});
    if(response.status===404){bundle=adaptResults(null);await response.text();}
    else {if(!response.ok)throw Error('Results request failed.');bundle=adaptResults(await response.json());}
  } catch(err) {bundle={...adaptResults(null),state:'unavailable',message:`Results unavailable: ${err instanceof SyntaxError?'invalid JSON.':err.message} No computed data are displayed.`};}
  finally {render();$('reload-results').disabled=false;}
}
$('case-picker').addEventListener('click',event=>{const button=event.target.closest('[data-case]');if(button){selected=button.dataset.case;render();}});
$('phase').addEventListener('input',geometry);
$('reload-results').addEventListener('click',reload);
$('compare-mode').addEventListener('change',figures);
$('receiver-direction').addEventListener('change',figures);
$('downloads').addEventListener('click',event=>{const a=event.target.closest('[data-download]');if(a){event.preventDefault();download(a);}});
$('downloads').addEventListener('keydown',event=>{if(event.key==='Enter' && event.target.getAttribute('aria-disabled')==='true')event.preventDefault();});
let resizeTimer;
window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(figures,100);});
render();reload();
