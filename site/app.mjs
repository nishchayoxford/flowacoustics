import {TAU, directivity, propagation, fieldPressure, loadingSignal, dipoleAmplitudes, distributedGain, coherentPower, aliasFrequency, pressureLevel, welch} from './physics.mjs';
import {renderPlot, formatNumber as fmt, escapeHTML as esc} from './plots.mjs';
import {loadResults} from './data.mjs';
const $ = id => document.getElementById(id);
const linspace = (a,b,n=181) => Array.from({length:n},(_,i)=>a+(b-a)*i/(n-1));
const state = {family:'dipole',f:0.4,r:2,angle:30,near:false,t:0,paused:matchMedia('(prefers-reduced-motion: reduce)').matches};
function draw(id, chart) {
  const el=$(id); if(!el) return;
  el.innerHTML=renderPlot(chart,el.clientWidth||500);
}
function updateWave() {
  const axis=linspace(-4,4,49);
  const z=axis.map(y=>axis.map(x=>{
    const r=Math.hypot(x,y);
    return r<0.3?null:fieldPressure({family:state.family,f:state.f,r,theta:Math.atan2(y,x),t:state.t,near:state.near});
  }));
  draw('wave-plot',{kind:'heatmap',title:'Analytical acoustic wave field',description:'Signed normalized pressure, not CFD.',equal_aspect:true,x:axis,y:axis,z,color_limit:1,color_label:'p* (clipped at ±1)',x_label:'x / ℓ',y_label:'y / ℓ'});
}
function updateSource() {
  const theta=state.angle*Math.PI/180,d=directivity(state.family,theta),prop=propagation(state.r,state.f);
  $('frequency-value').value=state.f.toFixed(2); $('range-value').value=state.r.toFixed(2); $('angle-value').value=`${state.angle}°`;
  $('near').disabled=state.family!=='dipole';
  $('experiment-feedback').textContent=`${state.family[0].toUpperCase()+state.family.slice(1)} · signed directivity ${Math.abs(d)<1e-10?'0 (ideal null)':d.toFixed(3)} · radiating amplitude |p*| = ${fmt(Math.abs(d)/state.r)} · delay t* = ${state.r.toFixed(2)} · kr = ${fmt(prop.kr)}. ${state.near&&state.family==='dipole'?'Dipole near term included in the map.':'Radiating term only in the map.'}`;
  $('source-feedback').textContent=state.family==='monopole'?'Equal in every direction. A stationary rigid cylinder has no moving-surface thickness monopole.':state.family==='dipole'?'The two lobes have opposite signs. An ideal null is directional cancellation—not an absence of unsteady loading.':'Four lobes for this tensor orientation only. Surface-only FW-H does not evaluate the omitted volume quadrupoles.';
  const angles=linspace(0,360,361);
  draw('directivity-plot',{kind:'polar',title:'Source directivity and observer direction',description:'Polar magnitude with observer ray',y_label:'|D(θ)| · arbitrary unit amplitude',traces:[{name:'Source magnitude',x:angles,y:angles.map(a=>Math.abs(directivity(state.family,a*Math.PI/180)))},{name:`Observer ${state.angle}°`,x:[state.angle,state.angle],y:[0,1]}]});
  $('pause').textContent=state.paused?'Play wave':'Pause wave';$('pause').setAttribute('aria-pressed',String(state.paused));
  updateWave();
}
function updatePropagation(){
  const times=linspace(0,3/state.f,241),signals=times.map(t=>loadingSignal(state.f,t));
  draw('loading-plot',{kind:'line',title:'Prescribed loading and its derivative',x_label:'Source time t*',y_label:'Independent normalized units',traces:[{name:'Force F*',x:times,y:signals.map(s=>s.force)},{name:'Derivative dF*/dt*',x:times,y:signals.map(s=>s.derivative)}]});
  $('loading-feedback').textContent=`For unit force amplitude at f* = ${fmt(state.f)}, the derivative amplitude is 2πf* = ${fmt(TAU*state.f)}. This is a unit-model result, not a measured cylinder force.`;
  draw('monopole-plot',{kind:'line',title:'Pulsating-sphere volume velocity and pressure driver',x_label:'Source time t*',y_label:'Independent normalized units',traces:[{name:'Volume velocity Q*',x:times,y:times.map(t=>Math.cos(TAU*state.f*t))},{name:'Derivative dQ*/dt*',x:times,y:times.map(t=>-TAU*state.f*Math.sin(TAU*state.f*t))}]});
  const clock=linspace(0,5+3/state.f,301);
  const signal=t=>t<0?0:(1-Math.exp(-state.f*t*3))*Math.sin(TAU*state.f*t);
  const gain=directivity(state.family,state.angle*Math.PI/180)/state.r;
  draw('retardation-plot',{kind:'line',title:'Causal source emission and far-field arrival',x_label:'Observer clock t*',y_label:'Signal (independent normalized units)',traces:[{name:'Emitted s*',x:clock,y:clock.map(signal)},{name:'Received p* · far term',x:clock,y:clock.map(t=>gain*signal(t-state.r))}]});
  $('retardation-feedback').textContent=`First arrival at t* = ${fmt(state.r)}. Signed far-field gain D/r* = ${fmt(gain)}. A negative gain is a phase reversal, not negative energy.`;
  const ranges=linspace(-1,1.2,151).map(v=>10**v),amps=ranges.map(r=>dipoleAmplitudes(state.f,r));
  draw('range-plot',{kind:'line',title:'Dipole radiating and reactive range decay',x_label:'Range r / ℓ (log)',y_label:'On-axis amplitude (log)',x_scale:'log',y_scale:'log',traces:[{name:'Radiating ∝ 1/r',x:ranges,y:amps.map(a=>a.radiating)},{name:'Reactive ∝ 1/r²',x:ranges,y:amps.map(a=>a.reactive)},{name:'Quadrature total',x:ranges,y:amps.map(a=>a.total)}]});
  const selected=dipoleAmplitudes(state.f,state.r);
  $('range-feedback').textContent=`At selected r* = ${fmt(state.r)}: radiating amplitude ${fmt(selected.radiating)}, reactive amplitude ${fmt(selected.reactive)}, quadrature total ${fmt(selected.total)}. On-axis dipole only; arbitrary pressure scale.`;
  const kr=linspace(-1.3,2,151).map(x=>10**x),observer=propagation(state.r,state.f).kr;
  draw('nearfield-plot',{kind:'line',title:'Reactive to radiating dipole ratio',x_label:'kr = 2πfr / c₀ (log)',y_label:'|p near| / |p far| (log)',x_scale:'log',y_scale:'log',traces:[{name:'Ratio = 1/(kr)',x:kr,y:kr.map(x=>1/x)},{name:'Selected observer',x:[observer],y:[1/observer],points:true}]});
}
function updateSummation(){
  const separation=+$('separation').value,coherence=+$('coherence').value;
  $('separation-value').value=separation.toFixed(2);$('coherence-value').value=coherence.toFixed(2);
  const angles=linspace(0,360,361);
  draw('compact-plot',{kind:'polar',title:'Compact and distributed coherent dipole pair',y_label:'|p| / compact maximum',traces:[{name:'Compact unit model',x:angles,y:angles.map(a=>Math.abs(Math.sin(a*Math.PI/180)))},{name:'Distributed pair model',x:angles,y:angles.map(a=>Math.abs(Math.sin(a*Math.PI/180)*distributedGain(state.f,a*Math.PI/180,separation)))}]});
  $('compact-feedback').textContent=`kd = ${fmt(TAU*state.f*separation)} at f* = ${fmt(state.f)} and d* = ${fmt(separation)}. ${TAU*state.f*separation<0.5?'Small phase variation: the compact approximation is close in this pair model.':'Phase variation matters: inspect angular cancellation in the pair model.'}`;
  const counts=Array.from({length:16},(_,i)=>i+1);
  draw('coherence-plot',{kind:'line',title:'Power growth with source count and coherence',x_label:'Number of equal sources N',y_label:'Total PSD / single-source PSD',traces:[{name:`Selected γ = ${coherence.toFixed(2)}`,x:counts,y:counts.map(n=>coherentPower(n,coherence))},{name:'Independent · N',x:counts,y:counts},{name:'In phase · N²',x:counts,y:counts.map(n=>n*n)}]});
  $('coherence-feedback').textContent=`For 8 sources, γ = ${coherence.toFixed(2)} gives ${fmt(coherentPower(8,coherence))} times one-source power. Independent: 8. Perfectly in phase: 64. No measured span coherence is implied.`;
}
let spectrumCache=null;
function updateSpectra(){
  const fs=+$('sample-rate').value,alias=aliasFrequency(state.f,fs);
  $('sample-rate-value').value=fs.toFixed(1);
  const t=linspace(0,5/state.f,451),sampleTimes=Array.from({length:Math.floor(5/state.f*fs)+1},(_,i)=>i/fs);
  draw('sampling-plot',{kind:'line',title:'True tone and discrete samples',x_label:'Time t*',y_label:'Unit-amplitude signal',traces:[{name:`True f* = ${fmt(state.f)}`,x:t,y:t.map(x=>Math.sin(TAU*state.f*x))},{name:`Samples · fₛ* = ${fmt(fs)}`,x:sampleTimes,y:sampleTimes.map(x=>Math.sin(TAU*state.f*x)),points:true}]});
  $('sampling-feedback').textContent=`Nyquist = ${fmt(fs/2)}. True f* = ${fmt(state.f)}; folded frequency = ${fmt(alias)}. ${state.f>=fs/2?'At or above Nyquist: these samples cannot reliably recover the original tone.':'Below Nyquist for this single tone; a real broadband source also needs anti-alias protection.'}`;
  const n=+$('segment').value,type=$('signal-type').value,key=`${state.f}/${n}/${type}`;
  if(spectrumCache?.key!==key){
    const signal=Array.from({length:1024},(_,i)=>{const t=i/16;return Math.sin(TAU*(state.f*t+(type==='chirp'?1.5*t*t/64:0)))+.35*Math.sin(TAU*2.3*state.f*t);});
    spectrumCache={key,result:welch(signal,16,n)};
  }
  const spec=spectrumCache.result,db=$('psd-units').value==='db',indices=spec.frequency.map((f,i)=>f<=Math.max(4,state.f*2.5+1)?i:-1).filter(i=>i>=0);
  draw('psd-plot',{kind:'line',title:'Welch estimate of the prescribed signal',x_label:'Frequency f*',y_label:db?'PSD (dB re unit density)':'PSD (p*² per unit f*)',traces:[{name:db?'Density · display floor −80 dB':'One-sided power density',x:indices.map(i=>spec.frequency[i]),y:indices.map(i=>db?10*Math.log10(Math.max(1e-8,spec.psd[i])):spec.psd[i])}]});
  const integral=spec.psd.reduce((a,b)=>a+b,0)*spec.df;
  $('psd-feedback').textContent=`Δf* = ${fmt(spec.df)} · ${spec.segments} overlapping segments · full-band ΣPSD Δf* = ${fmt(integral)} p*². ${db?'Display only: densities below 10⁻⁸ are floored at −80 dB; integration uses unfloored linear data.':'The density integral is a window-weighted estimate of mean-square signal, not a peak height.'}`;
  draw('spectrogram-plot',{kind:'heatmap',title:'Time-resolved density of prescribed samples',x_label:'Segment-center time t*',y_label:'Frequency f*',x:spec.times,y:indices.map(i=>spec.frequency[i]),z:indices.map(i=>spec.spectrogram[i]),color_label:'p*² per unit f*',diverging:false});
  $('spectrogram-feedback').textContent=`Window duration t* = ${fmt(n/16)}; hop t* = ${fmt(n/32)}. ${type==='chirp'?'The first tone sweeps upward by 3 units of f* over the record.':'Two stationary tones; edge-window effects and leakage depend on bin alignment.'}`;
  updateLevels();
}
function updateLevels(){
  const rms=+$('rms').value,unit=$('level-units').value;
  $('rms-value').value=String(rms);
  const r=linspace(1,10,181),pa=r.map(r=>rms*1e-6/r),y=pa.map(p=>unit==='db'?pressureLevel(p):unit==='upa'?p*1e6:p);
  draw('level-plot',{kind:'line',title:'Chosen RMS pressure represented in different units',x_label:'Range r (m)',y_label:unit==='db'?'SPL (dB re 1 µPa)':unit==='upa'?'RMS pressure (µPa)':'RMS pressure (Pa)',traces:[{name:'Chosen example · far-field 1/r',x:r,y}]});
  $('level-feedback').textContent=`At 1 m: ${rms} µPa = ${fmt(rms*1e-6)} Pa = ${pressureLevel(rms*1e-6).toFixed(2)} dB re 1 µPa. At 2 m: ${pressureLevel(rms*1e-6/2).toFixed(2)} dB, a ${(20*Math.log10(2)).toFixed(2)} dB decrease. This amplitude is chosen, not measured.`;
}
let results=null;
const displayValue=value=>value===null?'null':typeof value==='object'?JSON.stringify(value):String(value);
function drawResults(){
  if(!results)return;
  $('run-status').textContent=`RUN STATUS · ${results.status}`;
  $('run-status').className=`badge ${['pending','unavailable','failed'].includes(results.status)?'pending':'computed'}`;
  const computed=results.charts.filter(c=>c.origin==='computed').length;
  $('run-title').textContent=computed?`${computed} computed chart${computed===1?'':'s'} supplied · inspect limitations below`:'No computed history displayed';
  $('run-summary').textContent=results.summary;
  $('run-metadata').innerHTML=Object.keys(results.metadata).length?`<details class="model-note"><summary>Executed parameters, estimator, pressure statistics and provenance</summary><dl class="metadata">${Object.entries(results.metadata).map(([key,value])=>`<dt>${esc(key)}</dt><dd>${esc(displayValue(value))}</dd>`).join('')}</dl></details>`:'<p class="hint">No executed-case metadata supplied.</p>';
  $('run-checks').innerHTML=results.checks.length?`<details class="model-note"><summary>Inspect ${results.checks.length} checks and interpretation notes</summary><ul class="checks">${results.checks.map(check=>`<li><b>${esc(check.name)}</b><span class="badge">${esc(check.status)}</span><p>${esc(check.detail)}</p></li>`).join('')}</ul></details>`:'<p class="hint">No native checks supplied. This is not a pass.</p>';
  $('run-downloads').innerHTML=results.downloads.map(d=>`<a href="${esc(d.href)}" download>${esc(d.label)} ↓</a>`).join('');
  const container=$('computed-charts');container.replaceChildren();
  results.charts.forEach(chart=>{
    const figure=document.createElement('figure');figure.className='result-chart';figure.dataset.chartId=chart.id;
    const sampleInfo=Object.fromEntries(Object.entries(chart).filter(([key])=>/sample|coverage|exclud|includ|floor|window/i.test(key)));
    figure.innerHTML=`<figcaption><h3><span class="badge ${chart.origin}">${esc(chart.origin.toUpperCase())}</span>${esc(chart.title)}</h3><p>${esc(chart.description)}</p></figcaption><div class="plot"></div><p class="figure-note">${Object.keys(sampleInfo).length?esc(displayValue(sampleInfo)):'Included/excluded sample policy not supplied; do not infer full coverage.'}</p><p class="hint">Nulls are gaps; nonpositive log-axis values are omitted. No replacement floor is applied. ${chart.kind==='polar'?'Polar angles are degrees; radius is the absolute magnitude of supplied y.':''}</p>`;
    container.append(figure);
    const plot=figure.querySelector('.plot');plot.innerHTML=renderPlot(chart,plot.clientWidth||700);
  });
}
async function refreshResults(){
  $('retry-results').disabled=true;
  results=await loadResults();drawResults();
  $('retry-results').disabled=false;
}
function updateAll(){updateSource();updatePropagation();updateSummation();updateSpectra();}
$('retry-results').addEventListener('click',refreshResults);
for(const id of ['sample-rate','signal-type','segment','psd-units'])$(id).addEventListener('input',updateSpectra);
for(const id of ['rms','level-units'])$(id).addEventListener('input',updateLevels);
for(const id of ['separation','coherence'])$(id).addEventListener('input',updateSummation);
for(const id of ['family','frequency','range','angle','near'])$(id).addEventListener('input',()=>{
  state.family=$('family').value;state.f=+$('frequency').value;state.r=+$('range').value;state.angle=+$('angle').value;state.near=$('near').checked;
  updateAll();
});
$('pause').addEventListener('click',()=>{state.paused=!state.paused;updateSource();});
document.querySelectorAll('[data-answer]').forEach(button=>button.addEventListener('click',()=>{
  $('quiz-feedback').textContent=button.dataset.answer==='yes'?'Exactly. A fixed wall can carry changing traction. Loading noise survives even though moving-surface thickness is zero.':'Not quite. Surface motion is not required for changing loading. Zero thickness removes that monopole term, not the loading dipole.';
}));
$('reset').addEventListener('click',()=>{
  Object.assign(state,{family:'dipole',f:.4,r:2,angle:30,near:false,t:0});
  $('family').value='dipole';$('frequency').value=.4;$('range').value=2;$('angle').value=30;$('near').checked=false;updateAll();
});
let previous=0, visible=true;
new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;}).observe($('wave-plot'));
function animate(now){
  if(!previous)previous=now;
  if(now-previous>=180){if(!state.paused&&visible&&!document.hidden){state.t+=0.075;updateWave();}previous=now;}
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
let resizeTimer;
window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{updateAll();drawResults();},160);});
const links=[...document.querySelectorAll('.sidebar nav a')];
const sectionObserver=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){links.forEach(a=>a.removeAttribute('aria-current'));links.find(a=>a.hash===`#${entry.target.id}`)?.setAttribute('aria-current','location');}}, {rootMargin:'-10% 0px -65% 0px'});
document.querySelectorAll('main>section').forEach(section=>sectionObserver.observe(section));
updateAll();
refreshResults();
