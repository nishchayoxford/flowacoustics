/* Read-only figure adapter: no numerical recomputation or historical fallback. */
import {safeAssetHref} from './data.mjs';
import {escapeHTML as esc} from './plots.mjs';

const BASE='./assets/signature/';
const IDS=['I01_geometry','I02_record_support','I03_force_spectra','I04_receiver_spectra',
  'I05_spectrograms','I06_band_variability','I07_coherence_phase','I08_near_radiating','I09_directivity'];
const LABEL='NATIVE COMPLETED SHORT RECORD';
const object=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const sha=v=>typeof v==='string'&&/^[0-9a-f]{64}$/.test(v);
const text=v=>typeof v==='string'&&v.trim().length>0;
const require=(condition,message)=>{if(!condition)throw new TypeError(message);};
function finiteTree(value){
  if(typeof value==='number')require(Number.isFinite(value),'Nonfinite gallery metadata.');
  else if(value&&typeof value==='object')Object.values(value).forEach(finiteTree);
}
function runDigest(results){
  require(object(results)&&text(results.status)&&!['pending','unavailable','failed','running'].includes(results.status),
    'No readable native result identity.');
  const inputs=results.metadata?.inputs;
  require(Array.isArray(inputs),'No native run input identity.');
  const runs=inputs.filter(input=>input?.role==='run');
  require(runs.length===1&&sha(runs[0].sha256),'Missing or ambiguous native run digest.');
  return runs[0].sha256;
}
export function validateIndex(index,results){
  require(object(index)&&index.schema_version===1,'Unsupported signature index.');
  finiteTree(index);
  require(index.source_kind==='native_completed_short_record'&&index.evidence_label===LABEL,
    'Only current native reports are accepted; historical smoke is never a production gallery.');
  require(sha(index.native_run_sha256)&&index.native_run_sha256===runDigest(results),
    'Signature and displayed native run digests do not match.');
  require(index.figure_count===9&&Array.isArray(index.figures)&&index.figures.length===9,
    'Expected exactly nine signature figures.');
  require(Array.isArray(index.limitations)&&index.limitations.length>0&&index.limitations.every(text),
    'Missing short-record limitations.');
  const pins=index.output_sha256;
  require(object(pins)&&Object.keys(pins).length>0,'Missing output hashes.');
  for(const [path,hash] of Object.entries(pins))require(safeAssetHref(path)&&sha(hash),'Unsafe output pin.');
  const ids=new Set(), paths=new Set();
  const asset=(record,expected)=>{
    require(object(record)&&safeAssetHref(record.path)&&expected(record.path),'Unsafe figure asset path.');
    require(sha(record.sha256)&&pins[record.path]===record.sha256,'Missing/mismatched figure hash.');
    require(!paths.has(record.path),'Duplicate asset path.');paths.add(record.path);
  };
  index.figures.forEach((figure,i)=>{
    require(object(figure)&&figure.id===IDS[i]&&!ids.has(figure.id),'Wrong or duplicate figure ID.');
    ids.add(figure.id);
    require(text(figure.title)&&text(figure.caption)&&figure.evidence_label===LABEL,'Invalid figure text/label.');
    require(object(figure.outputs)&&Object.keys(figure.outputs).sort().join(',')==='pdf,png,svg','Missing figure formats.');
    for(const ext of ['png','svg','pdf'])asset(figure.outputs[ext],path=>path===`${figure.id}/figure.${ext}`);
    for(const axis of ['width','height'])require(Number.isInteger(figure.outputs.png[axis])&&figure.outputs.png[axis]>0,
      'Invalid PNG dimensions.');
    require(safeAssetHref(figure.provenance)&&figure.provenance===`${figure.id}/provenance.json`,'Unsafe provenance path.');
    require(Array.isArray(figure.numeric_tables)&&figure.numeric_tables.length>0,'Missing numeric tables.');
    for(const table of figure.numeric_tables){
      asset(table,path=>path.startsWith(figure.id+'/')&&path.split('/').length===2&&path.endsWith('.csv'));
      require(Number.isInteger(table.rows)&&table.rows>=0&&Array.isArray(table.columns)&&table.columns.every(text),
        'Invalid numeric table metadata.');
    }
  });
  require(paths.size===Object.keys(pins).length,'Unexpected pinned assets.');
  return index;
}
const empty=(status,summary)=>({status,summary,index:null});
export async function loadSignature(results,fetcher=globalThis.fetch){
  try{
    const response=await fetcher(BASE+'index.json',{cache:'no-store',signal:AbortSignal.timeout(8000)});
    if(response.status===404)return empty('pending','Acoustic signatures are pending. No matching verified figure bundle has been published.');
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    return {status:'ready',index:validateIndex(await response.json(),results)};
  }catch{
    return empty('unavailable','Acoustic signatures are unavailable: the bundle is missing, invalid, historical, or does not match the displayed native run. No substitute figures are shown.');
  }
}
export function showSignature(container,state){
  container.innerHTML=renderSignature(state);
  // A decoded-image failure must not leave a partial gallery looking complete.
  for(const image of container.querySelectorAll('img'))image.addEventListener('error',()=>{
    if(container.contains(image))container.innerHTML=renderSignature(empty('unavailable',
      'Acoustic signatures are unavailable: a figure asset could not be loaded. No partial gallery is displayed.'));
  },{once:true});
}

export function renderSignature(state){
  if(state?.status!=='ready'||!state.index)return `<p class="signature-status" role="status">${esc(state?.summary||'Acoustic signatures are pending. No verified figure bundle is displayed.')}</p>`;
  const index=state.index;
  const link=(path,label,extra='')=>`<a href="${esc(BASE+path)}" ${extra}>${esc(label)}</a>`;
  return `<div class="signature-context"><p class="signature-label">COMPUTED NATIVE HISTORIES + OFFLINE REDUCTIONS · SHORT RECORD</p><p>Nine figures from the same native run as the evidence panel. These are unvalidated loading-analogy predictions, not measured sensor performance. Open the SVG or PDF for readable axes and detail.</p><ul>${index.limitations.map(note=>`<li>${esc(note)}</li>`).join('')}</ul><details><summary>Native run identity</summary><code>${esc(index.native_run_sha256)}</code></details></div><div class="signature-gallery">${index.figures.map((figure,i)=>{
    const offline=['I08_near_radiating','I09_directivity'].includes(figure.id);
    const png=figure.outputs.png;
    const format=figure.rendering?.pdf?`<p class="signature-note">PDF: ${esc(figure.rendering.pdf)}${figure.rendering.raster_dpi?` (${esc(figure.rendering.raster_dpi)} dpi)`:''}.</p>`:'';
    return `<figure class="signature-card" data-signature-id="${esc(figure.id)}">
      <figcaption><p class="signature-label">SIGNATURE ${String(i+1).padStart(2,'0')} / 09 · ${offline?'OFFLINE COMPACT MODEL / NATIVE COMPARISON':'COMPUTED NATIVE HISTORIES / OFFLINE ANALYSIS'}</p><h3>${esc(figure.title)}</h3><p class="signature-caption">${esc(figure.caption)}</p></figcaption>
      <a class="signature-image" href="${esc(BASE+figure.outputs.svg.path)}" aria-label="${esc('Open full-size SVG: '+figure.title)}"><img src="${esc(BASE+png.path)}" width="${png.width}" height="${png.height}" loading="lazy" decoding="async" alt="${esc(figure.title+' — '+figure.caption)}"></a>
      <p class="signature-links">${link(figure.outputs.svg.path,'Open SVG')} ${link(figure.outputs.pdf.path,'Download PDF','download')} ${link(figure.provenance,'Sanitized provenance (JSON)')}</p>${format}
      <details class="signature-tables"><summary>Numeric CSV tables · original verified bytes</summary><ul>${figure.numeric_tables.map(table=>`<li>${link(table.path,table.path.split('/').pop(),'download')} <span>(${table.rows} rows)</span></li>`).join('')}</ul></details>
      <p class="signature-note">Short record · descriptive only. Native Curle and Ffowcs Williams–Hawkings results are alternatives, not additive contributions. Offline curves do not add CFD receivers.</p></figure>`;
  }).join('')}</div>`;
}
