/* Read-only integration boundary. Never create source or receiver histories here. */
const isObject=v=>v!==null&&typeof v==='object'&&!Array.isArray(v);
const finiteOrNull=v=>v===null||(typeof v==='number'&&Number.isFinite(v));
function require(condition,message){if(!condition)throw new TypeError(message);}
export function safeAssetHref(href){
  if(typeof href!=='string'||!href||/[:\\%?#\s\x00-\x1f]/.test(href)||href.startsWith('/'))return false;
  const path=href.startsWith('./')?href.slice(2):href;
  return path.length>0&&path.split('/').every(p=>p!==''&&p!=='.'&&p!=='..');
}
export function validateResults(data){
  require(isObject(data)&&data.schema_version===1,'Unsupported results schema; expected schema_version 1.');
  require(typeof data.status==='string'&&data.status.length>0,'Missing actual run status.');
  require(typeof data.summary==='string'&&isObject(data.metadata),'Missing summary or metadata.');
  for(const field of ['charts','checks','downloads'])require(Array.isArray(data[field]),`${field} must be an array.`);
  const ids=new Set();
  for(const chart of data.charts){
    require(isObject(chart),'Invalid chart.');
    for(const field of ['id','title','description','x_label','y_label'])require(typeof chart[field]==='string',`Chart ${field} must be text.`);
    require(!ids.has(chart.id),'Duplicate chart id.');ids.add(chart.id);
    require(['computed','analytical'].includes(chart.origin),'Chart needs an explicit computed or analytical origin.');
    require(['line','polar','heatmap','scatter'].includes(chart.kind),'Unsupported chart kind.');
    for(const axis of ['x_scale','y_scale'])require(chart[axis]===undefined||['linear','log'].includes(chart[axis]),'Unsupported scale.');
    require(Array.isArray(chart.traces),'Chart traces must be an array.');
    for(const trace of chart.traces){
      require(typeof trace.name==='string'&&Array.isArray(trace.x)&&Array.isArray(trace.y),'Invalid trace.');
      require(trace.x.length===trace.y.length,'Trace x/y length mismatch.');
      require(trace.x.every(finiteOrNull)&&trace.y.every(finiteOrNull),'Trace values must be finite numbers or null.');
    }
    if(chart.kind==='heatmap'){
      require(Array.isArray(chart.x)&&Array.isArray(chart.y)&&Array.isArray(chart.z)&&chart.x.length>0&&chart.y.length>0,'Invalid heatmap axes.');
      require(chart.x.every(v=>v!==null&&finiteOrNull(v))&&chart.y.every(v=>v!==null&&finiteOrNull(v)),'Invalid heatmap axis values.');
      require(chart.z.length===chart.y.length&&chart.z.every(row=>Array.isArray(row)&&row.length===chart.x.length&&row.every(finiteOrNull)),'Invalid heatmap row-major z[y][x] shape or values.');
    }
  }
  for(const check of data.checks)for(const field of ['name','status','detail'])require(typeof check?.[field]==='string',`Check ${field} must be text.`);
  for(const download of data.downloads)require(typeof download?.label==='string'&&safeAssetHref(download.href),'Download must be a public-safe relative asset path.');
  return data;
}
function empty(status,summary){return {schema_version:1,status,summary,metadata:{},charts:[],checks:[],downloads:[]};}
export async function loadResults(fetcher=globalThis.fetch){
  try{
    const response=await fetcher('./data/results.json',{cache:'no-store',signal:AbortSignal.timeout(8000)});
    if(response.status===404)return empty('pending','Native cylinder results have not yet been integrated. No computed histories are displayed. The analytical lessons above work independently.');
    if(!response.ok)throw new Error(`HTTP ${response.status}`);
    return validateResults(await response.json());
  }catch(error){
    return empty('unavailable',`The results bundle could not be read safely (${error.message}). No substitute data have been generated. Retry when a valid results.json is available.`);
  }
}
