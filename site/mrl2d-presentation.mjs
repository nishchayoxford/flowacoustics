/* Optional presentation-only extension; the frozen numerical adapter is unchanged. */
import {adaptResults,safeAsset} from './mrl2d-model.mjs';
function image(value) {
  if(!value || typeof value.caption!=='string' || value.caption.length>20000)throw Error('Invalid presentation image.');
  return {image:safeAsset(value.image),caption:value.caption};
}
export function adaptPresentation(raw) {
  const bundle=adaptResults(raw);
  return {...bundle,cases:bundle.cases.map(c=>{
    const supplied=raw?.cases.find(row=>row.id===c.id)?.presentation;
    let presentation=null;
    if(supplied!=null){
      const source_contours=image(supplied.source_contours),field_history={...image(supplied.field_history),download:safeAsset(supplied.field_history.download)};
      if(c.status==='completed')presentation={source_contours,field_history};
    }
    return {...c,presentation};
  })};
}
/** Filter display traces only, after the unmodified full-support comparison guard. */
export function receiverChart(chart,angle) {
  if(!Number.isInteger(angle)||angle<0||angle>=360||angle%30!==0)throw Error('Invalid receiver direction.');
  if(!/^(?:receiver(?:-|$)|spectrum$)/.test(chart.id))return chart;
  const direction=name=>name.match(/(?:Receiver|Observer) (\d+)(?: deg|°)$/)?.[1];
  const traces=chart.traces.filter(t=>Number(direction(t.name))===angle);
  // A non-directional legacy chart is not relabelled or reduced by guesswork.
  if(chart.traces.some(t=>direction(t.name)===undefined))return chart;
  return {...chart,traces};
}
