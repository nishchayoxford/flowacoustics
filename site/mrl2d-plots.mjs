/* Small read-only SVG renderer. Axis titles wrap outside the tick region. */
export const escapeHTML = v=>String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const formatNumber = v=>Number.isFinite(v) ? v === 0 ? '0' : (Math.abs(v) < .001 || Math.abs(v) >= 10000) ? v.toExponential(2) : Number(v.toPrecision(4)).toString() : 'Not recorded';
const palette = ['#006d70','#ad4c2e','#314b73','#78511b'];
const text = (x,y,s,anchor='middle')=>`<text x="${x}" y="${y}" text-anchor="${anchor}">${escapeHTML(s)}</text>`;
const color = (t,i)=>t.color || palette[i%palette.length];
const dash = t=>t.dash === true || t.dash === 'dashed' ? '7 4' : !t.dash || t.dash === 'solid' ? '' : t.dash;
function extent(values, log) {
  const usable = values.filter(v=>Number.isFinite(v) && (!log || v>0));
  if (!usable.length) return log ? [1,10] : [0,1];
  let lo=Infinity,hi=-Infinity; for (const v of usable) {lo=Math.min(lo,v);hi=Math.max(hi,v);}
  if (lo === hi) {if(log) {lo/=2;hi*=2;} else {const pad=Math.abs(lo)*.1 || 1;lo-=pad;hi+=pad;}}
  return [lo,hi];
}
export function renderChart(chart, requestedWidth=720) {
  const width=Math.max(240,Math.round(requestedWidth)), height=280, e=escapeHTML;
  if(!chart.traces.some(t=>t.x.some((x,j)=>Number.isFinite(x)&&Number.isFinite(t.y[j])&&(chart.x_scale!=='log'||x>0)&&(chart.y_scale!=='log'||t.y[j]>0))))return '<p class="empty-state">No plottable samples on the supplied scales. No replacement values are drawn.</p>';
  let content='',note='';
  if (chart.kind === 'polar') {
    const cx=width/2,cy=136,R=Math.min(width/2-42,102);
    let max=0; for (const t of chart.traces) for (const y of t.y) if (Number.isFinite(y)) max=Math.max(max,y);
    const scale=max || 1;
    for (const f of [.25,.5,.75,1]) content+=`<circle cx="${cx}" cy="${cy}" r="${f*R}" class="grid"/>`;
    for (const a of [0,90,180,270]) {
      const rad=a*Math.PI/180;
      content+=`<path class="grid" d="M${cx},${cy} L${cx+R*Math.cos(rad)},${cy-R*Math.sin(rad)}"/>`;
      content+=text(cx+(R+24)*Math.cos(rad),cy-(R+24)*Math.sin(rad)+4,`${a}°`);
    }
    chart.traces.forEach((t,i)=>t.x.forEach((angle,j)=>{
      const y=t.y[j]; if(!Number.isFinite(angle)||!Number.isFinite(y))return;
      const a=angle*Math.PI/180,r=y/scale*R;
      const label=`${t.name}: ${formatNumber(angle)}°; ${formatNumber(y)} ${chart.y_label}`;
      content+=`<circle data-sample="${j}" cx="${cx+r*Math.cos(a)}" cy="${cy-r*Math.sin(a)}" r="${3.5+i*1.3}" fill="none" stroke="${color(t,i)}" stroke-width="2"><title>${e(label)}</title></circle>`;
    }));
    note=`Outer ring: ${formatNumber(max)}. Rings at 25, 50, 75 and 100%. No interpolation between receivers.`;
  } else {
    const xl=chart.x_scale==='log',yl=chart.y_scale==='log';
    const xd=extent(chart.traces.flatMap(t=>t.x),xl),yd=extent(chart.traces.flatMap(t=>t.y),yl);
    const trans=(v,l)=>l?Math.log10(v):v;
    const tick=(d,l,f)=>l?10**(trans(d[0],l)+(trans(d[1],l)-trans(d[0],l))*f):d[0]+(d[1]-d[0])*f;
    const yt=[0,.25,.5,.75,1].map(f=>tick(yd,yl,f));
    const left=Math.max(52,Math.max(...yt.map(y=>formatNumber(y).length))*7.5+13),right=14,top=15,bottom=34;
    const w=width-left-right,h=height-top-bottom;
    const map=(v,d,l)=>(trans(v,l)-trans(d[0],l))/(trans(d[1],l)-trans(d[0],l));
    const X=v=>left+map(v,xd,xl)*w,Y=v=>top+h-map(v,yd,yl)*h;
    yt.forEach(y=>{content+=`<path class="grid" d="M${left},${Y(y)} H${width-right}"/>`+text(left-10,Y(y)+4,formatNumber(y),'end');});
    (width<500?[0,.5,1]:[0,.25,.5,.75,1]).forEach(f=>{
      const x=tick(xd,xl,f); content+=`<path class="grid" d="M${X(x)},${top} V${height-bottom}"/>`+text(X(x),height-9,formatNumber(x),f===0?'start':f===1?'end':'middle');
    });
    chart.traces.forEach((t,i)=>{
      let d='',active=false;
      t.x.forEach((x,j)=>{const y=t.y[j];
        if(!Number.isFinite(x)||!Number.isFinite(y)||(xl&&x<=0)||(yl&&y<=0)){active=false;return;}
        d+=` ${active?'L':'M'}${X(x)},${Y(y)}`;active=true;
        if (t.x.length<=24) content+=`<circle cx="${X(x)}" cy="${Y(y)}" r="2.4" fill="${color(t,i)}"/>`;
      });
      content+=`<path data-trace="${i}" d="${d}" fill="none" stroke="${color(t,i)}" stroke-width="2" stroke-dasharray="${dash(t)}"/>`;
    });
    if(xl||yl)note='Nonpositive log samples are omitted, not floored. Null samples break the line.';
  }
  const metadata={id:chart.id,origin:chart.origin,window:chart.window,x_label:chart.x_label,y_label:chart.y_label,x_scale:chart.x_scale||'linear',y_scale:chart.y_scale||'linear'};
  return `<div class="axis-heading">${e(chart.y_label)}</div><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="${e(chart.title)}"><title>${e(chart.title)}</title><desc>${e(chart.description)} X: ${e(chart.x_label)}. Y: ${e(chart.y_label)}. ${e(note)}</desc><metadata>${e(JSON.stringify(metadata))}</metadata>${content}</svg><div class="axis-bottom">${e(chart.x_label)}</div><ul class="plot-legend">${chart.traces.map((t,i)=>`<li><svg viewBox="0 0 30 16" aria-hidden="true">${chart.kind==='polar'?`<circle class="legend-marker" cx="15" cy="8" r="${Math.min(6,3.5+i*1.3)}" fill="none" stroke="${color(t,i)}" stroke-width="2"/>`:`<path d="M0,8 H30" stroke="${color(t,i)}" stroke-width="2.5" stroke-dasharray="${dash(t)}"/>`}</svg>${e(t.name)}</li>`).join('')}</ul>${note?`<p class="figure-note">${e(note)}</p>`:''}`;
}
