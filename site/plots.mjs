/* Dependency-free SVG charts. Labels are text-escaped; missing values break paths. */
export const COLORS = ['#006d70', '#b35432', '#233f5a', '#856224'];
export const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export function formatNumber(v) {
  if (!Number.isFinite(v)) return '—';
  if (v === 0) return '0';
  if (Math.abs(v) >= 10000 || Math.abs(v) < 0.001) return v.toExponential(1);
  return Number(v.toPrecision(3)).toString();
}
const valid = v => typeof v === 'number' && Number.isFinite(v);
const text = (x,y,label,extra='') => `<text x="${x}" y="${y}" ${extra}>${escapeHTML(label)}</text>`;
function limits(values, log=false) {
  const usable=values.filter(v=>valid(v)&&(!log||v>0));
  if (!usable.length) return log ? [0.1,1] : [0,1];
  let lo=Math.min(...usable), hi=Math.max(...usable);
  if(lo===hi) { if(log){lo/=2;hi*=2;} else {lo-=Math.abs(lo)*0.1||1;hi+=Math.abs(hi)*0.1||1;} }
  return [lo,hi];
}
function legend(traces) {
  return `<div class="plot-legend">${traces.map((t,i)=>`<span><i style="background:${COLORS[i%COLORS.length]}"></i>${escapeHTML(t.name)}</span>`).join('')}</div>`;
}
export function renderPlot(chart, requestedWidth=680) {
  const width=Math.max(270, Math.round(requestedWidth)), height=340;
  const e=escapeHTML;
  let content='', after='';
  const traces=[...(chart.traces||[]),...(chart.reference_traces||[])];
  const title=`<title>${e(chart.title)}</title><desc>${e(chart.description||'')} ${e(chart.x_label)}; ${e(chart.y_label)}.</desc>`;
  if(chart.kind==='polar') {
    // Reserve the bottom caption row below the 270° label, including text height.
    const cx=width/2,cy=154,R=Math.min(width/2-55,115);
    const max=Math.max(1e-12,...traces.flatMap(t=>t.y.filter(valid).map(Math.abs)));
    for(const fraction of [0.25,0.5,0.75,1]) content+=`<circle cx="${cx}" cy="${cy}" r="${R*fraction}" fill="none" class="grid"/>`+text(cx+5,cy-R*fraction+14,formatNumber(max*fraction));
    for(let deg=0;deg<360;deg+=45) {
      const a=deg*Math.PI/180, x=cx+R*Math.cos(a),y=cy-R*Math.sin(a);
      content+=`<path d="M${cx},${cy} L${x},${y}" class="grid"/>`;
      content+=text(cx+(R+23)*Math.cos(a),cy-(R+23)*Math.sin(a)+4,`${deg}°`,'text-anchor="middle"');
    }
    traces.forEach((trace,index)=>{
      let d='',active=false;
      trace.x.forEach((deg,j)=>{
        const value=trace.y[j];
        if(!valid(deg)||!valid(value)){active=false;return;}
        const a=deg*Math.PI/180,r=Math.abs(value)/max*R;
        const x=cx+r*Math.cos(a),y=cy-r*Math.sin(a);
        if(chart.connect_points===false){
          const observer=chart.samples?.observer_names?.[j]||`Observer ${j+1}`;
          const label=`${observer} · ${trace.name} · ${formatNumber(deg)}° · ${formatNumber(value)}`;
          content+=`<circle cx="${x}" cy="${y}" r="${4+2*index}" fill="none" stroke="${COLORS[index%COLORS.length]}" stroke-width="2" data-observer="${e(observer)}" aria-label="${e(label)}"><title>${e(label)}</title></circle>`;
        }else {d+=`${active?'L':'M'}${x},${y} `;active=true;}
      });
      if(d)content+=`<path d="${d}" fill="none" stroke="${COLORS[index%COLORS.length]}" stroke-width="2.4"/>`;
    });
    content+=text(cx,324,chart.y_label||'Magnitude (normalized)','text-anchor="middle"');
    after=legend(traces);
    if(chart.connect_points===false)after+=`<div class="plot-legend">${(chart.samples?.observer_names||[]).map((name,j)=>`<span>${e(name)} · ${e(formatNumber(traces[0]?.x[j]))}°</span>`).join('')}</div>`;
  } else {
    const right=18,bottom=61;
    let left=64,top=24,w=width-left-right,h=height-top-bottom;
    const xl=chart.x_scale==='log',yl=chart.y_scale==='log';
    const xs=chart.kind==='heatmap'?chart.x:traces.flatMap(t=>t.x);
    const ys=chart.kind==='heatmap'?chart.y:traces.flatMap(t=>t.y);
    const xd=chart.x_domain||limits(xs,xl),yd=chart.y_domain||limits(ys,yl);
    const yTicks=Array.from({length:5},(_,i)=>yl
      ?10**(Math.log10(yd[0])+(Math.log10(yd[1])-Math.log10(yd[0]))*i/4)
      :yd[0]+(yd[1]-yd[0])*i/4);
    // Reserve separate lanes for the rotated title and signed/exponential ticks.
    left=Math.max(left,30+7.5*Math.max(...yTicks.map(y=>formatNumber(y).length)));
    w=width-left-right;
    if(chart.equal_aspect&&!xl&&!yl){
      const ratio=(xd[1]-xd[0])/(yd[1]-yd[0]);
      if(w/h>ratio){const next=h*ratio;left+=(w-next)/2;w=next;}
      else {const next=w/ratio;top+=(h-next)/2;h=next;}
    }
    const trans=(v,log)=>log?Math.log10(v):v;
    const map=(v,domain,log)=> (trans(v,log)-trans(domain[0],log))/(trans(domain[1],log)-trans(domain[0],log));
    const X=v=>left+map(v,xd,xl)*w,Y=v=>top+h-map(v,yd,yl)*h;
    if(chart.kind==='heatmap') {
      const values=(chart.z||[]).flat().filter(valid), bound=chart.color_limit||Math.max(1e-12,...values.map(Math.abs));
      const diverging=chart.diverging!==false;
      chart.y.forEach((y,j)=>chart.x.forEach((x,i)=>{
        const z=chart.z[j]?.[i]; if(!valid(z))return;
        const u=Math.max(-1,Math.min(1,z/bound));
        const rgb=diverging?(u>=0?[0,109,112]:[179,84,50]):[0,109,112];
        const alpha=diverging?Math.abs(u):Math.max(0,u);
        const color=rgb.map(c=>Math.round(250+(c-250)*alpha));
        const x0=i===0?left:(X(chart.x[i-1])+X(x))/2;
        const x1=i===chart.x.length-1?left+w:(X(chart.x[i+1])+X(x))/2;
        const y0=j===chart.y.length-1?top:(Y(chart.y[j+1])+Y(y))/2;
        const y1=j===0?top+h:(Y(chart.y[j-1])+Y(y))/2;
        content+=`<rect x="${x0}" y="${y0}" width="${Math.min(x1-x0+0.4,left+w-x0)}" height="${Math.min(y1-y0+0.4,top+h-y0)}" fill="rgb(${color})"/>`;
      }));
      after=`<div class="color-key"><span>${diverging?'−'+formatNumber(bound):'0'}</span><i class="${diverging?'':'sequential'}"></i><span>${diverging?'+':''}${formatNumber(bound)}</span><span>Color range · ${e(chart.color_label||'z (supplied units)')}</span></div>`;
    }
    for(let i=0;i<=4;i++){
      const inv=(d,l)=>l?10**(Math.log10(d[0])+(Math.log10(d[1])-Math.log10(d[0]))*i/4):d[0]+(d[1]-d[0])*i/4;
      const x=inv(xd,xl),y=yTicks[i];
      content+=`<path d="M${X(x)},${top} V${top+h} M${left},${Y(y)} H${left+w}" class="grid"/>`;
      content+=text(X(x),top+h+23,formatNumber(x),`text-anchor="${i===0?'start':i===4?'end':'middle'}"`);
      content+=text(left-10,Y(y)+4,formatNumber(y),'text-anchor="end"');
    }
    content+=text(left+w/2,height-10,chart.x_label,'text-anchor="middle"');
    content+=`<text transform="translate(17 ${top+h/2}) rotate(-90)" text-anchor="middle">${e(chart.y_label)}</text>`;
    traces.forEach((trace,index)=>{
      let d='',active=false;
      const usable=j=>valid(trace.x[j])&&valid(trace.y[j])&&(!xl||trace.x[j]>0)&&(!yl||trace.y[j]>0);
      const connected=(a,b)=>usable(a)&&usable(b)&&
        !(chart.phase_wrap_period>0&&Math.abs(trace.y[b]-trace.y[a])>chart.phase_wrap_period/2);
      trace.x.forEach((x,j)=>{
        const y=trace.y[j];
        if(!valid(x)||!valid(y)||(xl&&x<=0)||(yl&&y<=0)){active=false;return;}
        if(!connected(j-1,j))active=false;
        if(chart.kind==='scatter'||trace.points||(!connected(j-1,j)&&!connected(j,j+1))) content+=`<circle cx="${X(x)}" cy="${Y(y)}" r="3.5" fill="${COLORS[index%COLORS.length]}"/>`;
        else {d+=`${active?'L':'M'}${X(x)},${Y(y)} `;active=true;}
      });
      if(d)content+=`<path d="${d}" fill="none" stroke="${COLORS[index%COLORS.length]}" stroke-width="2.3" ${index%2?'stroke-dasharray="7 4"':''}/>`;
    });
    if(chart.kind!=='heatmap')after=legend(traces);
    if(chart.comparison_rows){
      const precise=v=>valid(v)?v.toPrecision(9):'—';
      after+=`<table style="width:100%;table-layout:fixed;overflow-wrap:anywhere;font-size:12px"><caption>Same-ray ratios and signed residuals (dimensionless)</caption><thead><tr><th scope="col">Model / receiver pair</th><th scope="col">RMS ratio</th><th scope="col">Δ from reactive</th></tr></thead><tbody>${chart.comparison_rows.map(row=>`<tr><th scope="row">${e(row.label)}</th><td>${e(precise(row.ratio))}</td><td>${e(precise(row.delta_reactive))}</td></tr>`).join('')}</tbody></table>`;
    }
  }
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="${e(chart.title)}">${title}${content}</svg>${after}`;
}
