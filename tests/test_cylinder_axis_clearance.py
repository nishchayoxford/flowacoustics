"""Browser layout checks use synthetic scales, never replacement CFD data."""

from test_cylinder_site import CDP, isolated_browser, node_check


def test_cartesian_axis_title_clears_signed_and_exponential_ticks():
    with isolated_browser() as env:
        node_check(
            CDP
            + r"""
const charts=[
 {title:'Synthetic force density',kind:'line',x_label:'Frequency (Hz)',
  y_label:'Force PSD (N²/Hz)',y_scale:'log',
  traces:[{name:'test',x:[0,1],y:[3e-16,2.6e-7]}]},
 {title:'Synthetic signed pressure',kind:'line',x_label:'Time (s)',y_label:'Pressure (Pa)',
  traces:[{name:'test',x:[0,1],y:[-4.5e-5,4.5e-5]}]},
 {title:'Synthetic coefficient',kind:'line',x_label:'Time (s)',y_label:'Coefficient',
  traces:[{name:'test',x:[0,1],y:[-0.00143,0.0191]}]}
];
const before=JSON.stringify(charts);
for(const width of [1440,390]){
 await call('Emulation.setDeviceMetricsOverride',
   {width,height:1100,deviceScaleFactor:1,mobile:false});
 for(const chart of charts){
  const collisions=await evaluate(`(async()=>{
    const {renderPlot}=await import('./plots.mjs');
    const host=document.createElement('div');host.className='plot';
    host.style.width='${Math.min(900, width - 40)}px';
    document.body.appendChild(host);
    host.innerHTML=renderPlot(${JSON.stringify(chart)},${Math.min(900, width - 40)});
    const svg=host.querySelector('svg'),title=svg.querySelector('text[transform]');
    const a=title.getBoundingClientRect();
    const hits=Array.from(svg.querySelectorAll('text[text-anchor="end"]')).flatMap(t=>{
      const b=t.getBoundingClientRect();
      return Math.min(a.bottom,b.bottom)>Math.max(a.top,b.top)&&b.left-a.right<3
        ? [{tick:t.textContent,clearance:b.left-a.right}] : [];
    });
    host.remove();return hits;
  })()`);
  assert.deepEqual(collisions,[],`${width}px ${chart.title}: axis title/tick clearance`);
 }
}
assert.equal(JSON.stringify(charts),before);
assert.deepEqual(exceptions,[]);
ws.close();
""",
            env=env,
        )
