"""Static learning-site tests. All numerical fixtures here are analytical, never CFD."""
from pathlib import Path
import subprocess
import json
import os
import shutil
import socket
import time
import urllib.request
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def node_check(program, env=None):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", program],
        cwd=ROOT, text=True, capture_output=True, check=False, env=env, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@contextmanager
def isolated_browser():
    """Chrome CDP fallback: no real-browser profile, no dependencies, no global edits."""
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    from functools import partial
    import threading
    import tempfile
    chrome = shutil.which('google-chrome') or shutil.which('chromium')
    assert chrome, 'A Chromium browser is required for the actual UI acceptance test'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    cache = ROOT / '.cache' / 'cylinder-site-verification'
    cache.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(SITE)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with tempfile.TemporaryDirectory(prefix='browser-', dir=cache) as profile:
        process = subprocess.Popen([chrome, '--headless=new', '--disable-gpu', '--no-first-run',
            '--no-default-browser-check', '--disable-background-networking',
            f'--user-data-dir={profile}', f'--remote-debugging-port={port}', 'about:blank'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(100):
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version', timeout=1):
                        break
                except OSError:
                    time.sleep(.1)
            else:
                raise AssertionError('Isolated Chromium failed to start')
            yield {**os.environ, 'SITE_TEST_URL':f'http://127.0.0.1:{server.server_port}/',
                   'SITE_CDP_PORT':str(port), 'SITE_SCREENSHOTS':str(cache)}
        finally:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            server.shutdown(); server.server_close()


CDP = r"""
import assert from 'node:assert/strict';
import fs from 'node:fs';
const tab = await (await fetch(`http://127.0.0.1:${process.env.SITE_CDP_PORT}/json/new?about:blank`, {method:'PUT'})).json();
const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise(resolve => ws.addEventListener('open', resolve, {once:true}));
let seq=0; const pending=new Map(), exceptions=[];
ws.addEventListener('message',event=>{
  const m=JSON.parse(event.data);
  if(m.method==='Runtime.exceptionThrown')exceptions.push(m.params);
  if(m.id&&pending.has(m.id)){const [resolve,reject]=pending.get(m.id);pending.delete(m.id);m.error?reject(m.error):resolve(m.result);}
});
const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++seq;pending.set(id,[resolve,reject]);ws.send(JSON.stringify({id,method,params}));});
const evaluate=async(expression)=>{
  const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
};
await call('Runtime.enable');await call('Page.enable');
await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
await call('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]});
await call('Page.navigate',{url:process.env.SITE_TEST_URL});
for(let i=0;i<100;i++){
 if(await evaluate(`!!document.querySelector('#wave-plot svg') && !document.getElementById('retry-results').disabled`))break;
 await new Promise(r=>setTimeout(r,80));
}
"""


def test_browser_first_learning_loop():
    with isolated_browser() as env:
        node_check(CDP + r"""
          assert.ok(await evaluate(`document.querySelectorAll('.plot svg').length >= 2`));
          assert.equal(await evaluate(`document.getElementById('pause').textContent`),'Play wave');
          await evaluate(`document.getElementById('angle').value=90;document.getElementById('angle').dispatchEvent(new Event('input',{bubbles:true}))`);
          assert.match(await evaluate(`document.getElementById('experiment-feedback').textContent`),/0 \(ideal null\)/);
          await evaluate(`document.getElementById('family').value='monopole';document.getElementById('family').dispatchEvent(new Event('input',{bubbles:true}))`);
          assert.match(await evaluate(`document.getElementById('experiment-feedback').textContent`),/directivity 1.000/);
          assert.equal(await evaluate(`document.getElementById('near').disabled`),true);
          assert.deepEqual(exceptions,[]);
          const shot=await call('Page.captureScreenshot',{format:'png'});
          fs.writeFileSync(process.env.SITE_SCREENSHOTS+'/desktop-initial.png',Buffer.from(shot.data,'base64'));
          ws.close();
        """, env=env)


def test_first_experiment_directivity_and_retarded_pressure():
    assert (SITE / "physics.mjs").exists(), "The independent acoustic learning model is missing"
    node_check("""
        import assert from 'node:assert/strict';
        import {directivity, fieldPressure, propagation} from './site/physics.mjs';
        assert.equal(directivity('monopole', 0), 1);
        assert.ok(Math.abs(directivity('dipole', Math.PI/2)) < 1e-12);
        assert.equal(directivity('dipole', Math.PI), -1);
        assert.ok(Math.abs(directivity('quadrupole', Math.PI/4)) < 1e-12);
        assert.equal(directivity('quadrupole', Math.PI/2), -1);
        const a = fieldPressure({family:'dipole', f:0.5, r:1, theta:0, t:1.5, near:false});
        const b = fieldPressure({family:'dipole', f:0.5, r:2, theta:0, t:2.5, near:false});
        assert.ok(Math.abs(a/b-2) < 1e-12);
        assert.deepEqual(propagation(2, 0.5), {delay:2, wavelength:2, kr:2*Math.PI});
        assert.throws(() => propagation(0, 1), RangeError);
    """)


def test_first_learning_surface_and_accessible_renderers():
    assert (SITE / "index.html").exists(), "The learning surface is missing"
    html = (SITE / "index.html").read_text()
    for token in ['id="family"', 'id="frequency"', 'id="range"', 'id="angle"',
                  'id="pause"', 'id="near"', 'id="directivity-plot"',
                  'id="wave-plot"', 'id="experiment-feedback"']:
        assert token in html
    node_check("""
        import assert from 'node:assert/strict';
        import {renderPlot} from './site/plots.mjs';
        const base = {title:'<unsafe>',description:'test',x_label:'Time (s)',y_label:'p (Pa)',
          traces:[{name:'A',x:[0,1,2],y:[0,1,0]}]};
        for(const kind of ['line','scatter','polar']) {
          const svg = renderPlot({...base, kind}, 340);
          assert.ok(svg.includes('role="img"'));
          assert.ok(svg.includes('&lt;unsafe&gt;'));
          assert.ok(!svg.includes('<unsafe>'));
          assert.ok(!svg.includes('NaN'));
        }
        const map = renderPlot({...base,kind:'heatmap',x:[0,1],y:[0,1],z:[[0,1],[2,3]]},340);
        assert.ok(map.includes('Color range'));
        assert.ok(!map.includes('NaN'));
    """)


def test_loading_to_sound_and_range_laws():
    node_check("""
        import assert from 'node:assert/strict';
        import * as p from './site/physics.mjs';
        assert.equal(typeof p.loadingSignal, 'function', 'Force-to-sound model is missing');
        const a=p.loadingSignal(0.5,0), b=p.loadingSignal(1,0);
        assert.equal(a.force,0);
        assert.ok(Math.abs(a.derivative-Math.PI)<1e-12);
        assert.equal(b.derivative/a.derivative,2);
        const x=p.dipoleAmplitudes(0.5,1),y=p.dipoleAmplitudes(0.5,2);
        assert.equal(x.radiating/y.radiating,2);
        assert.equal(x.reactive/y.reactive,4);
        assert.ok(Math.abs(p.dipoleAmplitudes(1/(2*Math.PI),1).ratio-1)<1e-12);
        assert.throws(()=>p.dipoleAmplitudes(0,1),RangeError);
    """)


def test_distributed_loading_and_coherence_are_distinct_models():
    node_check("""
        import assert from 'node:assert/strict';
        import * as p from './site/physics.mjs';
        assert.equal(typeof p.distributedGain,'function','Distributed source lesson missing');
        assert.equal(p.distributedGain(.5,0,0),1);
        assert.ok(Math.abs(p.distributedGain(.5,0,1))<1e-12);
        assert.equal(p.distributedGain(.5,Math.PI/2,1),1);
        assert.equal(p.coherentPower(4,0),4);
        assert.equal(p.coherentPower(4,1),16);
        assert.equal(p.coherentPower(4,.5),10);
        assert.throws(()=>p.coherentPower(4,2),RangeError);
    """)


def test_spectral_density_parseval_aliasing_and_underwater_reference():
    node_check("""
        import assert from 'node:assert/strict';
        import * as p from './site/physics.mjs';
        assert.equal(typeof p.welch,'function','Welch density lesson missing');
        const signal=Array.from({length:128},(_,i)=>Math.sin(2*Math.PI*4*i/64));
        const result=p.welch(signal,64,128,'boxcar');
        const integral=result.psd.reduce((a,b)=>a+b,0)*result.df;
        assert.ok(Math.abs(integral-.5)<1e-12);
        assert.equal(result.frequency[result.psd.indexOf(Math.max(...result.psd))],4);
        assert.equal(result.segments,1);
        assert.equal(p.aliasFrequency(9,10),1);
        assert.equal(p.pressureLevel(1e-6),0);
        assert.equal(p.pressureLevel(1),120);
        assert.equal(p.pressureLevel(0),-Infinity);
        assert.throws(()=>p.pressureLevel(-1),RangeError);
        assert.throws(()=>p.welch(signal,64,256),RangeError);
        const silence=p.welch(Array(128).fill(0),64,64);
        assert.ok(silence.psd.every(v=>v===0));
        assert.equal(silence.spectrogram.length,silence.frequency.length);
        assert.equal(silence.spectrogram[0].length,silence.times.length);
    """)


def test_pending_results_adapter_validates_contract_without_filling_history():
    assert (SITE / 'data.mjs').exists(), 'Truthful native-results adapter is missing'
    node_check("""
        import assert from 'node:assert/strict';
        import {loadResults,validateResults,safeAssetHref} from './site/data.mjs';
        const pending=await loadResults(async()=>({ok:false,status:404}));
        assert.equal(pending.status,'pending'); assert.deepEqual(pending.charts,[]);
        const failed=await loadResults(async()=>{throw new Error('offline')});
        assert.equal(failed.status,'unavailable');assert.deepEqual(failed.charts,[]);
        const fixture={schema_version:1,status:'partial',summary:'TEST-ONLY fixture, not CFD',metadata:{pressure_reference_Pa:1e-6},checks:[{name:'Coverage',status:'limited',detail:'test only'}],downloads:[{label:'test',href:'./data/history.csv'}],charts:[{id:'unit',title:'Test only',origin:'computed',kind:'line',description:'test-only schema fixture',x_label:'s',y_label:'Pa',traces:[{name:'fixture',x:[0,1,2],y:[0,null,2]}],included_samples:2,excluded_samples:1}]};
        assert.equal(validateResults(fixture).charts[0].traces[0].y[1],null);
        for(const href of ['https://example.org/x','//evil.test/a','../secret','data/%2e%2e/secret','javascript:alert(1)','/etc/passwd','data/..\\\\secret'])assert.equal(safeAssetHref(href),false,href);
        assert.equal(safeAssetHref('./data/force_history.csv'),true);
        const invalid=structuredClone(fixture);invalid.charts[0].traces[0].y=[1];
        assert.throws(()=>validateResults(invalid),/length/);
        invalid.charts[0].traces[0].y=[1,NaN,2];
        assert.throws(()=>validateResults(invalid),/finite/);
        const map=structuredClone(fixture);map.charts=[{...fixture.charts[0],kind:'heatmap',x:[0,1],y:[0,1],z:[[1,2],[3]]}];
        assert.throws(()=>validateResults(map),/heatmap/);
    """)


def test_wave_map_preserves_equal_spatial_scales():
    node_check("""
        import assert from 'node:assert/strict';
        import {renderPlot} from './site/plots.mjs';
        const svg=renderPlot({kind:'heatmap',title:'Equal coordinate units',equal_aspect:true,
          x_label:'x',y_label:'y',x:[-1,0,1],y:[-1,0,1],z:[[1,1,1],[1,1,1],[1,1,1]]},700);
        const cells=[...svg.matchAll(/<rect[^>]+width="([0-9.]+)" height="([0-9.]+)"/g)];
        assert.equal(cells.length,9);
        assert.ok(Math.abs(+cells[4][1]-cells[4][2])<1e-9,'Equal spatial coordinates must have equal screen scale');
    """)


def test_complete_browser_interactions_and_mobile_layout():
    with isolated_browser() as env:
        node_check(CDP + r"""
          const change=async(id,value)=>evaluate(`document.getElementById(${JSON.stringify(id)}).value=${JSON.stringify(value)};document.getElementById(${JSON.stringify(id)}).dispatchEvent(new Event('input',{bubbles:true}))`);
          const content=id=>evaluate(`document.getElementById(${JSON.stringify(id)}).textContent`);
          assert.equal(await evaluate(`document.querySelectorAll('main>.lesson .figure-num').length`),13);
          assert.equal(await evaluate(`document.querySelectorAll('main>.lesson:not(#evidence) .plot svg').length`),13);
          await change('frequency','1');await change('range','4');await change('angle','0');
          assert.match(await content('retardation-feedback'),/t\* = 4/);
          assert.match(await content('loading-feedback'),/6.28/);
          const beforeNear=await evaluate(`document.getElementById('wave-plot').innerHTML`);
          await evaluate(`document.getElementById('near').click()`);
          assert.notEqual(await evaluate(`document.getElementById('wave-plot').innerHTML`),beforeNear);
          await change('family','quadrupole');assert.match(await content('source-feedback'),/tensor orientation/);
          await change('coherence','1');assert.match(await content('coherence-feedback'),/64 times/);
          await change('separation','0');assert.match(await content('compact-feedback'),/Small phase variation/);
          await change('sample-rate','0.5');assert.match(await content('sampling-feedback'),/At or above Nyquist/);
          await change('signal-type','chirp');assert.match(await content('spectrogram-feedback'),/sweeps upward/);
          await change('segment','256');assert.match(await content('psd-feedback'),/7 overlapping segments/);
          await change('psd-units','db');assert.match(await content('psd-feedback'),/floored at −80 dB/);
          await change('level-units','pa');assert.match(await content('level-plot'),/RMS pressure \(Pa\)/);
          await change('rms','1');await change('level-units','db');assert.match(await content('level-feedback'),/= 0.00 dB re 1 µPa/);
          await evaluate(`document.querySelector('[data-answer="no"]').click()`);assert.match(await content('quiz-feedback'),/Not quite/);
          await evaluate(`document.querySelector('[data-answer="yes"]').click()`);assert.match(await content('quiz-feedback'),/Exactly/);
          await evaluate(`document.getElementById('reset').click(); document.getElementById('angle').focus()`);
          await call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
          await call('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
          assert.equal(await evaluate(`document.getElementById('angle').value`),'35');
          await evaluate(`window.scrollTo(0,document.getElementById('sources').offsetTop-90)`);
          const snap=async name=>{const shot=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(process.env.SITE_SCREENSHOTS+'/'+name+'.png',Buffer.from(shot.data,'base64'));};
          await snap('desktop-source-lab');
          await evaluate(`document.getElementById('pause').click()`);
          const moving=await evaluate(`document.getElementById('wave-plot').innerHTML`);
          await new Promise(r=>setTimeout(r,500));
          assert.notEqual(await evaluate(`document.getElementById('wave-plot').innerHTML`),moving);
          await evaluate(`document.getElementById('pause').click()`);
          const frozen=await evaluate(`document.getElementById('wave-plot').innerHTML`);
          await new Promise(r=>setTimeout(r,350));
          assert.equal(await evaluate(`document.getElementById('wave-plot').innerHTML`),frozen);
          await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
          await new Promise(r=>setTimeout(r,300));
          assert.equal(await evaluate(`document.documentElement.scrollWidth<=390`),true,'Horizontal overflow at 390 px');
          for(const id of ['sources','loading','propagation','summation','spectra','evidence']){
            await evaluate(`window.scrollTo(0,document.getElementById('${id}').offsetTop-80)`);
            assert.equal(await evaluate(`document.documentElement.scrollWidth<=390`),true,id);
          }
          await evaluate(`window.scrollTo(0,document.getElementById('directivity-plot').getBoundingClientRect().top+window.scrollY-160)`);await snap('mobile-directivity');
          await evaluate(`window.scrollTo(0,document.getElementById('psd-plot').getBoundingClientRect().top+window.scrollY-160)`);await snap('mobile-spectrum');
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)


def test_browser_results_fixture_pending_invalid_and_downloads():
    """Fixtures only enter the in-memory fetch boundary; never write results.json."""
    with isolated_browser() as env:
        node_check(CDP + r"""
          await evaluate(`window.originalFetch=window.fetch; window.fetch=async()=>new Response('',{status:404});document.getElementById('retry-results').click()`);
          await new Promise(r=>setTimeout(r,80));
          assert.match(await evaluate(`document.getElementById('run-status').textContent`),/pending/);
          assert.equal(await evaluate(`document.querySelectorAll('#computed-charts svg').length`),0);
          const fixture={schema_version:1,status:'partial',summary:'TEST-ONLY SCHEMA FIXTURE — NOT CFD',metadata:{pressure_reference_Pa:1e-6,notes:'<script>unsafe</script>'},checks:[{name:'Test coverage',status:'limited',detail:'Incomplete fixture coverage'}],downloads:[{label:'Test-only download',href:'data/test.csv'}],charts:[]};
          for(const kind of ['line','polar','scatter','heatmap'])fixture.charts.push({id:'test-'+kind,title:'TEST ONLY '+kind,origin:kind==='polar'?'analytical':'computed',kind,description:'Schema fixture only',x_label:kind==='polar'?'Angle (deg)':'Time (s)',y_label:'Pressure (Pa)',traces:[{name:'fixture',x:[0,1,2],y:[0,1,0]}],x:[0,1],y:[0,1],z:[[0,1],[1,0]],included_samples:3,excluded_samples:1});
          await evaluate(`window.fetch=async()=>new Response(${JSON.stringify(JSON.stringify(fixture))},{status:200,headers:{'Content-Type':'application/json'}});document.getElementById('retry-results').click()`);
          await new Promise(r=>setTimeout(r,80));
          assert.equal(await evaluate(`document.querySelectorAll('#computed-charts svg').length`),4);
          assert.match(await evaluate(`document.getElementById('run-title').textContent`),/3 computed charts/);
          assert.match(await evaluate(`document.getElementById('run-metadata').textContent`),/0.000001/);
          assert.equal(await evaluate(`document.querySelectorAll('#run-metadata script').length`),0);
          assert.equal(await evaluate(`document.querySelector('#run-downloads a').getAttribute('href')`),'data/test.csv');
          assert.match(await evaluate(`document.getElementById('computed-charts').textContent`),/excluded_samples/);
          const before=await evaluate(`document.getElementById('computed-charts').textContent`);
          await evaluate(`document.getElementById('frequency').value=1.5;document.getElementById('frequency').dispatchEvent(new Event('input',{bubbles:true}))`);
          assert.equal(await evaluate(`document.getElementById('computed-charts').textContent`),before,'Analytical controls must not change computed data');
          await evaluate(`window.scrollTo(0,document.getElementById('evidence').offsetTop-90)`);
          const shot=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(process.env.SITE_SCREENSHOTS+'/test-only-results-adapter.png',Buffer.from(shot.data,'base64'));
          await evaluate(`window.fetch=async()=>new Response('{bad json',{status:200});document.getElementById('retry-results').click()`);
          await new Promise(r=>setTimeout(r,80));
          assert.match(await evaluate(`document.getElementById('run-status').textContent`),/unavailable/);
          assert.equal(await evaluate(`document.querySelectorAll('#computed-charts svg').length`),0);
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)


def test_local_assets_and_scope():
    from html.parser import HTMLParser
    class Assets(HTMLParser):
        def __init__(self):
            super().__init__(); self.assets=[]; self.ids=[]; self.anchors=[]
        def handle_starttag(self, tag, attrs):
            values={key: value or '' for key, value in attrs}
            if 'id' in values: self.ids.append(values['id'])
            if tag=='script': self.assets.append(values.get('src',''))
            if tag=='link' and values.get('rel')=='stylesheet': self.assets.append(values['href'])
            if tag=='a' and values.get('href','').startswith('#'): self.anchors.append(values['href'][1:])
    html=(SITE/'index.html').read_text(); parser=Assets();parser.feed(html)
    assert len(parser.ids)==len(set(parser.ids)), 'Duplicate HTML id'
    assert set(parser.anchors).issubset(parser.ids), 'Broken in-page navigation'
    for asset in parser.assets:
        assert asset.startswith('./') and (SITE/asset).is_file()
    for excluded in ['Amiet','cavitation','airfoil','propeller','streamlit','cdn.jsdelivr','unpkg']:
        assert excluded.lower() not in html.lower()
    assert '10.1098/rspa.1955.0191' in html
    assert '10.1098/rsta.1969.0031' in html
    assert 'https://github.com/unicfdlab/libAcoustics' in html


def test_nonuniform_heatmap_cells_and_isolated_samples_are_not_lost():
    node_check("""
        import assert from 'node:assert/strict';
        import {renderPlot} from './site/plots.mjs';
        const chart={kind:'heatmap',title:'Nonuniform test grid',x_label:'x',y_label:'y',
          x:[0,1,3],y:[0,1],z:[[0,1,2],[3,4,5]],traces:[]};
        const svg=renderPlot(chart,700);
        const cells=[...svg.matchAll(/<rect[^>]+width="([0-9.]+)" height="([0-9.]+)"/g)];
        assert.ok(+cells[1][1]>2.5*+cells[0][1], 'Nonuniform grid must use neighboring midpoint edges');
        const single=renderPlot({kind:'line',title:'One sample',x_label:'s',y_label:'Pa',traces:[{name:'one',x:[0,1,2],y:[null,1,null]}]},340);
        assert.ok(single.includes('<circle'), 'An isolated valid sample must remain visible');
    """)
