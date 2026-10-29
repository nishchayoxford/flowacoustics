"""Standalone Case 4 browser contract; fixtures never enter the served site."""

import json
import math
import subprocess
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "site/case4-spectra.mjs"


def fixture_summary():
    """Deterministic, structural-only in-memory data, not CFD evidence."""
    frequencies = [0.625 + i * 0.125 for i in range(196)]
    rms = [[(j + 1) * (1e-5 if i == 35 else 1e-6) for j in range(12)] for i in range(196)]
    psd = [[v * v / 0.1875 for v in row] for row in rms]
    mean_square = [sum(row[j] ** 2 for row in rms) for j in range(12)]
    band = [math.sqrt(v) for v in mean_square]
    return {
        "schema": "mrl2d-case4-spectral-v1",
        "case_id": "mrl2d_les_permeable",
        "frequency_hz": frequencies,
        "angles_deg": [j * 30 for j in range(12)],
        "bin_rms_pa": rms,
        "psd_pa2_per_hz": psd,
        "band_rms_pa": band,
        "band_level_db_re_1upa": [20 * math.log10(v / 1e-6) for v in band],
        "peaks": [
            {
                "angle_deg": j * 30,
                "index": 35,
                "frequency_hz": 5,
                "bin_rms_pa": rms[35][j],
                "psd_pa2_per_hz": psd[35][j],
                "is_band_edge": False,
                "tied_maximum_count": 1,
            }
            for j in range(12)
        ],
        "estimator": {
            "sample_count": 4000,
            "sample_interval_s": 0.002,
            "frequency_spacing_hz": 0.125,
            "enbw_hz": 0.1875,
            "sample_window_s": [2.002, 10],
            "band_hz": [0.625, 25],
            "window": "periodic Hann",
            "psd_definition": "bin_rms_pa^2 / ENBW",
            "bin_rms_definition": "sqrt(2) abs(c)",
            "band_level_definition": "20 log10(band_rms_pa / 1e-6)",
            "psd_band_integral_pa2": [sum(row[j] * 0.125 for row in psd) for j in range(12)],
            "coherent_band_mean_square_pa2": mean_square,
        },
        "provenance": {
            "source": "site/data/mrl2d/mrl2d_les_permeable.npz",
            "sha256": "a" * 64,
        },
    }


def node_assert(body, fixture=True):
    assert MODULE.is_file(), "Case 4 standalone module is missing"
    source = "import assert from 'node:assert/strict';\n"
    source += "import * as ui from './site/case4-spectra.mjs';\n"
    if fixture:
        source += "const raw = " + json.dumps(fixture_summary()) + ";\n"
    source += body
    result = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_summary_admission_preserves_original_arrays():
    node_assert("""
        const data = ui.validateSummary(raw);
        assert.equal(data, raw);
        assert.equal(data.frequency_hz.length, 196);
        assert.equal(data.angles_deg.length, 12);
        assert.deepEqual(data.bin_rms_pa, raw.bin_rms_pa);
        assert.throws(() => ui.validateSummary(null), /Unavailable/);
        assert.throws(() => ui.validateSummary({...raw, case_id: 'cavitation'}), /case/);
        assert.throws(() => ui.validateSummary({...raw, schema: 'unknown'}), /schema/);
    """)


def test_inconsistent_spectra_metadata_peaks_and_provenance_fail_closed():
    node_assert("""
        const mutations = [
          d => d.frequency_hz.pop(), d => d.angles_deg.pop(),
          d => d.frequency_hz[3] = d.frequency_hz[2],
          d => d.angles_deg[1] = 0, d => d.angles_deg[1] = 30.5,
          d => d.bin_rms_pa.pop(), d => d.bin_rms_pa[0].pop(),
          d => d.psd_pa2_per_hz[0][0] = -1,
          d => d.bin_rms_pa[0][0] = NaN, d => d.band_rms_pa[0] = Infinity,
          d => d.bin_rms_pa[0][0] = '1', d => d.psd_pa2_per_hz[0][0] *= 1.5,
          d => d.band_rms_pa[0] *= 2, d => d.band_level_db_re_1upa[0] = null,
          d => d.band_level_db_re_1upa[0] += 1,
          d => d.estimator.sample_count = 3999,
          d => d.estimator.sample_interval_s = .003,
          d => d.estimator.frequency_spacing_hz = .1875,
          d => d.estimator.enbw_hz = .125,
          d => d.estimator.sample_window_s = [2,10],
          d => d.estimator.band_hz = [.5,25], d => d.estimator.window = 'Hann',
          d => delete d.estimator.psd_definition,
          d => d.estimator.psd_band_integral_pa2[0] *= 1.5,
          d => d.estimator.coherent_band_mean_square_pa2[0] *= 2,
          d => d.peaks.pop(), d => d.peaks[0].index = 34,
          d => d.peaks[0].frequency_hz = 4.99,
          d => d.peaks[0].angle_deg = 20, d => d.peaks[0].bin_rms_pa *= 2,
          d => d.peaks[0].psd_pa2_per_hz *= 2,
          d => d.peaks[0].is_band_edge = true,
          d => d.peaks[0].tied_maximum_count = 2,
          d => d.provenance = {}, d => d.provenance.sha256 = 'invalid',
          d => d.provenance.source = '/home/private/input.npz',
          d => d.provenance.source = '../private/input.npz',
        ];
        for (const mutate of mutations) {
          const invalid = structuredClone(raw); mutate(invalid);
          assert.throws(() => ui.validateSummary(invalid), undefined, String(mutate));
        }
        // Zero spectra stay zero and have no finite dB level or candidate peak.
        const zero = structuredClone(raw);
        zero.bin_rms_pa.forEach(row => row[0] = 0);
        zero.psd_pa2_per_hz.forEach(row => row[0] = 0);
        zero.band_rms_pa[0] = 0; zero.band_level_db_re_1upa[0] = null;
        zero.estimator.psd_band_integral_pa2[0] = 0;
        zero.estimator.coherent_band_mean_square_pa2[0] = 0;
        zero.peaks[0] = {angle_deg:0,index:null,frequency_hz:null,bin_rms_pa:0,
          psd_pa2_per_hz:0,is_band_edge:false,tied_maximum_count:0};
        assert.equal(ui.validateSummary(zero), zero);
    """)


def test_provenance_requires_the_original_case4_source_path():
    node_assert("""
        const orphan = structuredClone(raw);
        delete orphan.provenance.source;
        assert.throws(() => ui.validateSummary(orphan), /source/);
        const wrong = structuredClone(raw);
        wrong.provenance.source = 'site/data/cavitation-loading/primary-on.npz';
        assert.throws(() => ui.validateSummary(wrong), /source/);
        const nested = structuredClone(raw);
        nested.provenance = {sources: [{path: raw.provenance.source,
          sha256: raw.provenance.sha256}]};
        assert.equal(ui.validateSummary(nested), nested);
    """)


def test_polar_uses_absolute_db_ticks_and_unconnected_receivers():
    node_assert("""
        assert.equal(typeof ui.renderPolar, 'function', 'dB polar renderer is missing');
        const html = ui.renderPolar(raw, 'db');
        assert.equal((html.match(/data-polar-angle=/g) || []).length, 12);
        assert.ok(html.includes('dB re 1 µPa'));
        assert.ok(html.includes('band RMS'));
        assert.ok(!html.includes('%'));
        assert.ok(!html.includes('data-trace='));
        const ticks = [...html.matchAll(/data-radial-tick="([^"]+)"/g)].map(m => +m[1]);
        assert.ok(ticks.length >= 3);
        assert.ok(ticks.every((v, i) => !i || v > ticks[i-1]));
        for (const v of raw.band_level_db_re_1upa) assert.ok(html.includes(`data-value="${v}"`));
        // Below-reference levels must have finite positions, never a pressure floor.
        const low = structuredClone(raw);
        low.band_level_db_re_1upa = raw.angles_deg.map((_,j) => -40+j);
        low.band_level_db_re_1upa[0] = null;
        const negative = ui.renderPolar(low, 'db');
        assert.equal((negative.match(/data-polar-angle=/g) || []).length, 11);
        assert.ok(negative.includes('data-radial-tick="-40"'));
        assert.ok(negative.includes('zero RMS'));
        assert.ok(!negative.includes('NaN') && !negative.includes('Infinity'));
        assert.ok(ui.renderPolar(raw, 'pa').includes('Band RMS (Pa)'));
        assert.throws(() => ui.renderPolar(raw, 'percent'));
    """)


def test_every_observer_card_changes_metric_and_marks_sampled_peak():
    node_assert("""
        assert.equal(typeof ui.renderSpectra, 'function', 'spectral cards are missing');
        for (const metric of ['rms','psd']) {
          const html = ui.renderSpectra(raw, metric);
          assert.equal((html.match(/data-observer-card=/g) || []).length, 12);
          assert.equal((html.match(/data-spectrum-path=/g) || []).length, 12);
          assert.equal((html.match(/data-peak-marker=/g) || []).length, 12);
          assert.equal((html.match(/data-frequency-hz="5"/g) || []).length, 12);
          assert.ok(html.includes(metric === 'rms' ? 'Bin RMS (Pa)' : 'PSD (Pa²/Hz)'));
          const maxima = [...html.matchAll(/data-y-max="([^"]+)"/g)].map(m => +m[1]);
          assert.equal(new Set(maxima).size, 1, 'all cards share the same amplitude scale');
          for (let j=0; j<12; j++) {
            assert.ok(html.includes(`data-observer-card="${j*30}"`));
            const value = metric === 'rms' ? raw.peaks[j].bin_rms_pa : raw.peaks[j].psd_pa2_per_hz;
            assert.ok(html.includes(`data-amplitude="${value}"`));
          }
          const paths = [...html.matchAll(/data-spectrum-path="[^\"]+" d="([^\"]+)"/g)];
          assert.ok(paths.every(m => (m[1].match(/[ML]/g)||[]).length === 196));
          assert.ok(!html.includes('NaN') && !html.includes('Infinity'));
        }
        assert.throws(() => ui.renderSpectra(raw, 'db'));
        const zero = structuredClone(raw);
        zero.bin_rms_pa.forEach(row => row.fill(0));
        zero.peaks.forEach(p => {p.index=null;p.frequency_hz=null;p.bin_rms_pa=0;});
        const empty = ui.renderSpectra(zero, 'rms');
        assert.equal((empty.match(/data-peak-marker=/g)||[]).length, 0);
        assert.ok(empty.includes('No nonzero peak'));
    """)


def test_native_roundoff_angles_are_formatted_only_for_display():
    node_assert("""
        raw.angles_deg[1] = 29.999999999999996;
        raw.peaks[1].angle_deg = raw.angles_deg[1];
        ui.validateSummary(raw);
        for (const metric of ['rms', 'psd']) {
          const html = ui.renderSpectra(raw, metric);
          assert.ok(html.includes('<h3>Observer 30°</h3>'));
          assert.ok(html.includes('at 30 degrees'));
          assert.ok(html.includes('<title>Observer 30°'));
          assert.ok(html.includes('data-observer-card="29.999999999999996"'));
        }
        assert.ok(ui.renderPeakTable(raw).includes('<th scope="row">30°</th>'));
        assert.ok(ui.renderPolar(raw, 'db').includes('<title>30°:'));
        assert.equal(raw.angles_deg[1], 29.999999999999996);
    """)


def test_peak_table_preserves_twelve_observers_and_qualifications():
    node_assert("""
        assert.equal(typeof ui.renderPeakTable, 'function', 'peak table is missing');
        const html = ui.renderPeakTable(raw);
        assert.equal((html.match(/data-peak-row=/g)||[]).length, 12);
        for (const angle of raw.angles_deg) assert.ok(html.includes(`data-peak-row="${angle}"`));
        for (const text of ['Peak frequency (Hz)', 'Bin RMS (Pa)', 'PSD (Pa²/Hz)',
          'Band RMS (Pa)', 'dB re 1 µPa', 'scope="col"', 'scope="row"']) {
          assert.ok(html.includes(text), text);
        }
        const special = structuredClone(raw);
        special.peaks[0].is_band_edge = true;
        special.peaks[1].tied_maximum_count = 2;
        special.peaks[2].index = null; special.peaks[2].frequency_hz = null;
        special.band_level_db_re_1upa[2] = null;
        const qualified = ui.renderPeakTable(special);
        assert.ok(qualified.includes('Band-edge candidate'));
        assert.ok(qualified.includes('2 tied maxima'));
        assert.ok(qualified.includes('No nonzero peak'));
        assert.ok(qualified.includes('−∞ (zero RMS)'));
        assert.ok(!qualified.includes('Not recorded'));
    """)


def test_page_exposes_fixed_case_scope_definitions_controls_and_downloads():
    page = ROOT / "site/case4-spectra.html"
    assert page.is_file(), "Standalone Case 4 page is missing"
    html = page.read_text()

    class Elements(HTMLParser):
        def __init__(self):
            super().__init__()
            self.elements = []
            self.text = []

        def handle_starttag(self, tag, attrs):
            self.elements.append((tag, dict(attrs)))

        def handle_data(self, data):
            self.text.append(data)

    parsed = Elements()
    parsed.feed(html)
    text = " ".join(" ".join(parsed.text).split())
    ids = {attrs["id"]: attrs for _, attrs in parsed.elements if "id" in attrs}
    for key in [
        "main",
        "spectra-status",
        "reload-spectra",
        "spectra-results",
        "polar-db",
        "polar-pa",
        "observer-spectra",
        "peak-table",
        "metric-rms",
        "metric-psd",
    ]:
        assert key in ids, key
    assert "hidden" in ids["spectra-results"]
    assert ids["spectra-status"]["role"] == "status"
    assert ids["spectra-status"]["aria-live"] == "polite"
    assert ids["metric-rms"]["aria-pressed"] == "true"
    assert ids["metric-psd"]["aria-pressed"] == "false"
    for phrase in [
        "mrl2d_les_permeable",
        "single-phase",
        "WALE",
        "stationary permeable",
        "not the matched cavitation",
        "0.625–25 Hz",
        "2.002–10 s",
        "1 m",
        "0.1875 Hz",
        "0.125 Hz",
        "dB re 1 µPa",
        "Pa²/Hz",
        "periodic Hann",
        "2/3",
        "single-record",
        "not a Welch",
        "near-field",
        "span-invariant",
        "not proof of a blade-passing mechanism",
        "No second window",
    ]:
        assert phrase in text, phrase
    downloads = {
        attrs["href"] for tag, attrs in parsed.elements if tag == "a" and "download" in attrs
    }
    expected = {
        "summary.json",
        "spectra.csv",
        "peaks.csv",
        "directivity.csv",
        "plots.pdf",
        "directivity.png",
        "bin-rms.png",
        "psd.png",
        "manifest.json",
    }
    assert downloads == {f"./data/case4-spectra/{name}" for name in expected}
    scripts = [attrs for tag, attrs in parsed.elements if tag == "script"]
    assert scripts == [{"type": "module", "src": "./case4-spectra.mjs"}]


def test_browser_keyboard_metrics_mobile_layout_and_failed_reload():
    """Real Chromium, in-memory fixture only; never writes a served data file."""
    source = r"""
      import assert from 'node:assert/strict';
      import {createServer} from 'node:http';
      import {readFile,writeFile,mkdir,rm} from 'node:fs/promises';
      import {spawn} from 'node:child_process';
      import {once} from 'node:events';
      import {resolve,extname,join} from 'node:path';
      const out=resolve('.cache/mrl2d/case4-spectra-001/ui-evidence/browser-'+process.pid);
      await mkdir(out,{recursive:true});
      let stdin='';for await(const chunk of process.stdin)stdin+=chunk;
      const profile=join(out,'profile'),site=resolve('site'),errors=[],checks=[];
      const server=createServer(async(req,res)=>{
        const file=resolve(site,'.'+new URL(req.url,'http://local').pathname);
        try {
          assert.ok(file.startsWith(site+'/'));
          const body=await readFile(file);
          res.writeHead(200,{'Content-Type':({'.html':'text/html','.css':'text/css',
            '.mjs':'text/javascript'})[extname(file)]||'application/octet-stream'});
          res.end(body);
        } catch {res.writeHead(404);res.end('Missing');}
      });
      server.listen(0,'127.0.0.1');await once(server,'listening');
      const url=`http://127.0.0.1:${server.address().port}/case4-spectra.html`;
      const chrome=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox',
        '--disable-gpu','--disable-dev-shm-usage','--no-first-run','--remote-debugging-port=0',
        '--user-data-dir='+profile,'about:blank'],{stdio:'ignore',detached:true});
      const delay=ms=>new Promise(r=>setTimeout(r,ms));
      async function until(fn,label){
        for(let i=0;i<100;i++){try{if(await fn())return;}catch{}await delay(50);}
        throw Error('Timed out: '+label);
      }
      let ws,id=0;const pending=new Map();
      function send(method,params={}){
        return new Promise((ok,no)=>{const key=++id;pending.set(key,{ok,no});
          ws.send(JSON.stringify({id:key,method,params}));});
      }
      async function evaluate(expression){
        const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
        if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));
        return r.result.value;
      }
      async function key(selector){
        await evaluate(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});
          e.scrollIntoView({block:'center'});e.focus();})()`);
        await send('Input.dispatchKeyEvent',{type:'keyDown',key:'Enter',code:'Enter',
          text:'\r',windowsVirtualKeyCode:13});
        await send('Input.dispatchKeyEvent',{type:'keyUp',key:'Enter',code:'Enter',
          windowsVirtualKeyCode:13});
      }
      async function shot(name,selector){
        await evaluate(`document.querySelector(${JSON.stringify(selector)}).scrollIntoView()`);
        const {data}=await send('Page.captureScreenshot',{format:'png'});
        await writeFile(join(out,name+'.png'),Buffer.from(data,'base64'));
      }
      try {
        await until(async()=>{await readFile(join(profile,'DevToolsActivePort'));return true;},
          'isolated Chromium startup');
        const port=(await readFile(join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0];
        const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
        ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
        await once(ws,'open');
        ws.onmessage=({data})=>{const m=JSON.parse(data);
          if(m.id){const p=pending.get(m.id);pending.delete(m.id);
            if(m.error)p.no(Error(JSON.stringify(m.error)));else p.ok(m.result);}
          else if(m.method==='Runtime.exceptionThrown')errors.push(m.params);
        };
        await send('Page.enable');await send('Runtime.enable');
        await send('Page.addScriptToEvaluateOnNewDocument',{source:
          `window.testSummary=${stdin};window.fetch=async()=>new Response(
            JSON.stringify(window.testSummary),{status:200});`});
        await send('Page.navigate',{url});
        await until(()=>evaluate(`document.querySelector('#spectra-status')?.dataset.state
          ==='ready'`),'valid summary renders instead of remaining in loading');
        for(const width of [1440,390]) {
          await send('Emulation.setDeviceMetricsOverride',{
            width,height:900,deviceScaleFactor:1,mobile:false});
          for(const metric of ['psd','rms']) {
            await key('#metric-'+metric);
            assert.equal(await evaluate(`document.querySelector('#metric-${metric}')
              .getAttribute('aria-pressed')`),'true');
            assert.equal(await evaluate(`document.activeElement.id`),'metric-'+metric);
            assert.equal(await evaluate(`document.querySelectorAll(
              '[data-observer-card][data-metric="${metric}"]').length`),12);
            assert.equal(await evaluate(`document.querySelectorAll('[data-peak-row]').length`),12);
            assert.equal(await evaluate(`document.querySelectorAll(
              '[data-peak-marker]').length`),12);
            assert.equal(await evaluate(`document.querySelectorAll(
              '#polar-db [data-polar-angle]').length`),12);
            assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true,
              'no document-wide horizontal overflow');
            const clipped=await evaluate(`(()=>{const bad=[];
              for(const svg of document.querySelectorAll('#spectra-results svg')){
                const box=svg.getBoundingClientRect();
                for(const t of svg.querySelectorAll('text')){
                  const r=t.getBoundingClientRect();
                  if(r.left<box.left-1||r.right>box.right+1||r.top<box.top-1
                    ||r.bottom>box.bottom+1)bad.push(t.textContent);
                  if(t.classList.contains('radial-label')) {
                    for(const marker of svg.querySelectorAll('[data-polar-angle]')) {
                      const m=marker.getBoundingClientRect();
                      if(Math.min(r.right,m.right)-Math.max(r.left,m.left)>1
                        &&Math.min(r.bottom,m.bottom)-Math.max(r.top,m.top)>1)
                        bad.push('radial label covers receiver '+t.textContent);
                    }
                  }
                }
              }return bad;})()`);
            assert.deepEqual(clipped,[],'axis labels fit in every rendered SVG');
            await shot(`${width}-${metric}`,'#spectra');
            checks.push({width,metric,cards:12,peaks:12,noOverflow:true});
          }
          await shot(width+'-polar','#directivity');
        }
        assert.equal(await evaluate(`document.querySelectorAll(
          '#downloads a[download]').length`),9);
        // A held reload clears all stale charts before any response is available.
        await evaluate(`window.fetch=()=>new Promise(resolve=>window.resolveTestFetch=resolve)`);
        await key('#reload-spectra');
        assert.equal(await evaluate(`document.querySelector('#spectra-status').dataset.state`),
          'loading');
        assert.equal(await evaluate(`document.querySelectorAll('[data-observer-card]').length`),0);
        assert.equal(await evaluate(`document.querySelector('#spectra-results').hidden`),true);
        await evaluate(`window.resolveTestFetch(new Response('',{status:404}))`);
        await until(()=>evaluate(`document.querySelector('#spectra-status').dataset.state
          ==='unavailable'`),'honest missing summary');
        for(const response of ["new Response('{',{status:200})",
          "new Response(JSON.stringify({...window.testSummary,case_id:'wrong'}),{status:200})"]) {
          await evaluate('window.fetch=async()=>'+response);await key('#reload-spectra');
          await until(()=>evaluate(`document.querySelector('#spectra-status').dataset.state
            ==='error'`),'malformed summary fails closed');
          assert.equal(await evaluate(`document.querySelectorAll(
            '[data-observer-card]').length`),0);
          assert.equal(await evaluate(`document.querySelector('#metric-psd').disabled`),true);
        }
        await evaluate(`window.fetch=async()=>new Response(JSON.stringify(window.testSummary))`);
        await key('#reload-spectra');
        await until(()=>evaluate(`document.querySelector('#spectra-status').dataset.state
          ==='ready'`),'retry restores valid original data');
        assert.deepEqual(errors,[]);
        await writeFile(join(out,'report.json'),JSON.stringify({checks,errors,passed:true},null,2));
        console.log(JSON.stringify({checks,errors,passed:true,evidence:out}));
      } finally {
        ws?.close();try{process.kill(-chrome.pid,'SIGTERM');}catch{}
        await new Promise(r=>server.close(r));await delay(200);
        await rm(profile,{recursive:true,force:true});
      }
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-e", source],
        input=json.dumps(fixture_summary()),
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
