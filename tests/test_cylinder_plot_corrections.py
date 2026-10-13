"""Audit regressions: analytical fixtures stay in memory; native records are read-only."""

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from flowacoustics.cylinder_acoustics import build_dataset


@pytest.fixture(scope="module")
def native_dataset():
    """Rebuild the frozen audited 6 s data in memory, never publish it."""
    baseline = ROOT / "docs/cylinder_60s/plot_audit/baseline/site/data/results.json"
    if not baseline.is_file():
        pytest.skip("Optional frozen 6 s native audit archive is not shipped publicly")
    meta = json.loads(baseline.read_text())["metadata"]
    source = ROOT / "results/cylinder/analysis/circular_v1"
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ("forces.csv", "observers.csv"):
        expected = next(x["sha256"] for x in manifest["outputs"] if x["path"].endswith(name))
        assert hashlib.sha256((source / "downloads" / name).read_bytes()).hexdigest() == expected
    run = {
        key: meta[key]
        for key in (
            "medium",
            "geometry",
            "force_convention",
            "force_content",
            "clock",
            "evidence_kind",
        )
    }
    run.update(
        schema_version=1,
        status=meta["native_status"],
        observers=[
            {"name": name, "position_m": info["position_m"]}
            for name, info in meta["comparisons"].items()
        ],
    )
    with (source / "downloads/forces.csv").open() as stream:
        force = np.array(
            [
                [float(row[k]) for k in ("time_s", "force_x_N", "force_y_N", "force_z_N")]
                for row in csv.DictReader(stream)
            ]
        )
    with (source / "downloads/observers.csv").open() as stream:
        observers = list(csv.DictReader(stream))
    data, _ = build_dataset(run, force, observers, **manifest["analysis_options"])
    return data


def test_native_spectrogram_builder_sets_sequential_shared_unit_scale(native_dataset):
    charts = [c for c in native_dataset["charts"] if c["id"].startswith("spectrogram_")]
    assert len(charts) == 5
    bound = max(value for chart in charts for row in chart["z"] for value in row)
    assert bound > 0
    for chart in charts:
        assert chart.get("color_label") == "PSD (Pa²/Hz)"
        assert chart.get("diverging") is False
        assert chart.get("color_limit") == bound
    node(
        """
        import assert from 'node:assert/strict';
        import fs from 'node:fs';
        import {renderPlot,formatNumber} from './site/plots.mjs';
        for(const chart of JSON.parse(fs.readFileSync(0,'utf8'))){
          const html=renderPlot(chart,340);
          const spans=[...html.matchAll(/<span>(.*?)<\\/span>/g)].map(m=>m[1]);
          assert.equal(spans[0],'0','sequential legend lower endpoint');
          assert.equal(spans[1],formatNumber(chart.color_limit),'upper endpoint only');
          assert.equal(spans[2],'Color range · PSD (Pa²/Hz)');
        }
        """,
        payload=charts,
    )


def test_native_msc_uses_full_zero_to_one_domain(native_dataset):
    charts = [c for c in native_dataset["charts"] if c["id"].startswith("coherence_")]
    assert len(charts) == 5
    for chart in charts:
        assert chart.get("y_domain") == [0, 1]
        assert "not a significance" in chart["description"]
    node(
        """
        import assert from 'node:assert/strict';
        import fs from 'node:fs';
        import {renderPlot} from './site/plots.mjs';
        for(const chart of JSON.parse(fs.readFileSync(0,'utf8'))){
          const html=renderPlot(chart,340);
          const ticks=[...html.matchAll(/<text x="54"[^>]*text-anchor="end"[^>]*>(.*?)<\\/text>/g)]
            .map(m=>m[1]);
          assert.deepEqual(ticks,['0','0.25','0.5','0.75','1']);
        }
        """,
        payload=charts,
    )


def test_polar_270_label_has_separate_caption_row_in_browser():
    from test_cylinder_site import CDP, isolated_browser

    with isolated_browser() as env:
        node(
            CDP
            + """
            for(const width of [1440,390]){
              await call('Emulation.setDeviceMetricsOverride',{
                width,height:1000,deviceScaleFactor:1,mobile:width<500});
              await new Promise(r=>setTimeout(r,350));
              const checks=await evaluate(`Array.from(document.querySelectorAll('.plot svg'))
                .filter(s=>Array.from(s.querySelectorAll('text')).some(t=>t.textContent==='270°'))
                .map(s=>{
                  const texts=Array.from(s.querySelectorAll('text'));
                  const angle=texts.find(t=>t.textContent==='270°').getBoundingClientRect();
                  const caption=texts.at(-1).getBoundingClientRect();
                  return {title:s.getAttribute('aria-label'),gap:caption.top-angle.bottom};
                })`);
              assert.equal(checks.length,4);
              for(const check of checks)assert.ok(check.gap>=6,
                `${width}: ${check.title}: caption overlaps 270° (${check.gap})`);
            }
            assert.deepEqual(exceptions,[]);
            ws.close();
            """,
            env=env,
        )


def test_distance_ratio_exposes_reactive_and_radiating_references(native_dataset):
    chart = next(c for c in native_dataset["charts"] if c["id"] == "distance_ratios")
    refs = chart.get("reference_traces", [])
    assert [r["y"] for r in refs] == [[0.25], [0.5]]
    assert chart["y_domain"][0] == 0
    assert chart["y_domain"][1] > 0.5
    assert "one actual receiver pair" in chart["description"]
    rows = chart["comparison_rows"]
    assert len(rows) == 3
    for row, pair in zip(rows, native_dataset["metadata"]["distance_pairs"]):
        assert row["ratio"] == pair["rms_ratio_far_over_near"]
        assert row["delta_reactive"] == row["ratio"] - 0.25
    node(
        """
        import assert from 'node:assert/strict';
        import fs from 'node:fs';
        import {renderPlot} from './site/plots.mjs';
        const chart=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(chart);
        const html=renderPlot(chart,340);
        assert.equal((html.match(/<circle /g)||[]).length,5,'three models plus two references');
        assert.ok(html.includes('Reactive 1/r² reference: 0.25'));
        assert.ok(html.includes('Radiating 1/r reference: 0.5'));
        assert.ok(html.includes('<table'),'separate precise residual table');
        assert.ok(html.includes('Δ from reactive'));
        for(const row of chart.comparison_rows)assert.ok(html.includes(row.ratio.toPrecision(9)));
        assert.equal(JSON.stringify(chart),before);
        """,
        payload=chart,
    )


def test_large_range_endpoint_tick_stays_inside_browser_svg():
    from test_cylinder_site import CDP, isolated_browser

    with isolated_browser() as env:
        node(
            CDP
            + """
            for(const width of [1440,390]){
              await call('Emulation.setDeviceMetricsOverride',{
                width,height:1000,deviceScaleFactor:1,mobile:width<500});
              await new Promise(r=>setTimeout(r,350));
              const check=await evaluate(`(()=>{
                const svg=document.querySelector('[data-chart-id="lesson_near_far"] svg');
                const tick=Array.from(svg.querySelectorAll('text'))
                  .find(t=>t.textContent==='1.0e+5');
                const a=svg.getBoundingClientRect(),b=tick.getBoundingClientRect();
                return {left:b.left-a.left,right:a.right-b.right};
              })()`);
              assert.ok(check.left>=0 && check.right>=0,
                `${width}: endpoint tick exceeds SVG ${JSON.stringify(check)}`);
            }
            assert.deepEqual(exceptions,[]);
            ws.close();
            """,
            env=env,
        )


def test_phase_wraps_break_without_changing_gates_or_samples(native_dataset):
    node(
        """
        import assert from 'node:assert/strict';
        import {renderPlot} from './site/plots.mjs';
        const chart={kind:'line',title:'Wrapped phase',x_label:'Hz',y_label:'Phase (rad)',
          phase_wrap_period:2*Math.PI,traces:[{name:'phase',
          x:[0,1,2,3,4,5,6,7,8],y:[2.8,3.1,-3.1,-2.8,null,.2,null,3,-3]}]};
        const before=JSON.stringify(chart),html=renderPlot(chart,340);
        const path=[...html.matchAll(/<path d="([^"]*)"[^>]*stroke=/g)].map(m=>m[1]).join('');
        assert.equal((path.match(/L/g)||[]).length,2,'never bridge ±π wraps');
        assert.equal((path.match(/M/g)||[]).length,2,'restart on the other phase branch');
        assert.equal((html.match(/<circle /g)||[]).length,3,'isolated gated/wrapped samples');
        assert.equal(JSON.stringify(chart),before,'keep numeric phase and null gates unchanged');
        """
    )
    phases = [c for c in native_dataset["charts"] if c["id"].startswith("phase_")]
    assert len(phases) == 5
    assert all(c.get("phase_wrap_period") == 2 * math.pi for c in phases)


def test_presentation_metadata_preserves_every_native_chart_array(native_dataset):
    """Verification-only: compare full arrays with the immutable historical analysis."""
    old = json.loads((ROOT / "results/cylinder/analysis/circular_v1/results.json").read_text())
    current = {c["id"]: c for c in native_dataset["charts"]}
    assert len(current) == len(old["charts"]) == 39
    for chart in old["charts"]:
        for key in ("traces", "x", "y", "z", "samples"):
            if key in chart:
                assert current[chart["id"]][key] == chart[key], (chart["id"], key)
    for key in ("comparisons", "distance_pairs", "spectral_estimator"):
        assert native_dataset["metadata"][key] == old["metadata"][key]


def test_real_dataset_rendered_at_desktop_and_mobile(native_dataset):
    """Verification-only integration: actual frozen native data, fetch boundary only."""
    import importlib.util

    from test_cylinder_site import CDP, isolated_browser

    spec = importlib.util.spec_from_file_location(
        "display", ROOT / "scripts/prepare_cylinder_site.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    display = module.make_display_dataset(native_dataset)
    with isolated_browser() as env:
        env["PLOT_FIX_REPORT"] = str(REPORT)
        crops = node(
            CDP
            + r"""
            const data=JSON.parse(fs.readFileSync(0,'utf8'));
            const out=process.env.PLOT_FIX_REPORT;
            const crops=[];
            fs.mkdirSync(out+'/renders',{recursive:true});
            const encoded=JSON.stringify(JSON.stringify(data));
            await evaluate(`window.fetch=async()=>new Response(${encoded},
              {status:200,headers:{'Content-Type':'application/json'}});
              document.getElementById('retry-results').click()`);
            for(let i=0;i<100;i++){
              if(await evaluate(`!document.getElementById('retry-results').disabled`))break;
              await new Promise(r=>setTimeout(r,80));
            }
            const report={provenance:'Frozen audited 6 s native data rebuilt in memory; '+
              'not the active 10 s run',widths:{}};
            for(const width of [1440,390]){
              await call('Emulation.setDeviceMetricsOverride',{
                width,height:1000,deviceScaleFactor:1,mobile:width<500});
              await new Promise(r=>setTimeout(r,400));
              const checks=await evaluate(`(()=>{
                const boxes=Array.from(document.querySelectorAll('.plot svg')).map(svg=>{
                  const box=svg.getBoundingClientRect();
                  const texts=Array.from(svg.querySelectorAll('text'));
                  const clipped=texts.filter(t=>{
                    const b=t.getBoundingClientRect();return b.left<box.left-.5||
                      b.right>box.right+.5||b.top<box.top-.5||b.bottom>box.bottom+.5;
                  }).map(t=>t.textContent);
                  const angle=texts.find(t=>t.textContent==='270°');
                  return {id:svg.closest('[data-chart-id]')?.dataset.chartId||svg.parentElement.id,
                    clipped,nonfinite:/NaN|Infinity/.test(svg.innerHTML),
                    polarGap:angle?texts.at(-1).getBoundingClientRect().top-
                      angle.getBoundingClientRect().bottom:null};
                });
                const pick=id=>document.querySelector('[data-chart-id="'+id+'"]');
                const maps=Array.from(document.querySelectorAll('[data-chart-id^="spectrogram_"]'))
                  .map(f=>({id:f.dataset.chartId,key:f.querySelector('.color-key').textContent,
                    ends:Array.from(f.querySelectorAll('.color-key span')).map(s=>s.textContent)}));
                const coherence=Array.from(document.querySelectorAll(
                  '[data-chart-id^="coherence_"]')).map(f=>({id:f.dataset.chartId,
                    ticks:Array.from(f.querySelectorAll('svg text[x="54"]'))
                      .map(t=>t.textContent)}));
                const polar=pick('directivity_0');
                return {boxes,maps,coherence,documentWidth:document.documentElement.scrollWidth,
                  markers:polar.querySelectorAll('circle[data-observer]').length,
                  polarPaths:polar.querySelectorAll('path[stroke]').length,
                  ratios:pick('distance_ratios').querySelectorAll('circle').length,
                  ratioTable:pick('distance_ratios').querySelector('table').textContent};
              })()`);
              assert.equal(checks.boxes.length,52);
              assert.ok(checks.documentWidth<=width,'no horizontal document overflow');
              assert.equal(checks.markers,12);assert.equal(checks.polarPaths,0);
              assert.equal(checks.ratios,5);
              assert.ok(checks.ratioTable.includes('0.250015898'));
              assert.equal(checks.maps.length,5);assert.equal(checks.coherence.length,5);
              assert.equal(new Set(checks.maps.map(m=>m.ends[1])).size,1);
              for(const map of checks.maps){
                assert.equal(map.ends[0],'0');
                assert.equal(map.ends[2],'Color range · PSD (Pa²/Hz)');
                assert.ok(!map.key.includes('−')&&!map.key.includes('+'));
              }
              for(const c of checks.coherence)
                assert.deepEqual(c.ticks,['0','0.25','0.5','0.75','1']);
              for(const box of checks.boxes){
                assert.deepEqual(box.clipped,[],`${width}: ${box.id} clipped labels`);
                assert.equal(box.nonfinite,false);
                if(box.polarGap!==null)assert.ok(box.polarGap>=6);
              }
              report.widths[width]=checks;
              const ids=['directivity_0','distance_ratios','lesson_near_far','lesson_directivity',
                ...checks.maps.map(m=>m.id),...checks.coherence.map(c=>c.id)];
              for(const id of ids){
                await evaluate(`document.querySelector('[data-chart-id="${id}"] .plot')
                  .scrollIntoView({behavior:'instant',block:'center'})`);
                await evaluate(`new Promise(resolve=>requestAnimationFrame(
                  ()=>requestAnimationFrame(resolve)))`);
                await new Promise(r=>setTimeout(r,100));
                const clip=await evaluate(`(()=>{
                  const p=document.querySelector('[data-chart-id="${id}"] .plot');
                  const b=p.getBoundingClientRect();
                  return {x:b.left,y:b.top,width:b.width,height:b.height};
                })()`);
                assert.ok(clip.y>=0&&clip.y+clip.height<=1000,'capture fits viewport');
                const shot=await call('Page.captureScreenshot',{format:'png'});
                const filename=`${out}/renders/${width}_${id}.png`;
                fs.writeFileSync(filename,Buffer.from(shot.data,'base64'));
                crops.push({filename,...clip});
              }
            }
            assert.deepEqual(exceptions,[]);
            report.exceptions=exceptions;
            fs.writeFileSync(out+'/browser_checks.json',JSON.stringify(report,null,2));
            ws.close();
            console.log(JSON.stringify(crops));
            """,
            payload=display,
            env=env,
        )
        # Crop actual viewport captures rather than CDP beyond-viewport repositioning.
        from PIL import Image

        assert isinstance(crops, list) and len(crops) == 28
        for crop in crops:
            with Image.open(crop["filename"]) as image:
                image.crop(
                    (
                        round(crop["x"]),
                        round(crop["y"]),
                        round(crop["x"] + crop["width"]),
                        round(crop["y"] + crop["height"]),
                    )
                ).save(crop["filename"])


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/cylinder_60s/plot_fixes"


def node(program, *, payload=None, env=None):
    result = subprocess.run(
        ["node", "--input-type=module", "-e", program],
        cwd=ROOT,
        input=json.dumps(payload, default=lambda x: x.item()) if payload is not None else None,
        text=True,
        capture_output=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout) if result.stdout.strip() else None


def test_signed_harmonic_force_fixes_relative_reactive_phase():
    # Independent force-on-fluid: F=-4π cos(ωτ)/ω, c*=1.
    # Its exact derivative is +4π sin(ωτ). Neither uses fieldPressure.
    cases = []
    for f in (0.4, 1.1):
        omega = 2 * math.pi * f
        for r in (0.7, 2.0):
            for theta in (0.0, 0.8, math.pi):
                for phase in (0.0, 0.4, math.pi / 2, math.pi, 4.4):
                    tau = phase / omega
                    force = -4 * math.pi * math.cos(phase) / omega
                    derivative = 4 * math.pi * math.sin(phase)
                    projection = math.cos(theta)
                    near = projection * force / (4 * math.pi * r**2)
                    far = projection * derivative / (4 * math.pi * r)
                    cases.append(dict(f=f, r=r, theta=theta, t=tau + r, near=near, far=far))
    node(
        """
        import assert from 'node:assert/strict';
        import fs from 'node:fs';
        import {fieldPressure} from './site/physics.mjs';
        const cases=JSON.parse(fs.readFileSync(0,'utf8'));
        for(const c of cases){
          const far=fieldPressure({...c,near:false});
          const total=fieldPressure({...c,near:true});
          assert.ok(Math.abs(far-c.far)<1e-12,'signed radiating pressure');
          assert.ok(Math.abs(total-far-c.near)<1e-12,'signed reactive pressure');
          assert.ok(Math.abs(total-c.near-c.far)<1e-12,'signed total pressure');
        }
        """,
        payload=cases,
    )
    html = (ROOT / "site/index.html").read_text()
    assert "sin(ω*τ*) − cos(ω*τ*)/(kr)" in html
    assert "force proportional to −cos(ω*τ*)" in html


def test_sparse_polar_is_four_identified_markers_without_paths_per_trace():
    node(
        """
        import assert from 'node:assert/strict';
        import {renderPlot} from './site/plots.mjs';
        const names=['cross_minus','downstream','oblique','cross_plus'];
        const chart={kind:'polar',title:'Actual receivers',connect_points:false,
          x_label:'Azimuth (degrees)',y_label:'AC RMS pressure (Pa)',
          samples:{observer_names:names},traces:['compact','Curle','FW-H'].map(name=>({
            name,x:[270,0,53.130102,90],y:[1,0.2,0.8,1]}))};
        const before=JSON.stringify(chart), html=renderPlot(chart,340);
        assert.equal((html.match(/<circle[^>]*data-observer=/g)||[]).length,12,
          'four independent markers per trace');
        assert.equal((html.match(/<path[^>]*stroke=/g)||[]).length,0,
          'sparse samples must not have inter-observer paths');
        for(const name of names){
          assert.equal((html.match(new RegExp('data-observer="'+name+'"','g'))||[]).length,3);
          assert.ok(html.includes(name+' ·'),'visible observer key');
        }
        assert.equal(JSON.stringify(chart),before,'renderer must not mutate arrays');
        const dense=renderPlot({...chart,connect_points:true},340);
        assert.equal((dense.match(/<path[^>]*stroke=/g)||[]).length,3);
        """
    )
