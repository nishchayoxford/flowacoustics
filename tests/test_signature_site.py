"""Publication/UI contract tests; historical assets never enter the public site."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/cylinder/signature/historical_smoke/report"
SCRIPT = ROOT / "scripts/prepare_signature_site.py"
SMOKE = ROOT / ".cache/signature-site-smoke"


@pytest.fixture(autouse=True)
def require_optional_historical_report(request):
    """Skip archive-bound integration only; never synthesize native evidence."""
    if "REPORT" in request.function.__code__.co_names and not (REPORT / "manifest.json").is_file():
        pytest.skip("Optional historical native signature report is not shipped publicly")


def adapter():
    assert SCRIPT.is_file(), "The safe signature publication adapter is missing"
    spec = importlib.util.spec_from_file_location("signature_site", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(data):
    return hashlib.sha256(data).hexdigest()


@contextmanager
def scratch():
    SMOKE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-only-", dir=SMOKE) as directory:
        yield Path(directory)


def test_historical_smoke_bundle_preserves_numerical_bytes_and_sanitizes_provenance():
    m = adapter()
    root = json.loads((REPORT / "manifest.json").read_text())
    with scratch() as work:
        output = work / "bundle"
        m.build_bundle(REPORT, output, root["run_sha256"], historical_smoke=True)
        index = json.loads((output / "index.json").read_text())
        assert index["source_kind"] == "historical_smoke_only"
        assert index["evidence_label"] == "HISTORICAL SMOKE ONLY"
        assert index["native_run_sha256"] == root["run_sha256"]
        assert index["figure_count"] == len(index["figures"]) == 9
        assert [f["id"] for f in index["figures"]] == [f["id"] for f in root["figures"]]
        copied = {
            k: v
            for k, v in root["output_sha256"].items()
            if Path(k).suffix in {".csv", ".png", ".pdf", ".svg"}
        }
        assert index["output_sha256"] == copied
        for name, expected in copied.items():
            assert digest((output / name).read_bytes()) == expected
        assert not (output / "run_metadata.json").exists()
        for figure in index["figures"]:
            original = json.loads((REPORT / figure["id"] / "manifest.json").read_text())
            assert figure["title"] == original["title"]
            assert figure["caption"] == original["caption"]
            assert figure["evidence_label"] == "HISTORICAL SMOKE ONLY"
            assert figure["outputs"]["png"]["width"] == 1728
            assert figure["outputs"]["png"]["height"] in {928, 1184}
            public = json.loads((output / figure["provenance"]).read_text())
            assert public["equations"] == original["equations"]
            assert public["window"] == original["window"]
            assert public["native_run_sha256"] == root["run_sha256"]
            assert not (output / figure["id"] / "manifest.json").exists()
        for path in output.rglob("*.json"):
            text = path.read_text()
            assert "/home/" not in text and str(ROOT) not in text
            assert '"sources"' not in text and '"source_metadata"' not in text
        assert not list(work.glob(".signature-site-stage-*"))


@pytest.mark.parametrize("expected", ["", "a" * 63, "G" * 64, "0" * 64, " " + "a" * 64])
def test_run_identity_gate_rejects_invalid_or_mismatched_digest_without_writes(expected):
    with scratch() as work:
        output = work / "absent-parent" / "bundle"
        with pytest.raises(ValueError, match="SHA|sha|digest"):
            adapter().build_bundle(REPORT, output, expected, historical_smoke=True)
        assert not output.parent.exists()


@pytest.mark.parametrize(
    "historical,location", [(False, "cache"), (True, "site"), (True, "outside")]
)
def test_historical_label_cannot_be_published_outside_explicit_smoke_root(historical, location):
    root = json.loads((REPORT / "manifest.json").read_text())
    with scratch() as work:
        m = adapter()
        m.ROOT = work / "test-only-project"
        output = {
            "cache": m.ROOT / ".cache/signature-site-smoke/bundle",
            "site": m.ROOT / "site/bundle",
            "outside": work / "outside/bundle",
        }[location]
        try:
            with pytest.raises(ValueError, match="historical|smoke|output|traversal"):
                m.build_bundle(REPORT, output, root["run_sha256"], historical_smoke=historical)
            assert not output.exists()
        finally:
            if output.exists():
                shutil.rmtree(output)


def rewrite_json(path, value):
    path.write_text(json.dumps(value))


@pytest.mark.parametrize(
    "fault",
    [
        "empty-pins",
        "missing-pins",
        "missing-pin",
        "bad-pin",
        "count",
        "missing-figure",
        "duplicate-id",
        "wrong-id",
        "traversal",
        "absolute",
        "symlink",
        "parent-symlink",
        "corrupt-png",
        "corrupt-metadata",
        "perfigure-id",
        "perfigure-output-pin",
        "perfigure-run-pin",
        "actual-run-bytes",
        "nonfinite",
        "private-public-text",
        "unexpected-asset",
        "wrong-evidence-label",
    ],
)
def test_report_integrity_preflight_leaves_no_destination_or_staging(fault):
    with scratch() as work:
        source = work / "report"
        shutil.copytree(REPORT, source)
        root = json.loads((source / "manifest.json").read_text())
        item = root["figures"][0]
        perpath = source / item["manifest"]
        per = json.loads(perpath.read_text())
        png = "I01_geometry/figure.png"
        if fault == "empty-pins":
            root["output_sha256"] = {}
        elif fault == "missing-pins":
            root.pop("output_sha256")
        elif fault == "missing-pin":
            root["output_sha256"].pop(png)
        elif fault == "bad-pin":
            root["output_sha256"][png] = "bad"
        elif fault == "count":
            root["figure_count"] = 8
        elif fault == "missing-figure":
            root["figures"].pop()
        elif fault == "duplicate-id":
            root["figures"][-1] = root["figures"][0]
        elif fault == "wrong-id":
            item["id"] = "I00_unknown"
        elif fault in {"traversal", "absolute"}:
            root["output_sha256"][
                "../outside.png" if fault == "traversal" else str(work / "outside.png")
            ] = digest(b"test")
            (work / "outside.png").write_bytes(b"test")
        elif fault == "symlink":
            (source / png).unlink()
            (source / png).symlink_to(REPORT / png)
        elif fault == "parent-symlink":
            shutil.rmtree(source / "I01_geometry")
            (source / "I01_geometry").symlink_to(REPORT / "I01_geometry")
        elif fault == "corrupt-png":
            (source / png).write_bytes(b"not the verified bytes")
        elif fault == "corrupt-metadata":
            (source / "run_metadata.json").write_bytes(b"{}")
        elif fault == "perfigure-id":
            per["figure_id"] = "I00_unknown"
        elif fault == "perfigure-output-pin":
            per["outputs"]["png"]["sha256"] = "0" * 64
        elif fault == "perfigure-run-pin":
            for pin in per["sources"].values():
                if pin["identity"] == "run_manifest":
                    pin["sha256"] = "0" * 64
        elif fault == "actual-run-bytes":
            old = next(k for k, v in per["sources"].items() if v["identity"] == "run_manifest")
            per["sources"][str(work / "fake-run.json")] = per["sources"].pop(old)
            (work / "fake-run.json").write_text('{"test_fixture_only": true}')
        elif fault == "nonfinite":
            per["estimator"]["dt_s"] = float("inf")
        elif fault == "private-public-text":
            per["title"] = "/home/private/data"
        elif fault == "unexpected-asset":
            (source / "paper.pdf").write_bytes(b"not a figure")
            root["output_sha256"]["paper.pdf"] = digest(b"not a figure")
        elif fault == "wrong-evidence-label":
            per["evidence_label"] = "NATIVE COMPLETED SHORT RECORD"
        if fault.startswith("perfigure") or fault in {
            "actual-run-bytes",
            "nonfinite",
            "private-public-text",
            "wrong-evidence-label",
        }:
            rewrite_json(perpath, per)
            root["output_sha256"][item["manifest"]] = digest(perpath.read_bytes())
        rewrite_json(source / "manifest.json", root)
        output = work / "absent-parent" / "bundle"
        with pytest.raises((ValueError, OSError)):
            adapter().build_bundle(source, output, root["run_sha256"], historical_smoke=True)
        assert not output.parent.exists()


@pytest.mark.parametrize(
    "fault", ["existing", "symlink-destination", "symlink-ancestor", "source-tree"]
)
def test_output_cannot_overwrite_or_follow_links_or_touch_source(fault):
    m = adapter()
    root = json.loads((REPORT / "manifest.json").read_text())
    with scratch() as work:
        output = work / "bundle"
        if fault == "existing":
            output.mkdir()
        if fault == "symlink-destination":
            output.symlink_to(work / "missing")
        if fault == "symlink-ancestor":
            (work / "real").mkdir()
            (work / "link").symlink_to(work / "real")
            output = work / "link" / "bundle"
        source = REPORT
        if fault == "source-tree":
            source = work / "report"
            shutil.copytree(REPORT, source)
            output = source / "child"
        with pytest.raises((ValueError, OSError)):
            m.build_bundle(source, output, root["run_sha256"], historical_smoke=True)
        assert not list(work.rglob(".signature-site-stage-*"))


def test_relabelling_old_report_cannot_create_native_evidence():
    with scratch() as work:
        source = work / "report"
        shutil.copytree(REPORT, source)
        root = json.loads((source / "manifest.json").read_text())
        root["evidence_label"] = "native_completed_short_record"
        for item in root["figures"]:
            path = source / item["manifest"]
            per = json.loads(path.read_text())
            per["evidence_label"] = "NATIVE COMPLETED SHORT RECORD"
            rewrite_json(path, per)
            root["output_sha256"][item["manifest"]] = digest(path.read_bytes())
        rewrite_json(source / "manifest.json", root)
        with pytest.raises(ValueError, match="native|Native|schema|record|label"):
            adapter().build_bundle(source, work / "bundle", root["run_sha256"])
        assert not (work / "bundle").exists()


def test_native_schema_contract_fixture_is_accepted_only_as_test_in_cache():
    """TEST ONLY contract fixture; historical plots retain their smoke watermark."""
    with scratch() as work:
        source = work / "TEST-ONLY-report"
        shutil.copytree(REPORT, source)
        native = {
            "schema_version": 2,
            "status": "completed_native_reconstructed",
            "exports": {"status": "completed"},
            "reconstruction": {"status": "completed"},
            "evidence_kind": "native_execution",
            "note": "TEST ONLY SCHEMA FIXTURE, NOT A CFD RUN",
        }
        run_path = work / "TEST-ONLY-run.json"
        rewrite_json(run_path, native)
        expected = digest(run_path.read_bytes())
        root = json.loads((source / "manifest.json").read_text())
        root.update(evidence_label="native_completed_short_record", run_sha256=expected)
        rewrite_json(source / "run_metadata.json", native)
        root["output_sha256"]["run_metadata.json"] = digest(
            (source / "run_metadata.json").read_bytes()
        )
        for item in root["figures"]:
            path = source / item["manifest"]
            per = json.loads(path.read_text())
            per["evidence_label"] = "10 s native record · descriptive only"  # actual producer label
            per["title"] = "TEST ONLY SCHEMA FIXTURE — " + per["title"]
            per["sources"] = {str(run_path): {"identity": "run_manifest", "sha256": expected}}
            rewrite_json(path, per)
            root["output_sha256"][item["manifest"]] = digest(path.read_bytes())
        rewrite_json(source / "manifest.json", root)
        result = adapter().build_bundle(source, work / "TEST-ONLY-bundle", expected)
        assert result["source_kind"] == "native_completed_short_record"
        assert result["native_run_sha256"] == expected
        assert all("TEST ONLY" in f["title"] for f in result["figures"])


def test_cli_requires_identity_and_builds_only_explicit_historical_smoke():
    root = json.loads((REPORT / "manifest.json").read_text())
    with scratch() as work:
        args = [str(SCRIPT), "--report", str(REPORT), "--output", str(work / "bundle")]
        missing = subprocess.run(
            [__import__("sys").executable, *args], capture_output=True, text=True
        )
        assert missing.returncode != 0 and "expected-run-sha256" in missing.stderr
        done = subprocess.run(
            [
                __import__("sys").executable,
                *args,
                "--historical-smoke",
                "--expected-run-sha256",
                root["run_sha256"],
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        assert "HISTORICAL SMOKE ONLY" in done.stdout
        assert (work / "bundle/index.json").is_file()


def test_gallery_index_requires_current_native_run_identity_and_safe_assets():
    assert (ROOT / "site/signature.mjs").is_file(), "The run-bound gallery module is missing"
    import os

    from test_cylinder_site import node_check

    root = json.loads((REPORT / "manifest.json").read_text())
    with scratch() as work:
        adapter().build_bundle(REPORT, work / "bundle", root["run_sha256"], historical_smoke=True)
        node_check(
            r"""
          import assert from 'node:assert/strict';
          import fs from 'node:fs';
          import {validateIndex,loadSignature,renderSignature} from './site/signature.mjs';
          const historical=JSON.parse(fs.readFileSync(process.env.SIGNATURE_FIXTURE));
          const results={status:'completed_native_reconstructed',metadata:{inputs:[{role:'run',sha256:historical.native_run_sha256}]}};
          assert.throws(()=>validateIndex(historical,results),/historical|native/i);
          // Explicit in-memory contract fixture, never written to a public results path.
          const fixture=structuredClone(historical);
          fixture.source_kind='native_completed_short_record';
          fixture.evidence_label='NATIVE COMPLETED SHORT RECORD';
          for(const figure of fixture.figures){figure.evidence_label=fixture.evidence_label;figure.title='TEST ONLY <unsafe> — '+figure.title;}
          assert.equal(validateIndex(fixture,results).figures.length,9);
          const ready=await loadSignature(results,async url=>{
            assert.equal(url,'./assets/signature/index.json');
            return {ok:true,json:async()=>fixture};
          });
          assert.equal(ready.status,'ready');
          const html=renderSignature(ready);
          assert.equal((html.match(/class="signature-card"/g)||[]).length,9);
          assert.equal((html.match(/loading="lazy"/g)||[]).length,9);
          assert.ok(html.includes('&lt;unsafe&gt;')&&!html.includes('<unsafe>'));
          assert.match(html,/OFFLINE/);assert.match(html,/short record/i);
          assert.match(html,/\.svg/);assert.match(html,/\.pdf/);assert.match(html,/\.csv/);
          for(const [status,reply] of [['pending',{ok:false,status:404}],['unavailable',{ok:false,status:500}]]){
            const bundle=await loadSignature(results,async()=>reply);
            assert.equal(bundle.status,status);assert.ok(!renderSignature(bundle).includes('<img'));
          }
          for(const mutate of [
            x=>x.figure_count=8, x=>x.figures.pop(),x=>x.figures[8]=x.figures[0],
            x=>x.figures[0].id='I00_fake',x=>x.output_sha256={},x=>delete x.output_sha256,
            x=>x.native_run_sha256='0'.repeat(64),x=>x.figures[0].outputs.png.width=Infinity,
            x=>x.figures[0].outputs.png.path='../secret.png',x=>x.figures[0].outputs.svg.path='https://evil.test/a.svg',
            x=>x.figures[0].numeric_tables[0].path='data/%2e%2e/secret.csv',
            x=>x.figures[0].outputs.png.sha256='0'.repeat(64),x=>x.figures[0].provenance='../private.json',
            x=>x.source_kind='historical_smoke_only',x=>x.figures[0].evidence_label='HISTORICAL SMOKE ONLY',
          ]){
            const bad=structuredClone(fixture);mutate(bad);
            assert.throws(()=>validateIndex(bad,results));
            const state=await loadSignature(results,async()=>({ok:true,json:async()=>bad}));
            assert.equal(state.status,'unavailable');assert.ok(!renderSignature(state).includes('<img'));
          }
          assert.throws(()=>validateIndex(fixture,{metadata:{}}));
          assert.throws(()=>validateIndex(fixture,{...results,status:'pending'}));
          const ambiguous=structuredClone(results);ambiguous.metadata.inputs.push(ambiguous.metadata.inputs[0]);
          assert.throws(()=>validateIndex(fixture,ambiguous));
          const malformed=await loadSignature(results,async()=>({ok:true,json:async()=>{throw Error('bad JSON')}}));
          assert.equal(malformed.status,'unavailable');
        """,
            env={**os.environ, "SIGNATURE_FIXTURE": str(work / "bundle/index.json")},
        )


def test_browser_gallery_is_run_bound_full_width_and_independent_of_analytical_controls(
    monkeypatch,
):
    """Private test server only; real smoke image bytes + in-memory TEST ONLY index."""

    import test_cylinder_site as browser

    html = (ROOT / "site/index.html").read_text()
    assert 'id="signature"' in html, "Acoustic signatures section is missing"
    assert html.index('id="evidence"') < html.index('id="signature"') < html.index("<footer>")
    root = json.loads((REPORT / "manifest.json").read_text())
    original_results = digest((ROOT / "site/data/results.json").read_bytes())
    with scratch() as work:
        website = work / "TEST-ONLY-site"
        # Exclude the real gallery: the no-clobber adapter needs a fresh test target.
        shutil.copytree(ROOT / "site", website, ignore=shutil.ignore_patterns("signature"))
        # No test asset or fixture is ever written into the real site.
        adapter().build_bundle(
            REPORT, website / "assets/signature", root["run_sha256"], historical_smoke=True
        )
        monkeypatch.setattr(browser, "SITE", website)
        with browser.isolated_browser() as env:
            env["SIGNATURE_FIXTURE"] = str(website / "assets/signature/index.json")
            env["SITE_SCREENSHOTS"] = str(ROOT / "docs/cylinder_60s/gallery")
            Path(env["SITE_SCREENSHOTS"]).mkdir(parents=True, exist_ok=True)
            browser.node_check(
                browser.CDP
                + r"""
              assert.equal(await evaluate(`document.querySelectorAll('#signature img').length`),0);
              assert.match(await evaluate(`document.getElementById('signature-content').textContent`),/unavailable|pending/i);
              const historical=JSON.parse(fs.readFileSync(process.env.SIGNATURE_FIXTURE));
              const fixture=structuredClone(historical);fixture.source_kind='native_completed_short_record';fixture.evidence_label='NATIVE COMPLETED SHORT RECORD';
              fixture.limitations.unshift('TEST ONLY browser fixture — historical smoke assets, NOT new CFD results.');
              for(const f of fixture.figures){f.evidence_label=fixture.evidence_label;f.title='TEST ONLY — '+f.title;f.caption='TEST ONLY contract fixture. Historical smoke image below. '+f.caption;}
              const resultFixture={schema_version:1,status:'completed_native_reconstructed',summary:'TEST ONLY schema fixture — not new CFD',metadata:{inputs:[{role:'run',sha256:fixture.native_run_sha256}]},charts:[],checks:[],downloads:[]};
              await evaluate(`window.testIndex=${JSON.stringify(fixture)};window.testResults=${JSON.stringify(resultFixture)};window.signatureMode='ready';window.originalFetch=window.fetch;
                window.fetch=async(url,...args)=>{
                  if(url==='./data/results.json')return new Response(JSON.stringify(window.testResults));
                  if(url==='./assets/signature/index.json'){
                    if(window.signatureMode==='wait')await new Promise(resolve=>window.releaseSignature=resolve);
                    if(window.signatureMode==='missing')return new Response('',{status:404});
                    if(window.signatureMode==='malformed')return new Response('{not JSON');
                    if(window.signatureMode==='wrong-schema')return new Response(JSON.stringify(window.testResults));
                    return new Response(JSON.stringify(window.testIndex));
                  }
                  return window.originalFetch(url,...args);
                };`);
              const waitDone=async()=>{
                for(let i=0;i<150;i++){
                  if(await evaluate(`!document.getElementById('retry-results').disabled`))return;
                  await new Promise(r=>setTimeout(r,60));
                }
                assert.fail('Refresh did not finish');
              };
              const reload=async()=>{await evaluate(`document.getElementById('retry-results').click()`);await waitDone();};
              await evaluate(`window.signatureMode='wait';document.getElementById('retry-results').click()`);
              for(let i=0;i<100;i++){
                if(await evaluate(`typeof window.releaseSignature==='function'`))break;
                await new Promise(r=>setTimeout(r,30));
              }
              assert.equal(await evaluate(`document.getElementById('retry-results').disabled`),true,'Retry must await signature fetch too');
              assert.equal(await evaluate(`document.querySelectorAll('#signature img').length`),0);
              await evaluate(`window.releaseSignature();window.signatureMode='ready'`);await waitDone();
              assert.equal(await evaluate(`document.querySelectorAll('#signature .signature-card').length`),9);
              const snap=async name=>{const shot=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(process.env.SITE_SCREENSHOTS+'/'+name+'.png',Buffer.from(shot.data,'base64'));};
              for(const viewport of [{width:1440,height:1100,mobile:false},{width:390,height:844,mobile:true}]){
                await call('Emulation.setDeviceMetricsOverride',{...viewport,deviceScaleFactor:1});
                await new Promise(r=>setTimeout(r,250));
                for(let i=0;i<9;i++){
                  await evaluate(`document.querySelectorAll('.signature-card')[${i}].scrollIntoView({block:'start',behavior:'instant'})`);
                  for(let j=0;j<100;j++){
                    if(await evaluate(`(()=>{const image=document.querySelectorAll('.signature-card img')[${i}];return image.complete&&image.naturalWidth>0})()`))break;
                    await new Promise(r=>setTimeout(r,40));
                  }
                  assert.equal(await evaluate(`document.querySelectorAll('.signature-card img')[${i}].naturalWidth`),1728);
                  assert.equal(await evaluate(`document.documentElement.scrollWidth<=${viewport.width}`),true,'Horizontal overflow');
                  assert.equal(await evaluate(`(()=>{const card=document.querySelectorAll('.signature-card')[${i}];return card.clientWidth/document.querySelector('.signature-gallery').clientWidth>.95})()`),true,'Figure must use full content column');
                  if([0,4,8].includes(i))await snap(`signature-TEST-ONLY-${viewport.width}-${i+1}`);
                }
              }
              const before=await evaluate(`window.firstSignature=document.querySelector('.signature-card');document.getElementById('signature-content').innerHTML`);
              await evaluate(`document.getElementById('frequency').value=1;document.getElementById('frequency').dispatchEvent(new Event('input',{bubbles:true}))`);
              assert.equal(await evaluate(`document.querySelector('.signature-card')===window.firstSignature`),true,'Analytical controls must not rebuild gallery');
              assert.equal(await evaluate(`document.getElementById('signature-content').innerHTML`),before);
              const links=await evaluate(`[...document.querySelectorAll('#signature a')].map(a=>a.href)`);
              for(const url of links){const response=await fetch(url);assert.equal(response.status,200,url);assert.ok((await response.arrayBuffer()).byteLength>0,url);}
              // Click the real SVG link and inspect the served vector document.
              const svg=await evaluate(`document.querySelector('.signature-image').href`);
              await evaluate(`document.querySelector('.signature-image').click()`);
              for(let i=0;i<100;i++){if(await evaluate(`location.href===${JSON.stringify(svg)} && document.documentElement.tagName.toLowerCase()==='svg'`))break;await new Promise(r=>setTimeout(r,50));}
              assert.equal(await evaluate(`document.documentElement.tagName.toLowerCase()`),'svg');
              await call('Page.navigate',{url:process.env.SITE_TEST_URL});
              await new Promise(r=>setTimeout(r,200));await waitDone();
              // After navigation the actual historical index is again refused.
              assert.equal(await evaluate(`document.querySelectorAll('#signature img').length`),0);
              // Reinstall fixtures after navigation, then test every failure transition.
              await evaluate(`window.testIndex=${JSON.stringify(fixture)};window.testResults=${JSON.stringify(resultFixture)};window.signatureMode='ready';window.fetch=async url=>url==='./data/results.json'?new Response(JSON.stringify(window.testResults)):window.signatureMode==='missing'?new Response('',{status:404}):window.signatureMode==='malformed'?new Response('{bad'):window.signatureMode==='wrong-schema'?new Response(JSON.stringify(window.testResults)):new Response(JSON.stringify(window.testIndex));`);
              for(const mode of ['missing','malformed','wrong-schema','mismatch','historical']){
                await evaluate(`window.testIndex=${JSON.stringify(fixture)};window.signatureMode='ready'`);await reload();
                assert.equal(await evaluate(`document.querySelectorAll('#signature img').length`),9);
                await evaluate(`window.signatureMode=${JSON.stringify(mode)};if(window.signatureMode==='mismatch')window.testIndex.native_run_sha256='0'.repeat(64);if(window.signatureMode==='historical')window.testIndex.source_kind='historical_smoke_only';`);
                await reload();
                assert.equal(await evaluate(`document.querySelectorAll('#signature img').length`),0,mode);
              }
              // A missing image must clear the gallery, not leave plausible stale plots.
              await call('Network.enable');await call('Network.setCacheDisabled',{cacheDisabled:true});
              fs.unlinkSync(process.env.SIGNATURE_FIXTURE.replace(/index\.json$/,'I01_geometry/figure.png'));
              await evaluate(`window.testIndex=${JSON.stringify(fixture)};window.signatureMode='ready'`);await reload();
              await evaluate(`document.getElementById('signature').scrollIntoView({behavior:'instant'})`);
              for(let i=0;i<100;i++){
                if(await evaluate(`document.querySelectorAll('#signature img').length===0`))break;
                await new Promise(r=>setTimeout(r,30));
              }
              assert.equal(await evaluate(`document.querySelectorAll('#signature img').length`),0,'A missing figure must clear the entire gallery');
              assert.match(await evaluate(`document.getElementById('signature-content').textContent`),/unavailable/i);
              assert.deepEqual(exceptions,[]);ws.close();
            """,
                env=env,
            )
    assert digest((ROOT / "site/data/results.json").read_bytes()) == original_results


def test_interpolation_and_cadence_provenance_survives_explicit_nested_whitelist():
    with scratch() as work:
        source = work / "TEST-ONLY-report"
        shutil.copytree(REPORT, source)
        root = json.loads((source / "manifest.json").read_text())
        item = root["figures"][2]
        path = source / item["manifest"]
        per = json.loads(path.read_text())
        flags = {
            "native_to_uniform": True,
            "retarded_interpolation": False,
            "local_spectral": False,
        }
        cadence = {
            "dt_s": 0.0009,
            "nominal_dt_s": None,
            "observed_mean_dt_s": 0.0009,
            "observed_min_dt_s": 0.00089999999999,
            "observed_max_dt_s": 0.00090000000001,
            "max_interval_jitter_s": 1e-14,
            "max_lattice_residual_s": 1e-13,
            "validation_tolerance_s": 1e-9,
            "sample_support_s": [2.0, 6.0],
            "sample_count": 4444,
            "clock_replaced": False,
        }
        bin_contract = {
            "frequency_origin_Hz": 0.0,
            "bin_count": 1601,
            "bin_spacing_Hz": 0.5,
            "edge_snap_eps_multiplier": 32,
            "edge_snap_max_bins": 1e-9,
            "edge_coordinate": "q=(edge_Hz-frequency_origin_Hz)/bin_spacing_Hz",
            "edge_snap_rule": "bounded integer snapping",
            "interior_rule": "half_open_right_owns_boundary",
            "final_upper_edge_inclusive": True,
        }
        for estimator in [per["estimator"], per["numeric_tables"][0]["estimator"]]:
            estimator.update(flags, cadence={**cadence, "/home/private/secret": 123})
            estimator["frequency_bin_contract"] = {
                **bin_contract,
                "private_source": "/home/private/secret",
            }
            estimator["unapproved_path"] = "/home/private/secret"
        rewrite_json(path, per)
        root["output_sha256"][item["manifest"]] = digest(path.read_bytes())
        rewrite_json(source / "manifest.json", root)
        output = work / "bundle"
        result = adapter().build_bundle(source, output, root["run_sha256"], historical_smoke=True)
        table = result["figures"][2]["numeric_tables"][0]
        assert all(table["estimator"].get(key) == value for key, value in flags.items())
        assert table["estimator"].get("cadence") == cadence
        assert table["estimator"].get("frequency_bin_contract") == bin_contract
        public = json.loads((output / result["figures"][2]["provenance"]).read_text())
        assert public["estimator"]["cadence"] == cadence
        assert public["estimator"]["frequency_bin_contract"] == bin_contract
        assert all(public["estimator"][key] == value for key, value in flags.items())
        assert "/home/" not in json.dumps(result) + json.dumps(public)


def test_raster_format_disclosure_survives_publication_and_is_visible():
    import os

    from test_cylinder_site import node_check

    with scratch() as work:
        source = work / "TEST-ONLY-report"
        shutil.copytree(REPORT, source)
        root = json.loads((source / "manifest.json").read_text())
        item = root["figures"][4]
        path = source / item["manifest"]
        per = json.loads(path.read_text())
        rendering = {
            "pdf": "mixed vector text/axes with rasterized fields and colorbars",
            "raster_dpi": 160,
            "reason": "TEST ONLY display-format fixture",
            "scientific_arrays_changed": False,
        }
        per["rendering"] = {**rendering, "private_source": "/home/private/secret"}
        rewrite_json(path, per)
        root["output_sha256"][item["manifest"]] = digest(path.read_bytes())
        rewrite_json(source / "manifest.json", root)
        output = work / "bundle"
        result = adapter().build_bundle(source, output, root["run_sha256"], historical_smoke=True)
        figure = result["figures"][4]
        assert figure.get("rendering") == rendering
        public = json.loads((output / figure["provenance"]).read_text())
        assert public["rendering"] == rendering
        assert "/home/" not in json.dumps(result) + json.dumps(public)
        node_check(
            r"""
            import assert from 'node:assert/strict';
            import fs from 'node:fs';
            import {renderSignature} from './site/signature.mjs';
            const index=JSON.parse(fs.readFileSync(process.env.SIGNATURE_FIXTURE));
            const html=renderSignature({status:'ready',index});
            assert.match(html,/rasterized fields and colorbars/);
            assert.match(html,/160 dpi/);
            assert.ok(!html.includes('Open vector SVG'));
            """,
            env={**os.environ, "SIGNATURE_FIXTURE": str(output / "index.json")},
        )


@pytest.mark.parametrize("failure", ["write", "raced-target"])
def test_atomic_staging_never_exposes_partial_or_overwrites_raced_target(monkeypatch, failure):
    """Verification-only of no-replace commit and cleanup; no new behavior implied."""
    m = adapter()
    root = json.loads((REPORT / "manifest.json").read_text())
    with scratch() as work:
        output = work / "bundle"
        real_write = Path.write_bytes
        written = []

        def intercept(path, data):
            assert not output.exists() or failure == "raced-target"
            written.append(path)
            if len(written) == 2:
                if failure == "write":
                    raise OSError("injected staging failure")
                output.mkdir()
                real_write(output / "marker", b"must survive")
            return real_write(path, data)

        monkeypatch.setattr(Path, "write_bytes", intercept)
        with pytest.raises(OSError):
            m.build_bundle(REPORT, output, root["run_sha256"], historical_smoke=True)
        assert not list(work.glob(".signature-site-stage-*"))
        if failure == "write":
            assert not output.exists()
        else:
            assert (output / "marker").read_bytes() == b"must survive"
