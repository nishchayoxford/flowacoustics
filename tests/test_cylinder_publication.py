"""Public presentation transforms, using in-memory test-only data."""
import copy
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    path = ROOT / "scripts/prepare_cylinder_site.py"
    assert path.exists(), "Public presentation builder is missing"
    spec = importlib.util.spec_from_file_location("prepare_cylinder_site", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_display_envelope_keeps_extrema_without_changing_analysis():
    x = list(range(100))
    y = [0.0] * 100
    y[13], y[77] = 19.0, -11.0
    source = {"metadata": {"test_only": True}, "charts": [{
        "id": "forces", "kind": "line", "origin": "computed",
        "x_label": "Time (s)", "description": "TEST ONLY",
        "traces": [{"name": "force", "x": x, "y": y}],
    }]}
    original = copy.deepcopy(source)
    display = load_builder().make_display_dataset(source, max_time_points=12)
    trace = display["charts"][0]["traces"][0]
    assert len(trace["x"]) <= 12
    assert trace["x"][0] == 0 and trace["x"][-1] == 99
    assert max(trace["y"]) == 19.0 and min(trace["y"]) == -11.0
    assert all(y[x.index(t)] == value for t, value in zip(trace["x"], trace["y"]))
    assert source == original
    assert "display_sample_policy" in display["charts"][0]


def test_frequency_display_band_preserves_full_source_and_origins():
    source = {"charts": [
        {"kind": "line", "origin": "computed", "x_label": "Frequency (Hz)",
         "traces": [{"name": "PSD", "x": [0, 10, 20, 30], "y": [1, 2, 3, 4]}]},
        {"kind": "heatmap", "origin": "computed", "x_label": "Observer time (s)",
         "y_label": "Frequency (Hz)", "x": [2, 3], "y": [0, 10, 20, 30],
         "z": [[1, 1], [2, 2], [3, 3], [4, 4]], "traces": []},
    ]}
    original = copy.deepcopy(source)
    display = load_builder().make_display_dataset(source, max_frequency_hz=20)
    assert display["charts"][0]["traces"][0]["x"] == [0, 10, 20]
    assert display["charts"][1]["y"] == [0, 10, 20]
    assert display["charts"][1]["z"] == [[1, 1], [2, 2], [3, 3]]
    assert all(c["display_frequency_window_Hz"] == [0, 20] for c in display["charts"])
    assert source == original


def test_analytical_controls_do_not_rebuild_computed_evidence():
    from test_cylinder_site import CDP, isolated_browser, node_check

    with isolated_browser() as env:
        node_check(CDP + r"""
          for(let i=0;i<100;i++) {
            if(await evaluate(`document.querySelectorAll('.result-chart').length===39`))break;
            await new Promise(r=>setTimeout(r,80));
          }
          assert.equal(await evaluate(`document.querySelectorAll('.result-chart').length`),39);
          assert.equal(await evaluate(`!!document.querySelector('#run-metadata details:not([open])')`),true,
            'Detailed numerical metadata should be expandable, not bury the charts');
          await evaluate(`window.originalEvidence=document.querySelector('.result-chart');
            document.getElementById('angle').value=90;
            document.getElementById('angle').dispatchEvent(new Event('input',{bubbles:true}));`);
          assert.equal(await evaluate(`window.originalEvidence===document.querySelector('.result-chart')`),true,
            'Analytical controls must not rebuild the independent computed evidence');
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)


def test_published_native_mesh_and_flow_images_load_on_mobile():
    from test_cylinder_site import CDP, isolated_browser, node_check

    with isolated_browser() as env:
        node_check(CDP + r"""
          assert.equal(await evaluate(`document.querySelectorAll('#native-geometry img').length`),3);
          await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
          await evaluate(`document.getElementById('native-geometry').scrollIntoView()`);
          await new Promise(r=>setTimeout(r,350));
          assert.equal(await evaluate(`Array.from(document.querySelectorAll('#native-geometry img')).every(x=>x.complete&&x.naturalWidth>0)`),true);
          assert.equal(await evaluate(`document.documentElement.scrollWidth<=390`),true);
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)
