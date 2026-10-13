"""Reader-facing separation of acoustic lessons from this executed CFD case."""
from test_cylinder_site import CDP, isolated_browser, node_check


def test_source_mechanisms_are_explicitly_mapped_to_cfd():
    with isolated_browser() as env:
        node_check(CDP + r"""
          const text=await evaluate(`document.getElementById('cfd-source-connection')?.textContent || ''`);
          assert.match(text,/Which parts come from this CFD/);
          assert.match(text,/13 interactive figures are analytical/);
          assert.match(text,/Monopole[\s\S]*stationary[\s\S]*zero thickness/);
          assert.match(text,/Dipole[\s\S]*CFD surface loading[\s\S]*Curle[\s\S]*FW-H/);
          assert.match(text,/Quadrupole[\s\S]*not evaluated/);
          assert.match(text,/35 computed charts/);
          await call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true});
          await evaluate(`document.getElementById('cfd-source-connection').scrollIntoView()`);
          assert.equal(await evaluate(`document.documentElement.scrollWidth<=390`),true);
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)


def test_accuracy_note_distinguishes_mesh_checks_from_validation():
    with isolated_browser() as env:
        node_check(CDP + r"""
          const text=await evaluate(`document.getElementById('accuracy-note')?.textContent || ''`);
          assert.match(text,/physical accuracy has not been established/);
          assert.match(text,/mesh[\s\S]*time-step[\s\S]*domain/);
          assert.match(text,/not[\s\S]*validated sound levels/);
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)


def test_native_output_note_explains_zero_and_lagged_diagnostics():
    with isolated_browser() as env:
        node_check(CDP + r"""
          const text=await evaluate(`document.getElementById('native-output-note')?.textContent || ''`);
          assert.match(text,/FW-H[\s\S]*inherited[\s\S]*zero/);
          assert.match(text,/surface pressure[\s\S]*nonzero/);
          assert.match(text,/Curle[\s\S]*one timestep/);
          assert.match(text,/forces.csv/);
          assert.match(text,/adaptive[\s\S]*derivative/);
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)


def test_live_flow_figure_identifies_full_circle_and_true_planes():
    with isolated_browser() as env:
        node_check(CDP + r"""
          const text=await evaluate(`document.getElementById('circular-flow-figure')?.textContent || ''`);
          assert.match(text,/full circular domain/);
          assert.match(text,/0.035 m/);
          assert.match(text,/no solid/);
          assert.equal(await evaluate(`document.querySelector('#circular-flow-figure img')?.getAttribute('src')`),'./assets/flow_vorticity_circular.png');
          assert.equal(await evaluate(`document.querySelector('#circular-flow-figure img').naturalWidth>0`),true);
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)


def test_comparison_note_distinguishes_support_and_loading_bases():
    with isolated_browser() as env:
        node_check(CDP + r"""
          const text=await evaluate(`document.getElementById('comparison-contract')?.textContent || ''`);
          assert.match(text,/force spectrum uses seven[\s\S]*receiver spectra use six/);
          assert.match(text,/last used spectral samples[\s\S]*9\.001/);
          assert.match(text,/signature[\s\S]*raw 2–10 s[\s\S]*seven segments/);
          assert.match(text,/Curle[\s\S]*on-body total traction[\s\S]*unretarded/);
          assert.match(text,/FW-H[\s\S]*pressure-only[\s\S]*facewise/);
          assert.match(text,/Website compact[\s\S]*negative exported total body force/);
          assert.match(text,/signature compact[\s\S]*pressure-only/);
          assert.match(text,/not summed/);
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)


def test_current_case_copy_matches_completed_refined_run():
    with isolated_browser() as env:
        node_check(CDP + r"""
          const geometry=await evaluate(`document.getElementById('native-geometry').textContent`);
          assert.match(geometry,/459,264/);
          assert.match(geometry,/10 s/);
          assert.match(geometry,/2–10 s/);
          assert.match(geometry,/unresolved/);
          assert.ok(!/138,240|1\.45 Hz|5\.6 s|six-second/.test(geometry));
          const native=await evaluate(`document.getElementById('native-output-note').textContent`);
          assert.match(native,/80,000/);
          assert.match(native,/11 initial/);
          assert.match(native,/fixed[\s\S]*0\.000625/);
          assert.match(native,/no restart/);
          assert.match(await evaluate(`document.getElementById('run-status').textContent`),/ANALYSIS STATUS/);
          assert.deepEqual(exceptions,[]);
          ws.close();
        """, env=env)
