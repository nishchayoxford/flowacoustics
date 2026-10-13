"""Structural contract for the editable, evidence-bounded project map."""

import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "docs/structure/flowacoustics-map.svg"
NS = {"s": "http://www.w3.org/2000/svg"}


def test_flowchart_is_standalone_and_edges_resolve():
    assert SVG.is_file(), "The requested editable project SVG is missing"
    root = ET.parse(SVG).getroot()
    assert root.tag == f"{{{NS['s']}}}svg"
    ids = [element.attrib["id"] for element in root.iter() if "id" in element.attrib]
    assert len(ids) == len(set(ids)), "SVG identifiers must be unique"
    nodes = {element.attrib["id"] for element in root.findall(".//s:g[@data-node='true']", NS)}
    assert {
        "case",
        "mesh",
        "les",
        "fields",
        "forces",
        "surface",
        "reconstruct",
        "curle",
        "fwh",
        "flow",
        "compact",
        "receivers",
        "analysis",
        "release",
    } == nodes
    edges = root.findall(".//s:path[@data-from]", NS)
    assert len(edges) >= 15
    for edge in edges:
        assert edge.attrib["data-from"] in nodes
        assert edge.attrib["data-to"] in nodes
    assert not root.findall(".//s:script", NS)
    assert not root.findall(".//s:image", NS), "Keep the SVG vector-only and offline"


def test_flowchart_preserves_acoustic_and_evidence_boundaries():
    assert SVG.is_file(), "The requested editable project SVG is missing"
    root = ET.parse(SVG).getroot()
    text = " ".join(root.itertext())
    for label in (
        "reconstructPar",
        "Curle",
        "FW-H",
        "500,000",
        "10 s",
        "force on body",
        "force on fluid",
        "not summed",
        "volume quadrupoles",
        "analytical lessons",
        "not experimental validation",
    ):
        assert label in text, f"Missing scientific boundary: {label}"
    edges = {
        (e.attrib["data-from"], e.attrib["data-to"])
        for e in root.findall(".//s:path[@data-from]", NS)
    }
    assert ("forces", "compact") in edges
    assert ("curle", "compact") not in edges, "The independent oracle is not native output"
    assert ("curle", "receivers") in edges and ("fwh", "receivers") in edges
    assert ("fields", "reconstruct") in edges


def test_flowchart_identifies_completed_ten_second_execution():
    root = ET.parse(SVG).getroot()
    status = root.find(".//s:text[@id='run-status']", NS).text
    assert "10 s completed" in status
    assert "reconstructed" in status
    assert "running" not in " ".join(root.itertext())


def test_flowchart_rendered_text_fits_each_card(monkeypatch):
    import test_cylinder_site as browser

    monkeypatch.setattr(browser, "SITE", ROOT)
    output = ROOT / ".cache/flowchart-verification"
    output.mkdir(parents=True, exist_ok=True)
    program = (
        browser.CDP.split("await call('Page.navigate'")[0]
        + r"""
        await call('Emulation.setDeviceMetricsOverride',
          {width:1600,height:1640,deviceScaleFactor:1,mobile:false});
        await call('Page.navigate',
          {url:process.env.SITE_TEST_URL+'docs/structure/flowacoustics-map.svg'});
        for(let i=0;i<100;i++) {
          if(await evaluate(`document.querySelectorAll('g[data-node]').length===14`))break;
          await new Promise(r=>setTimeout(r,50));
        }
        const report=await evaluate(`(()=>{
          const nodes=[...document.querySelectorAll('g[data-node]')], bad=[];
          for(const g of nodes) {
            const x=+g.dataset.x,y=+g.dataset.y,w=+g.dataset.width,h=+g.dataset.height;
            for(const t of g.querySelectorAll('text')) {
              const b=t.getBBox();
              if(b.x<x+12||b.x+b.width>x+w-12||b.y<y+8||b.y+b.height>y+h-6)
                bad.push({node:g.id,text:t.textContent});
            }
          }
          return {node_count:nodes.length,overflows:bad};
        })()`);
        assert.equal(report.node_count,14);
        assert.deepEqual(report.overflows,[]);
        assert.deepEqual(exceptions,[]);
        fs.writeFileSync(process.env.FLOWCHART_OUTPUT+'/layout.json',
          JSON.stringify(report,null,2)+'\n');
        const shot=await call('Page.captureScreenshot',
          {format:'png',captureBeyondViewport:true});
        fs.writeFileSync(process.env.FLOWCHART_OUTPUT+'/flowacoustics-map.png',
          Buffer.from(shot.data,'base64'));
        ws.close();
    """
    )
    with browser.isolated_browser() as environment:
        browser.node_check(program, env={**environment, "FLOWCHART_OUTPUT": str(output)})
