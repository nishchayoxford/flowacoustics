"""Scoped conversion regressions; synthetic sources only, no native writes."""

import xml.etree.ElementTree as ET

import pytest
from test_cylinder_exports import module, raw


def test_selected_case_converts_only_its_postprocessing(tmp_path):
    mod = module()
    cfd = tmp_path / "cfd"
    for case in ["cylinder_circular", "cylinder_circular_60s_pilot", "cylinder_circular_60s"]:
        for time in ["10", "2"]:
            raw(cfd / case / "postProcessing/flowSlice" / time / "U_midspan.raw")
    # Invalid non-sampled file inside selected case must never be discovered/read.
    irrelevant = cfd / "cylinder_circular_60s/other/bad.raw"
    irrelevant.parent.mkdir()
    irrelevant.write_text("not a sample")
    output = tmp_path / "allowed/selected"
    report = mod.convert_tree(
        cfd, output, allowed_root=tmp_path / "allowed", case_names=["cylinder_circular_60s"]
    )
    assert report["selected_case_names"] == ["cylinder_circular_60s"]
    assert report["cases"] == {"cylinder_circular_60s": 2}
    assert report["counts"]["raw"] == 2
    assert report["counts"]["pvd"] == 1
    assert all(
        r["path"].startswith("cylinder_circular_60s/postProcessing/") for r in report["sources"]
    )
    series = report["collections"][0]
    assert series["times_s"] == [2, 10]
    assert [
        float(n.get("timestep"))
        for n in ET.parse(output / series["path"]).findall("./Collection/DataSet")
    ] == [2, 10]


@pytest.mark.parametrize(
    "bad",
    [
        "empty",
        "string",
        "duplicate",
        "traversal",
        "absolute",
        "missing",
        "case_link",
        "raw_link",
        "root_link",
        "root_traversal",
    ],
)
def test_scoped_input_preflight_rejects_before_any_writes(tmp_path, bad):
    mod = module()
    cfd = tmp_path / "cfd"
    source = raw(cfd / "selected/postProcessing/flow/2/U_midspan.raw")
    names = ["selected"]
    if bad == "empty":
        names = []
    elif bad == "string":
        names = "selected"
    elif bad == "duplicate":
        names = ["selected", "selected"]
    elif bad == "traversal":
        names = ["../cfd/selected"]
    elif bad == "absolute":
        names = [str(cfd / "selected")]
    elif bad == "missing":
        names = ["absent"]
    elif bad == "case_link":
        (cfd / "alias").symlink_to(cfd / "selected", target_is_directory=True)
        names = ["alias"]
    elif bad == "raw_link":
        (source.parent / "external.raw").symlink_to(source)
    elif bad == "root_link":
        (tmp_path / "link").symlink_to(cfd, target_is_directory=True)
        cfd = tmp_path / "link"
    elif bad == "root_traversal":
        (cfd / "child").mkdir()
        cfd = cfd / "child/.."
    original = source.read_bytes()
    allowed = tmp_path / "not_created"
    with pytest.raises(ValueError):
        mod.convert_tree(cfd, allowed / "out", allowed_root=allowed, case_names=names)
    assert not allowed.exists(), "Input-scope rejection must precede staging/mkdir"
    assert source.read_bytes() == original


def test_scoped_manifest_verifies_with_real_paraview_without_unselected_reads(tmp_path):
    import json
    import subprocess

    from test_cylinder_exports import SCRIPT

    mod = module()
    cfd = tmp_path / "cfd"
    for time in ["2", "10"]:
        raw(cfd / "selected/postProcessing/flow" / time / "U_midspan.raw")
    unselected = raw(cfd / "pilot/postProcessing/flow/2/U_midspan.raw")
    unselected.write_text("deliberately unreadable export")
    allowed = tmp_path / "allowed"
    output = allowed / "exports"
    mod.convert_tree(cfd, output, allowed_root=allowed, case_names=["selected"])
    code = (
        f"import runpy,json;m=runpy.run_path({str(SCRIPT)!r});"
        f"print(json.dumps(m['verify_exports']({str(cfd)!r},{str(output)!r})))"
    )
    result = subprocess.run(
        ["/usr/bin/pvpython", "--force-offscreen-rendering", "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["counts"]["raw"] == report["pvd_time_steps_verified"] == 2
    assert report["binary64_bitwise_equal"]
    assert report["paraview_max_abs_error"] == 0
