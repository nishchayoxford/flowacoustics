"""Independent synthetic tests; no native solver or original-output writes."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/view_cylinder_exports.py"


def module():
    assert SCRIPT.is_file(), "the reusable export converter has not been implemented"
    spec = importlib.util.spec_from_file_location("cylinder_exports", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def raw(path, field="U", association="POINT_DATA", rows=None):
    rows = rows or ["0.1 -0.2 0 1.2345678901234567 -0.0 3e-12", "0.2 0.3 0 -4 5 6"]
    columns = "x y z " + (field if field == "p" else " ".join(field + "_" + k for k in "xyz"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {field}  {association} {len(rows)}\n# {columns}\n" + "\n".join(rows) + "\n")
    return path


def test_raw_parser_preserves_source_precision_and_association(tmp_path):
    mod = module()
    path = raw(tmp_path / "5.6/U_midspan.raw")
    data = mod.load_raw(path)
    assert data.field == "U"
    assert data.association == "POINT_DATA"
    assert data.time_text == "5.6"
    assert data.tokens[0] == ["0.1", "-0.2", "0", "1.2345678901234567", "-0.0", "3e-12"]
    np.testing.assert_array_equal(data.values, np.loadtxt(path))
    assert data.units == "m s^-1"


@pytest.mark.parametrize(
    "corruption", ["count", "columns", "nonfinite", "header", "time", "unknown_field"]
)
def test_raw_parser_rejects_invalid_exports(tmp_path, corruption):
    mod = module()
    path = raw(tmp_path / "5.6/U_midspan.raw")
    text = path.read_text()
    if corruption == "count":
        text = text.replace("POINT_DATA 2", "POINT_DATA 3")
    elif corruption == "columns":
        text = text.replace("U_x U_y U_z", "U_z U_y U_x")
    elif corruption == "nonfinite":
        text = text.replace("1.2345678901234567", "nan")
    elif corruption == "header":
        text = text.replace("POINT_DATA", "CELL_DATA")
    elif corruption == "time":
        path = tmp_path / "nan/U_midspan.raw"
        path.parent.mkdir()
    elif corruption == "unknown_field":
        text = text.replace("U", "unknown")
    path.write_text(text)
    with pytest.raises(ValueError):
        mod.load_raw(path)


def test_convert_tree_losslessly_keeps_cases_and_orders_series(tmp_path):
    import csv
    import hashlib
    import json
    import xml.etree.ElementTree as ET

    mod = module()
    cfd = tmp_path / "cfd"
    sources = [
        raw(cfd / f"{case}/postProcessing/flowSlice/{t}/U_midspan.raw")
        for case in ["cylinder_demo", "cylinder_circular"]
        for t in ["10", "2"]
    ]
    sources.append(
        raw(
            cfd / "cylinder_circular/postProcessing/wallHistory/2/p_body.raw",
            "p",
            "FACE_DATA",
            ["0 0 0.03 -0.12345678901234567"],
        )
    )
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    output = tmp_path / "allowed/bundle"
    manifest = mod.convert_tree(cfd, output, allowed_root=tmp_path / "allowed")
    assert manifest["counts"] == {
        "raw": 5,
        "csv": 5,
        "vtp": 5,
        "pvd": 3,
        "sample_rows": 9,
        "source_numeric_values": 52,
    }
    assert json.loads((output / "manifest.json").read_text()) == manifest
    assert len(manifest["sources"]) == len(sources)
    for record in manifest["sources"]:
        source = cfd / record["path"]
        assert record["sha256"] == hashes[source]
        with (output / record["csv"]).open() as f:
            rows = list(csv.reader(f))
        assert rows[0][:4] == ["time_s", "x_m", "y_m", "z_m"]
        expected = np.loadtxt(source, ndmin=2)
        actual = np.asarray([r[1:] for r in rows[1:]], dtype=float)
        assert actual.tobytes() == expected.tobytes()
        root = ET.parse(output / record["vtp"]).getroot()
        piece = root.find("./PolyData/Piece")
        assert int(piece.get("NumberOfPoints")) == len(expected)
        assert int(piece.get("NumberOfVerts")) == len(expected)
        assert int(piece.get("NumberOfPolys")) == 0
        points = np.fromstring(piece.find("./Points/DataArray").text, sep=" ").reshape(-1, 3)
        field = np.fromstring(piece.find("./PointData/DataArray").text, sep=" ").reshape(
            len(expected), -1
        )
        assert np.column_stack([points, field]).tobytes() == expected.tobytes()
        assert record["checks"]["csv_max_abs_error"] == 0
        assert record["association"] in ["POINT_DATA", "FACE_DATA"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == hashes[source]
    for series in manifest["collections"]:
        nodes = ET.parse(output / series["path"]).findall("./Collection/DataSet")
        times = [float(n.get("timestep")) for n in nodes]
        assert times == sorted(times)
        assert all(
            (output / series["path"]).parent.joinpath(n.get("file")).is_file() for n in nodes
        )
    assert manifest["representation"] == "samples_only_not_native_mesh"


@pytest.mark.parametrize("unsafe", ["outside", "symlink", "dotdot", "existing", "source_tree"])
def test_output_preflight_precedes_source_reads_and_writes(tmp_path, unsafe):
    mod = module()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("original")
    output = allowed / "bundle"
    if unsafe == "outside":
        output = outside / "bundle"
    elif unsafe == "symlink":
        (allowed / "link").symlink_to(outside, target_is_directory=True)
        output = allowed / "link/bundle"
    elif unsafe == "dotdot":
        output = allowed / "child/../bundle"
    elif unsafe == "existing":
        output.mkdir()
    else:
        output = allowed / "source/bundle"
    source = allowed / "source" if unsafe == "source_tree" else tmp_path / "MISSING"
    with pytest.raises(ValueError, match="output|Output|destination|Destination"):
        mod.convert_tree(source, output, allowed_root=allowed)
    assert (outside / "keep").read_text() == "original"
    assert not (outside / "bundle").exists()
    assert not (allowed / "child").exists()


def test_failed_conversion_does_not_publish_partial_bundle(tmp_path):
    mod = module()
    source = tmp_path / "cfd"
    raw(source / "case/postProcessing/flow/1/U_midspan.raw")
    bad = raw(source / "case/postProcessing/flow/2/U_midspan.raw")
    bad.write_text(bad.read_text().replace("POINT_DATA 2", "POINT_DATA 3"))
    allowed = tmp_path / "allowed"
    output = allowed / "bundle"
    with pytest.raises(ValueError):
        mod.convert_tree(source, output, allowed_root=allowed)
    assert not output.exists()
    assert not list(allowed.glob(".stage-*"))


def test_triangle_solid_intersection_not_centroid_mask_and_tip_is_fluid():
    mod = module()
    # The first triangle has its centroid outside the disk but an edge through it.
    xy = np.array(
        [
            [[-0.02, 0.002], [0.02, 0.002], [0.02, 0.025]],
            [[0.011, 0], [0.02, 0], [0.015, 0.01]],
            [[-0.02, -0.02], [0.02, -0.02], [0, 0.03]],
        ]
    )
    assert np.linalg.norm(xy[0].mean(0)) > 0.01
    np.testing.assert_array_equal(mod.triangles_intersect_disk(xy, 0.01), [True, False, True])
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = np.column_stack([0.24 * np.cos(theta), 0.24 * np.sin(theta)])
    inner = ring / 24
    points = np.column_stack([np.vstack([ring, inner]), np.zeros(64)])
    tri, audit = mod.circular_triangulation(points)
    assert audit["solid_intersecting_triangles_removed"] > 0
    assert not mod.triangles_intersect_disk(points[tri.get_masked_triangles(), :2], 0.01).any()
    tip = np.vstack(
        [np.column_stack([np.vstack([ring, inner]), np.full(64, 0.035)]), [0, 0, 0.035]]
    )
    tri, audit = mod.circular_triangulation(tip)
    assert audit["solid_intersecting_triangles_removed"] == 0
    assert tri.get_trifinder()(0, 0) >= 0
    assert audit["z_m"] == pytest.approx(0.035)


def test_circular_snapshot_saves_full_equal_scale_planes_without_tip_mask(tmp_path):
    mod = module()
    cfd = tmp_path / "cfd"
    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.column_stack([np.cos(theta), np.sin(theta)])
    for plane, z in [("midspan", 0), ("tipplane", 0.035)]:
        xy = np.vstack([ring * r for r in [0.01, 0.02, 0.08, 0.24]])
        if plane == "tipplane":
            xy = np.vstack([xy, [0, 0]])
        rows = [" ".join(map(str, [x, y, z, 0, 0, 100 * x - 20 * y])) for x, y in xy]
        raw(
            cfd / f"cylinder_circular/postProcessing/flowSlice/5.6/vorticity_{plane}.raw",
            "vorticity",
            rows=rows,
        )
    output = tmp_path / "allowed/figures"
    report = mod.plot_circular_snapshot(cfd, output, allowed_root=tmp_path / "allowed")
    assert (output / "flow_vorticity_circular.png").stat().st_size > 10000
    assert (output / "flow_vorticity_circular.pdf").read_bytes().startswith(b"%PDF")
    import subprocess

    pdf_images = subprocess.run(
        ["pdfimages", "-list", str(output / "flow_vorticity_circular.pdf")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert " image " not in pdf_images.stdout, "PDF must not rasterize contours or colourbar"
    text = subprocess.run(
        ["pdftotext", str(output / "flow_vorticity_circular.pdf"), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "scale limits" in text, "Symmetric colour limits are not the asymmetric sample extrema"
    assert report["time_s"] == 5.6
    assert report["layout"]["header_title_gap_pt"] >= 4
    assert report["vorticity_component"] == "vorticity_z"
    assert len(report["planes"]) == 2
    for plane in report["planes"]:
        assert plane["axis_limits_m"] == [[-0.24, 0.24], [-0.24, 0.24]]
        assert plane["display_xy_scale_ratio"] == pytest.approx(1)
    assert report["planes"][0]["solid_present"]
    assert not report["planes"][1]["solid_present"]
    assert report["planes"][1]["origin_supported"]
    assert not report["planes"][0]["origin_supported"]
    with pytest.raises(ValueError, match="Output"):
        mod.plot_circular_snapshot(tmp_path / "MISSING", output, allowed_root=tmp_path / "allowed")


def test_pinned_identity_gate_rejects_missing_malformed_and_changed_data(tmp_path):
    import hashlib
    import json

    mod = module()
    source = tmp_path / "original"
    source.write_bytes(b"finite 1.23\n")
    pin = hashlib.sha256(source.read_bytes()).hexdigest()
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"sha256": {"original": pin}}))
    result = mod.verify_pins(tmp_path, run, expected_count=1)
    assert result["verified_paths"] == 1
    assert result["run_sha256"] == hashlib.sha256(run.read_bytes()).hexdigest()
    for bad in [{}, {"original": "bad"}, {"missing": pin}, {"../outside": pin}]:
        run.write_text(json.dumps({"sha256": bad}))
        with pytest.raises(ValueError):
            mod.verify_pins(tmp_path, run, expected_count=1)
    run.write_text(json.dumps({"sha256": {"original": pin}}))
    source.write_bytes(b"finite 9.87\n")
    with pytest.raises(ValueError, match="identity"):
        mod.verify_pins(tmp_path, run, expected_count=1)


def test_actual_vtk_and_paraview_readback_of_all_sample_types(tmp_path):
    import json
    import subprocess

    mod = module()
    cfd = tmp_path / "cfd"
    for field in ["U", "vorticity"]:
        raw(cfd / f"case/postProcessing/flow/2/{field}_midspan.raw", field)
    raw(
        cfd / "case/postProcessing/wall/2/p_body.raw",
        "p",
        "FACE_DATA",
        ["0 0 0.03 -0.12345678901234567"],
    )
    output = tmp_path / "allowed/bundle"
    mod.convert_tree(cfd, output, allowed_root=tmp_path / "allowed")
    assert hasattr(mod, "verify_exports"), "independent installed-VTK verifier is missing"
    code = (
        "import runpy,json; m=runpy.run_path(" + repr(str(SCRIPT)) + "); "
        "print(json.dumps(m['verify_exports']("
        + repr(str(cfd))
        + ","
        + repr(str(output))
        + ",paraview=True)))"
    )
    result = subprocess.run(
        ["/usr/bin/pvpython", "--force-offscreen-rendering", "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["vtp_files_verified"] == 3
    assert report["pvd_time_steps_verified"] == 3
    assert report["vtk_max_abs_error"] == 0
    assert report["csv_max_abs_error"] == 0
    assert report["paraview_max_abs_error"] == 0


def test_real_cli_build_is_atomic_preserves_403_pins_and_replays(tmp_path):
    import hashlib
    import json
    import subprocess

    project = tmp_path / "project"
    cfd = project / "cfd"
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = np.column_stack([np.cos(theta), np.sin(theta)])
    for plane, z in [("midspan", 0), ("tipplane", 0.035)]:
        xy = np.vstack([ring * r for r in [0.01, 0.02, 0.08, 0.24]])
        if plane == "tipplane":
            xy = np.vstack([xy, [0, 0]])
        raw(
            cfd / f"cylinder_circular/postProcessing/flowSlice/5.6/vorticity_{plane}.raw",
            "vorticity",
            rows=[" ".join(map(str, [x, y, z, 0, 0, 100 * x - 20 * y])) for x, y in xy],
        )
    raw(
        cfd / "cylinder_demo/postProcessing/wallHistory/2/p_body.raw",
        "p",
        "FACE_DATA",
        ["0 0 0.03 2.5"],
    )
    pins = {
        p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in cfd.rglob("*.raw")
    }
    for i in range(400):
        p = project / "pinned" / str(i)
        p.parent.mkdir(exist_ok=True)
        p.write_text(str(i))
        pins[p.relative_to(project).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    run = project / "results/cylinder/native_circular/run.json"
    run.parent.mkdir(parents=True)
    run.write_text(json.dumps({"sha256": pins}))
    assert len(pins) == 403
    site = project / "site/assets/flow_vorticity.png"
    site.parent.mkdir(parents=True)
    site.write_bytes(b"existing site asset")
    destination = project / "results/cylinder/viewable/current"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--project", str(project)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (destination / "manifest.json").is_file(), "CLI did not produce a verified bundle"
    report = json.loads((destination / "manifest.json").read_text())
    assert report["preservation"]["verified_paths"] == 403
    assert report["preservation"]["raw_sources_unchanged"] == 3
    assert (
        report["preservation"]["extra_original_sha256"]["site/assets/flow_vorticity.png"]
        == hashlib.sha256(site.read_bytes()).hexdigest()
    )
    replay = destination.with_name("replay")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "build", "--project", str(project), "--output", str(replay)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (destination / "manifest.json").read_bytes() == (replay / "manifest.json").read_bytes()
    for name in [
        "exports/manifest.json",
        "figures/manifest.json",
        "figures/flow_vorticity_circular.png",
        "figures/flow_vorticity_circular.pdf",
    ]:
        assert (destination / name).read_bytes() == (replay / name).read_bytes()
    assert {p: hashlib.sha256((project / p).read_bytes()).hexdigest() for p in pins} == pins
    verify_report = project / "results/cylinder/viewable/verified.json"
    result = subprocess.run(
        [
            "/usr/bin/pvpython",
            "--force-offscreen-rendering",
            str(SCRIPT),
            "verify",
            "--project",
            str(project),
            "--bundle",
            str(destination),
            "--report",
            str(verify_report),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(verify_report.read_text())["counts"]["raw"] == 3
