#!/usr/bin/env python3
"""Lossless sampled-export conversion and conservative circular-domain plots.

No OpenFOAM commands are run; no native source or accepted output is modified.
"""

import csv
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RawSample:
    field: str
    association: str
    time_text: str
    tokens: list
    values: np.ndarray
    units: str


UNITS = {"U": "m s^-1", "vorticity": "s^-1", "p": "m^2 s^-2 (kinematic pressure)"}


def load_raw(path):
    """Read exported ASCII coordinates/values without altering decimal tokens."""
    path = Path(path)
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Empty/truncated raw export: {path}")
    match = re.fullmatch(r"#\s+(\w+)\s+(POINT_DATA|FACE_DATA)\s+(\d+)", lines[0])
    if not match:
        raise ValueError(f"Unsupported raw header: {path}")
    field, association, count = match.groups()
    if field not in UNITS:
        raise ValueError(f"Unknown field/units: {field}")
    columns = ["x", "y", "z"] + ([field] if field == "p" else [field + "_" + c for c in "xyz"])
    if lines[1].split() != ["#", *columns]:
        raise ValueError(f"Unexpected component order: {path}")
    time_value = float(path.parent.name)
    if not np.isfinite(time_value):
        raise ValueError(f"Nonfinite source time: {path}")
    tokens = [line.split() for line in lines[2:] if line.strip()]
    if len(tokens) != int(count) or any(len(row) != len(columns) for row in tokens):
        raise ValueError(f"Declared count/column mismatch: {path}")
    values = np.asarray(tokens, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite source values: {path}")
    return RawSample(field, association, path.parent.name, tokens, values, UNITS[field])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_sample(sample, csv_path, vtp_path):
    """Serialize original decimal tokens, including signed zeros; no resampling."""
    field = sample.field
    suffix = {"U": "m_s", "vorticity": "s_inv", "p": "kinematic_m2_s2"}[field]
    names = [f"p_{suffix}"] if field == "p" else [f"{field}_{c}_{suffix}" for c in "xyz"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["time_s", "x_m", "y_m", "z_m", *names])
        writer.writerows([sample.time_text, *row] for row in sample.tokens)
    root = ET.Element("VTKFile", type="PolyData", version="0.1", byte_order="LittleEndian")
    poly = ET.SubElement(root, "PolyData")
    fd = ET.SubElement(poly, "FieldData")
    ET.SubElement(
        fd, "DataArray", type="Float64", Name="TimeValue", NumberOfTuples="1", format="ascii"
    ).text = sample.time_text
    n = len(sample.tokens)
    piece = ET.SubElement(
        poly,
        "Piece",
        NumberOfPoints=str(n),
        NumberOfVerts=str(n),
        NumberOfLines="0",
        NumberOfStrips="0",
        NumberOfPolys="0",
    )
    pd = ET.SubElement(
        piece, "PointData", **({"Scalars": field} if field == "p" else {"Vectors": field})
    )
    ET.SubElement(
        pd,
        "DataArray",
        type="Float64",
        Name=field,
        NumberOfComponents=str(sample.values.shape[1] - 3),
        format="ascii",
    ).text = "\n" + "\n".join(" ".join(r[3:]) for r in sample.tokens) + "\n"
    points = ET.SubElement(piece, "Points")
    ET.SubElement(
        points, "DataArray", type="Float64", NumberOfComponents="3", format="ascii"
    ).text = "\n" + "\n".join(" ".join(r[:3]) for r in sample.tokens) + "\n"
    verts = ET.SubElement(piece, "Verts")
    ET.SubElement(
        verts, "DataArray", type="Int64", Name="connectivity", format="ascii"
    ).text = " ".join(map(str, range(n)))
    ET.SubElement(verts, "DataArray", type="Int64", Name="offsets", format="ascii").text = " ".join(
        map(str, range(1, n + 1))
    )
    ET.ElementTree(root).write(vtp_path, encoding="utf-8", xml_declaration=True)
    with csv_path.open() as stream:
        reader = csv.reader(stream)
        next(reader)
        reread = np.asarray([row[1:] for row in reader], dtype=np.float64)
    if reread.tobytes() != sample.values.tobytes():
        raise ValueError("CSV failed exact binary64 round trip")
    return {
        "csv_max_abs_error": float(np.max(np.abs(reread - sample.values))),
        "csv_bitwise_equal": True,
    }


def verify_pins(project, run_path, expected_count=403):
    """Check retained historical pins; never create or replace expected identities."""
    project, run_path = Path(project).resolve(), Path(run_path)
    pins = json.loads(run_path.read_text())["sha256"]
    if not isinstance(pins, dict) or len(pins) != expected_count or expected_count < 1:
        raise ValueError("Incomplete pinned identity map")
    for name, expected in pins.items():
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError("Malformed pinned identity")
        path = project / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Pinned identity mismatch: {name}")
    return dict(verified_paths=len(pins), run_sha256=sha256(run_path))


def preflight_output(output, allowed_root, source=None):
    """Reject existing destinations, links and traversal before any source read."""
    output, allowed_root = Path(output), Path(allowed_root)
    for path in (output, allowed_root):
        if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Unsafe output path: symlink or parent traversal")
    output, allowed_root = output.absolute(), allowed_root.absolute()
    if output == allowed_root or not output.is_relative_to(allowed_root) or output.exists():
        raise ValueError("Output destination must be a new child of the allowed root")
    if source is not None:
        source = Path(source).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Output destination overlaps source tree")
    return output


def convert_tree(cfd, output, *, allowed_root, case_names=None):
    """Stage a new bundle, optionally limited to named cases' postProcessing.

    Omitting case_names preserves historical whole-tree discovery.
    """
    import shutil
    import tempfile

    output = preflight_output(output, allowed_root, cfd)
    sources = _sample_sources(Path(cfd), case_names)
    cfd = Path(cfd).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=output.parent))
    try:
        manifest = _convert_into(cfd, stage, case_names=case_names, sources=sources)
        if output.exists():
            raise ValueError("Output destination appeared during conversion")
        stage.rename(output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _sample_sources(cfd, case_names=None):
    """Validate explicit source scope before staging; never follow scoped links."""
    if case_names is None:
        return sorted(cfd.resolve().rglob("*.raw"))
    if (
        not isinstance(case_names, (list, tuple))
        or not case_names
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name)
            for name in case_names
        )
        or len(set(case_names)) != len(case_names)
    ):
        raise ValueError("Expected unique, nonempty single-directory case names")
    if ".." in cfd.parts or any(p.is_symlink() for p in (cfd, *cfd.parents)):
        raise ValueError("Unsafe scoped source root")
    cfd = cfd.resolve()
    sources = []
    for name in case_names:
        root = cfd / name / "postProcessing"
        if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
            raise ValueError(f"Missing or linked selected postProcessing root: {name}")
        paths = sorted(root.rglob("*"))
        if any(p.is_symlink() for p in paths):
            raise ValueError(f"Linked source inside selected postProcessing: {name}")
        selected = [p for p in paths if p.is_file() and p.suffix == ".raw"]
        if not selected:
            raise ValueError(f"No .raw exports for selected case: {name}")
        sources.extend(selected)
    return sorted(sources)


def _convert_into(cfd, output, case_names=None, sources=None):
    sources = _sample_sources(cfd, case_names) if sources is None else sources
    if not sources:
        raise ValueError("No .raw exports discovered")
    records, series = [], defaultdict(list)
    outputs = {}
    for source in sources:
        sample = load_raw(source)
        relative = source.relative_to(cfd)
        csv_rel = Path("samples") / relative.with_suffix(".csv")
        vtp_rel = Path("samples") / relative.with_suffix(".vtp")
        checks = write_sample(sample, output / csv_rel, output / vtp_rel)
        record = dict(
            path=relative.as_posix(),
            sha256=sha256(source),
            field=sample.field,
            association=sample.association,
            units=sample.units,
            time_s=float(sample.time_text),
            time_text=sample.time_text,
            rows=len(sample.values),
            numeric_values=int(sample.values.size),
            coordinate_bounds_m=[
                sample.values[:, :3].min(0).tolist(),
                sample.values[:, :3].max(0).tolist(),
            ],
            value_bounds=[
                sample.values[:, 3:].min(0).tolist(),
                sample.values[:, 3:].max(0).tolist(),
            ],
            csv=csv_rel.as_posix(),
            vtp=vtp_rel.as_posix(),
            checks=checks,
        )
        records.append(record)
        series[(relative.parent.parent, source.stem)].append(record)
        outputs[csv_rel.as_posix()] = sha256(output / csv_rel)
        outputs[vtp_rel.as_posix()] = sha256(output / vtp_rel)
    collections = []
    for (parent, stem), entries in sorted(series.items()):
        relative = Path("collections") / parent / (stem + ".pvd")
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
        collection = ET.SubElement(root, "Collection")
        entries.sort(key=lambda r: r["time_s"])
        for entry in entries:
            ET.SubElement(
                collection,
                "DataSet",
                timestep=entry["time_text"],
                group="",
                part="0",
                file=os.path.relpath(output / entry["vtp"], path.parent),
            )
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        collections.append(
            dict(
                path=relative.as_posix(), count=len(entries), times_s=[e["time_s"] for e in entries]
            )
        )
        outputs[relative.as_posix()] = sha256(path)
    counts = dict(
        raw=len(sources),
        csv=len(sources),
        vtp=len(sources),
        pvd=len(collections),
        sample_rows=sum(r["rows"] for r in records),
        source_numeric_values=sum(r["numeric_values"] for r in records),
    )
    manifest = dict(
        schema_version=1,
        representation="samples_only_not_native_mesh",
        counts=counts,
        cases=dict(sorted(Counter(r["path"].split("/")[0] for r in records).items())),
        sources=records,
        collections=collections,
        output_sha256=outputs,
    )
    if case_names is not None:
        manifest["selected_case_names"] = list(case_names)
    (output / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return manifest


def verify_exports(cfd, output, *, paraview=True):
    """Independently read every source with NumPy and every VTP with installed VTK.

    With pvpython, also exercise PVDReader at EVERY archived time, not merely XML.
    Returns a report; this read-only API never writes or alters a bundle.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    cfd, output = Path(cfd).resolve(), Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    records = manifest["sources"]
    discovered = {
        p.relative_to(cfd).as_posix()
        for p in _sample_sources(cfd, manifest.get("selected_case_names"))
    }
    if len(records) != len(discovered) or {r["path"] for r in records} != discovered:
        raise ValueError("Source inventory mismatch")
    for name, expected in manifest["output_sha256"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"Output identity mismatch: {name}")
    csv_error = vtk_error = pv_error = 0.0
    total_rows = total_values = 0
    lookup = {}

    def check_polydata(poly, expected, field):
        if poly.IsA("vtkMultiBlockDataSet"):
            if poly.GetNumberOfBlocks() != 1:
                raise ValueError("Unexpected multi-block PVD result")
            poly = poly.GetBlock(0)
        if (
            poly.GetNumberOfPoints() != len(expected)
            or poly.GetNumberOfVerts() != len(expected)
            or poly.GetNumberOfPolys() != 0
        ):
            raise ValueError("VTP must contain sample vertices, not native mesh polygons")
        coords = vtk_to_numpy(poly.GetPoints().GetData())
        values = vtk_to_numpy(poly.GetPointData().GetArray(field)).reshape(len(expected), -1)
        actual = np.column_stack([coords, values])
        if actual.tobytes() != expected.tobytes():
            raise ValueError("VTK/ParaView values failed exact binary64 round trip")
        return float(np.max(np.abs(actual - expected)))

    for record in records:
        source = cfd / record["path"]
        if sha256(source) != record["sha256"]:
            raise ValueError(f"Source identity mismatch: {source}")
        expected = np.loadtxt(source, ndmin=2)
        # Separate parser from the producer, preserving all serialized binary64 values.
        with (output / record["csv"]).open() as stream:
            rows = list(csv.reader(stream))
        actual = np.asarray([row[1:] for row in rows[1:]], dtype=np.float64)
        times = np.asarray([row[0] for row in rows[1:]], dtype=np.float64)
        if actual.tobytes() != expected.tobytes() or not np.all(times == record["time_s"]):
            raise ValueError("CSV values/time failed exact readback")
        csv_error = max(csv_error, float(np.max(np.abs(actual - expected))))
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(output / record["vtp"]))
        reader.Update()
        if reader.GetErrorCode():
            raise ValueError("VTK reader returned an error")
        vtk_error = max(vtk_error, check_polydata(reader.GetOutput(), expected, record["field"]))
        if reader.GetOutput().GetFieldData().GetArray("TimeValue").GetValue(0) != record["time_s"]:
            raise ValueError("VTP time mismatch")
        total_rows += len(expected)
        total_values += expected.size
        lookup[(output / record["vtp"]).resolve()] = record
    steps = 0
    version = None
    if paraview:
        from paraview import servermanager
        from paraview.simple import Delete, GetParaViewVersion, PVDReader

        version = str(GetParaViewVersion())
        for series in manifest["collections"]:
            pvd = output / series["path"]
            reader = PVDReader(FileName=str(pvd))
            if list(reader.TimestepValues) != series["times_s"]:
                raise ValueError("ParaView time controls disagree with source timestamps")
            for node in ET.parse(pvd).findall("./Collection/DataSet"):
                record = lookup[(pvd.parent / node.attrib["file"]).resolve()]
                time = float(node.attrib["timestep"])
                if time != record["time_s"]:
                    raise ValueError("PVD source/time mismatch")
                reader.UpdatePipeline(time=time)
                actual = servermanager.Fetch(reader)
                expected = np.loadtxt(cfd / record["path"], ndmin=2)
                pv_error = max(pv_error, check_polydata(actual, expected, record["field"]))
                steps += 1
            Delete(reader)
        if steps != len(records):
            raise ValueError("PVD collections do not cover all sources exactly once")
    counts = dict(
        raw=len(records),
        csv=len(list(output.rglob("*.csv"))),
        vtp=len(list(output.rglob("*.vtp"))),
        pvd=len(list(output.rglob("*.pvd"))),
        sample_rows=total_rows,
        source_numeric_values=int(total_values),
    )
    if counts != manifest["counts"]:
        raise ValueError("Manifest counts disagree with actual readback")
    return dict(
        schema_version=1,
        counts=counts,
        vtp_files_verified=len(records),
        csv_files_verified=len(records),
        pvd_time_steps_verified=steps,
        vtk_max_abs_error=vtk_error,
        csv_max_abs_error=csv_error,
        paraview_max_abs_error=pv_error if paraview else None,
        binary64_bitwise_equal=True,
        vtk_version=vtk.vtkVersion.GetVTKVersion(),
        paraview_version=version,
        manifest_sha256=sha256(output / "manifest.json"),
    )


def triangles_intersect_disk(vertices, radius, tolerance=1e-12):
    """Test edges AND interior; the 1 pm tolerance is wall decimal precision."""
    a = np.asarray(vertices, dtype=float)
    b = np.roll(a, -1, axis=1)
    edge = b - a
    t = np.clip(-np.sum(a * edge, axis=2) / np.sum(edge * edge, axis=2), 0, 1)
    nearest = a + t[:, :, None] * edge
    distance = np.linalg.norm(nearest, axis=2).min(axis=1)
    cross = a[:, :, 0] * b[:, :, 1] - a[:, :, 1] * b[:, :, 0]
    contains_origin = np.all(cross >= 0, axis=1) | np.all(cross <= 0, axis=1)
    return contains_origin | (distance < radius - tolerance)


def circular_triangulation(points, outer_radius=0.24, body_radius=0.01, half_span=0.03):
    """Delaunay display support for complete circular cutting-plane samples.

    Native cutting-plane connectivity was not exported. No new sample values,
    mesh reconstruction, long-edge clipping or extrapolation is performed.
    """
    import matplotlib.tri as mtri

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite xyz points")
    if np.ptp(points[:, 2]) > 1e-9:
        raise ValueError("Sample points do not share a z plane")
    if len(np.unique(points[:, :2], axis=0)) != len(points):
        raise ValueError("Duplicate xy samples cannot be triangulated unambiguously")
    radii = np.linalg.norm(points[:, :2], axis=1)
    if radii.max() > outer_radius + 1e-9:
        raise ValueError("Samples fall outside the circular domain")
    tri = mtri.Triangulation(points[:, 0], points[:, 1])
    z = float(np.median(points[:, 2]))
    solid = abs(z) <= half_span
    mask = (
        triangles_intersect_disk(points[tri.triangles, :2], body_radius)
        if solid
        else np.zeros(len(tri.triangles), dtype=bool)
    )
    tri.set_mask(mask)
    retained = points[tri.get_masked_triangles(), :2]
    ab, ac = retained[:, 1] - retained[:, 0], retained[:, 2] - retained[:, 0]
    areas = np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]) / 2
    edge_lengths = np.linalg.norm(retained - np.roll(retained, -1, axis=1), axis=2)
    analytic_area = np.pi * (outer_radius**2 - (body_radius**2 if solid else 0))
    audit = dict(
        z_m=z,
        points=len(points),
        triangles=len(tri.triangles),
        retained_triangles=int((~mask).sum()),
        solid_intersecting_triangles_removed=int(mask.sum()),
        solid_present=solid,
        body_radius_m=body_radius if solid else 0,
        radius_range_m=[float(radii.min()), float(radii.max())],
        xy_bounds_m=[points[:, :2].min(0).tolist(), points[:, :2].max(0).tolist()],
        retained_area_m2=float(areas.sum()),
        analytic_fluid_area_m2=float(analytic_area),
        supported_area_fraction=float(areas.sum() / analytic_area),
        max_retained_edge_m=float(edge_lengths.max()),
        boundary_serialization_tolerance_m=1e-9,
        solid_intersection_tolerance_m=1e-12,
        interpolation="piecewise linear on masked Delaunay triangles; no extrapolation",
        outer_support="sample convex hull; unfilled chord-to-circle rim",
        masked_area=(
            "whole triangles intersecting midspan solid; conservative fluid-side slivers unfilled"
        ),
    )
    return tri, audit


def plot_circular_snapshot(
    cfd, output, *, allowed_root, time_text="5.6", case_name="cylinder_circular"
):
    """Plot one shared-time pair over the full circular domain, in a new folder."""
    import shutil
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    output = preflight_output(output, allowed_root, cfd)
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", time_text):
        raise ValueError("Unsafe/non-numeric snapshot time")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", case_name):
        raise ValueError("Unsafe case name: expected a single CFD subdirectory name")
    cfd = Path(cfd)
    selected = []
    for plane, expected_z in [("midspan", 0), ("tipplane", 0.035)]:
        source = cfd / case_name / "postProcessing/flowSlice" / time_text / f"vorticity_{plane}.raw"
        sample = load_raw(source)
        if sample.field != "vorticity" or not np.allclose(
            sample.values[:, 2], expected_z, atol=1e-9, rtol=0
        ):
            raise ValueError(f"Unexpected {plane} field or z coordinate")
        tri, audit = circular_triangulation(sample.values[:, :3])
        audit.update(
            plane=plane,
            source=source.relative_to(cfd).as_posix(),
            source_sha256=sha256(source),
            origin_supported=bool(tri.get_trifinder()(0, 0) >= 0),
            vorticity_z_range_s_inv=[
                float(sample.values[:, 5].min()),
                float(sample.values[:, 5].max()),
            ],
        )
        selected.append((sample, tri, audit))
    vmax = max(float(np.abs(s.values[:, 5]).max()) for s, _, _ in selected)
    vmax = max(vmax, 1e-12)
    norm = SymLogNorm(linthresh=1, vmin=-vmax, vmax=vmax, base=10)
    levels = norm.inverse(np.linspace(0, 1, 65))
    levels[0], levels[-1] = -vmax, vmax
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=output.parent))
    try:
        with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "pdf.fonttype": 42}):
            fig, axes = plt.subplots(1, 2, figsize=(12, 7.2))
            fig.subplots_adjust(left=0.07, right=0.88, bottom=0.20, top=0.80, wspace=0.18)
            for ax, (sample, tri, audit) in zip(axes, selected):
                image = ax.tricontourf(
                    tri, sample.values[:, 5], levels=levels, cmap="RdBu_r", norm=norm
                )
                ax.add_patch(plt.Circle((0, 0), 0.24, fill=False, color="#56616c", linewidth=0.7))
                if audit["solid_present"]:
                    ax.add_patch(
                        plt.Circle(
                            (0, 0), 0.01, facecolor="#333333", edgecolor="#333333", linewidth=0.5
                        )
                    )
                label = (
                    "Midspan" if audit["solid_present"] else "Beyond the +z tip (fluid across axis)"
                )
                ax.set(
                    xlim=(-0.24, 0.24),
                    ylim=(-0.24, 0.24),
                    xlabel="Streamwise x [m]",
                    title=f"{label}\nz = {audit['z_m']:g} m",
                    aspect="equal",
                )
                ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
                ax.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
                ax.spines[["top", "right"]].set_visible(False)
                audit["axis_limits_m"] = [list(ax.get_xlim()), list(ax.get_ylim())]
            axes[0].set_ylabel("Cross-flow y [m]")
            cax = fig.add_axes([0.91, 0.22, 0.017, 0.52])
            colorbar = fig.colorbar(image, cax=cax)
            colorbar.solids.set_rasterized(False)
            ticks = [t for t in [-1000, -100, -10, -1, 0, 1, 10, 100, 1000] if -vmax <= t <= vmax]
            colorbar.set_ticks(ticks)
            colorbar.set_label(r"Spanwise vorticity $\omega_z$ [s$^{-1}$]")
            fig.suptitle(f"Circular-domain cylinder flow | t = {time_text} s", y=0.97, fontsize=16)
            subtitle = fig.text(
                0.5,
                0.90,
                "Full sampled domain: R = 0.24 m  |  Cylinder: D = 0.02 m, z = −0.03 to +0.03 m",
                ha="center",
            )
            fig.text(
                0.5,
                0.085,
                "Sample-based linear contours; no extrapolation through solid "
                "or beyond sampled support.",
                ha="center",
                fontsize=9,
            )
            fig.text(
                0.5,
                0.048,
                "Shared symmetric-log colour scale (linear within ±1 s⁻¹); "
                f"scale limits ±{vmax:.3f} s⁻¹.  LES demonstration, not validation.",
                ha="center",
                fontsize=9,
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            title_top = max(ax.title.get_window_extent(renderer).y1 for ax in axes)
            layout = {
                "header_title_gap_pt": float(
                    (subtitle.get_window_extent(renderer).y0 - title_top) * 72 / fig.dpi
                )
            }
            for ax, (_, _, audit) in zip(axes, selected):
                origin, xunit, yunit = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
                audit["display_xy_scale_ratio"] = float(
                    np.linalg.norm(xunit - origin) / np.linalg.norm(yunit - origin)
                )
            fig.savefig(stage / "flow_vorticity_circular.png", dpi=180)
            fig.savefig(
                stage / "flow_vorticity_circular.pdf",
                metadata={"CreationDate": None, "ModDate": None},
            )
            plt.close(fig)
        report = dict(
            schema_version=1,
            time_s=float(time_text),
            vorticity_component="vorticity_z",
            layout=layout,
            colour_scale=dict(
                type="symmetric_log",
                linthresh_s_inv=1,
                limits_s_inv=[-vmax, vmax],
                levels=65,
                clipped_values=0,
            ),
            planes=[audit for _, _, audit in selected],
            output_sha256={p.name: sha256(p) for p in sorted(stage.iterdir())},
        )
        (stage / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        if output.exists():
            raise ValueError("Output destination appeared during plotting")
        stage.rename(output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def build_bundle(project, output):
    """Build exports, plots and identities as one new no-overwrite bundle."""
    import shutil
    import tempfile

    project = Path(project).resolve()
    allowed = project / "results/cylinder/viewable"
    output = preflight_output(output, allowed, project / "cfd")
    cfd = project / "cfd"
    run = project / "results/cylinder/native_circular/run.json"
    before = verify_pins(project, run)
    raw_before = {p.relative_to(cfd).as_posix(): sha256(p) for p in sorted(cfd.rglob("*.raw"))}
    protected = [
        "scripts/cylinder_circular.py",
        "scripts/cylinder_native.py",
        "results/cylinder/native_circular/flow_vorticity.png",
        "site/assets/flow_vorticity.png",
    ]
    extra_before = {p: sha256(project / p) for p in protected if (project / p).is_file()}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=output.parent))
    try:
        exports = convert_tree(cfd, stage / "exports", allowed_root=stage)
        figures = plot_circular_snapshot(cfd, stage / "figures", allowed_root=stage)
        raw_after = {p.relative_to(cfd).as_posix(): sha256(p) for p in sorted(cfd.rglob("*.raw"))}
        after = verify_pins(project, run)
        extra_after = {p: sha256(project / p) for p in extra_before}
        if raw_before != raw_after or before != after or extra_before != extra_after:
            raise ValueError("Original identities changed during conversion; bundle not published")
        if {r["path"]: r["sha256"] for r in exports["sources"]} != raw_before:
            raise ValueError("Conversion source identities disagree with task-start snapshot")
        report = dict(
            schema_version=1,
            counts=exports["counts"],
            cases=exports["cases"],
            representation="samples_only_not_native_mesh",
            generator_sha256=sha256(Path(__file__)),
            preservation={
                **after,
                "raw_sources_unchanged": len(raw_after),
                "extra_original_sha256": extra_after,
            },
            source_inventory="exports/manifest.json",
            plot_details="figures/manifest.json",
            snapshot_time_s=figures["time_s"],
            output_sha256={
                p.relative_to(stage).as_posix(): sha256(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        )
        (stage / "manifest.json").write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        )
        if output.exists():
            raise ValueError("Output destination appeared during build")
        stage.rename(output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Convert all cfd/*.raw and plot the circular 5.6 s pair"
    )
    build.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    build.add_argument(
        "--output", type=Path, help="New child under results/cylinder/viewable; default current"
    )
    verify = commands.add_parser("verify", help="Read every CSV/VTP/PVD using installed pvpython")
    verify.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    verify.add_argument("--bundle", type=Path)
    verify.add_argument(
        "--report", type=Path, required=True, help="New JSON under results/cylinder/viewable"
    )
    args = parser.parse_args(argv)
    project = args.project.resolve()
    default = project / "results/cylinder/viewable/current"
    if args.command == "build":
        report = build_bundle(project, args.output or default)
        print(
            json.dumps(
                {"counts": report["counts"], "preservation": report["preservation"]}, sort_keys=True
            )
        )
    else:
        target = preflight_output(
            args.report, project / "results/cylinder/viewable", project / "cfd"
        )
        bundle = args.bundle or default
        summary = json.loads((bundle / "manifest.json").read_text())
        for name, expected in summary["output_sha256"].items():
            if sha256(bundle / name) != expected:
                raise ValueError(f"Bundle identity mismatch: {name}")
        report = verify_exports(project / "cfd", bundle / "exports")
        report["preservation"] = verify_pins(
            project, project / "results/cylinder/native_circular/run.json"
        )
        if report["preservation"]["run_sha256"] != summary["preservation"]["run_sha256"]:
            raise ValueError("Original run manifest changed")
        report["bundle_manifest_sha256"] = sha256(bundle / "manifest.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.write("\n")
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
