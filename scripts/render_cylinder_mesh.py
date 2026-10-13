"""Read-only views of prelaunch hash-bound OpenFOAM mesh connectivity.

No CFD commands, generator imports, fabricated closure, or field/result analysis.
Historical parsers reopen paths and/or filter out non-quads, so this standalone
reader instead parses the exact hash-verified bytes and preserves every face ID.
"""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MESH_NAMES = ("points", "faces", "boundary")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _list_body(raw):
    text = raw.decode("ascii")
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    if re.search(r"\bformat\s+binary", text):
        raise ValueError("ASCII mesh required")
    text = re.sub(r"FoamFile\s*\{[^}]*\}", "", text).strip()
    match = re.fullmatch(r"(\d+)\s*\((.*)\)\s*;?", text, re.S)
    if not match:
        raise ValueError("invalid native list header/body")
    return int(match[1]), match[2]


def load_mesh(mesh_dir, expected_sha256):
    """Verify required identities before parsing; consume each file only once."""
    if set(expected_sha256) != set(MESH_NAMES) or any(
        not re.fullmatch(r"[0-9a-f]{64}", h) for h in expected_sha256.values()
    ):
        raise ValueError("complete expected mesh SHA256 identities required")
    payloads = {name: (Path(mesh_dir) / name).read_bytes() for name in MESH_NAMES}
    actual = {name: _sha(raw) for name, raw in payloads.items()}
    if actual != expected_sha256:
        raise ValueError("mesh SHA256 mismatch against prelaunch identities")
    count, body = _list_body(payloads["points"])
    rows = re.findall(r"\(([^()]*)\)", body)
    points = np.array([list(map(float, row.split())) for row in rows])
    if points.shape != (count, 3) or not np.isfinite(points).all():
        raise ValueError("finite native point count/coordinates required")
    count, body = _list_body(payloads["faces"])
    rows = re.findall(r"(\d+)\s*\(([^()]*)\)", body)
    if len(rows) != count or any(int(size) != 4 for size, _ in rows):
        raise ValueError("all native faces must be quadrilateral; never drop or renumber")
    faces = np.array([list(map(int, row.split())) for _, row in rows], dtype=np.int64)
    if faces.shape != (count, 4) or faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("invalid native face point indices/count")
    count, body = _list_body(payloads["boundary"])
    rows = re.findall(r"(\w+)\s*\{([^}]+)\}", body)
    if len(rows) != count or {name for name, _ in rows} != {"cylinder", "spanEnds", "farfield"}:
        raise ValueError("expected cylinder, spanEnds, farfield boundary patches")
    patch_ids = {}
    for name, patch in rows:
        start = int(re.search(r"startFace\s+(\d+)", patch)[1])
        size = int(re.search(r"nFaces\s+(\d+)", patch)[1])
        if start + size > len(faces):
            raise ValueError("boundary face range outside native faces")
        patch_ids[name] = np.arange(start, start + size)
    joined = np.concatenate(list(patch_ids.values()))
    if len(joined) != len(np.unique(joined)):
        raise ValueError("overlapping native boundary face ranges")
    for array in (points, faces, *patch_ids.values()):
        array.setflags(write=False)
    return dict(points=points, faces=faces, patch_ids=patch_ids, native_mesh_sha256=actual)


def _safe_path(path, root, kind):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError(f"{kind}: parent traversal forbidden")
    path = path.absolute()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"{kind} must be below {root}")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"{kind}: symlink path forbidden")
    return path


def _preflight(run_record, output):
    # Validate every destination before even opening source metadata.
    output = _safe_path(output, ROOT / "results/cylinder/mesh_views", "output")
    if output.exists():
        raise ValueError("existing output forbidden: no overwrite")
    return _safe_path(run_record, ROOT, "source"), output


def prepare_inputs(run_record, output):
    """Read only immutable mesh/report inputs, never pending CFD output fields."""
    run_record, _ = _preflight(run_record, output)
    run_bytes = run_record.read_bytes()
    record = json.loads(run_bytes)
    case = _safe_path(record["case_path"], ROOT, "source case")
    mesh = case / "constant/polyMesh"
    paths = {name: _safe_path(mesh / name, ROOT, "source mesh") for name in MESH_NAMES}
    pins = record.get("input_sha256", {})
    if any(str(path) not in pins for path in paths.values()):
        raise ValueError("missing prelaunch input_sha256 mesh identity")
    # Freeze expected identities before opening points/faces/boundary.
    expected = {name: pins[str(path)] for name, path in paths.items()}
    audit_path = _safe_path(run_record.parent / "mesh_audit.json", ROOT, "source audit")
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    command = audit["checkMesh_command"]
    log_path = _safe_path(command["log"], ROOT, "source checkMesh log")
    log_bytes = log_path.read_bytes()
    if _sha(log_bytes) != command["log_sha256"]:
        raise ValueError("checkMesh log identity mismatch")
    text = log_bytes.decode("utf-8")
    cells = int(record["cell_count"])
    if (
        cells <= 0
        or cells != audit["cell_count"]
        or not re.search(rf"\bcells:\s+{cells}\b", text)
        or "Mesh OK." not in text
        or re.search(r"Failed\s+\d+\s+mesh", text)
    ):
        raise ValueError("recorded mesh count/checkMesh acceptance mismatch")
    data = load_mesh(mesh, expected)
    counts = {"points": len(data["points"]), "faces": len(data["faces"]), "cells": cells}
    for key in ("points", "faces"):
        if not re.search(rf"(?m)^\s*{key}:\s+{counts[key]}\b", text):
            raise ValueError("native mesh counts differ from recorded checkMesh")
    data.update(
        cell_count=cells,
        counts=counts,
        wall_audit=audit["wall_geometry"],
        source_evidence_kind=record.get("evidence_kind", "unknown"),
        source_record=str(run_record.relative_to(ROOT)),
        source_record_sha256=_sha(run_bytes),
        mesh_audit_sha256=_sha(audit_bytes),
        checkMesh_log_sha256=_sha(log_bytes),
    )
    return data


def _edge_ids(faces):
    pairs = np.stack((faces, np.roll(faces, -1, axis=1)), axis=-1).reshape(-1, 2)
    return np.unique(np.sort(pairs, axis=1), axis=0)


def select_geometry(data):
    """Native face/edge IDs only; cuts select faces, never manufacture caps."""
    p, f, patch = data["points"], data["faces"], data["patch_ids"]
    wall = f[patch["cylinder"]]
    wall_z = p[wall, 2]
    directed = np.stack((wall, np.roll(wall, -1, axis=1)), axis=-1).reshape(-1, 2)
    _, incidence = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    oriented = directed[np.lexsort(directed.T)]
    reversed_edges = directed[:, ::-1]
    reversed_edges = reversed_edges[np.lexsort(reversed_edges.T)]
    closed = bool(np.all(incidence == 2) and np.array_equal(oriented, reversed_edges))
    cap_minus = int(np.all(np.isclose(wall_z, -0.03, atol=1e-10, rtol=0), axis=1).sum())
    cap_plus = int(np.all(np.isclose(wall_z, 0.03, atol=1e-10, rtol=0), axis=1).sum())
    audit = data["wall_audit"]
    if (
        not closed
        or not audit["closed"]
        or (len(wall), cap_minus, cap_plus)
        != (audit["faces"], audit["cap_minus_faces"], audit["cap_plus_faces"])
    ):
        raise ValueError("native wall/cap counts or closure differ from mesh audit")
    section = np.flatnonzero(np.all(np.isclose(p[f, 2], 0, atol=1e-10, rtol=0), axis=1))
    if not len(section):
        raise ValueError("no native z=0 faces; section cannot be fabricated")
    far = patch["farfield"][p[f[patch["farfield"]], 1].mean(axis=1) > 0]
    bottom = patch["spanEnds"][p[f[patch["spanEnds"]], 2].mean(axis=1) < 0]
    radius = np.hypot(p[:, 0], p[:, 1])
    near = np.flatnonzero(
        np.all(radius[f] < 0.016, axis=1)
        & (p[f, 1].mean(axis=1) > 0)
        & (np.abs(p[f, 2].mean(axis=1)) < 0.045)
    )
    geom = dict(
        section_ids=section,
        wall_ids=patch["cylinder"],
        far_ids=far,
        bottom_ids=bottom,
        near_ids=near,
    )
    for name in ("section", "far", "bottom", "near"):
        geom[name + "_edges"] = _edge_ids(f[geom[name + "_ids"]])
    geom["info"] = dict(
        section_z_m=0.0,
        section_tolerance_m=1e-10,
        section_faces=len(section),
        section_edges=len(geom["section_edges"]),
        wall_faces=len(wall),
        cap_minus_faces=cap_minus,
        cap_plus_faces=cap_plus,
        wall_closed_edge_incidence=closed,
        wall_unique_edges=len(incidence),
        farfield_faces=len(patch["farfield"]),
        span_end_faces=len(patch["spanEnds"]),
        displayed_rear_farfield_faces=len(far),
        displayed_bottom_faces=len(bottom),
        displayed_near_faces=len(near),
        point_bounds_m=[p.min(axis=0).tolist(), p.max(axis=0).tolist()],
        display_treatment="Display-only cutaway: rear farfield (face mean y>0), lower span "
        "end (face mean z<0), native z=0 faces; detail retains faces with all r<0.016 m, "
        "mean y>0 and |mean z|<0.045 m. Complete original cylinder wall in both 3D panels; "
        "no invented caps, interpolation, renumbering or mesh modification.",
    )
    for array in geom.values():
        if isinstance(array, np.ndarray):
            array.setflags(write=False)
    return geom


def build_figures(data, geom):
    """Fixed export dimensions and dimensional box aspects; layout only."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator, FormatStrFormatter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    p, f = data["points"], data["faces"]
    bounds = np.array(geom["info"]["point_bounds_m"])
    section = p[geom["section_edges"]]
    cross = Figure(figsize=(8, 8), dpi=200)
    FigureCanvasAgg(cross)
    ax = cross.add_axes((0.13, 0.12, 0.81, 0.79))
    ax.add_collection(LineCollection(section[:, :, :2], colors="#223963", linewidths=0.35))
    ax.set(
        xlim=bounds[:, 0],
        ylim=bounds[:, 1],
        aspect="equal",
        xlabel="streamwise x [m]",
        ylabel="cross-flow y [m]",
        title="Native midspan mesh | z = 0 m",
    )
    cross.text(
        0.5,
        0.025,
        f"{data['cell_count']:,} cells in full mesh | Actual points / faces",
        ha="center",
        fontsize=10,
    )
    three = Figure(figsize=(14, 7.5), dpi=200)
    FigureCanvasAgg(three)
    axes = [
        three.add_axes((0.02, 0.18, 0.46, 0.67), projection="3d"),
        three.add_axes((0.53, 0.18, 0.43, 0.67), projection="3d"),
    ]
    for ax in axes:
        # All accepted faces are quads: no ragged Poly3DCollection padding.
        ax.add_collection3d(
            Poly3DCollection(
                p[f[geom["wall_ids"]]], facecolor="#d8e3ec", edgecolor="#314f6a", linewidths=0.22
            )
        )
        ax.view_init(elev=24, azim=-55)
        ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        ax.tick_params(labelsize=9, pad=3)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.labelpad = 13
            axis.set_major_formatter(FormatStrFormatter("%.2f"))
    for name, color, alpha in (
        ("far", "#8d9eac", 0.6),
        ("bottom", "#8d9eac", 0.5),
        ("section", "#2b5479", 0.7),
    ):
        if len(geom[name + "_edges"]):
            axes[0].add_collection3d(
                Line3DCollection(
                    p[geom[name + "_edges"]], colors=color, linewidths=0.22, alpha=alpha
                ),
                autolim=False,
            )
    axes[0].set(
        xlim=bounds[:, 0],
        ylim=bounds[:, 1],
        zlim=bounds[:, 2],
        title="Full domain | display-only cutaway",
    )
    axes[0].set_box_aspect(bounds[1] - bounds[0])
    axes[1].add_collection3d(
        Line3DCollection(p[geom["near_edges"]], colors="#608cac", linewidths=0.17, alpha=0.4)
    )
    axes[1].set(
        xlim=(-0.016, 0.016),
        ylim=(-0.016, 0.016),
        zlim=(-0.045, 0.045),
        title="Body detail | native closed caps",
    )
    axes[1].set_box_aspect((0.032, 0.032, 0.09))
    # Tall physical aspect reduces projected tick offsets; keep z glyphs off the spine.
    axes[1].tick_params(axis="z", pad=16)
    axes[1].zaxis.labelpad = 32
    for ax, xy, z in (
        (axes[0], [-0.15, 0, 0.15], [-0.05, 0, 0.05]),
        (axes[1], [-0.01, 0.01], [-0.03, 0, 0.03]),
    ):
        ax.xaxis.set_major_locator(FixedLocator(xy))
        ax.yaxis.set_major_locator(FixedLocator(xy))
        ax.zaxis.set_major_locator(FixedLocator(z))
    three.suptitle(f"Native mesh connectivity | {data['cell_count']:,} cells", y=0.95, fontsize=15)
    three.text(
        0.5,
        0.06,
        "Front fluid mesh omitted for visibility; complete body retained. "
        "Axes preserve physical proportions.",
        ha="center",
        fontsize=11,
    )
    return {"mesh_cross_section": cross, "mesh_3d": three}


def text_layout_audit(fig):
    """Measure projected *drawn* ticks, axis labels, titles and notes, in pixels."""
    from itertools import combinations

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.append(ax.title)
        for name in ("x", "y", "z"):
            axis = getattr(ax, name + "axis", None)
            if axis is None:
                continue
            artists.extend((axis.label, axis.get_offset_text()))
            lo, hi = sorted(getattr(ax, "get_" + name + "lim")())
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi:
                    artists.extend((tick.label1, tick.label2))
    boxes = [
        (a.get_text(), a.get_window_extent(renderer))
        for a in artists
        if a.get_visible() and a.get_text().strip()
    ]
    collisions, distances = [], []
    for (a, x), (b, y) in combinations(boxes, 2):
        dx = max(x.x0 - y.x1, y.x0 - x.x1, 0)
        dy = max(x.y0 - y.y1, y.y0 - x.y1, 0)
        distances.append(float(np.hypot(dx, dy)))
        if x.overlaps(y):
            collisions.append([a, b])
    strokes = []
    for ax in fig.axes:
        if getattr(ax, "name", "") == "3d":
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                strokes.append(axis.line)
                strokes.extend(t.tick1line for t in axis.get_major_ticks())
    stroke_collisions = []
    for text, box in boxes:
        for line in strokes:
            path = line.get_path().transformed(line.get_transform())
            if line.get_visible() and path.intersects_bbox(box.padded(4), filled=False):
                stroke_collisions.append(text)
    return dict(
        text_count=len(boxes),
        collisions=collisions,
        text_axis_stroke_collisions=stroke_collisions,
        axis_stroke_clearance_test_px=4,
        minimum_clearance_px=min(distances, default=0.0),
        outside_canvas=[
            a
            for a, b in boxes
            if b.x0 < 0 or b.y0 < 0 or b.x1 > fig.bbox.width or b.y1 > fig.bbox.height
        ],
        boxes=[dict(text=a, bounds_px=list(b.bounds)) for a, b in boxes],
    )


def _geometry_hashes(data, geom):
    arrays = {k: v for k, v in geom.items() if isinstance(v, np.ndarray)}
    arrays.update(points=data["points"], faces=data["faces"])
    return {k: _sha(v.tobytes()) for k, v in sorted(arrays.items())}


def _publish_no_replace(stage, output):
    """Atomic Linux directory publication; even a competing empty target fails."""
    import ctypes
    import os

    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(output), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def render(run_record, output):
    """Build and check six artifacts in project-local staging; never overwrite."""
    import shutil
    import tempfile

    run_record, output = _preflight(run_record, output)
    data = prepare_inputs(run_record, output)
    geom = select_geometry(data)
    before = _geometry_hashes(data, geom)
    # No directory writes until source, output, identities and geometry pass.
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".mesh-stage-", dir=output.parent))
    try:
        import platform

        import matplotlib as mpl

        with mpl.rc_context(
            {
                "svg.hashsalt": "refined-native-mesh-v1",
                "font.family": "DejaVu Sans",
                "text.usetex": False,
                "font.size": 10,
                "savefig.bbox": None,
            }
        ):
            figures = build_figures(data, geom)
            audits = {}
            for stem, fig in figures.items():
                audits[stem] = text_layout_audit(fig)
                if (
                    audits[stem]["collisions"]
                    or audits[stem]["text_axis_stroke_collisions"]
                    or audits[stem]["outside_canvas"]
                    or (audits[stem]["minimum_clearance_px"] < 4)
                ):
                    raise ValueError(f"projected text layout failed: {stem}: {audits[stem]}")
                fig.savefig(stage / f"{stem}.png", dpi=200)
                fig.savefig(stage / f"{stem}.pdf", metadata={"CreationDate": None, "ModDate": None})
                fig.savefig(stage / f"{stem}.svg", metadata={"Date": None})
            if before != _geometry_hashes(data, geom):
                raise ValueError("layout/export mutated numerical mesh arrays")
        hashes = {p.name: _sha(p.read_bytes()) for p in sorted(stage.iterdir())}
        report = dict(
            schema_version=1,
            evidence_kind="native_mesh_views_only",
            source_evidence_kind=data["source_evidence_kind"],
            source_record=data["source_record"],
            source_record_sha256=data["source_record_sha256"],
            mesh_audit_sha256=data["mesh_audit_sha256"],
            checkMesh_log_sha256=data["checkMesh_log_sha256"],
            native_mesh_sha256=data["native_mesh_sha256"],
            mesh={**data["counts"], **geom["info"]},
            geometry_array_sha256=before,
            geometry_unchanged_after_export=True,
            layout=audits,
            artifacts={
                "mesh_cross_section": {"inches": [8, 8], "png_pixels": [1600, 1600]},
                "mesh_3d": {"inches": [14, 7.5], "png_pixels": [2800, 1500]},
            },
            artifact_sha256=hashes,
            renderer_sha256=_sha(Path(__file__).read_bytes()),
            versions={
                "python": platform.python_version(),
                "numpy": np.__version__,
                "matplotlib": mpl.__version__,
            },
            limitations="Connectivity and layout verification only. No new checkMesh/CFD commands; "
            "no LES adequacy, flow completion, convergence, or physical validation claims.",
        )
        (stage / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        _preflight(run_record, output)
        _publish_no_replace(stage, output)
        return report
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "results/cylinder/native_circular_60s/run.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/cylinder/mesh_views/refined_10s"
    )
    args = parser.parse_args()
    try:
        report = render(args.run, args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Mesh render rejected: {exc}\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "mesh": report["mesh"],
                "artifact_count": len(report["artifact_sha256"]),
                "text_clearance_px": {
                    k: v["minimum_clearance_px"] for k, v in report["layout"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
