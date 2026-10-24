"""Geometry-first one-layer 2D MRL mesh and native motion admission.

No dependencies on retired turbine/cylinder code. Five disconnected point
regions meet in four paired circular cyclicAMIs; no artificial span caps.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

OMEGA = 10.472
DEPTH = 0.01
CENTRES = np.array(
    [[0, 0.055], [math.sqrt(3) * 0.055 / 2, -0.0275], [-math.sqrt(3) * 0.055 / 2, -0.0275]]
)
ANGLES = (90, 30, -30)
ZONES = (None, "carrierZone", "blade0Zone", "blade1Zone", "blade2Zone")


def header(name: str, kind: str = "dictionary") -> str:
    return (
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        f"    class {kind};\n    object {name};\n}}\n"
    )


def read_points(path: Path) -> np.ndarray:
    text = path.read_text().split("}", 1)[1]
    rows = re.findall(r"\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", text)
    # Match native v2206 strtold -> scalar conversion, not Python direct double.
    return np.asarray(rows, dtype=np.longdouble).astype(float)


def dynamic_dictionary() -> str:
    def rotating(origin, omega):
        return (
            "solidBodyMotionFunction rotatingMotion;\nrotatingMotionCoeffs "
            f"{{ origin ({origin[0]:.20g} {origin[1]:.20g} 0); axis (0 0 1); "
            f"omega {omega:g}; }}\n"
        )

    text = header("dynamicMeshDict") + (
        'dynamicFvMesh dynamicMotionSolverFvMesh;\nmotionSolverLibs ("libfvMotionSolvers.so");\n'
        "solver multiSolidBodyMotionSolver;\nmultiSolidBodyMotionSolverCoeffs\n{\n"
        "carrierZone\n{\n" + rotating((0, 0), -OMEGA) + "}\n"
    )
    for i, centre in enumerate(CENTRES):
        text += (
            f"blade{i}Zone\n{{\nsolidBodyMotionFunction multiMotion;\n"
            "multiMotionCoeffs\n{\norbit\n{\n" + rotating((0, 0), -OMEGA) + "}\n"
            "relativePitch\n{\n" + rotating(centre, OMEGA / 2) + "}\n}\n}\n"
        )
    return text + "}\n"


def _gmsh():
    try:
        return importlib.import_module("gmsh")
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[2]
        paths = sorted((root / ".cache/mrl-gmsh-netgen/lib").glob("python*/site-packages"))
        if not paths:
            raise RuntimeError("Existing isolated gmsh dependency not found") from None
        sys.path.insert(0, str(paths[0]))
        return importlib.import_module("gmsh")


def _planar_geometry():
    gmsh = _gmsh()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("mrl2d")
        cad = gmsh.model.occ
        far = cad.addCircle(0, 0, 0, 0.6)
        carrier = cad.addCircle(0, 0, 0, 0.105)
        circles = [cad.addCircle(x, y, 0, 0.04) for x, y in CENTRES]
        ellipses = [cad.addEllipse(x, y, 0, 0.035, 0.0035) for x, y in CENTRES]
        for ellipse, (x, y), angle in zip(ellipses, CENTRES, ANGLES, strict=True):
            cad.rotate([(1, ellipse)], x, y, 0, 0, 0, 1, math.radians(angle))
        loops = {c: cad.addCurveLoop([c]) for c in [far, carrier, *circles, *ellipses]}
        surfaces = [
            cad.addPlaneSurface([loops[far], loops[carrier]]),
            cad.addPlaneSurface([loops[carrier], *(loops[c] for c in circles)]),
        ]
        surfaces += [
            cad.addPlaneSurface([loops[c], loops[e]])
            for c, e in zip(circles, ellipses, strict=True)
        ]
        cad.synchronize()
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.00006)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.025)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 40)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        field = gmsh.model.mesh.field
        distance = field.add("Distance")
        field.setNumbers(distance, "CurvesList", ellipses)
        field.setNumber(distance, "Sampling", 500)
        wall_size = field.add("Threshold")
        for key, value in {
            "InField": distance,
            "SizeMin": 0.0004,
            "SizeMax": 0.025,
            "DistMin": 0.001,
            "DistMax": 0.20,
        }.items():
            field.setNumber(wall_size, key, value)
        interface_distance = field.add("Distance")
        field.setNumbers(interface_distance, "CurvesList", [carrier, *circles])
        field.setNumber(interface_distance, "Sampling", 500)
        interface_size = field.add("Threshold")
        for key, value in {
            "InField": interface_distance,
            "SizeMin": 0.002,
            "SizeMax": 0.025,
            "DistMin": 0.005,
            "DistMax": 0.25,
        }.items():
            field.setNumber(interface_size, key, value)
        minimum = field.add("Min")
        field.setNumbers(minimum, "FieldsList", [wall_size, interface_size])
        field.setAsBackgroundMesh(minimum)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        nodes = dict(
            zip(map(int, tags), np.asarray(coordinates).reshape(-1, 3)[:, :2], strict=True)
        )
        regions = []
        for surface in surfaces:
            types, _, connectivity = gmsh.model.mesh.getElements(2, surface)
            if list(types) != [2]:
                raise ValueError(f"Expected linear triangles, got {types}")
            regions.append(np.asarray(connectivity[0], dtype=int).reshape(-1, 3))
        curves = {}
        for curve in [far, carrier, *circles, *ellipses]:
            types, _, conn = gmsh.model.mesh.getElements(1, curve)
            if list(types) != [1]:
                raise ValueError("Expected linear curve segments")
            curves[curve] = {
                tuple(sorted(map(int, pair))) for pair in np.asarray(conn[0]).reshape(-1, 2)
            }
        boundaries = [
            {far: "farfield", carrier: "carrierAMI_slave"},
            {
                carrier: "carrierAMI_master",
                **{c: f"blade{i}AMI_slave" for i, c in enumerate(circles)},
            },
        ]
        boundaries += [
            {c: f"blade{i}AMI_master", e: f"blade{i}"}
            for i, (c, e) in enumerate(zip(circles, ellipses, strict=True))
        ]
        return nodes, regions, curves, boundaries
    finally:
        gmsh.finalize()


def build_base(case: Path, *, max_cells: int = 250000) -> dict:
    """Build a fresh geometry-based ASCII native mesh; does not claim admission."""
    case = Path(case).resolve()
    if case.exists():
        raise FileExistsError(case)
    nodes, regions, curves, boundaries = _planar_geometry()
    count = sum(len(region) for region in regions)
    if count > min(max_cells, 250000):
        raise ValueError(f"Initial fluid-cell ceiling exceeded: {count}")
    mesh_dir = case / "constant/polyMesh"
    mesh_dir.mkdir(parents=True)
    points, point_region, cells, cell_region = [], [], [], []
    boundary_edges = {}
    farfield_nodes = set()
    for region_id, triangles in enumerate(regions):
        unique = sorted(set(map(int, triangles.ravel())))
        index = {tag: len(points) + i * 2 for i, tag in enumerate(unique)}
        for tag in unique:
            x, y = nodes[tag]
            points.extend([(x, y, -DEPTH / 2), (x, y, DEPTH / 2)])
            point_region.extend([region_id, region_id])
        for curve, name in boundaries[region_id].items():
            for a, b in curves[curve]:
                key = tuple(sorted((index[a], index[b], index[a] + 1, index[b] + 1)))
                boundary_edges[key] = name
                if name == "farfield":
                    farfield_nodes.update([index[a], index[b]])
        for triangle in triangles:
            tri = [int(n) for n in triangle]
            a, b, c = (nodes[n] for n in tri)
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0:
                tri.reverse()
            cells.append([index[n] for n in tri])
            cell_region.append(region_id)
    face_map = {}
    for cell, tri in enumerate(cells):
        a, b, c = tri
        cell_faces = [(c, b, a), (a + 1, b + 1, c + 1)]
        cell_faces += [(p, q, q + 1, p + 1) for p, q in [(a, b), (b, c), (c, a)]]
        for face in cell_faces:
            key = tuple(sorted(face))
            if key in face_map:
                face_map[key][2] = cell
            else:
                face_map[key] = [face, cell, None]
    internal = sorted(
        (entry for entry in face_map.values() if entry[2] is not None),
        key=lambda entry: (entry[1], entry[2]),
    )
    patch_faces = {}
    for key, entry in face_map.items():
        if entry[2] is None:
            patch = "frontAndBack" if len(key) == 3 else boundary_edges[key]
            patch_faces.setdefault(patch, []).append(entry)
    patches, ordered = {}, internal.copy()
    for name in sorted(patch_faces):
        kind = (
            "cyclicAMI"
            if "AMI_" in name
            else (
                "empty" if name == "frontAndBack" else ("patch" if name == "farfield" else "wall")
            )
        )
        patches[name] = {"type": kind, "faces": len(patch_faces[name]), "startFace": len(ordered)}
        ordered.extend(patch_faces[name])

    def write_list(name, kind, items):
        (mesh_dir / name).write_text(
            header(name, kind) + f"{len(items)}\n(\n" + "\n".join(items) + "\n)\n"
        )

    write_list(
        "points", "vectorField", ["(" + " ".join(f"{n:.20g}" for n in p) + ")" for p in points]
    )
    write_list(
        "faces", "faceList", [f"{len(f)}(" + " ".join(map(str, f)) + ")" for f, _, _ in ordered]
    )
    write_list("owner", "labelList", [str(owner) for _, owner, _ in ordered])
    write_list("neighbour", "labelList", [str(neighbour) for _, _, neighbour in internal])
    blocks = []
    for name, patch in patches.items():
        extra = ""
        if patch["type"] == "cyclicAMI":
            other = (
                name.replace("_master", "_slave")
                if "_master" in name
                else name.replace("_slave", "_master")
            )
            extra = (
                f"neighbourPatch {other}; transform noOrdering;\n"
                "matchTolerance 0.0001; lowWeightCorrection -1;\n"
            )
        blocks.append(
            f"{name}\n{{\ntype {patch['type']};\nnFaces {patch['faces']};\n"
            f"startFace {patch['startFace']};\n{extra}}}"
        )
    write_list("boundary", "polyBoundaryMesh", blocks)
    zone_counts, blocks = {}, []
    for region_id, name in enumerate(ZONES):
        if name is None:
            continue
        labels = [i for i, r in enumerate(cell_region) if r == region_id]
        zone_counts[name] = len(labels)
        blocks.append(
            f"{name}\n{{\ntype cellZone;\ncellLabels List<label>\n{len(labels)}\n(\n"
            + "\n".join(map(str, labels))
            + "\n);\n}"
        )
    write_list("cellZones", "regIOobject", blocks)
    (case / "constant/dynamicMeshDict").write_text(dynamic_dictionary())
    points_array = np.array(points)
    np.savez_compressed(
        case / "mesh_reference.npz",
        points=points_array,
        point_region=np.array(point_region),
        cell_region=np.array(cell_region),
    )
    manifest = {
        "case": str(case),
        "cells": count,
        "points": len(points),
        "span_layers": 1,
        "depth_m": DEPTH,
        "patches": patches,
        "zones": zone_counts,
        "farfield_radius_m": 0.6,
        "carrier_radius_m": 0.105,
        "blade_zone_radius_m": 0.04,
        "farfield_max_radius_error_m": float(
            np.max(np.abs(np.linalg.norm(points_array[sorted(farfield_nodes), :2], axis=1) - 0.6))
        ),
        "admitted": False,
        "geometry": "three elliptical blades; circular open farfield; one-layer triangular prisms",
    }
    (case / "mesh_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _check_summary(text: str) -> dict:
    if "Mesh OK." not in text or re.search(r"Failed\s+\d+\s+mesh checks", text):
        raise ValueError("Full native checkMesh did not admit the mesh")
    cells = int(re.search(r"^\s*cells:\s+(\d+)", text, re.MULTILINE).group(1))
    directions = {}
    for kind in ("geometric", "solution"):
        match = re.search(rf"Mesh has 2 {kind} .*?\(([01]) ([01]) ([01])\)", text)
        if not match or list(map(int, match.groups())) != [1, 1, 0]:
            raise ValueError(f"Expected x-y-only {kind} directions")
        directions[kind] = list(map(int, match.groups()))
    patterns = {
        "max_aspect_ratio": r"Max aspect ratio = ([0-9.eE+-]+)",
        "max_nonorthogonality_deg": r"Mesh non-orthogonality Max: ([0-9.eE+-]+)",
        "max_skewness": r"Max skewness = ([0-9.eE+-]+)",
        "min_edge_m": r"Min/max edge length = ([0-9.eE+-]+)",
        "min_determinant": r"Cell determinant .*?minimum: ([0-9.eE+-]+)",
        "min_interpolation_weight": r"Face interpolation weight : minimum: ([0-9.eE+-]+)",
        "min_volume_ratio": r"Face volume ratio : minimum: ([0-9.eE+-]+)",
    }
    quality = {name: float(re.search(pattern, text).group(1)) for name, pattern in patterns.items()}
    return {"cells": cells, "directions": directions, "quality": quality}


def _write_probe_system(case: Path) -> None:
    system = case / "system"
    system.mkdir(exist_ok=True)
    (system / "controlDict").write_text(
        header("controlDict")
        + """
application moveDynamicMesh;
startFrom startTime; startTime 0; stopAt endTime; endTime 0.6;
deltaT 0.05; writeControl timeStep; writeInterval 1;
purgeWrite 0; writeFormat ascii; writePrecision 20; writeCompression off;
timeFormat general; timePrecision 10; runTimeModifiable false;
"""
    )
    (system / "fvSchemes").write_text(
        header("fvSchemes")
        + """
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    )
    (system / "fvSolution").write_text(
        header("fvSolution")
        + """
solvers {}
PIMPLE { nOuterCorrectors 1; nCorrectors 1; nNonOrthogonalCorrectors 0; }
"""
    )


def _native_stage(case: Path, name: str, args: list[str], stages: list[dict]) -> str:
    import os
    import shlex
    import subprocess
    import time

    bashrc = os.environ.get("MRL2D_FOAM_BASHRC")
    if not bashrc or not Path(bashrc).is_file():
        raise ValueError("MRL2D_FOAM_BASHRC must name an existing OpenFOAM v2206 bashrc")
    logs = case / "logs"
    logs.mkdir(exist_ok=True)
    command = (
        f"source {shlex.quote(bashrc)} >/dev/null 2>&1 || exit 121; "
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; exec "
        + shlex.join([*args, "-case", str(case)])
    )
    start = time.monotonic()
    result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=120)
    text = result.stdout + result.stderr
    log = logs / f"{name}.log"
    log.write_text(text)
    stages.append(
        {
            "name": name,
            "case": str(case),
            "argv": [*args, "-case", str(case)],
            "return_code": result.returncode,
            "wall_seconds": time.monotonic() - start,
            "log": str(log),
        }
    )
    if result.returncode:
        raise RuntimeError(f"Native {name} returned {result.returncode}; see {log}")
    return text


def _weights(path: Path) -> np.ndarray:
    import base64
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    data = root.find('.//CellData/DataArray[@Name="weightsSum"]')
    if data is None:
        raise ValueError(f"Missing per-face weightsSum: {path}")
    if data.get("format") == "ascii":
        return np.fromstring(data.text, sep=" ")
    raw = base64.b64decode("".join(data.text.split()))
    prefix = "<" if root.get("byte_order", "LittleEndian") == "LittleEndian" else ">"
    header_size = 8 if root.get("header_type", "UInt32") == "UInt64" else 4
    dtype = prefix + {"Float32": "f4", "Float64": "f8"}[data.get("type")]
    count = int.from_bytes(raw[:header_size], "little" if prefix == "<" else "big")
    if len(raw) != count + header_size:
        raise ValueError(f"Unexpected VTK byte count: {path}")
    return np.frombuffer(raw[header_size:], dtype=dtype).astype(float)


def verify_native(case: Path) -> dict:
    """Admit full 2D native quality and 12 independent saved motion states.

    Leaves the base constant mesh unchanged; the sibling ``-motion`` case holds
    native points, per-face AMI weights and full per-state checkMesh logs.
    """
    import shutil

    case = Path(case).resolve()
    manifest_path = case / "mesh_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    stages = []
    manifest["native_stages"] = stages
    _write_probe_system(case)
    try:
        text = _native_stage(
            case,
            "checkMesh-initial",
            ["checkMesh", "-allGeometry", "-allTopology", "-constant"],
            stages,
        )
        initial = _check_summary(text)
        if initial["cells"] != manifest["cells"] or initial["cells"] > 250000:
            raise ValueError("Actual native count disagrees or exceeds cell ceiling")
        manifest["native_cells"] = initial["cells"]
        manifest["native_directions"] = initial["directions"]
        manifest["quality"] = initial["quality"]
        motion_case = case.with_name(case.name + "-motion")
        if motion_case.exists():
            raise FileExistsError(motion_case)
        motion_case.mkdir()
        shutil.copytree(case / "constant", motion_case / "constant")
        _write_probe_system(motion_case)
        _native_stage(motion_case, "moveDynamicMesh", ["moveDynamicMesh", "-checkAMI"], stages)
        references = np.load(case / "mesh_reference.npz")
        points0, regions = references["points"], references["point_region"]
        z_tolerance = 8 * np.finfo(float).eps * DEPTH
        evidence = {
            "case": str(motion_case),
            "states": [],
            "max_xy_error_m": 0.0,
            "max_z_error_m": 0.0,
            "all_faces_covered": True,
            "z_roundoff_tolerance_m": z_tolerance,
            "z_roundoff_reason": (
                "Native quaternion q*(0,v)*conjugate(q) roundoff; no coordinate projection"
            ),
        }
        for step in range(1, 13):
            time_value = step / 20
            time_name = f"{time_value:g}"
            written = read_points(motion_case / time_name / "polyMesh/points")
            expected = points0.copy()

            def rotate(xy, angle):
                c, s = math.cos(angle), math.sin(angle)
                return xy @ np.array([[c, s], [-s, c]])

            expected[regions == 1, :2] = rotate(points0[regions == 1, :2], -OMEGA * time_value)
            for blade, centre in enumerate(CENTRES):
                selected = regions == blade + 2
                expected[selected, :2] = rotate(
                    points0[selected, :2] - centre, -OMEGA * time_value / 2
                ) + rotate(centre, -OMEGA * time_value)
            xy_error = float(np.max(np.abs(written[:, :2] - expected[:, :2])))
            z_error = float(np.max(np.abs(written[:, 2] - points0[:, 2])))
            if xy_error > 1e-12 or z_error > z_tolerance:
                raise ValueError(f"Native motion law failed at {time_name}: {xy_error}, {z_error}")
            coverage_files = sorted(
                (motion_case / "postProcessing/checkAMI").glob(f"*_{time_name}.vtp")
            )
            if len(coverage_files) != 8:
                raise ValueError(
                    f"Expected all 8 AMI coverage files at {time_name}, got {len(coverage_files)}"
                )
            coverage = []
            names = sorted(manifest["patches"])
            for path in coverage_files:
                values = _weights(path)
                match = re.fullmatch(r"patch(\d+)-(src|tgt)_.+", path.stem)
                patch_name = names[int(match[1])]
                if match[2] == "tgt":
                    patch_name = patch_name.replace("_master", "_slave")
                if len(values) != manifest["patches"][patch_name]["faces"]:
                    raise ValueError(f"AMI array count mismatch: {path}")
                if not np.all(np.isfinite(values)) or values.min() < 0.995 or values.max() > 1.005:
                    raise ValueError(
                        f"AMI coverage outside strict 0.995..1.005 at {path}: "
                        f"{values.min()}, {values.max()}"
                    )
                coverage.append(
                    {
                        "patch": patch_name,
                        "faces": len(values),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        "file": str(path),
                    }
                )
            text = _native_stage(
                motion_case,
                f"checkMesh-{time_name}",
                ["checkMesh", "-allGeometry", "-allTopology", "-time", time_name],
                stages,
            )
            checked = _check_summary(text)
            if any(checked[key] != initial[key] for key in ("cells", "directions")):
                raise ValueError("Native count or 2D directions changed through motion")
            evidence["states"].append(
                {
                    "time_s": time_value,
                    "xy_error_m": xy_error,
                    "z_error_m": z_error,
                    "full_check_mesh_ok": True,
                    "coverage": coverage,
                    "quality": checked["quality"],
                }
            )
            evidence["max_xy_error_m"] = max(evidence["max_xy_error_m"], xy_error)
            evidence["max_z_error_m"] = max(evidence["max_z_error_m"], z_error)
        manifest["motion"] = evidence
        manifest["admitted"] = True
    except Exception as error:
        manifest["admitted"] = False
        manifest["failure"] = str(error)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
