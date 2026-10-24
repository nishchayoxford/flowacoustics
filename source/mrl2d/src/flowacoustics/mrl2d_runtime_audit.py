"""Read-only runtime evidence checks, separate from production admission."""

import base64
import struct
import xml.etree.ElementTree as ET

import numpy as np


def metrics_summary(paths, *, dt, end):
    """Require every post-PIMPLE rank/time/kind, including the last solved state."""
    from pathlib import Path

    expected = np.arange(1, round(end / dt) + 1) * dt
    kinds = ("relative", "absolute", "meshRaw", "meshEffective")
    collected = {kind: [] for kind in kinds}
    for rank, path in enumerate(sorted(map(Path, paths))):
        rows = [line.split("\t") for line in path.read_text().splitlines()]
        if len(rows) != len(expected) * len(kinds) or any(len(r) != 16 for r in rows):
            raise ValueError("missing post-solve metric rows")
        for kind in kinds:
            group = [r for r in rows if r[2] == kind]
            if (
                len(group) != len(expected)
                or not np.allclose([float(r[0]) for r in group], expected, rtol=0, atol=1e-10)
                or any(int(r[1]) != rank for r in group)
            ):
                raise ValueError("missing/repeated metric time or rank")
            numeric = np.array([[float(r[i]) for i in (0, 1, 3, *range(5, 16))] for r in group])
            if not np.isfinite(numeric).all():
                raise ValueError("nonfinite post-solve metrics")
            collected[kind].extend(group)
    result = {}
    for kind, group in collected.items():
        if not group:
            raise ValueError("missing post-solve metrics")
        maximum = max(group, key=lambda r: float(r[11]))
        value = float(maximum[11])
        over = sum(int(r[15]) for r in group)
        if kind != "absolute" and (value > 1 or over):
            raise ValueError("post-solve fluid/mesh Courant gate failed")
        result[kind] = {
            "maximum": value,
            "time": float(maximum[0]),
            "rank": int(maximum[1]),
            "cell": int(maximum[3]),
            "zone": maximum[4],
            "over_one_cell_steps": over,
            "rank_step_records": len(group),
        }
    return result


def section_summary(path, *, diagnose_v2206_header=False):
    """Check native VTP coordinates, connectivity and finite p/U without projection."""
    root = ET.parse(path).getroot()
    if root.attrib.get("byte_order") != "LittleEndian":
        raise ValueError("unsupported byte order")
    piece = root.find("./PolyData/Piece")
    if piece is None:
        raise ValueError("missing PolyData piece")
    header_valid = True
    arrays = {}
    types = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8"}
    for node in piece.findall(".//DataArray"):
        if node.text is None:
            raise ValueError("missing VTK data")
        if node.attrib["format"] == "binary":
            raw = base64.b64decode(node.text.strip())
            if root.attrib.get("header_type") != "UInt64":
                raise ValueError("unsupported binary header")
            length = struct.unpack("<Q", raw[:8])[0]
            if length != len(raw) - 8:
                if (
                    diagnose_v2206_header
                    and node.attrib.get("Name") == "connectivity"
                    and node.attrib["type"] == "Int32"
                    and len(raw) - 8 == int(piece.attrib["NumberOfPolys"]) * 3 * 4
                    and length == 4 * (len(raw) - 8)
                ):
                    # Installed v2206 polyWriter multiplies sizeof(label) twice.
                    # Forensic geometry read only: the file stays INVALID/unchanged.
                    header_valid = False
                else:
                    raise ValueError("truncated VTK payload")
            array = np.frombuffer(raw[8:], dtype=types[node.attrib["type"]])
        else:
            array = np.fromstring(node.text, sep=" ", dtype=types[node.attrib["type"]])
        if not np.isfinite(array).all():
            raise ValueError("nonfinite slice data")
        arrays[node.attrib.get("Name", "Points")] = array
    points = arrays["Points"].reshape(-1, 3)
    n = int(piece.attrib["NumberOfPoints"])
    m = int(piece.attrib["NumberOfPolys"])
    if points.shape != (n, 3) or arrays["p"].shape != (n,) or arrays["U"].shape != (3 * n,):
        raise ValueError("incomplete point field coverage")
    z = float(np.abs(points[:, 2]).max())
    if z > 1e-12:
        raise ValueError("native slice off requested z=0 plane")
    offsets = arrays["offsets"]
    if not np.array_equal(offsets, np.arange(1, m + 1) * 3):
        raise ValueError("expected native triangular cutting-plane faces")
    faces = arrays["connectivity"].reshape(-1, 3)
    if faces.shape != (m, 3) or faces.min() < 0 or faces.max() >= n:
        raise ValueError("invalid slice connectivity")
    tri = points[faces, :2].astype(float)
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    return {
        "vtk_header_valid": header_valid,
        "points": n,
        "triangles": m,
        "maximum_abs_z_m": z,
        "area_m2": float(area.sum()),
        "minimum_triangle_area_m2": float(area.min()),
        "maximum_abs_Uz_m_s": float(np.abs(arrays["U"].reshape(-1, 3)[:, 2]).max()),
        "p_range_m2_s2": [float(arrays["p"].min()), float(arrays["p"].max())],
    }
