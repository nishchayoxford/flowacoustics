"""Synthetic parser fixtures only; never reported as native flow evidence."""

import base64
import struct

import numpy as np
import pytest


def test_complete_post_solve_metrics_including_last_step(tmp_path):
    from flowacoustics import mrl2d_runtime_audit as audit

    assert hasattr(audit, "metrics_summary"), "all-step post-solve audit missing"
    path = tmp_path / "runtime-metrics.tsv"
    rows = []
    for time in [0.001, 0.002]:
        for kind in ["relative", "absolute", "meshRaw", "meshEffective"]:
            rows.append(f"{time}\t0\t{kind}\t0\tzone\t0\t0\t1\t0\t1\t1\t0.8\t1\t1\t1\t0")
    path.write_text("\n".join(rows))
    summary = audit.metrics_summary([path], dt=0.001, end=0.002)
    assert summary["relative"]["maximum"] == 0.8
    path.write_text("\n".join(rows[:-1]))
    with pytest.raises(ValueError, match="missing"):
        audit.metrics_summary([path], dt=0.001, end=0.002)
    path.write_text("\n".join(rows).replace("0.8", "1.1"))
    with pytest.raises(ValueError, match="Courant"):
        audit.metrics_summary([path], dt=0.001, end=0.002)


def test_vtk_binary_plane_checks_coordinates_and_both_fields(tmp_path):
    import importlib.util

    assert importlib.util.find_spec("flowacoustics.mrl2d_runtime_audit") is not None, (
        "slice audit module missing"
    )
    from flowacoustics import mrl2d_runtime_audit as audit

    assert hasattr(audit, "section_summary"), "native slice-coordinate audit missing"

    def encoded(values):
        raw = np.asarray(values, dtype="<f4").tobytes()
        return base64.b64encode(struct.pack("<Q", len(raw)) + raw).decode()

    xml = """<VTKFile byte_order='LittleEndian' header_type='UInt64'><PolyData>
    <Piece NumberOfPoints='3' NumberOfPolys='1'><Points>
    <DataArray type='Float32' NumberOfComponents='3' format='binary'>{points}</DataArray>
    </Points><Polys><DataArray type='Int32' Name='connectivity' format='ascii'>0 1 2</DataArray>
    <DataArray type='Int32' Name='offsets' format='ascii'>3</DataArray></Polys>
    <PointData><DataArray type='Float32' Name='p' format='binary'>{p}</DataArray>
    <DataArray type='Float32' Name='U' NumberOfComponents='3' format='binary'>{u}</DataArray>
    </PointData></Piece></PolyData></VTKFile>"""
    path = tmp_path / "section.vtp"
    for z in [0, 0.001]:
        path.write_text(
            xml.format(
                points=encoded([[0, 0, z], [1, 0, z], [0, 1, z]]),
                p=encoded([0, 1, 2]),
                u=encoded([[1, 0, 0]] * 3),
            )
        )
        if z:
            with pytest.raises(ValueError, match="plane"):
                audit.section_summary(path)
        else:
            result = audit.section_summary(path)
            assert result["points"] == 3
            assert result["area_m2"] == 0.5
            assert result["maximum_abs_z_m"] == 0
            broken = base64.b64encode(
                struct.pack("<Q", 48) + np.array([0, 1, 2], dtype="<i4").tobytes()
            ).decode()
            xml_bad = path.read_text().replace("format='ascii'>0 1 2", f"format='binary'>{broken}")
            path.write_text(xml_bad)
            with pytest.raises(ValueError, match="truncated"):
                audit.section_summary(path)
            diagnosed = audit.section_summary(path, diagnose_v2206_header=True)
            assert diagnosed["vtk_header_valid"] is False
            assert diagnosed["area_m2"] == 0.5
