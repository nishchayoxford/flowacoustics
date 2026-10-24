"""Read-only native 2D reductions and staged, public-safe numerical exports.

Never launches CFD. See docs/mrl2d/postprocessing/README.md for estimator semantics.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from flowacoustics.acoustics2d import (
    ContourHistory,
    Spectrum2D,
    common_receivers,
    contour_spectrum,
)
from flowacoustics.mrl2d_case import acoustic_source_kwargs, audit_source_motion, read_sources

CASE_IDS = tuple(
    f"mrl2d_{model}_{surface}" for model in ("urans", "les") for surface in ("body", "permeable")
)


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("unsafe relative input path")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("input symlink escapes case root")
    return result


def validate_completion(spec):
    """Read the versioned detached view; old unbound receipts cannot prove completion."""
    from flowacoustics.mrl2d_completion import direct_native_completion

    if "native_completion" in spec:
        return direct_native_completion(spec)
    # Retain identity diagnostics for legacy receipts, but never accept their assertions.
    root = Path(spec["root"]).resolve()
    content = _relative(root, spec["completion"]).read_bytes()
    if hashlib.sha256(content).hexdigest() != spec["completion_sha256"]:
        raise ValueError("completion identity mismatch")
    raise ValueError("versioned direct-native completion with original report pins required")


def load_native_case(spec, *, mode, memory_limit_bytes=2_000_000_000, panel_chunk=96):
    """Read immutable native archives only; verify expected identity on both sides."""
    end = float(spec["end_s"])
    if mode not in {"production", "diagnostic"}:
        raise ValueError("unknown mode")
    if mode == "production" and end != 10.0:
        raise ValueError("production requires a complete native 10 s run")
    completion = validate_completion(spec) if mode == "production" else None
    if spec["id"] not in CASE_IDS or not 0 < end <= 10:
        raise ValueError("invalid case identity or endpoint")
    root = Path(spec["root"]).resolve()
    files = sorted(root.glob("postProcessing/mrl2dSources/rank*.bin")) + sorted(
        root.glob("processor*/postProcessing/mrl2dSources/rank*.bin")
    )
    pins = spec.get("source_sha256", {})
    if not files or set(pins) != {str(x.relative_to(root)) for x in files}:
        raise ValueError("source identity map must cover exactly every native rank file")
    cadence = float(spec.get("cadence_s", 0.002))
    n = round(end / cadence)
    if n < 4:
        raise ValueError("at least four source frames required")
    panels = sum(x.stat().st_size for x in files) // (n * 16 * 8)
    budget = resource_estimate(n, panels, max(1, round(end * 25)), panel_chunk=panel_chunk)
    if budget["conservative_peak_bytes"] > memory_limit_bytes:
        raise ValueError("native reader / reduction exceeds explicit memory budget")
    for path in files:
        if _sha(path) != pins[str(path.relative_to(root))]:
            raise ValueError("native source identity mismatch before reading")
    data = read_sources(root, envelope=spec["envelope_recorded"], end=end, cadence=cadence)
    motion = audit_source_motion(data)
    for path in files:
        if _sha(path) != pins[str(path.relative_to(root))]:
            raise ValueError("native source identity changed during reading")
    body = data["kind"] == 0
    slip = np.sum(
        (data["fluid_velocity"][:, body] - data["surface_velocity"][:, body])
        * data["normal"][:, body],
        axis=-1,
    )
    kwargs = acoustic_source_kwargs(data, kind=spec["id"].rsplit("_", 1)[1])
    # Whitelist provenance rather than recursively copying private native paths.
    kwargs["provenance"] = {
        "source_type": "native_cfd",
        "pressure_conversion": "native_kinematic_times_rho_once",
        "pressure_units": "Pa",
        "length_units": "m",
        "complete_panel_coverage": True,
    }
    completion_audit = {}
    if completion is not None:
        fields = verify_field_frames(root, completion["field_frames"])
        completion_audit = {
            k: completion[k] for k in ("cells", "ranks", "dt_s", "physical_input_fingerprint")
        }
        completion_audit.update(
            completion_verified=True,
            field_readback=fields,
            completion={
                k: completion[k]
                for k in (
                    "schema",
                    "original_reports",
                    "input_sha256",
                    "evidence_sha256",
                    "stages",
                    "time_roundoff_atol_s",
                )
            },
        )
    return ContourHistory(**kwargs), {
        **completion_audit,
        "native_coupling_verified": True,
        "source_identity_basis": "expected hashes supplied before reading",
        "source_sha256": list(pins.values()),
        "motion": motion,
        "maximum_wall_normal_slip_m_s": float(np.max(abs(slip))),
        "resource_estimate": budget,
        "source_frames": n,
        "source_panels_all_contours": panels,
        "mode": mode,
        "caveat": "Native consistency, not physical accuracy or acoustic-band validation",
    }


def verify_field_frames(root, frames):
    """One plane at a time with the installed VTK reader, not a native solver."""
    import importlib.util
    import os
    import subprocess

    if not frames:
        raise ValueError("empty field frame collection")
    if importlib.util.find_spec("vtkmodules") is None:
        code = (
            "import json,sys; from flowacoustics.mrl2d_postprocess import verify_field_frames; "
            "a=json.load(sys.stdin); print(json.dumps(verify_field_frames(a['root'],a['frames'])))"
        )
        result = subprocess.run(
            ["/usr/bin/python3", "-c", code],
            input=json.dumps({"root": str(root), "frames": frames}),
            text=True,
            capture_output=True,
            timeout=1800,
            env={
                **os.environ,
                "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
                "PYTHONDONTWRITEBYTECODE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
        if result.returncode:
            raise ValueError("VTK field readback failed: " + result.stderr[-2000:])
        return json.loads(result.stdout)
    root = Path(root)
    zeros = []
    for frame in frames:
        path = _relative(root, frame["path"])
        if _sha(path) != frame["sha256"]:
            raise ValueError("field identity mismatch before readback")
        plane = read_plane(path)
        if plane["metadata"]["source_sha256"] != frame["sha256"]:
            raise ValueError("field identity changed during readback")
        if not np.isclose(plane["metadata"]["native_time_s"], frame["time_s"], atol=5e-7, rtol=0):
            raise ValueError("field native time differs from declared frame time")
        zeros.append(plane["metadata"]["display_only_zero_area_triangles_removed"])
        del plane
    return {
        "frames_read_with_vtk": len(frames),
        "first_time_s": frames[0]["time_s"],
        "last_time_s": frames[-1]["time_s"],
        "zero_area_xy_triangles_per_frame": zeros,
        "native_geometry_preserved": True,
    }


def read_plane(path):
    """Actual VTK readback. Reject binary before VTK can recover bad connectivity."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(Path(path).read_bytes())
    if root.get("type") != "PolyData" or any(
        x.get("format") != "ascii" for x in root.iter("DataArray")
    ):
        raise ValueError("only native ASCII PolyData accepted; binary VTP is not repaired")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkIOXML import vtkXMLPolyDataReader

    reader = vtkXMLPolyDataReader()
    errors = []
    reader.AddObserver("ErrorEvent", lambda *_: errors.append("VTK reader error"))
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    if errors or reader.GetErrorCode() or mesh.GetNumberOfPoints() == 0:
        raise ValueError("VTK failed to read complete native plane")
    points = vtk_to_numpy(mesh.GetPoints().GetData()).copy()
    cells = vtk_to_numpy(mesh.GetPolys().GetData()).copy()
    npolys = mesh.GetNumberOfPolys()
    if cells.size != 4 * npolys or not np.all(cells.reshape(-1, 4)[:, 0] == 3):
        raise ValueError("native plane must contain complete triangle connectivity")
    triangles = cells.reshape(-1, 4)[:, 1:]
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise ValueError("invalid native connectivity index")
    fields = {}
    for name, shape in (("p", (len(points),)), ("U", (len(points), 3))):
        field = mesh.GetPointData().GetArray(name)
        if field is None:
            raise ValueError("native point p/U required")
        value = vtk_to_numpy(field).copy()
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("incomplete or nonfinite native field")
        fields[name] = value
    if not np.isfinite(points).all() or np.max(abs(points[:, 2])) > 1e-10:
        raise ValueError("native field is not finite z=0 plane within 1e-10 m tolerance")
    vertices = points.astype(float)[triangles, :2]
    a, b = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    area = abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2
    retain = area > 0  # Exactly zero in displayed x-y plane, not a size threshold.
    native_time = mesh.GetFieldData().GetArray("TimeValue")
    if native_time is None or native_time.GetNumberOfTuples() != 1:
        raise ValueError("native TimeValue required")
    return {
        "points_m": points,
        "triangles": triangles,
        "display_triangles": triangles[retain],
        "pressure_pa": fields["p"].astype(float) * 1000,
        "velocity_m_s": fields["U"],
        "metadata": {
            "reader": "vtkXMLPolyDataReader",
            "reader_errors": len(errors),
            "native_time_s": float(native_time.GetTuple1(0)),
            "native_points": len(points),
            "native_triangles": int(npolys),
            "display_only_zero_area_triangles_removed": int(np.sum(~retain)),
            "connectivity": "native only; no interpolation across solids or AMI gaps",
            "point_dtype": str(points.dtype),
            "pressure_dtype": str(fields["p"].dtype),
            "velocity_dtype": str(fields["U"].dtype),
            "area_m2": float(np.sum(area)),
            "maximum_abs_z_m": float(np.max(abs(points[:, 2]))),
            "source_sha256": _sha(path),
            "pressure_conversion": "kinematic p times 1000 once for display",
        },
    }


def resource_estimate(samples, panels, bins, *, panel_chunk=96):
    """Conservative array budget, including the existing native reader's copies."""
    if min(samples, panels, bins) <= 0 or panel_chunk < 3:
        raise ValueError("positive dimensions and panel_chunk >= 3 required")
    return {
        "samples": samples,
        "panels": panels,
        "bins": bins,
        "panel_chunk": panel_chunk,
        "forbidden_dense_complex_tensor_bytes": samples * panels * bins * 12 * 16,
        "conservative_peak_bytes": (
            samples * panels * 8 * 48
            + samples * min(panels, panel_chunk) * 8 * 40
            + samples * 12 * 8
            + bins * 12 * 16
            + 64 * 1024**2
        ),
        "strategy": "one case, panel chunk, receiver at a time; bounded time synthesis",
        "runtime": "no dense tensor; benchmark separately, not a CFD runtime estimate",
    }


def reduce_history(history, *, mode, panel_chunk=96, startup_s=2.0, fmax=25.0):
    """Linear sum of chunked exact fixed / low-kR series surface integrals."""
    h = history
    if not isinstance(panel_chunk, int) or panel_chunk < 3:
        raise ValueError("panel_chunk must be an integer >= 3")
    plan = analysis_plan(h.time, mode=mode, startup_s=startup_s, fmax=fmax)
    keep = h.time >= plan["sample_window_s"][0]
    receivers = common_receivers()
    chunks = np.array_split(np.arange(len(h.ds)), max(1, int(np.ceil(len(h.ds) / panel_chunk))))
    # ContourHistory needs >=3 panels; merge a possible tiny final partition.
    if len(chunks) > 1 and len(chunks[-1]) < 3:
        chunks[-2] = np.concatenate(chunks[-2:])
        chunks.pop()
    total = None
    bound = None
    pressure_force = np.zeros((len(h.time), 2))
    source_loading = np.zeros_like(pressure_force)
    methods = set()
    for panels in chunks:
        arrays = {}
        for name in ("xy", "normal", "surface_velocity"):
            value = getattr(h, name)
            arrays[name] = value[:, panels][keep] if value.ndim == 3 else value[panels]
        part = ContourHistory(
            time=h.time[keep],
            ds=h.ds[panels],
            p=h.p[:, panels][keep],
            U=h.U[:, panels][keep],
            kind=h.kind,
            rho0=h.rho0,
            c0=h.c0,
            **arrays,
            provenance={"source_type": (h.provenance or {}).get("source_type", "unknown")},
        )
        moving = part.xy.ndim == 3 and not np.array_equal(
            part.xy, np.broadcast_to(part.xy[0], part.xy.shape)
        )
        # Explicitly forbid auto fallback to expensive direct evaluation.
        spec = contour_spectrum(
            part,
            receivers,
            fmin=plan["requested_band_hz"][0],
            fmax=fmax,
            method="series" if moving else "fixed",
            window="hann",
        )
        methods.add(spec.metadata["method"])
        if total is None:
            total = np.zeros_like(spec.coefficient)
            bound = np.zeros(total.shape)
        total += spec.coefficient
        bound += np.asarray(spec.metadata.get("series_truncation_bound_pa", 0))
        normal = h.normal[:, panels] if h.normal.ndim == 3 else h.normal[panels]
        velocity = (
            h.surface_velocity[:, panels]
            if h.surface_velocity.ndim == 3
            else h.surface_velocity[panels]
        )
        pressure_force -= np.sum(h.p[:, panels, None] * normal * h.ds[panels, None], axis=1)
        slip = np.sum((h.U[:, panels] - velocity) * normal, axis=-1)
        source_loading += np.sum(
            (h.p[:, panels, None] * normal + h.rho0 * h.U[:, panels] * slip[..., None])
            * h.ds[panels, None],
            axis=1,
        )
    combined = Spectrum2D(spec.frequency, total, receivers, h.time[keep], {})
    pressure = np.concatenate(
        [
            combined.synthesize(t)
            for t in np.array_split(h.time[keep], max(1, int(np.ceil(np.sum(keep) / 256))))
        ]
    )
    rms = combined.band_rms
    levels = [float(20 * np.log10(x / 1e-6)) if x > 0 else None for x in rms]
    return {
        "time_s": h.time[keep],
        "source_time_s": h.time,
        "frequency_hz": spec.frequency,
        "coefficients_pa": total,
        "receivers_m": receivers,
        "pressure_pa": pressure,
        "band_rms_pa": rms,
        "band_level_db_re_1uPa": levels,
        "pressure_force_n_per_m": pressure_force,
        "integrated_source_loading_n_per_m": source_loading,
        "metadata": {
            **plan,
            "pressure_units": "Pa",
            "loading_units": "N/m",
            "reference_pressure_pa": 1e-6,
            "chunk_count": len(chunks),
            "algorithms": sorted(methods),
            "series_terms": 6,
            "actual_band_hz": spec.metadata["actual_band_hz"],
            "bin_indices": spec.metadata["frequency_bin_indices"],
            "series_truncation_bound_pa": bound.tolist(),
            "mean_conditioning": spec.metadata["mean_conditioning"],
            "source_type": (h.provenance or {}).get("source_type", "unknown"),
            "loading_sign": (
                "pressure force on solid = -integral(p*n ds); source L has opposite pressure sign"
            ),
            "spectrum_quantity": "sqrt(2)*abs(coefficient), coherent-window bin RMS Pa; not PSD",
        },
    }


REPOSITORY = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = REPOSITORY / ".cache/mrl2d/postprocessing"


def safe_destination(destination):
    """Preflight the exact write tree; never write site/final results or overwrite."""
    raw = Path(destination)
    if ".." in raw.parts:
        raise ValueError("unsafe output path traversal")
    path = raw.absolute()
    if not path.is_relative_to(OUTPUT_ROOT) or path == OUTPUT_ROOT:
        raise ValueError("output must be a new child of .cache/mrl2d/postprocessing")
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("output symlink is forbidden")
    if path.exists():
        raise ValueError("output already exists; preserve prior artifacts")
    return path


def render_plane(source, output, *, pressure_limit_pa, speed_limit_m_s):
    """Diagnostic stills only. Native triangles, no Delaunay/AMI/solid bridging."""
    import tempfile

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    destination = safe_destination(output)
    if (
        not np.isfinite([pressure_limit_pa, speed_limit_m_s]).all()
        or min(pressure_limit_pa, speed_limit_m_s) <= 0
    ):
        raise ValueError("positive finite common colour limits required")
    plane = read_plane(source)
    points = plane["points_m"]
    tri = Triangulation(points[:, 0], points[:, 1], plane["display_triangles"])
    speed = np.linalg.norm(plane["velocity_m_s"].astype(float), axis=1)
    meta = {
        **plane["metadata"],
        "title_prefix": "DIAGNOSTIC native CFD",
        "triangulation": "native triangles only, zero x-y area removed for display",
        "pressure_limits_pa": [-pressure_limit_pa, pressure_limit_pa],
        "speed_limits_m_s": [0, speed_limit_m_s],
        "pressure_range_pa": [float(plane["pressure_pa"].min()), float(plane["pressure_pa"].max())],
        "speed_range_m_s": [float(speed.min()), float(speed.max())],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".plane-") as temporary:
        stage = Path(temporary) / "images"
        stage.mkdir()
        for name, values, limits, cmap, label in (
            (
                "pressure",
                plane["pressure_pa"],
                [-pressure_limit_pa, pressure_limit_pa],
                "RdBu_r",
                "Gauge flow pressure (Pa)",
            ),
            ("speed", speed, [0, speed_limit_m_s], "viridis", "Flow speed (m/s)"),
        ):
            fig, ax = plt.subplots(figsize=(7.2, 6), constrained_layout=True)
            field = ax.tripcolor(
                tri,
                values,
                shading="gouraud",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
                rasterized=True,
            )
            fig.colorbar(field, ax=ax, label=label)
            ax.set(
                xlim=(-0.61, 0.61),
                ylim=(-0.61, 0.61),
                xlabel="x (m)",
                ylabel="y (m)",
                title=f"DIAGNOSTIC native CFD · t={meta['native_time_s']:.3f} s",
            )
            ax.set_aspect("equal")
            fig.savefig(stage / (name + ".png"), dpi=150)
            plt.close(fig)
        meta["assets_sha256"] = {x.name: _sha(x) for x in sorted(stage.iterdir())}
        (stage / "plane.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
        safe_destination(destination)
        stage.rename(destination)
    return meta


def export_study(specification, output, *, mode, panel_chunk=96, memory_limit_bytes=2_000_000_000):
    """Execute native reductions into a new private staging bundle, never publish."""
    import tempfile

    destination = safe_destination(output)
    spec = json.loads(Path(specification).read_text())
    entries = spec.get("cases", [])
    if (
        spec.get("schema") != "mrl2d-postprocess-input-v1"
        or len(entries) != 4
        or {e["id"] for e in entries} != set(CASE_IDS)
    ):
        raise ValueError("four exact case identities required")
    by_id = {e["id"]: e for e in entries}
    if mode == "production" and any(e["end_s"] != 10 for e in entries):
        raise ValueError("production requires four complete native 10 s runs")
    reductions, audits = {}, {}
    for identity in CASE_IDS:
        h, audit = load_native_case(
            by_id[identity],
            mode=mode,
            panel_chunk=panel_chunk,
            memory_limit_bytes=memory_limit_bytes,
        )
        reductions[identity] = reduce_history(h, mode=mode, panel_chunk=panel_chunk)
        audits[identity] = audit
        del h
    bundle = build_bundle(reductions, audits, mode=mode)
    filename = "diagnostic-results.json" if mode == "diagnostic" else "study.json"
    # Build and validate all products before the atomic, no-overwrite directory promotion.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".staging-") as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for case in bundle["cases"]:
            identity = case["id"]
            result = reductions[identity]
            arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("nonfinite reduction cannot be exported")
            np.savez(stage / (identity + ".npz"), **arrays)
            detail = {
                "analysis": result["metadata"],
                "native_audit": audits[identity],
                "band_level_db_re_1uPa": result["band_level_db_re_1uPa"],
            }
            (stage / (identity + "-analysis.json")).write_text(
                json.dumps(detail, indent=2, allow_nan=False) + "\n"
            )
            np.savetxt(
                stage / (identity + "-receivers.csv"),
                np.column_stack((result["time_s"], result["pressure_pa"])),
                delimiter=",",
                header="time_s," + ",".join(f"p_{j * 30}deg_Pa" for j in range(12)),
                comments="",
            )
            np.savetxt(
                stage / (identity + "-loading.csv"),
                np.column_stack(
                    (
                        result["source_time_s"],
                        result["pressure_force_n_per_m"],
                        result["integrated_source_loading_n_per_m"],
                    )
                ),
                delimiter=",",
                header="source_time_s,negative_pressure_integral_x_N_m,negative_pressure_integral_y_N_m,source_Lx_N_m,source_Ly_N_m",
                comments="",
            )
            coefficients = result["coefficients_pa"]
            np.savetxt(
                stage / (identity + "-spectrum.csv"),
                np.column_stack((result["frequency_hz"], coefficients.real, coefficients.imag)),
                delimiter=",",
                header="frequency_Hz,"
                + ",".join(
                    f"c_{part}_{j * 30}deg_Pa" for part in ("real", "imag") for j in range(12)
                ),
                comments="",
            )
            np.savetxt(
                stage / (identity + "-directivity.csv"),
                np.column_stack((np.arange(12) * 30, result["receivers_m"], result["band_rms_pa"])),
                delimiter=",",
                header="angle_deg,x_m,y_m,periodic_band_rms_Pa",
                comments="",
            )
            case["downloads"] = [
                {
                    "label": "Numerical arrays and complex coefficients",
                    "href": f"data/mrl2d/{identity}.npz",
                },
                {
                    "label": "Analysis and source audit",
                    "href": f"data/mrl2d/{identity}-analysis.json",
                },
            ]
        (stage / filename).write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
        hashes = {x.name: _sha(x) for x in sorted(stage.iterdir())}
        manifest = {
            "schema": "mrl2d-postprocess-artifacts-v1",
            "mode": mode,
            "artifacts_sha256": hashes,
            "case_count": 4,
            "implementation_sha256": {
                x.name: _sha(x)
                for x in (
                    Path(__file__),
                    Path(__file__).with_name("acoustics2d.py"),
                    Path(__file__).with_name("mrl2d_case.py"),
                    Path(__file__).with_name("mrl2d_completion.py"),
                )
            },
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        # Read back exact saved arrays and manifest-bound payloads, including complex phases.
        for identity in CASE_IDS:
            with np.load(stage / (identity + ".npz")) as data:
                for key in data.files:
                    if not np.array_equal(data[key], reductions[identity][key]):
                        raise ValueError("numerical serialization changed values")
        if any(_sha(stage / name) != digest for name, digest in hashes.items()):
            raise ValueError("artifact identity changed before promotion")
        safe_destination(destination)
        stage.rename(destination)
    return {
        "mode": mode,
        "cases": 4,
        "artifacts": len(hashes) + 1,
        "bundle": filename,
        "output": str(destination.relative_to(REPOSITORY)),
    }


def build_bundle(reductions, audits, *, mode):
    """Public whitelist matching the current DOM-free site model (four groups)."""
    if set(reductions) != set(CASE_IDS) or set(audits) != set(CASE_IDS):
        raise ValueError("exactly four case reductions and audits required")
    if mode not in {"diagnostic", "production"}:
        raise ValueError("unknown export mode")
    if mode == "production" and any(
        not audits[c].get("completion_verified")
        or not audits[c].get("native_coupling_verified")
        or reductions[c]["metadata"]["mode"] != "production"
        for c in CASE_IDS
    ):
        raise ValueError("all four production completion/source gates required")
    limitations = [
        "Ideal infinite-span 2D pressure is Pa, not Pa/m; loading is N/m.",
        "2D WALE / LES-model calculation, not resolved 3D turbulence or broadband prediction.",
        (
            "Body and permeable are overlapping alternatives: never sum them or call "
            "their difference a quadrupole."
        ),
        (
            "Viscous traction and exterior volume quadrupoles omitted; zero acoustic "
            "propagation mean flow."
        ),
        (
            "Finite Hann-window coherent-gain coefficients and selected-band synthesis; "
            "not PSD, full sound or causal arrivals."
        ),
        (
            "Startup exclusion is not demonstrated stationarity; no mesh/time/bandwidth "
            "convergence or experimental validation."
        ),
    ]
    if mode == "diagnostic":
        limitations.insert(
            0,
            (
                "DIAGNOSTIC ONLY. Short startup sources establish no useful acoustic band; "
                "four production cases remain not run."
            ),
        )
    cases = []
    for case in CASE_IDS:
        r, audit = reductions[case], audits[case]
        meta = r["metadata"]
        flow_model = "SST URANS" if "_urans_" in case else "WALE"
        surface = "on-body" if case.endswith("_body") else "permeable"
        prefix = "DIAGNOSTIC ONLY · " if mode == "diagnostic" else ""
        window = {
            k: meta[k]
            for k in (
                "sample_window_s",
                "nominal_period_s",
                "cadence_s",
                "window",
                "requested_band_hz",
                "mean_conditioning",
                "signal_semantics",
            )
        }

        def chart(identity, title, kind, x_label, y_label, traces, description):
            return {
                "id": identity,
                "title": prefix + title,
                "description": description,
                "kind": kind,
                "x_label": x_label,
                "y_label": y_label,
                "traces": traces,
                "origin": "computed",
                "window": window,
                "x_scale": "linear",
                "y_scale": "linear",
            }

        def trace(name, x, y):
            return {"name": name, "x": np.asarray(x).tolist(), "y": np.asarray(y).tolist()}

        charts = [
            chart(
                "loading",
                "Contour source loading",
                "line",
                "Source time (s)",
                "Integrated L (N/m)",
                [
                    trace(
                        f"L{axis}", r["source_time_s"], r["integrated_source_loading_n_per_m"][:, j]
                    )
                    for j, axis in enumerate("xy")
                ],
                (
                    "Integral of p n + rho U (U-v)·n over the selected contour; not the force "
                    "on an imaginary envelope. Includes startup."
                ),
            ),
            chart(
                "receiver",
                "Window-conditioned band pressure",
                "line",
                "Periodic synthesis time (s)",
                "Pressure (Pa)",
                [
                    trace(f"Receiver {30 * j} deg", r["time_s"], r["pressure_pa"][:, j])
                    for j in range(12)
                ],
                meta["interpretation"] + "; not recovered full-band acoustic arrivals.",
            ),
            chart(
                "spectrum",
                "Selected-bin pressure spectrum",
                "line",
                "Frequency (Hz)",
                "Coherent-window bin RMS (Pa)",
                [
                    trace(
                        f"Receiver {30 * j} deg",
                        r["frequency_hz"],
                        np.sqrt(2) * abs(r["coefficients_pa"][:, j]),
                    )
                    for j in range(12)
                ],
                "sqrt(2)|c|; coherent Hann normalization, not a stationary-noise PSD. "
                + meta["interpretation"],
            ),
            chart(
                "directivity",
                "Selected-band receiver magnitudes",
                "polar",
                "Angle (deg)",
                "Periodic band RMS (Pa)",
                [trace("Twelve receivers at 1 m", np.arange(12) * 30, r["band_rms_pa"])],
                "Twelve sampled directions; connecting lines are a display guide. "
                "No far-field assumption. " + meta["interpretation"],
            ),
        ]
        charts[0]["window"] = {
            "sample_window_s": [float(r["source_time_s"][0]), float(r["source_time_s"][-1])],
            "window": "none; native source history",
            "cadence_s": meta["cadence_s"],
        }
        cases.append(
            {
                "id": case,
                "label": f"{flow_model} · {surface}",
                "flow_model": flow_model,
                "acoustic_surface": surface,
                "status": "completed" if mode == "production" else "not_run",
                "summary": prefix + meta["interpretation"],
                "metadata": {
                    "cells": audit.get("cells"),
                    "ranks": audit.get("ranks"),
                    "requested_end_s": 10.0,
                    "achieved_end_s": float(r["source_time_s"][-1]),
                    "dt_s": audit.get("dt_s"),
                    "source_cadence_s": meta["cadence_s"],
                    "density_kg_m3": 1000.0,
                    "sound_speed_m_s": 1480.0,
                    "spanwise_model": "ideal span-invariant 2D",
                    "analysis_window_s": meta["sample_window_s"],
                    "acoustic_band_hz": meta["requested_band_hz"] if mode == "production" else None,
                    "native_returncode": 0 if audit.get("completion_verified") else None,
                    "native_End": True if audit.get("completion_verified") else None,
                },
                "mesh": None,
                "flow": None,
                "charts": charts,
                "checks": [
                    {
                        "name": "Source provenance",
                        "status": meta["source_type"],
                        "detail": "Native consistency gates passed"
                        if audit.get("native_coupling_verified")
                        else "Numerical fixture / not native evidence",
                    }
                ],
                "downloads": [],
            }
        )
    comparisons = []
    for a, b in (
        (CASE_IDS[0], CASE_IDS[1]),
        (CASE_IDS[2], CASE_IDS[3]),
        (CASE_IDS[0], CASE_IDS[2]),
        (CASE_IDS[1], CASE_IDS[3]),
    ):
        ra, rb = reductions[a], reductions[b]
        matched = all(
            np.array_equal(ra[key], rb[key]) for key in ("time_s", "frequency_hz", "receivers_m")
        )
        matched = matched and all(
            ra["metadata"][key] == rb["metadata"][key]
            for key in ("window", "requested_band_hz", "mean_conditioning")
        )
        physical = audits[a].get("physical_input_fingerprint")
        matched_physics = bool(physical) and physical == audits[b].get("physical_input_fingerprint")
        if mode == "production" and not (matched and matched_physics):
            raise ValueError("unmatched production comparison support/physical inputs")
        ratios = (
            [
                float(y / x) if x > 0 else None
                for x, y in zip(ra["band_rms_pa"], rb["band_rms_pa"], strict=True)
            ]
            if matched
            else None
        )
        comparisons.append(
            {
                "case_a": a,
                "case_b": b,
                "matched_numerical_support": matched,
                "matched_physical_inputs": matched_physics,
                "physical_interpretation_allowed": mode == "production"
                and matched
                and matched_physics,
                "band_rms_b_over_a": ratios,
                "meaning": "alternative surface or model comparison, not an isolated quadrupole",
            }
        )
    return {
        "schema_version": 2,
        "study": "mrl2d",
        "title": "Four-case 2D turbine study",
        "summary": "Diagnostic numerical export; not production CFD results"
        if mode == "diagnostic"
        else "Completed native-source numerical export; ideal 2D demonstration",
        "cases": cases,
        "limitations": limitations,
        "comparisons": comparisons,
        "provenance": {
            "mode": mode,
            "estimator": "2D free-space moving-contour surface analogy",
            "reference_pressure_pa": 1e-6,
            "source_types": sorted({r["metadata"]["source_type"] for r in reductions.values()}),
        },
        "downloads": [],
    }


def analysis_plan(time, *, mode, startup_s=2.0, fmax=25.0):
    """Select an explicit common window; startup exclusion is not stationarity."""
    time = np.asarray(time, dtype=float)
    if mode not in {"production", "diagnostic"}:
        raise ValueError("mode must be production or diagnostic")
    if time.ndim != 1 or len(time) < 4 or not np.isfinite(time).all():
        raise ValueError("at least four finite timestamps required")
    dt = (time[-1] - time[0]) / (len(time) - 1)
    if dt <= 0 or not np.allclose(np.diff(time), dt, rtol=1e-9, atol=1e-12):
        raise ValueError("uniform increasing timestamps required")
    if mode == "production" and (
        not np.isclose(time[-1], 10, rtol=0, atol=1e-10)
        or not np.isclose(time[0], dt, rtol=0, atol=1e-10)
    ):
        raise ValueError("production requires the complete native 10 s run")
    if not np.isfinite(startup_s) or startup_s < 0:
        raise ValueError("startup_s must be finite and nonnegative")
    start = startup_s if mode == "production" else 0.0
    keep = time > start + dt * 1e-8
    n = int(np.sum(keep))
    if n < 4:
        raise ValueError("insufficient retained samples")
    period = n * dt
    fmin = 5 / period if mode == "production" else 0.0
    if not np.isfinite(fmax) or not fmin < fmax <= min(25, 0.1 / dt):
        raise ValueError("band must fit 25 Hz cap, ten samples/cycle and five-cycle lower edge")
    return {
        "mode": mode,
        "sample_window_s": [float(time[keep][0]), float(time[-1])],
        "startup_excluded_s": start,
        "sample_count": n,
        "cadence_s": float(dt),
        "nominal_period_s": float(period),
        "frequency_spacing_hz": float(1 / period),
        "requested_band_hz": [float(fmin), float(fmax)],
        "window": "hann",
        "window_coherent_gain": 0.5,
        "window_rms_gain": float(np.sqrt(3 / 8)),
        "hann_enbw_hz": float(1.5 / period),
        "nyquist_hz": float(0.5 / dt),
        "stationarity_verified": False,
        "acoustic_band_claim_valid": False,
        "interpretation": (
            "DIAGNOSTIC ONLY: startup transient; no useful acoustic-band claim"
            if mode == "diagnostic"
            else "Selected exploratory band; no CFD bandwidth or stationarity validation"
        ),
        "signal_semantics": "periodic/window-conditioned band-limited, not causal arrivals",
    }
