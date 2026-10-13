#!/usr/bin/env python3
"""Publish verified figure bytes, never raw native metadata or analysis sources."""

import ctypes
import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESTIMATOR_KEYS = (
    "bin_spacing_Hz",
    "dt_s",
    "hop_s",
    "noverlap",
    "nperseg",
    "resampled",
    "native_to_uniform",
    "retarded_interpolation",
    "local_spectral",
    "sample_count",
    "segment_detrend",
    "segment_duration_s",
    "segments",
    "unused_tail_samples",
    "window",
    "integrated_power",
    "weighted_power",
)
WINDOW_KEYS = (
    "actual_samples_s",
    "requested_s",
    "exclusions_s",
    "native_selected_samples",
    "resample_rule",
    "resampled",
    "samples",
)
TABLE_KEYS = ("columns", "path", "rows", "sha256", "full_frequency_support_Hz", "sample_window_s")
RENDERING_KEYS = ("pdf", "raster_dpi", "reason", "scientific_arrays_changed")
CADENCE_KEYS = (
    "dt_s",
    "nominal_dt_s",
    "observed_mean_dt_s",
    "observed_min_dt_s",
    "observed_max_dt_s",
    "max_interval_jitter_s",
    "max_lattice_residual_s",
    "validation_tolerance_s",
    "sample_support_s",
    "sample_count",
    "clock_replaced",
)
BIN_CONTRACT_KEYS = (
    "frequency_origin_Hz",
    "bin_count",
    "bin_spacing_Hz",
    "edge_snap_eps_multiplier",
    "edge_snap_max_bins",
    "edge_coordinate",
    "edge_snap_rule",
    "interior_rule",
    "final_upper_edge_inclusive",
)


FIGURE_IDS = (
    "I01_geometry",
    "I02_record_support",
    "I03_force_spectra",
    "I04_receiver_spectra",
    "I05_spectrograms",
    "I06_band_variability",
    "I07_coherence_phase",
    "I08_near_radiating",
    "I09_directivity",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def directory_fd(path, *, create=False):
    """Walk from / with no-follow directory handles, including every ancestor."""
    path = Path(path).absolute()
    require(".." not in path.parts, "Path traversal refused")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_bytes(path):
    path = Path(path).absolute()
    fd = directory_fd(path.parent)
    try:
        file_fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(file_fd, "rb") as stream:
            require(
                stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "Source must be a regular file"
            )
            return stream.read()
    finally:
        os.close(fd)


def finite_tree(value):
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite metadata refused")


def decoded(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    result = json.loads(raw, object_pairs_hook=pairs)
    finite_tree(result)
    require(isinstance(result, dict), "Expected metadata object")
    return result


def safe_relative(name):
    return (
        isinstance(name, str)
        and bool(re.fullmatch(r"[A-Za-z0-9_./-]+", name))
        and not name.startswith("/")
        and all(part not in ("", ".", "..") for part in name.split("/"))
    )


def check_destination(output, report, historical_smoke):
    allowed = (
        [ROOT / ".cache/signature-site-smoke"]
        if historical_smoke
        else [ROOT / "site", ROOT / ".cache"]
    )
    require(
        ".." not in output.parts
        and any(output != base and output.is_relative_to(base) for base in allowed),
        "Unsafe output: use a fresh child of the allowed site/cache or smoke root",
    )
    require(
        not output.is_relative_to(report) and not report.is_relative_to(output),
        "Output must not overlap source tree",
    )
    for part in [output, *output.parents]:
        require(not part.is_symlink(), "Output symlink refused")
    require(not output.exists(), "Output destination already exists")


def select(record, keys):
    require(isinstance(record, dict), "Expected metadata object")
    return {key: record[key] for key in keys if key in record}


def encoded(value):
    def private_check(item):
        if isinstance(item, dict):
            for key, child in item.items():
                private_check(key)
                private_check(child)
        elif isinstance(item, list):
            for child in item:
                private_check(child)
        elif isinstance(item, str):
            require(
                not item.startswith(("/", "file:"))
                and not re.search(r"/(home|Users|private|tmp)/", item),
                "Private path in public metadata refused",
            )

    private_check(value)
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def estimator_metadata(record):
    """Preserve interpolation flags and numeric cadence, never arbitrary source-key maps."""
    result = select(record, ESTIMATOR_KEYS)
    if "cadence" in record:
        result["cadence"] = select(record["cadence"], CADENCE_KEYS)
    if "frequency_bin_contract" in record:
        result["frequency_bin_contract"] = select(record["frequency_bin_contract"], BIN_CONTRACT_KEYS)
    return result


def build_bundle(report, output, expected_run_sha256, *, historical_smoke=False):
    """Create a fresh bundle; asset paths are relative to its index.json."""
    report, output = Path(report).absolute(), Path(output).absolute()
    if not isinstance(expected_run_sha256, str) or not re.fullmatch(
        r"[0-9a-fA-F]{64}", expected_run_sha256
    ):
        raise ValueError("Expected run SHA-256 must be exactly 64 hexadecimal characters")
    root = decoded(read_bytes(report / "manifest.json"))
    if root.get("run_sha256") != expected_run_sha256:
        raise ValueError("Native run SHA-256 mismatch")
    kind = "historical_smoke_only" if historical_smoke else "native_completed_short_record"
    if root.get("evidence_label") != kind:
        raise ValueError(
            "Source label refused; historical reports require explicit historical-smoke"
        )
    check_destination(output, report, historical_smoke)
    require(root.get("schema_version") == 1, "Unsupported report schema")
    figures = root.get("figures")
    require(
        isinstance(figures, list) and root.get("figure_count") == len(figures) == 9,
        "Expected exactly nine figures",
    )
    require(all(isinstance(f, dict) for f in figures), "Invalid figure entries")
    require(
        [f.get("id") for f in figures] == list(FIGURE_IDS), "Wrong or duplicate figure IDs/order"
    )
    label = "HISTORICAL SMOKE ONLY" if historical_smoke else "NATIVE COMPLETED SHORT RECORD"
    pins = root.get("output_sha256")
    require(isinstance(pins, dict) and bool(pins), "Missing/empty output hash map")
    verified = {}
    for name, expected in pins.items():
        require(safe_relative(name), "Unsafe output pin path")
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            "Invalid output SHA-256",
        )
        raw = read_bytes(report / name)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Output hash mismatch")
        verified[name] = raw
    required = {"run_metadata.json"}
    originals = []
    for item in figures:
        fid = item["id"]
        require(item.get("manifest") == fid + "/manifest.json", "Wrong figure manifest path")
        require(item["manifest"] in verified, "Missing figure manifest pin")
        required.add(item["manifest"])
        original = decoded(verified[item["manifest"]])
        require(original.get("figure_id") == fid, "Per-figure ID mismatch")
        source_label = (
            "HISTORICAL SMOKE ONLY" if historical_smoke else "10 s native record · descriptive only"
        )
        require(
            original.get("evidence_label") == source_label, "Per-figure evidence label mismatch"
        )
        require(original.get("caption") == item.get("caption"), "Per-figure caption mismatch")
        sources = original.get("sources")
        require(isinstance(sources, dict), "Missing source identity map")
        runs = [
            (path, pin)
            for path, pin in sources.items()
            if isinstance(pin, dict) and pin.get("identity") == "run_manifest"
        ]
        require(
            len(runs) == 1 and runs[0][1].get("sha256") == expected_run_sha256,
            "Per-figure native run SHA-256 mismatch",
        )
        native_raw = read_bytes(runs[0][0])
        require(
            hashlib.sha256(native_raw).hexdigest() == expected_run_sha256,
            "Actual native run SHA-256 mismatch",
        )
        native_run = decoded(native_raw)
        require(
            native_run.get("evidence_kind") == "native_execution"
            and not native_run.get("test_fixture_only"),
            "Native execution identity required",
        )
        if historical_smoke:
            require(
                native_run.get("schema_version") == 1
                and native_run.get("status") == "completed_demo",
                "Historical smoke requires its completed schema-1 record",
            )
        else:
            require(
                native_run.get("schema_version") == 2
                and native_run.get("status") == "completed_native_reconstructed"
                and all(
                    native_run.get(key, {}).get("status") == "completed"
                    for key in ("exports", "reconstruction")
                ),
                "Require a completed reconstructed native record, not a relabelled historical run",
            )
        outputs = original.get("outputs")
        tables = original.get("numeric_tables")
        require(
            isinstance(outputs, dict) and set(outputs) == {"png", "svg", "pdf"},
            "Missing figure outputs",
        )
        require(isinstance(tables, list) and len(tables) > 0, "Missing numeric tables")
        records = []
        for ext, record in outputs.items():
            require(
                isinstance(record, dict) and record.get("path") == fid + "/figure." + ext,
                "Wrong figure output path",
            )
            records.append(record)
        for table in tables:
            require(
                isinstance(table, dict) and safe_relative(table.get("path")), "Unsafe table path"
            )
            require(
                Path(table["path"]).parent.as_posix() == fid and table["path"].endswith(".csv"),
                "Table outside figure",
            )
            require(
                type(table.get("rows")) is int and table["rows"] >= 0, "Invalid table row count"
            )
            records.append(table)
        for record in records:
            name = record["path"]
            require(name not in required, "Duplicate asset path")
            require(
                name in pins and record.get("sha256") == pins[name], "Missing/mismatched asset pin"
            )
            required.add(name)
        originals.append(original)
    require(set(pins) == required, "Unexpected or missing pinned outputs")
    # run_metadata is verified for completeness but never copied or exposed.
    decoded(verified["run_metadata.json"])
    payload = {
        name: raw
        for name, raw in verified.items()
        if Path(name).suffix in {".png", ".svg", ".pdf", ".csv"}
    }
    index = dict(
        schema_version=1,
        source_kind=root["evidence_label"],
        evidence_label=label,
        native_run_sha256=root["run_sha256"],
        figure_count=root["figure_count"],
        limitations=root["limitations"],
        figures=[],
        output_sha256={name: pins[name] for name in payload},
    )
    for item, original in zip(figures, originals, strict=True):
        outputs = {
            ext: select(record, ("path", "sha256")) for ext, record in original["outputs"].items()
        }
        png = verified[outputs["png"]["path"]]
        outputs["png"]["width"], outputs["png"]["height"] = struct.unpack(">II", png[16:24])
        tables = []
        for table in original["numeric_tables"]:
            public = select(table, TABLE_KEYS)
            if "estimator" in table:
                public["estimator"] = estimator_metadata(table["estimator"])
            tables.append(public)
        figure = dict(
            id=item["id"],
            title=original["title"],
            caption=original["caption"],
            evidence_label=label,
            outputs=outputs,
            numeric_tables=tables,
            provenance=item["id"] + "/provenance.json",
            rendering=select(original.get("rendering", {}), RENDERING_KEYS),
        )
        public = dict(
            figure_id=item["id"],
            title=figure["title"],
            caption=figure["caption"],
            evidence_label=label,
            native_run_sha256=root["run_sha256"],
            equations=original["equations"],
            limitations=original["limitations"],
            window=select(original["window"], WINDOW_KEYS),
            estimator=estimator_metadata(original["estimator"]),
            units=[select(axis, ("xlabel", "ylabel")) for axis in original["axes"]],
            numeric_tables=tables,
            outputs=outputs,
        )
        if figure["rendering"]:
            public["rendering"] = figure["rendering"]
        payload[figure["provenance"]] = encoded(public)
        index["figures"].append(figure)
    payload["index.json"] = encoded(index)
    # Every source byte is verified and cached before the first output write.
    check_destination(output, report, historical_smoke)
    parent_fd = directory_fd(output.parent, create=True)
    stage_name = ".signature-site-stage-" + uuid.uuid4().hex
    stage_fd = None
    try:
        os.mkdir(stage_name, mode=0o700, dir_fd=parent_fd)
        stage_fd = os.open(
            stage_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        stage = Path("/proc/self/fd") / str(stage_fd)
        for name, raw in payload.items():
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        # Linux no-replace rename is atomic and cannot overwrite a raced-in target.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.renameat2(
            parent_fd, os.fsencode(stage_name), parent_fd, os.fsencode(output.name), 1
        ):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
        try:
            shutil.rmtree(stage_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    return index


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", required=True, type=Path, help="Original pinned nine-figure report"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Fresh child under project site/ or .cache/"
    )
    parser.add_argument(
        "--expected-run-sha256",
        required=True,
        help="Independently selected exact native run digest",
    )
    parser.add_argument(
        "--historical-smoke",
        action="store_true",
        help="HISTORICAL SMOKE ONLY; restrict output to .cache/signature-site-smoke/",
    )
    args = parser.parse_args()
    try:
        result = build_bundle(
            args.report,
            args.output,
            args.expected_run_sha256,
            historical_smoke=args.historical_smoke,
        )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f"Publication refused: {error}\n")
    print(
        f"{result['evidence_label']}: {result['figure_count']} verified figures; "
        f"{args.output / 'index.json'}"
    )


if __name__ == "__main__":
    main()
