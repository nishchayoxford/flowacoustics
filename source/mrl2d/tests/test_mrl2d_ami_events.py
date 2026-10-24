"""SYNTHETIC native-order AMI streams; no native commands or histories."""

import json
import os
import shutil

import pytest
import test_mrl2d_completion as fixtures
from test_mrl2d_completion_pipeline import synthetic_binary_sources

from flowacoustics.mrl2d_case import audit_native_run
from flowacoustics.mrl2d_postprocess import load_native_case

seed = fixtures.seed

CREATE = "AMI: Creating addressing and weights between 127 source faces and 127 target faces\n"
SOURCE = "AMI: Patch source sum(weights) min:1 max:1 average:1\n"
TARGET = "AMI: Patch target sum(weights) min:1 max:1 average:1\n"
PAIR = CREATE + SOURCE + TARGET
GROUP = PAIR * 4


def stream(group=GROUP, index=2):
    groups = [GROUP] * 4
    groups[index] = group
    text = "SYNTHETIC native-order stream, NOT CFD\n" + groups[0]
    for i, time in enumerate(("0.1", "0.2", "0.3"), 1):
        text += (
            "Courant Number mean: 0.1 max: 0.2\n"
            + f"Time = {time}\nPIMPLE: iteration 1\n"
            + groups[i]
            + "Mesh Courant Number mean: 0.1 max: 0.2\nPIMPLE: iteration 2\n"
            + f"MRL2D_POST time={time} fluid=0.2 meshRaw=0.2 meshEffective=0.2 finite=1\n"
            + f"ExecutionTime = 1 s ClockTime = {i // 3} s\n"
        )
    return text + "End\nFinalising parallel run\n"


def audit(text):
    return audit_native_run(
        text, returncode=0, dt=0.1, end=0.3, elapsed=2, wall_limit=30, require_complete_ami=True
    )


def test_explicit_native_order_positive_with_no_update_iteration():
    assert audit(stream())["steps"] == 3


def test_source_source_cannot_replace_source_target():
    with pytest.raises(ValueError, match="AMI"):
        audit(stream((CREATE + SOURCE + SOURCE) * 4))


@pytest.mark.parametrize("index", [0, 1, 2, 3])
@pytest.mark.parametrize("events", [5, 6, 7])
def test_incomplete_extra_group_after_complete_group(index, events):
    # The first four events must not hide a partial additional mesh update.
    with pytest.raises(ValueError, match="AMI"):
        audit(stream(PAIR * events, index=index))


@pytest.mark.parametrize(
    "index,separator",
    [(i, "") for i in range(4)] + [(i, "PIMPLE: iteration 9\n") for i in (1, 2, 3)],
)
def test_additional_complete_group_positive(index, separator):
    assert audit(stream(GROUP + separator + GROUP, index=index))["steps"] == 3


@pytest.mark.parametrize(
    "group",
    [PAIR * 2 + CREATE + (SOURCE + TARGET) * 2, CREATE + (SOURCE + TARGET) * 4],
    ids=["missing_fourth_compensated_inside_third", "one_event_eight_rows"],
)
def test_surplus_pairs_cannot_replace_missing_events(group):
    # Verification-only after the role-pair correction: these already reject.
    with pytest.raises(ValueError, match="AMI"):
        audit(stream(group))


@pytest.mark.parametrize("index", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "group",
    [
        (CREATE + TARGET + TARGET) * 4,
        (CREATE + TARGET + SOURCE) * 4,
        CREATE + TARGET + PAIR * 3,
        PAIR * 3 + CREATE + SOURCE,
        (SOURCE + TARGET) * 4,
        "AMI: sum(weights) min:1 max:1 average:1\n" * 8,
        GROUP + SOURCE,
        GROUP + CREATE,
        PAIR * 3,
        "",
    ],
    ids=[
        "duplicate_target",
        "reversed_roles",
        "missing_source",
        "missing_target",
        "orphan_roles",
        "anonymous",
        "surplus_role",
        "empty_event",
        "three_events",
        "no_events",
    ],
)
def test_invalid_local_event_coverage_at_every_boundary(group, index):
    with pytest.raises(ValueError, match="AMI"):
        audit(stream(group, index=index))


@pytest.mark.parametrize("events", [1, 2, 3, 5, 6, 7])
def test_incomplete_additional_pimple_group_after_complete_update(events):
    with pytest.raises(ValueError, match="AMI"):
        audit(stream(GROUP + "PIMPLE: iteration 9\n" + PAIR * events))


@pytest.mark.parametrize("weight", ["0.98", "1.02", "1e309", "-1e309"])
def test_explicit_event_weights_remain_bounded(weight):
    with pytest.raises(ValueError, match="AMI"):
        audit(stream(GROUP.replace("min:1 ", f"min:{weight} ", 1)))


@pytest.fixture(scope="module")
def complete_sources(seed, tmp_path_factory):
    root = tmp_path_factory.mktemp("SYNTHETIC-event-full-loader") / seed.name
    shutil.copytree(seed, root, copy_function=os.link)
    manifest = json.loads((root / "mrl2d-input.json").read_bytes())
    run = json.loads(
        (root / "mrl2d-run.json").read_bytes().replace(str(seed).encode(), str(root).encode())
    )
    run["source"]["native_files_sha256"] = synthetic_binary_sources(root, False)
    fixtures.seal(root, manifest, run)
    return root


@pytest.fixture
def complete_case(complete_sources, tmp_path):
    root = tmp_path / complete_sources.name
    shutil.copytree(complete_sources, root, copy_function=os.link)
    manifest = json.loads((root / "mrl2d-input.json").read_bytes())
    run = json.loads(
        (root / "mrl2d-run.json")
        .read_bytes()
        .replace(str(complete_sources).encode(), str(root).encode())
    )
    return root, manifest, run, fixtures.seal(root, manifest, run)


def test_full_loader_explicit_events_positive(complete_case):
    root, manifest, _, spec = complete_case
    history, result = load_native_case(spec, mode="production")
    assert result["completion_verified"] is True
    assert len(history.time) == 5000 and history.p.shape == (5000, 12)
    assert result["field_readback"]["frames_read_with_vtk"] == 500
    assert result["physical_input_fingerprint"] == fixtures.independent_fingerprint(root, manifest)


@pytest.mark.parametrize(
    "group",
    [(CREATE + SOURCE + SOURCE) * 4, PAIR * 2 + CREATE + (SOURCE + TARGET) * 2],
    ids=["duplicate_source", "missing_fourth_event_compensated"],
)
def test_full_loader_rejects_only_mutated_local_event(complete_case, group):
    root, manifest, run, _ = complete_case
    log = root / "log.pimpleFoam"
    text = log.read_text()
    token = "Time = 0.5\nPIMPLE: iteration 1\n" + GROUP
    assert text.count(token) == 1
    changed = text.replace(token, "Time = 0.5\nPIMPLE: iteration 1\n" + group, 1)
    assert changed.count("sum(weights)") == text.count("sum(weights)")
    fixtures.put(log, changed)
    run["stages"]["pimpleFoam"]["log_sha256"] = fixtures.digest(changed.encode())
    with pytest.raises(ValueError, match="AMI addressing requires exactly source then target"):
        load_native_case(fixtures.seal(root, manifest, run), mode="production")
