"""Pure synthetic completion regressions; never run native commands/owner locks."""

import json
from pathlib import Path

import pytest
import test_mrl2d_completion as fixtures

from flowacoustics.mrl2d_case import audit_native_run
from flowacoustics.mrl2d_postprocess import validate_completion

seed = fixtures.seed
native_fixture = fixtures.native_fixture

AMI_PAIR = (
    "AMI: Creating addressing and weights between 127 source faces and 127 target faces\n"
    "AMI: Patch source sum(weights) min:1 max:1 average:1\n"
    "AMI: Patch target sum(weights) min:1 max:1 average:1\n"
)
AMI_GROUP = AMI_PAIR * 4


@pytest.mark.parametrize("name", ["checkMesh", "decomposePar", "pimpleFoam", "volumeReadback"])
@pytest.mark.parametrize(
    "field,value",
    [("cleanup_confirmed", x) for x in (False, None, 1, "true", "MISSING")]
    + [("descendants_reaped", x) for x in (-1, 0, True, 1.5, "1", None, "MISSING")],
)
def test_current_stage_requires_explicit_valid_cleanup_receipt(native_fixture, name, field, value):
    root, manifest, run, _ = native_fixture
    if value == "MISSING":
        run["stages"][name].pop(field)
    else:
        run["stages"][name][field] = value
    with pytest.raises(ValueError, match="cleanup"):
        validate_completion(fixtures.seal(root, manifest, run))


def test_current_stage_cleanup_schema_positive(native_fixture):
    _, _, _, spec = native_fixture
    assert validate_completion(spec)["stages"]["solver"]["steps"] == 160000


def short_log(clocks):
    ami = AMI_GROUP
    return (
        ami
        + "".join(
            f"Time = {i * 0.1:.12g}\nPIMPLE: iteration 1\n"
            + ami
            + "Courant Number mean: 0.1 max: 0.2\nPIMPLE: iteration 2\n"
            + f"MRL2D_POST time={i * 0.1:.12g} fluid=0.2 meshRaw=0.2 meshEffective=0.2 finite=1\n"
            + f"ExecutionTime = 1 s  ClockTime = {clock} s\n"
            for i, clock in enumerate(clocks, 1)
        )
        + "End\n"
    )


def audit_short(log, **kwargs):
    return audit_native_run(log, returncode=0, dt=0.1, end=0.3, elapsed=2, wall_limit=30, **kwargs)


@pytest.mark.parametrize("clock", ["-1e309", "-1", "1e309"])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_shared_native_clock_rejects_every_nonfinite_or_negative_row(clock, index):
    clocks = ["1", "1", "1"]
    clocks[index] = clock
    with pytest.raises(ValueError, match="clock"):
        audit_short(short_log(clocks))


@pytest.mark.parametrize("clocks", [(0, 0, 0), (0, 0, 1), (1, 1, 1)])
def test_shared_native_clock_allows_zero_and_repeated_readings(clocks):
    assert audit_short(short_log(clocks))["native_clock_seconds"] == clocks[-1]


@pytest.mark.parametrize("clock", ["-1e309", "-1"])
def test_completion_rejects_first_invalid_native_clock(native_fixture, clock):
    root, manifest, run, _ = native_fixture
    path = root / "log.pimpleFoam"
    text = path.read_text().replace("ClockTime = 1 s", f"ClockTime = {clock} s", 1)
    fixtures.put(path, text)
    run["stages"]["pimpleFoam"]["log_sha256"] = fixtures.digest(text.encode())
    with pytest.raises(ValueError, match="clock"):
        validate_completion(fixtures.seal(root, manifest, run))


@pytest.mark.parametrize("conflicting", [False, True])
@pytest.mark.parametrize("absolute_first", [False, True])
def test_completion_rejects_canonical_source_aliases(native_fixture, conflicting, absolute_first):
    root, _, run, spec = native_fixture
    pins = run["source"]["native_files_sha256"]
    absolute = next(iter(pins))
    relative = str(Path(absolute).relative_to(root))
    alias = {relative: "0" * 64 if conflicting else pins[absolute]}
    run["source"]["native_files_sha256"] = (
        {**pins, **alias} if absolute_first else {**alias, **pins}
    )
    # Preserve input order, including the previously overwritten contradictory pin.
    text = json.dumps(run)
    fixtures.put(root / "mrl2d-run.json", text)
    spec["native_completion"]["run_sha256"] = fixtures.digest(text.encode())
    with pytest.raises(ValueError, match="duplicate canonical source"):
        validate_completion(spec)


@pytest.mark.parametrize("step", ["0.5", "10", "initialization"])
@pytest.mark.parametrize("destination", ["before_post", "after_post", "after_clock", "after_end"])
def test_ami_missing_block_cannot_borrow_other_or_trailing_summaries(
    native_fixture, step, destination
):
    root, manifest, run, _ = native_fixture
    path = root / "log.pimpleFoam"
    text = path.read_text()
    ami = AMI_GROUP
    if step == "initialization":
        text = text.replace(ami, "", 1)
    else:
        token = f"Time = {step}\nPIMPLE: iteration 1\n"
        text = text.replace(token + ami, token, 1)
    if destination == "before_post":
        # Another time block; an extra complete update cannot repair the missing one.
        text = text.replace("MRL2D_POST time=0.25 ", ami + "MRL2D_POST time=0.25 ", 1)
    elif destination == "after_post":
        text = text.replace("ExecutionTime = 1 s", ami + "ExecutionTime = 1 s", 1)
    elif destination == "after_clock":
        text = text.replace("End\n", ami + "End\n")
    else:
        text += ami
    assert text.count("sum(weights)") == run["post_solve"]["ami_lines"]
    fixtures.put(path, text)
    run["stages"]["pimpleFoam"]["log_sha256"] = fixtures.digest(text.encode())
    with pytest.raises(ValueError, match="AMI"):
        validate_completion(fixtures.seal(root, manifest, run))


def test_shared_ami_gate_rejects_missing_step_with_compensating_total():
    ami = AMI_GROUP
    token = "Time = 0.2\nPIMPLE: iteration 1\n"
    text = short_log([0, 0, 1]).replace(token + ami, token)
    text = text.replace("End\n", ami + "End\n")
    with pytest.raises(ValueError, match="AMI"):
        audit_short(text, require_complete_ami=True)


@pytest.mark.parametrize("partial", [False, True])
def test_shared_ami_extra_pimple_updates_are_local_not_global(partial):
    ami = AMI_PAIR * (3 if partial else 4)
    text = short_log([0, 0, 1]).replace(
        "MRL2D_POST time=0.2 ", "PIMPLE: iteration 2\n" + ami + "MRL2D_POST time=0.2 "
    )
    if partial:
        with pytest.raises(ValueError, match="AMI"):
            audit_short(text, require_complete_ami=True)
    else:
        assert audit_short(text, require_complete_ami=True)["steps"] == 3


def test_legacy_native_audit_without_ami_keeps_its_explicitly_weaker_api():
    text = short_log([0, 0, 1]).replace(AMI_GROUP, "")
    assert audit_short(text)["steps"] == 3
    with pytest.raises(ValueError, match="AMI"):
        audit_short(text, require_complete_ami=True)


def test_producer_and_adapter_explicitly_opt_in_to_the_same_pure_gate():
    import ast
    import inspect

    from flowacoustics import mrl2d_completion, mrl2d_production

    for function in (mrl2d_production.run_cases, mrl2d_completion._stage_evidence):
        calls = [
            n
            for n in ast.walk(ast.parse(inspect.getsource(function)))
            if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("audit_native_run")
        ]
        assert len(calls) == 1
        assert any(
            kw.arg == "require_complete_ami" and ast.literal_eval(kw.value) is True
            for kw in calls[0].keywords
        )


@pytest.mark.parametrize("location", ["before_post", "after_post", "after_clock", "after_end"])
def test_shared_ami_final_step_boundary(location):
    ami = AMI_GROUP
    token = "Time = 0.3\nPIMPLE: iteration 1\n"
    text = short_log([0, 0, 1]).replace(token + ami, token)
    before = "MRL2D_POST time=0.3 fluid=0.2 meshRaw=0.2 meshEffective=0.2 finite=1\n"
    if location == "before_post":
        text = text.replace(before, ami + before)
        assert audit_short(text, require_complete_ami=True)["steps"] == 3
        return
    if location == "after_post":
        text = text.replace(before, before + ami)
    elif location == "after_clock":
        text = text.replace("End\n", ami + "End\n")
    else:
        text += ami
    with pytest.raises(ValueError, match="AMI"):
        audit_short(text, require_complete_ami=True)


def test_current_nine_generator_pins_are_compatible_without_rewriting_history(native_fixture):
    from flowacoustics.mrl2d_case import _source_hashes

    root, manifest, run, _ = native_fixture
    pins = _source_hashes()
    assert len(pins) == 9
    assert all(fixtures.digest(Path(name).read_bytes()) == sha for name, sha in pins.items())
    manifest["source_sha256"] = pins
    spec = fixtures.seal(root, manifest, run)
    originals = {
        name: (root / name).read_bytes() for name in ("mrl2d-input.json", "mrl2d-run.json")
    }
    assert validate_completion(spec)["original_reports"] == spec["native_completion"]
    assert all((root / name).read_bytes() == data for name, data in originals.items())


def test_shared_ami_explicit_mesh_update_cannot_hide_all_its_missing_summaries():
    # Native addressing events prove this iteration updates the mesh: it is not
    # the legitimate no-update second PIMPLE iteration seen in the retained logs.
    creating = (
        "AMI: Creating addressing and weights between 127 source faces and 127 target faces\n"
    )
    text = short_log([0, 0, 1]).replace(
        "MRL2D_POST time=0.2 ",
        "PIMPLE: iteration 2\n" + creating * 4 + "MRL2D_POST time=0.2 ",
    )
    with pytest.raises(ValueError, match="AMI"):
        audit_short(text, require_complete_ami=True)


@pytest.mark.parametrize("summaries", [1, 2, 3])
def test_shared_ami_addressing_pair_coverage_and_extra_complete_update(summaries):
    creating = (
        "AMI: Creating addressing and weights between 127 source faces and 127 target faces\n"
    )
    roles = ("source", "target", "source")[:summaries]
    pair = "".join(f"AMI: Patch {role} sum(weights) min:1 max:1 average:1\n" for role in roles)
    text = short_log([0, 0, 1]).replace(
        "MRL2D_POST time=0.2 ",
        "PIMPLE: iteration 2\n" + (creating + pair) * 4 + "MRL2D_POST time=0.2 ",
    )
    if summaries != 2:
        with pytest.raises(ValueError, match="AMI"):
            audit_short(text, require_complete_ami=True)
    else:
        assert audit_short(text, require_complete_ami=True)["steps"] == 3
