"""Fail-closed source isolation and native-execution prohibition for unit checks."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def verify_sources():
    identity = json.loads((ROOT / "SOURCE_IDENTITY.json").read_text())
    pins = {**identity["unchanged_source_sha256"],
            **identity.get("runtime_locator_adapted_sha256", {}),
            **identity.get("new_public_cli_sha256", {})}
    assert pins, "empty source pins"
    for name, digest in pins.items():
        path = ROOT / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        if name.startswith("src/") and name.endswith(".py"):
            module = name[4:-3].replace("/", ".").removesuffix(".__init__")
            location = importlib.import_module(module).__file__
            assert location is not None
            assert Path(location).resolve() == path.resolve(), module


def pytest_sessionstart(session):
    assert session.config.rootpath.resolve() == ROOT, "wrong pytest root"
    verify_sources()
    # The original guard test expects this parent; no result target is created.
    (ROOT / ".cache/mrl2d/postprocessing").mkdir(parents=True, exist_ok=True)


def pytest_sessionfinish(session, exitstatus):
    verify_sources()


@pytest.fixture(autouse=True)
def prohibit_native_execution(monkeypatch):
    import fcntl

    from flowacoustics import mrl2d_case, mrl2d_mesh, mrl2d_runtime

    def denied(*args, **kwargs):
        raise AssertionError("unit verification forbids native commands and owner locks")

    monkeypatch.setattr(mrl2d_case, "native_command", denied)
    monkeypatch.setattr(mrl2d_runtime, "native_command", denied)
    monkeypatch.setattr(mrl2d_mesh, "_native_stage", denied)
    monkeypatch.setattr(fcntl, "flock", denied)
