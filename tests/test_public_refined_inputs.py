"""Verification-only: accepted input replay and a clean public source boundary."""
from pathlib import Path

from test_cylinder_longrun import runner

ROOT = Path(__file__).resolve().parents[1]
INPUTS = (
    "0/U", "0/p", "0/nut", "constant/transportProperties",
    "constant/turbulenceProperties", "system/blockMeshDict", "system/controlDict",
    "system/decomposeParDict", "system/fvSchemes", "system/fvSolution",
)


def test_accepted_refined_dictionaries_match_explicit_generator_parameters(tmp_path):
    generated = tmp_path / "generated"
    runner().write_case(generated, end=10, dt=0.000625, ranks=8)
    accepted = ROOT / "cfd/cylinder_circular_60s"
    for name in INPUTS:
        assert (accepted / name).read_bytes() == (generated / name).read_bytes(), name
