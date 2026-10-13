"""Verification-only public metadata tests; no native archive or generated evidence."""
import json

import pytest
from test_signature_site import adapter


def test_nested_estimator_allowlist_keeps_bin_contract_without_private_fields():
    api = adapter()
    contract = {
        "frequency_origin_Hz": 0.0, "bin_count": 1601, "bin_spacing_Hz": 0.5,
        "edge_snap_eps_multiplier": 32, "edge_snap_max_bins": 1e-9,
        "edge_coordinate": "q=(edge_Hz-frequency_origin_Hz)/bin_spacing_Hz",
        "edge_snap_rule": "bounded integer snapping",
        "interior_rule": "half_open_right_owns_boundary",
        "final_upper_edge_inclusive": True,
    }
    source = {
        "cadence": {"dt_s": 0.000625, "private_source": "/home/private/example"},
        "frequency_bin_contract": {**contract, "private_source": "/home/private/example"},
        "private_source": "/home/private/example",
    }
    result = api.estimator_metadata(source)
    assert result["cadence"] == {"dt_s": 0.000625}
    assert result["frequency_bin_contract"] == contract
    assert "private_source" not in json.dumps(result)
    assert json.loads(api.encoded(result)) == result


def test_rendering_allowlist_preserves_mixed_pdf_disclosure():
    api = adapter()
    rendering = {
        "pdf": "mixed vector text/axes with rasterized fields and colorbars",
        "raster_dpi": 160, "reason": "TEST ONLY rendering contract",
        "scientific_arrays_changed": False,
    }
    assert api.select({**rendering, "private_source": "/home/private/example"}, api.RENDERING_KEYS) == rendering


@pytest.mark.parametrize("value", [
    {"path": "/home/private/example"}, {"/home/private/example": "key"},
    {"nested": ["file:///private/example"]},
])
def test_public_json_rejects_private_paths_in_keys_and_values(value):
    with pytest.raises(ValueError, match="Private path"):
        adapter().encoded(value)
