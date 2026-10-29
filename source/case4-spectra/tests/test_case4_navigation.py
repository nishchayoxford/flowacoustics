"""Additive homepage navigation for the fixed Case 4 spectral detail page."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_homepage_exposes_case4_spectra_without_replacing_original_controls():
    page = (ROOT / "site/index.html").read_text()
    assert 'href="./case4-spectra.html"' in page
    assert "Case 4" in page
    assert "all 12" in page
    for case in ("urans_body", "urans_permeable", "les_body", "les_permeable"):
        assert f'data-case="mrl2d_{case}"' in page
    assert 'id="receiver-direction"' in page
    assert 'id="compare-mode"' in page
