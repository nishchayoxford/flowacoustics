"""Presentation qualifications preserve the accepted numerical analysis."""

import copy

import pytest
from test_cylinder_publication import load_builder


def force_fixture(status):
    return {
        "metadata": {"shedding": {"status": status, "frequency_Hz": None}},
        "charts": [
            {
                "id": "force_psd",
                "title": "Force spectra and observed lift peak",
                "description": "One-sided Hann Welch density.",
                "origin": "computed",
                "kind": "line",
                "x_label": "Frequency (Hz)",
                "traces": [{"name": "lift", "x": [0, 0.5, 1], "y": [0, 2, 1]}],
            }
        ],
    }


def test_unresolved_lift_title_is_qualified_without_changing_arrays():
    source = force_fixture("unresolved")
    original = copy.deepcopy(source)
    display = load_builder().make_display_dataset(source)
    chart = display["charts"][0]
    assert "unresolved" in chart["title"].lower()
    assert "observed lift peak" not in chart["title"].lower()
    assert chart["traces"] == original["charts"][0]["traces"]
    assert display["metadata"]["shedding"] == original["metadata"]["shedding"]
    assert source == original


@pytest.mark.parametrize("status", ["resolved", "unavailable", None])
def test_title_qualification_does_not_invent_an_unresolved_result(status):
    source = force_fixture(status)
    title = load_builder().make_display_dataset(source)["charts"][0]["title"]
    assert title == source["charts"][0]["title"]


def test_global_estimator_is_explicitly_force_only_and_values_are_unchanged():
    source = force_fixture("unresolved")
    estimator = {
        "window": "hann",
        "nperseg": 3200,
        "noverlap": 1600,
        "segments": 7,
        "unused_tail_samples": 1,
    }
    source["metadata"]["spectral_estimator"] = estimator
    original = copy.deepcopy(source)
    display = load_builder().make_display_dataset(source)
    published = display["metadata"]["spectral_estimator"]
    assert published["scope"] == "force_spectrum_only"
    assert {key: published[key] for key in estimator} == estimator
    assert source == original
