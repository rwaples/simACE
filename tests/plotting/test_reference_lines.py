"""Reference lines on gathered (folder-level) plots and degenerate traits (ADR 0025)."""

import pandas as pd
import polars as pl
import pytest
import yaml

from simace.analysis.gather import main as gather
from simace.core.relationships import reference_corr, reference_midparent_slope
from simace.plotting.plot_correlations import _parametric_per_relationship_type
from simace.plotting.plot_validation import _reference_corr, _reference_slope

_PARAMS = {
    "A1": 0.4,
    "C1": 0.3,
    "E1": 0.3,
    "A2": 0.4,
    "C2": 0.3,
    "E2": 0.3,
    "assort1": 0.0,
    "assort2": 0.0,
    "mating_model": "standard",
    "Ctm_maternal1": 0.5,
    "Ctm_paternal1": 0.3,
}
_TRUTH_R, _TRUTH_SLOPE = 0.489, 0.71


def _gathered(tmp_path, params):
    """Write one rep's report with recorded truth, gather it, and read it back as plot_validation does."""
    report = {
        "inputs": {"parameters": params},
        "truth": {
            "recorded_pedigree": {
                "traits": {
                    "trait1": {
                        "relationships": {"PHS": {"r": _TRUTH_R}},
                        "midparent_slope": {"slope": _TRUTH_SLOPE},
                    }
                }
            }
        },
    }
    path = tmp_path / "results" / "f" / "s" / "rep1" / "report.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(report))
    tsv = tmp_path / "report_summary.tsv"
    gather([str(path)], str(tsv))
    return pl.read_csv(tsv, separator="\t", null_values=["NA", ""], infer_schema_length=None).to_pandas(), report


def test_matrix_only_assortment_keeps_the_recorded_reference_through_gather(tmp_path):
    """assort_matrix with zero assort1/assort2 is assortative mating: the gathered row must not read as random."""
    params = {**_PARAMS, "assort_matrix": [[0.0, 0.2], [0.2, 0.0]]}
    df, report = _gathered(tmp_path, params)
    truth = report["truth"]["recorded_pedigree"]["traits"]["trait1"]
    before = reference_corr("PHS", params, 1, truth["relationships"])
    assert before == _TRUTH_R
    assert _reference_corr("PHS", 1)(df) == pytest.approx(before)
    assert _reference_slope(1)(df) == pytest.approx(_TRUTH_SLOPE)


def test_random_mating_still_draws_the_analytic_reference_after_gather(tmp_path):
    df, _ = _gathered(tmp_path, _PARAMS)
    analytic = (0.25 * 0.4 + 0.09 * 0.3) / 1.0
    assert _reference_corr("PHS", 1)(df) == pytest.approx(analytic)


def test_per_generation_cells_parse_back_without_crashing(tmp_path):
    """Gathered dict cells are Python-literal text; the gate must see the dicts, not choke on the text."""
    no_ct = {**_PARAMS, "Ctm_maternal1": 0.0, "Ctm_paternal1": 0.0}
    per_gen_e, _ = _gathered(tmp_path / "e", {**no_ct, "E1": {0: 0.3, 3: 0.5}})
    assert _reference_corr("FS", 1)(per_gen_e) == pytest.approx(0.5 * 0.4 + 0.3)  # undivided: no single V_P
    per_gen_c, _ = _gathered(tmp_path / "c", {**no_ct, "C1": {0: 0.2, 3: 0.3}})
    assert _reference_corr("FS", 1)(per_gen_c) is None


def test_all_zero_per_generation_assortment_is_random_mating_after_gather(tmp_path):
    params = {**_PARAMS, "assort1": {0: 0.0, 4: 0.0}}
    df, _ = _gathered(tmp_path, params)
    assert _reference_corr("PHS", 1)(df) == pytest.approx(reference_corr("PHS", params, 1))
    assert _reference_corr("PHS", 1)(df) != pytest.approx(_TRUTH_R)


def test_missing_components_draw_no_line():
    df = pd.DataFrame([{"scenario": "s", "A1": float("nan"), "C1": 0.2, "E1": 0.3}])
    assert _reference_corr("FS", 1)(df) is None
    assert _reference_slope(1)(df) is None


def test_zero_variance_trait_has_no_reference_line():
    params = {**_PARAMS, "A1": 0.0, "C1": 0.0, "E1": 0.0}
    assert reference_corr("FS", params, 1) is None
    assert reference_midparent_slope(params, 1) is None
    assert reference_midparent_slope({**params, "Ctm_maternal1": 0.0, "Ctm_paternal1": 0.0}, 1) is None
    assert _parametric_per_relationship_type(params, 1, ["MZ", "FS", "MO"]) == {}
