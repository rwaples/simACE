"""Tests for artifact metadata consumed by phenotype plotting."""

import pytest

from simace.plotting.plot_phenotype import _resolve_artifact_max_degree


def _stats(max_degree: int | None) -> dict:
    parameters = {} if max_degree is None else {"max_degree": max_degree}
    return {"parameters": parameters}


def test_resolves_max_degree_from_all_replicates():
    assert _resolve_artifact_max_degree([_stats(5), _stats(5)]) == 5


def test_legacy_reports_omit_depth_claim():
    assert _resolve_artifact_max_degree([_stats(None), _stats(None)]) is None


@pytest.mark.parametrize(
    ("reports", "message"),
    [
        ([_stats(3), _stats(None)], "mix recorded and missing"),
        ([_stats(3), _stats(5)], "disagree on max_degree"),
    ],
)
def test_rejects_inconsistent_replicate_metadata(reports, message):
    with pytest.raises(ValueError, match=message):
        _resolve_artifact_max_degree(reports)
