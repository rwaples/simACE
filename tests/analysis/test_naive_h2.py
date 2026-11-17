"""Tests for the naive h² estimators, against hand-computed values."""

from __future__ import annotations

import math

import pytest

from simace.analysis.stats.naive_h2 import NaiveH2, falconer_h2, naive_h2, pooled_mean

_R = {"MZ": 0.75, "FS": 0.5, "MHS": 0.25, "PHS": 0.125, "1C": 0.0625}


def test_falconer_doubles_the_mz_fs_gap():
    assert falconer_h2(0.75, 0.5) == 0.5


@pytest.mark.parametrize(("r_mz", "r_fs"), [(math.nan, 0.5), (0.75, math.nan)])
def test_falconer_propagates_nan(r_mz: float, r_fs: float):
    assert math.isnan(falconer_h2(r_mz, r_fs))


def test_pooled_mean_skips_non_finite_values():
    assert pooled_mean([0.25, math.nan, 0.125, math.inf]) == 0.1875


def test_pooled_mean_of_no_finite_value_is_nan():
    assert math.isnan(pooled_mean([math.nan]))
    assert math.isnan(pooled_mean([]))


def test_naive_h2_closed_forms():
    assert naive_h2(_R, po_slope=0.4) == NaiveH2(falconer=0.5, sibs=1.0, po=0.4, hs=0.75, cousins=0.5)


def test_naive_h2_pools_half_sibs_over_the_finite_member():
    """A class with only PHS pairs still gives an HS estimate."""
    assert naive_h2({**_R, "MHS": math.nan}, po_slope=0.4).hs == 0.5


def test_naive_h2_marks_each_missing_input_nan_only_where_it_is_used():
    est = naive_h2({**_R, "MZ": math.nan}, po_slope=math.nan)
    assert math.isnan(est.falconer)
    assert math.isnan(est.po)
    assert (est.sibs, est.hs, est.cousins) == (1.0, 0.75, 0.5)
