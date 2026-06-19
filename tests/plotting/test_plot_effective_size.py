"""Tests for the effective-size atlas plot module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from simace.plotting.plot_effective_size import (
    _NE_KEYS_ORDERED,
    _build_subtitle,
    gather_effective_size,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh)


def _make_payload(
    *,
    g_ped: int = 6,
    n_ne: float | None = 7300.0,
) -> dict:
    """Synthetic effective_size.yaml mirroring the real schema."""
    n_trans = g_ped - 1
    return {
        "ne_inbreeding": {
            "ne": n_ne,
            "ne_per_gen": [None, None, 8000.0, 7400.0, None, 7100.0],
            "mean_f_per_gen": [0.0, 0.0, 5e-6, 1e-5, 1.5e-5, 2e-5],
            "slope": -1e-5,
            "n_generations_used": 5,
            "expected": None,
        },
        "ne_coancestry": {
            "ne": 7350.0,
            "ne_per_gen": [None, 7300.0, 7400.0, 7350.0, 7320.0, 7400.0],
            "mean_theta_per_gen": [0.0, 6e-6, 1.2e-5, 1.8e-5, 2.4e-5, 3.0e-5],
            "slope": -6e-6,
            "n_generations_used": 5,
            "expected": None,
        },
        "ne_variance_family_size": {
            "ne": 7350.0,
            "ne_per_transition": [7100.0, 7200.0, 7300.0, 7400.0, 7500.0],
            "v_mm": [1.18] * n_trans,
            "v_mf": [1.18] * n_trans,
            "v_fm": [1.18] * n_trans,
            "v_ff": [1.18] * n_trans,
            "cov_m": [0.18] * n_trans,
            "cov_f": [0.18] * n_trans,
            "expected": 7349.0,
        },
        "ne_sex_ratio": {
            "ne": 9999.5,
            "ne_per_gen": [9999.0] * g_ped,
            "n_male_per_gen": [5000] * g_ped,
            "n_female_per_gen": [5000] * g_ped,
            "expected": 10000.0,
        },
        "ne_individual_delta_f": {
            "ne": 7400.0,
            "ne_per_gen": [None, None, 7300.0, 7350.0, 7400.0, 7450.0],
            "mean_eqg_per_gen": [None, None, 2.0, 3.0, 4.0, 5.0],
            "n_used_per_gen": [0, 0, 100, 100, 100, 100],
            "expected": 7349.0,
        },
        "ne_long_term_contributions": {
            "ne": None,
            "asymptote_reached": False,
            "n_iterations": 5,
            "max_delta_final": 1e-4,
            "sum_c_squared": 2e-4,
            "expected": 3675.0,
        },
        "ne_hill_overlapping": {
            "ne": 7350.0,
            "generation_interval": 1.0,
            "collapses_to_ne_v": True,
            "expected": 7349.0,
        },
        "ne_caballero_toro": {
            "ne": 7400.0,
            "ne_per_gen": [None, 7100.0, 7200.0, 7300.0, 7400.0, 7500.0],
            "mean_self_coancestry_per_gen": [None, 0.5, 0.50001, 0.50002, 0.50003, 0.50004],
            "n_founders_with_descendants_per_gen": [0, 100, 100, 100, 100, 100],
            "slope": -5e-5,
            "expected": None,
        },
    }


@pytest.fixture
def two_rep_yamls(tmp_path: Path) -> list[Path]:
    paths = [tmp_path / f"rep{i}.yaml" for i in range(1, 3)]
    for p in paths:
        _write_yaml(p, _make_payload())
    return paths


@pytest.fixture
def params_path(tmp_path: Path) -> Path:
    p = tmp_path / "params.yaml"
    _write_yaml(p, {"scenario": "test_scenario", "N": 10000, "mating_lambda": 0.5, "G_ped": 6})
    return p


# ---------------------------------------------------------------------------
# gather_effective_size
# ---------------------------------------------------------------------------


def test_gather_returns_two_frames_with_distinct_granularity(two_rep_yamls):
    scalar_df, series_df = gather_effective_size(two_rep_yamls)

    # Scalar: one row per (rep, estimator) — 2 reps × 8 estimators = 16.
    assert len(scalar_df) == 2 * len(_NE_KEYS_ORDERED)
    assert set(scalar_df["estimator"]) == set(_NE_KEYS_ORDERED)
    assert set(scalar_df["rep"]) == {1, 2}

    # Series: 2 reps × (5 estimators × G_ped=6 + 1 estimator × G_ped−1=5) = 2 × 35 = 70.
    g_ped = 6
    expected_rows = 2 * (5 * g_ped + 1 * (g_ped - 1))
    assert len(series_df) == expected_rows


def test_gather_kind_column_distinguishes_gen_vs_transition(two_rep_yamls):
    _, series_df = gather_effective_size(two_rep_yamls)
    var_kinds = series_df.loc[series_df["estimator"] == "ne_variance_family_size", "kind"].unique()
    other_kinds = series_df.loc[series_df["estimator"] != "ne_variance_family_size", "kind"].unique()
    assert list(var_kinds) == ["transition"]
    assert list(other_kinds) == ["generation"]


def test_gather_handles_null_ne(two_rep_yamls):
    # ne_long_term_contributions has ne: None — must surface as NaN, no crash.
    scalar_df, _ = gather_effective_size(two_rep_yamls)
    ltc = scalar_df[scalar_df["estimator"] == "ne_long_term_contributions"]
    assert ltc["ne"].isna().all()


def test_gather_handles_missing_per_gen_entries(two_rep_yamls):
    # ne_inbreeding.ne_per_gen has explicit nulls at indices 0, 1, 4 — must be NaN.
    _, series_df = gather_effective_size(two_rep_yamls)
    inb = series_df[(series_df["estimator"] == "ne_inbreeding") & (series_df["rep"] == 1)]
    assert inb.loc[inb["index"].isin([0, 1, 4]), "ne"].isna().all()
    assert inb.loc[inb["index"] == 2, "ne"].notna().all()


def test_gather_drift_columns_filled_only_for_relevant_estimators(two_rep_yamls):
    _, series_df = gather_effective_size(two_rep_yamls)
    # mean_f only on ne_inbreeding rows
    assert series_df.loc[series_df["estimator"] != "ne_inbreeding", "mean_f"].isna().all()
    assert series_df.loc[series_df["estimator"] == "ne_inbreeding", "mean_f"].notna().any()
    # mean_theta only on ne_coancestry
    assert series_df.loc[series_df["estimator"] != "ne_coancestry", "mean_theta"].isna().all()
    # v_** only on ne_variance_family_size
    assert series_df.loc[series_df["estimator"] != "ne_variance_family_size", "v_mm"].isna().all()


# ---------------------------------------------------------------------------
# main: integration smoke
# ---------------------------------------------------------------------------


def test_main_writes_all_outputs(two_rep_yamls, params_path, tmp_path: Path):
    out_dir = tmp_path / "plots"
    main(
        yaml_paths=[str(p) for p in two_rep_yamls],
        params_path=str(params_path),
        output_dir=str(out_dir),
        plot_ext="png",
    )

    expected_files = [
        "effective_size.estimators.png",
        "effective_size.by_generation.png",
        "effective_size.drift.png",
        "effective_size.family_size_variance.png",
        "effective_size.atlas.html",
    ]
    for fname in expected_files:
        assert (out_dir / fname).exists(), f"missing {fname}"


# ---------------------------------------------------------------------------
# _build_subtitle: WF branch
# ---------------------------------------------------------------------------


class TestBuildSubtitleWF:
    """Subtitle omits λ and shows Ne_V≈N under WF."""

    def test_wf_omits_lambda_and_shows_n(self):
        params = {
            "mating_model": "wright_fisher",
            "N": 2000,
            "G_ped": 6,
            "mating_lambda": 0.5,  # inherited default — must be ignored
        }
        subtitle = _build_subtitle(params, scenario="wf_smoke")
        assert "λ=" not in subtitle
        assert "WF" in subtitle
        assert "Ne_V≈2,000" in subtitle
        assert "N=2,000" in subtitle

    def test_standard_unchanged(self):
        params = {
            "mating_model": "standard",
            "N": 2000,
            "G_ped": 6,
            "mating_lambda": 0.5,
        }
        subtitle = _build_subtitle(params, scenario="std_smoke")
        assert "λ=0.5" in subtitle
        # Standard ZTP(0.5) gives Ne_V ≈ 0.7349·N ≈ 1470, not N.
        assert "Ne_V≈1,470" in subtitle
