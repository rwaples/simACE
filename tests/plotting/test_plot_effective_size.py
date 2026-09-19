"""Tests for the effective-size atlas plot module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
import yaml

from simace.plotting.plot_effective_size import (
    _NE_KEYS_ORDERED,
    _build_subtitle,
    gather_effective_size,
    main,
    plot_estimators_overview,
    plot_ne_by_generation,
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
    gens = list(range(g_ped))
    n_trans = g_ped - 1
    src, dst = gens[:-1], gens[1:]
    return {
        "ne_inbreeding": {
            "ne": n_ne,
            "generations": gens,
            "mean_f_per_gen": [0.0, 0.0, 5e-6, 1e-5, 1.5e-5, 2e-5],
            "transition_from": src,
            "transition_to": dst,
            "ne_per_gen": [None, 8000.0, 7400.0, None, 7100.0],
            "slope": -1e-5,
            "n_generations_used": 5,
            "expected": None,
        },
        "ne_coancestry": {
            "ne": 7350.0,
            "generations": gens,
            "mean_theta_per_gen": [0.0, 6e-6, 1.2e-5, 1.8e-5, 2.4e-5, 3.0e-5],
            "transition_from": src,
            "transition_to": dst,
            "ne_per_gen": [7300.0, 7400.0, 7350.0, 7320.0, 7400.0],
            "slope": -6e-6,
            "n_generations_used": 5,
            "expected": None,
        },
        "ne_variance_family_size": {
            "ne": 7350.0,
            "parent_generations": gens,
            "ne_per_transition": [7100.0, 7200.0, 7300.0, 7400.0, 7500.0, None],
            "v_mm": [1.18] * n_trans + [None],
            "v_mf": [1.18] * n_trans + [None],
            "v_fm": [1.18] * n_trans + [None],
            "v_ff": [1.18] * n_trans + [None],
            "cov_m": [0.18] * n_trans + [None],
            "cov_f": [0.18] * n_trans + [None],
            "expected": 7349.0,
        },
        "ne_sex_ratio": {
            "ne": 9999.5,
            "generations": gens,
            "ne_per_gen": [9999.0] * g_ped,
            "n_male_per_gen": [5000] * g_ped,
            "n_female_per_gen": [5000] * g_ped,
            "expected": 10000.0,
        },
        "ne_individual_delta_f": {
            "ne": 7400.0,
            "generations": gens,
            "ne_per_gen": [None, None, 7300.0, 7350.0, 7400.0, 7450.0],
            "mean_eqg_per_gen": [None, None, 2.0, 3.0, 4.0, 5.0],
            "n_used_per_gen": [0, 0, 100, 100, 100, 100],
            "expected": 7349.0,
        },
        "ne_long_term_contributions": {
            "ne": 3700.0,
            "n_effective_founders": 1850.0,
            "sum_c_squared": 2e-4,
            "max_delta_final": 1e-4,
            "asymptote_reached": False,
            "n_cohorts": 5,
            "final_generation": g_ped - 1,
            "expected": 3675.0,
        },
        "ne_hill_overlapping": {
            "ne": 7350.0,
            "generation_interval": 1.0,
            "collapses_to_ne_v": True,
            "expected": 7349.0,
        },
        "ne_group_coancestry": {
            "ne": 7400.0,
            "generations": gens,
            "mean_group_coancestry_per_gen": [None, 0.5, 0.50001, 0.50002, 0.50003, 0.50004],
            "n_genomes_per_gen": [0, 100, 100, 100, 100, 100],
            "transition_from": src,
            "transition_to": dst,
            "ne_per_gen": [7100.0, 7200.0, 7300.0, 7400.0, 7500.0],
            "slope": -5e-5,
            "n_generations_used": 6,
            "census_ratio": 1.0,
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

    # Series, per rep, at G_ped=6: Ne_sr and Ne_iΔF give 6 cohort rows each;
    # Ne_V gives 6 parent rows; Ne_I, Ne_C and Ne_CT give 6 cohort rows (their
    # drift mean) plus 5 transition rows (their Ne) each.
    per_rep = 2 * 6 + 6 + 3 * (6 + 5)
    assert len(series_df) == 2 * per_rep


def test_gather_kind_column_separates_cohort_and_transition_axes(two_rep_yamls):
    _, series_df = gather_effective_size(two_rep_yamls)
    kinds = {
        est: set(series_df.filter(pl.col("estimator") == est)["kind"].to_list())
        for est in series_df["estimator"].unique()
    }
    assert kinds["ne_sex_ratio"] == {"cohort"}
    assert kinds["ne_individual_delta_f"] == {"cohort"}
    assert kinds["ne_variance_family_size"] == {"transition"}
    for est in ("ne_inbreeding", "ne_coancestry", "ne_group_coancestry"):
        assert kinds[est] == {"cohort", "transition"}


def test_gather_carries_the_records_own_labels(two_rep_yamls):
    _, series_df = gather_effective_size(two_rep_yamls)
    cohort = series_df.filter((pl.col("estimator") == "ne_sex_ratio") & (pl.col("rep") == 1))
    assert cohort["label"].to_list() == [str(g) for g in range(6)]
    assert cohort["x"].to_list() == [float(g) for g in range(6)]

    transition = series_df.filter(
        (pl.col("estimator") == "ne_inbreeding") & (pl.col("kind") == "transition") & (pl.col("rep") == 1)
    )
    assert transition["label"].to_list() == [f"{g}→{g + 1}" for g in range(5)]
    assert transition["x"].to_list() == [g + 0.5 for g in range(5)]


def test_gather_skips_an_unavailable_record(tmp_path):
    payload = _make_payload()
    payload["ne_coancestry"] = {"reason": "missing_metadata", "code": "missing_sex", "fields": {"status": "absent"}}
    path = tmp_path / "unavailable.yaml"
    _write_yaml(path, payload)

    scalar_df, series_df = gather_effective_size([path])
    assert scalar_df.filter(pl.col("estimator") == "ne_coancestry")["ne"].is_nan().all()
    assert series_df.filter(pl.col("estimator") == "ne_coancestry").is_empty()
    assert not series_df.filter(pl.col("estimator") == "ne_inbreeding").is_empty()


def test_gather_handles_an_unavailable_estimator(tmp_path):
    """An estimator that reported no value still gets a scalar row, with NaN.

    ``skip_ne_coancestry`` makes ``compute_effective_size`` write the
    ``{reason, code, fields}`` payload for ``ne_coancestry``, which carries no
    ``ne`` key at all. Under pedigree-graph 0.9 that is the case to cover:
    ``ne_long_term_contributions`` used to be the example here, and it now
    always reports an ``ne`` (its 0.9 changelog, the Ne_LTC asymptote entry).
    """
    payload = _make_payload()
    payload["ne_coancestry"] = {"reason": "not_requested", "code": "not_requested", "fields": []}
    path = tmp_path / "rep1.yaml"
    _write_yaml(path, payload)

    scalar_df, series_df = gather_effective_size([path])
    coancestry = scalar_df.filter(pl.col("estimator") == "ne_coancestry")
    assert len(coancestry) == 1
    assert coancestry["ne"].is_nan().all()
    assert series_df.filter(pl.col("estimator") == "ne_coancestry").is_empty()


def test_gather_handles_missing_per_gen_entries(two_rep_yamls):
    # ne_inbreeding.ne_per_gen has explicit nulls at transitions 0→1 and 3→4.
    _, series_df = gather_effective_size(two_rep_yamls)
    inb = series_df.filter(
        (pl.col("estimator") == "ne_inbreeding") & (pl.col("kind") == "transition") & (pl.col("rep") == 1)
    )
    assert inb.filter(pl.col("label").is_in(["0→1", "3→4"]))["ne"].is_nan().all()
    assert not inb.filter(pl.col("label") == "1→2")["ne"].is_nan().any()


def test_gather_drift_columns_filled_only_for_relevant_estimators(two_rep_yamls):
    _, series_df = gather_effective_size(two_rep_yamls)
    # mean_f only on ne_inbreeding's cohort rows
    assert series_df.filter(pl.col("estimator") != "ne_inbreeding")["mean_f"].is_nan().all()
    assert (
        not series_df.filter((pl.col("estimator") == "ne_inbreeding") & (pl.col("kind") == "cohort"))["mean_f"]
        .is_nan()
        .all()
    )
    # mean_theta only on ne_coancestry
    assert series_df.filter(pl.col("estimator") != "ne_coancestry")["mean_theta"].is_nan().all()
    # v_** only on ne_variance_family_size
    assert series_df.filter(pl.col("estimator") != "ne_variance_family_size")["v_mm"].is_nan().all()


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
# Expected-reference lines: plot_estimators_overview / plot_ne_by_generation
#
# Both read ``expected`` off scalar_df with a polars ``unique()``, whose order
# is only defined because these ask for ``maintain_order=True``.  If an
# estimator ever carries two distinct expected values across reps, the drawn
# reference must be the first-appearing one rather than whichever the hash
# table happened to yield.
# ---------------------------------------------------------------------------


@pytest.fixture
def recorded_reference_lines(monkeypatch):
    """Record y-values passed to ``Axes.hlines`` / ``Axes.axhline``, still drawing."""
    from matplotlib.axes import Axes

    drawn: dict[str, list[float]] = {"hlines": [], "axhline": []}
    real_hlines, real_axhline = Axes.hlines, Axes.axhline

    def spy_hlines(self, y, xmin, xmax, **kwargs):
        drawn["hlines"].append(float(y))
        return real_hlines(self, y, xmin, xmax, **kwargs)

    def spy_axhline(self, y=0, *args, **kwargs):
        drawn["axhline"].append(float(y))
        return real_axhline(self, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "hlines", spy_hlines)
    monkeypatch.setattr(Axes, "axhline", spy_axhline)
    return drawn


def _scalar_df(expected: list[float | None], estimator: str = "ne_sex_ratio") -> pl.DataFrame:
    """Minimal scalar frame: one row per rep for a single estimator."""
    return pl.DataFrame(
        {
            "rep": list(range(1, len(expected) + 1)),
            "estimator": [estimator] * len(expected),
            "ne": [7000.0 + 10 * i for i in range(len(expected))],
            "expected": expected,
        },
        schema={"rep": pl.Int64, "estimator": pl.Utf8, "ne": pl.Float64, "expected": pl.Float64},
    )


def test_overview_reference_line_takes_the_first_expected(recorded_reference_lines, tmp_path: Path):
    # The larger value appears first on purpose: an unordered unique() yields
    # the smallest here, so this fails if maintain_order is ever dropped.
    scalar_df = _scalar_df([11000.0, 9000.0, 9000.0])
    plot_estimators_overview(scalar_df, "subtitle", tmp_path, "png")

    assert recorded_reference_lines["hlines"] == [11000.0]


def test_overview_draws_no_reference_when_expected_is_all_null(recorded_reference_lines, tmp_path: Path):
    plot_estimators_overview(_scalar_df([None, None]), "subtitle", tmp_path, "png")

    assert recorded_reference_lines["hlines"] == []
    assert (tmp_path / "effective_size.estimators.png").exists()


def _series_df(estimator: str = "ne_sex_ratio", n_gen: int = 3) -> pl.DataFrame:
    """Minimal per-cohort series for a single estimator, two reps."""
    rows = [
        {"rep": rep, "estimator": estimator, "kind": "cohort", "x": float(g), "label": str(g), "ne": 7000.0 + g}
        for rep in (1, 2)
        for g in range(n_gen)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "rep": pl.Int64,
            "estimator": pl.Utf8,
            "kind": pl.Utf8,
            "x": pl.Float64,
            "label": pl.Utf8,
            "ne": pl.Float64,
        },
    )


def test_by_generation_reference_line_takes_the_first_expected(recorded_reference_lines, tmp_path: Path):
    # Larger value first, as above — an unordered unique() would draw 9000.
    plot_ne_by_generation(_series_df(), _scalar_df([11000.0, 9000.0]), tmp_path, "png")

    assert recorded_reference_lines["axhline"] == [11000.0]


def test_by_generation_renders_empty_panels_without_crashing(recorded_reference_lines, tmp_path: Path):
    # Only one of the six panels has data; the rest must fall to the "no data"
    # branch rather than raising on an empty frame.
    plot_ne_by_generation(_series_df(), _scalar_df([None]), tmp_path, "png")

    assert recorded_reference_lines["axhline"] == []
    assert (tmp_path / "effective_size.by_generation.png").exists()


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
