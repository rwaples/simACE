"""Correlation-related phenotype plots.

Contains: plot_tetrachoric_sibling, plot_tetrachoric_by_generation,
plot_cross_trait_tetrachoric, plot_parent_offspring_liability,
plot_tetrachoric_by_sex.  Heritability pages live in ``plot_heritability``.
"""

from __future__ import annotations

__all__ = [
    "plot_cross_trait_tetrachoric",
    "plot_parent_offspring_liability",
    "plot_tetrachoric_by_generation",
    "plot_tetrachoric_by_sex",
    "plot_tetrachoric_sibling",
]

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

from simace.core.relationships import PAIR_TYPES
from simace.plotting.plot_style import (
    COLOR_AFFECTED,
    COLOR_OBSERVED,
    COLOR_UNAFFECTED,
    COLOR_UNCENSORED,
)
from simace.plotting.plot_utils import (
    finalize_pair_type_panels,
    finalize_plot,
    pair_type_legend_handles,
    save_placeholder_plot,
    setup_pair_type_panel,
)

logger = logging.getLogger(__name__)

# Parametric expected liability correlations under the ACE model
_EXPECTED_R_COEFFICIENTS: dict[str, tuple[float, float]] = {
    # pair_type: (coefficient of A, coefficient of C)
    "MZ": (1.0, 1.0),
    "FS": (0.5, 1.0),
    "MHS": (0.25, 1.0),  # maternal half-sibs share household (assigned by mother)
    "PHS": (0.25, 0.0),
    "MO": (0.5, 0.0),
    "FO": (0.5, 0.0),
    "1C": (0.125, 0.0),
}


def _expected_liability_corr(A: float, C: float, pair_type: str) -> float | None:
    """Return parametric E[r] for the given pair type, or None if unknown."""
    coeffs = _EXPECTED_R_COEFFICIENTS.get(pair_type)
    if coeffs is None:
        return None
    return coeffs[0] * A + coeffs[1] * C


def _extract_pair_type_observed(
    all_stats: list[dict[str, Any]],
    container_key: str,
    trait_key: str,
    pair_types: list[str],
) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Pull per-rep ``r`` values and total pair counts for one trait."""
    observed: dict[str, list[float]] = {pt: [] for pt in pair_types}
    n_pairs: dict[str, int] = dict.fromkeys(pair_types, 0)
    for s in all_stats:
        for ptype in pair_types:
            entry = s.get(container_key, {}).get(trait_key, {}).get(ptype, {})
            r = entry.get("r")
            if r is not None:
                observed[ptype].append(float(r))
            n_pairs[ptype] += int(entry.get("n_pairs", 0) or 0)
    return observed, n_pairs


def _mean_per_pair_type(
    all_stats: list[dict[str, Any]],
    extractor,
    pair_types: list[str],
) -> dict[str, float]:
    """Average a per-rep value across reps for each pair type.

    Returns only pair types where at least one rep produced a value.
    """
    out: dict[str, float] = {}
    for ptype in pair_types:
        vals = [extractor(s, ptype) for s in all_stats]
        vals = [v for v in vals if v is not None]
        if vals:
            out[ptype] = float(np.mean(vals))
    return out


def _parametric_per_pair_type(
    params: dict[str, Any] | None,
    trait_num: int,
    pair_types: list[str],
) -> dict[str, float]:
    if params is None:
        return {}
    A = params.get(f"A{trait_num}")
    C = params.get(f"C{trait_num}")
    if A is None or C is None:
        return {}
    out: dict[str, float] = {}
    for ptype in pair_types:
        exp_r = _expected_liability_corr(float(A), float(C), ptype)
        if exp_r is not None:
            out[ptype] = float(exp_r)
    return out


def plot_tetrachoric_sibling(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Plot tetrachoric correlations by relationship type using marker-based references.

    For each pair type, draws shapes stacked at the same x position: gray dots
    per rep (observed r), a black wide cross (mean of observed), an open black
    diamond (mean liability r), a red star (parametric E[r]), and a green plus
    (frailty r on uncensored frailties, when available). Faint violins appear
    only when reps >= 4 so the spread is visible without dominating the panel.
    """
    pair_types = PAIR_TYPES
    n_reps = max(len(all_stats), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)

    has_uncens_any = any(s.get("frailty_corr_uncensored") for s in all_stats)
    has_parametric_any = bool(params) and any(params.get(f"A{t}") is not None for t in (1, 2))

    panel_states: list[dict] = []

    for col_idx, trait_num in enumerate([1, 2]):
        ax = axes[col_idx]
        trait_key = f"trait{trait_num}"

        observed, n_pairs = _extract_pair_type_observed(all_stats, "tetrachoric", trait_key, pair_types)
        liability = _mean_per_pair_type(
            all_stats,
            lambda s, pt, _tk=trait_key: s.get("liability_correlations", {}).get(_tk, {}).get(pt),
            pair_types,
        )
        frailty = (
            _mean_per_pair_type(
                all_stats,
                lambda s, pt, _tk=trait_key: s.get("frailty_corr_uncensored", {}).get(_tk, {}).get(pt, {}).get("r"),
                pair_types,
            )
            if has_uncens_any
            else None
        )
        parametric = _parametric_per_pair_type(params, trait_num, pair_types)

        state = setup_pair_type_panel(
            ax,
            pair_types=pair_types,
            n_pairs_per_ptype=n_pairs,
            n_reps=n_reps,
            observed_per_rep=observed,
            liability_r=liability or None,
            parametric_r=parametric or None,
            frailty_r=frailty,
        )
        if col_idx == 0:
            ax.set_ylabel("Tetrachoric correlation", fontsize=12)
        ax.set_title(f"Trait {trait_num}", fontsize=13)
        panel_states.append(state)

    finalize_pair_type_panels(panel_states)

    fig.legend(
        handles=pair_type_legend_handles(
            has_observed_mean=True,
            has_liability=True,
            has_frailty=has_uncens_any,
            has_parametric=has_parametric_any,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=10,
        frameon=False,
    )

    finalize_plot(output_path, scenario=scenario, tight_rect=[0, 0, 1, 0.94])


def plot_tetrachoric_by_generation(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    params: dict[str, Any] | None = None,
) -> None:
    """Plot tetrachoric correlations by relationship type, broken out by generation.

    2 rows (traits) x N cols (last 3 non-founder generations). Each cell shares
    the same marker conventions as :func:`plot_tetrachoric_sibling`. Y-axis is
    shared across cells of the same trait row so generation-to-generation drift
    is directly comparable.
    """
    gen_keys_sets = [set(s.get("tetrachoric_by_generation", {}).keys()) for s in all_stats]
    if not gen_keys_sets or not gen_keys_sets[0]:
        save_placeholder_plot(output_path, "No per-generation tetrachoric data")
        return

    gen_keys = sorted(set.intersection(*gen_keys_sets))
    if not gen_keys:
        save_placeholder_plot(output_path, "No per-generation tetrachoric data")
        return

    pair_types = PAIR_TYPES
    n_reps = max(len(all_stats), 1)
    n_cols = len(gen_keys)

    fig, axes = plt.subplots(2, n_cols, figsize=(6.0 * n_cols, 10), squeeze=False)

    has_parametric_any = bool(params) and any(params.get(f"A{t}") is not None for t in (1, 2))

    for row, trait_num in enumerate([1, 2]):
        trait_key = f"trait{trait_num}"
        row_states: list[dict] = []

        for col, gen_key in enumerate(gen_keys):
            ax = axes[row, col]

            observed: dict[str, list[float]] = {pt: [] for pt in pair_types}
            n_pairs: dict[str, int] = dict.fromkeys(pair_types, 0)
            for s in all_stats:
                cell = s.get("tetrachoric_by_generation", {}).get(gen_key, {}).get(trait_key, {})
                for ptype in pair_types:
                    entry = cell.get(ptype, {})
                    r = entry.get("r")
                    if r is not None:
                        observed[ptype].append(float(r))
                    n_pairs[ptype] += int(entry.get("n_pairs", 0) or 0)

            liability = _mean_per_pair_type(
                all_stats,
                lambda s, pt, _gk=gen_key, _tk=trait_key: (
                    s.get("tetrachoric_by_generation", {}).get(_gk, {}).get(_tk, {}).get(pt, {}).get("liability_r")
                ),
                pair_types,
            )
            parametric = _parametric_per_pair_type(params, trait_num, pair_types)

            state = setup_pair_type_panel(
                ax,
                pair_types=pair_types,
                n_pairs_per_ptype=n_pairs,
                n_reps=n_reps,
                observed_per_rep=observed,
                liability_r=liability or None,
                parametric_r=parametric or None,
            )

            if row == 0:
                gen_num = gen_key.replace("gen", "")
                ax.set_title(f"Gen {gen_num}", fontsize=13)
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nTetrachoric correlation", fontsize=12)
            row_states.append(state)

        finalize_pair_type_panels(row_states)

    fig.legend(
        handles=pair_type_legend_handles(
            has_observed_mean=True,
            has_liability=True,
            has_frailty=False,
            has_parametric=has_parametric_any,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=10,
        frameon=False,
    )

    finalize_plot(output_path, scenario=scenario, tight_rect=[0, 0, 1, 0.96])


def plot_cross_trait_tetrachoric(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Two-panel figure for cross-trait tetrachoric correlations.

    Left: Same-person cross-trait r by generation (dots per rep + mean line),
          with frailty cross-trait reference lines if available.
    Right: Cross-person cross-trait r by pair type (violin/dots), showing how
           relatedness induces cross-trait association.
    """
    pair_types = PAIR_TYPES

    _fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Left panel: same-person by generation ----
    ax_left = axes[0]

    # Collect generation data across reps
    gen_data: dict[int, list[float]] = {}
    for s in all_stats:
        ct = s.get("cross_trait_tetrachoric", {})
        by_gen = ct.get("same_person_by_generation", {})
        for gk, gv in by_gen.items():
            gen_num = int(gk.replace("gen", ""))
            r_g = gv.get("r")
            if r_g is not None:
                gen_data.setdefault(gen_num, []).append(r_g)

    if gen_data:
        generations = sorted(gen_data.keys())
        for gen in generations:
            rs = gen_data[gen]
            for rep_idx, r_val in enumerate(rs):
                jitter = np.random.default_rng(42 + rep_idx).uniform(-0.08, 0.08)
                ax_left.scatter(gen + jitter, r_val, color=COLOR_OBSERVED, alpha=0.9, s=15, zorder=5)

        mean_rs = [np.mean(gen_data[g]) for g in generations]
        ax_left.plot(
            generations,
            mean_rs,
            color=COLOR_OBSERVED,
            linewidth=1.2,
            marker="o",
            markersize=5,
            zorder=6,
            label="Per-gen mean",
        )

        # Overall same-person r (averaged across reps)
        overall_rs = [s.get("cross_trait_tetrachoric", {}).get("same_person", {}).get("r") for s in all_stats]
        overall_rs = [r for r in overall_rs if r is not None]
        if overall_rs:
            mean_overall = np.mean(overall_rs)
            ax_left.axhline(
                y=mean_overall,
                color="black",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
                label=f"Overall r = {mean_overall:.3f}",
            )

        # frailty cross-trait reference lines if available
        oracle_rs = [s.get("frailty_cross_trait_uncensored", {}).get("r") for s in all_stats]
        oracle_rs = [r for r in oracle_rs if r is not None]
        if oracle_rs:
            ax_left.axhline(
                y=np.mean(oracle_rs),
                color=COLOR_UNCENSORED,
                linestyle="-.",
                linewidth=1.0,
                alpha=0.7,
                label=f"Frailty oracle = {np.mean(oracle_rs):.3f}",
            )

        ax_left.set_xticks(generations)
        ax_left.legend(loc="best", fontsize=9)
    else:
        ax_left.text(0.5, 0.5, "No generation data", ha="center", va="center", transform=ax_left.transAxes)

    ax_left.set_xlabel("Generation")
    ax_left.set_ylabel("Cross-trait tetrachoric r")
    ax_left.set_title("Same-Person: affected1 vs affected2")

    # ---- Right panel: cross-person by pair type ----
    ax_right = axes[1]
    n_reps = max(len(all_stats), 1)

    observed: dict[str, list[float]] = {pt: [] for pt in pair_types}
    n_pairs: dict[str, int] = dict.fromkeys(pair_types, 0)
    for s in all_stats:
        cell = s.get("cross_trait_tetrachoric", {}).get("cross_person", {})
        for ptype in pair_types:
            entry = cell.get(ptype, {})
            r = entry.get("r")
            if r is not None:
                observed[ptype].append(float(r))
            n_pairs[ptype] += int(entry.get("n_pairs", 0) or 0)

    if any(observed.values()):
        right_state = setup_pair_type_panel(
            ax_right,
            pair_types=pair_types,
            n_pairs_per_ptype=n_pairs,
            n_reps=n_reps,
            observed_per_rep=observed,
        )
        finalize_pair_type_panels([right_state])
    else:
        ax_right.text(0.5, 0.5, "No cross-person data", ha="center", va="center", transform=ax_right.transAxes)

    ax_right.set_ylabel("Cross-trait tetrachoric r", fontsize=12)
    ax_right.set_title("Cross-Person: personA.affected1 vs personB.affected2", fontsize=13)

    finalize_plot(output_path, scenario=scenario)


def plot_parent_offspring_liability(
    df_samples: pd.DataFrame,
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    subsample_note: str = "",
    params: dict[str, Any] | None = None,
) -> None:
    """2 x 3 scatter grid: midparent vs offspring liability by generation."""
    from scipy.stats import t as t_dist

    from simace.plotting.plot_style import COLOR_FEMALE, COLOR_MALE

    if "generation" not in df_samples.columns:
        save_placeholder_plot(output_path, "No generation data")
        return

    # Build id -> row lookup within df_samples
    ids_arr = df_samples["id"].values
    max_id = int(ids_arr.max()) + 1
    id_to_row = np.full(max_id, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df_samples), dtype=np.int32)

    # Select non-founder generations whose parents are present in the sample.
    # The earliest phenotyped generation's parents may be outside the phenotype
    # window (e.g. G_pheno < G_ped), so we test each candidate generation.
    _sample_ids = set(ids_arr.tolist())
    min_gen = int(df_samples["generation"].min())
    max_gen = int(df_samples["generation"].max())
    candidate_gens = list(range(max(min_gen + 1, 1), max_gen + 1))
    plot_gens = []
    for gen in candidate_gens:
        gen_mask = df_samples["generation"].values == gen
        mothers = df_samples["mother"].values[gen_mask]
        # Check if any parents are present in the sample
        if np.any(np.isin(mothers[mothers >= 0], ids_arr)):
            plot_gens.append(gen)
    # Keep at most 3 generations for a readable grid
    plot_gens = plot_gens[-3:]

    if not plot_gens:
        save_placeholder_plot(output_path, "No generations with parent data available")
        return

    n_cols = len(plot_gens)
    _fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)

    for row, trait_num in enumerate([1, 2]):
        liability = df_samples[f"liability{trait_num}"].values

        for col, gen in enumerate(plot_gens):
            ax = axes[row, col]
            gen_idx = np.where(df_samples["generation"].values == gen)[0]

            mother_ids = df_samples["mother"].values[gen_idx]
            father_ids = df_samples["father"].values[gen_idx]

            has_m = (mother_ids >= 0) & (mother_ids < max_id)
            has_f = (father_ids >= 0) & (father_ids < max_id)

            m_rows = np.full(len(gen_idx), -1, dtype=np.int32)
            f_rows = np.full(len(gen_idx), -1, dtype=np.int32)
            m_rows[has_m] = id_to_row[mother_ids[has_m]]
            f_rows[has_f] = id_to_row[father_ids[has_f]]

            valid = (m_rows >= 0) & (f_rows >= 0)

            if valid.sum() < 2:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                if row == 0:
                    ax.set_title(f"Gen {gen}")
                continue

            offspring_liab = liability[gen_idx[valid]]
            midparent_liab = (liability[m_rows[valid]] + liability[f_rows[valid]]) / 2.0

            # Sex-stratified scatter: daughters in green, sons in blue
            sex_arr = df_samples["sex"].values
            offspring_sex = sex_arr[gen_idx[valid]]
            f_mask = offspring_sex == 0
            m_mask = offspring_sex == 1
            if f_mask.any():
                ax.plot(
                    midparent_liab[f_mask],
                    offspring_liab[f_mask],
                    "o",
                    ms=2,
                    mew=0,
                    alpha=0.25,
                    color=COLOR_FEMALE,
                    rasterized=True,
                )
            if m_mask.any():
                ax.plot(
                    midparent_liab[m_mask],
                    offspring_liab[m_mask],
                    "o",
                    ms=2,
                    mew=0,
                    alpha=0.25,
                    color=COLOR_MALE,
                    rasterized=True,
                )

            # Collect pre-computed stats (averaged across reps)
            r_vals, slope_vals, intercept_vals, n_vals = [], [], [], []
            stderr_vals: list[float] = []
            for s in all_stats:
                po = s.get("parent_offspring_corr", {}).get(f"trait{trait_num}", {}).get(f"gen{gen}", {})
                if po and po.get("r") is not None:
                    r_vals.append(po["r"])
                    slope_vals.append(po["slope"])
                    intercept_vals.append(po["intercept"])
                    n_vals.append(po["n_pairs"])
                    if po.get("stderr") is not None:
                        stderr_vals.append(po["stderr"])

            if r_vals:
                mean_r = np.mean(r_vals)
                mean_slope = np.mean(slope_vals)
                mean_intercept = np.mean(intercept_vals)
                mean_n = int(np.mean(n_vals))
                mean_stderr = float(np.mean(stderr_vals)) if stderr_vals else None
            else:
                from simace.core.numerics import fast_linregress

                mean_slope, mean_intercept, mean_r, mean_stderr, _mean_pvalue = fast_linregress(
                    midparent_liab, offspring_liab
                )
                mean_n = int(valid.sum())

            # Observed regression line
            x_line = np.array([midparent_liab.min(), midparent_liab.max()])
            ax.plot(x_line, mean_slope * x_line + mean_intercept, color=COLOR_AFFECTED, linewidth=1.2)

            # 95% confidence band around regression line
            if mean_stderr is not None and mean_n > 2:
                x_smooth = np.linspace(midparent_liab.min(), midparent_liab.max(), 200)
                y_hat = mean_slope * x_smooth + mean_intercept
                x_mean = np.mean(midparent_liab)
                ss_x = np.sum((midparent_liab - x_mean) ** 2)
                if ss_x > 1e-12:
                    # Reconstruct residual SE: stderr_slope = s / sqrt(SS_x)
                    s = mean_stderr * np.sqrt(ss_x)
                    t_crit = t_dist.ppf(0.975, df=mean_n - 2)
                    se_fit = s * np.sqrt(1.0 / mean_n + (x_smooth - x_mean) ** 2 / ss_x)
                    ax.fill_between(
                        x_smooth,
                        y_hat - t_crit * se_fit,
                        y_hat + t_crit * se_fit,
                        alpha=0.15,
                        color=COLOR_AFFECTED,
                        zorder=2,
                    )

            # Expected slope from configured A (h² = A for midparent-offspring)
            if params is not None:
                expected_slope = params.get(f"A{trait_num}")
                if expected_slope is not None:
                    x_mean = np.mean(midparent_liab)
                    y_mean = np.mean(offspring_liab)
                    expected_intercept = y_mean - float(expected_slope) * x_mean
                    ax.plot(
                        x_line,
                        float(expected_slope) * x_line + expected_intercept,
                        color=COLOR_UNAFFECTED,
                        linestyle="--",
                        linewidth=1.0,
                        zorder=4,
                    )

            # Sex-stratified regression lines
            sex_slopes: dict[str, float | None] = {}
            for sex_key, sex_color in [("female", COLOR_FEMALE), ("male", COLOR_MALE)]:
                sex_slope_vals = []
                for s in all_stats:
                    po_s = (
                        s.get("parent_offspring_corr_by_sex", {})
                        .get(sex_key, {})
                        .get(f"trait{trait_num}", {})
                        .get(f"gen{gen}", {})
                    )
                    if po_s and po_s.get("slope") is not None:
                        sex_slope_vals.append(po_s["slope"])
                if sex_slope_vals:
                    s_slope = np.mean(sex_slope_vals)
                    s_intercept = np.mean(offspring_liab) - s_slope * np.mean(midparent_liab)
                    ax.plot(
                        x_line,
                        s_slope * x_line + s_intercept,
                        color=sex_color,
                        linewidth=1.5,
                        alpha=0.8,
                        zorder=3,
                    )
                    sex_slopes[sex_key] = s_slope
                else:
                    sex_slopes[sex_key] = None

            # Annotation: lead with h² (slope = heritability estimate)
            ann_lines = []
            if mean_stderr is not None:
                ann_lines.append(f"h\u00b2 = {mean_slope:.4f} \u00b1 {mean_stderr:.4f}")
            else:
                ann_lines.append(f"h\u00b2 = {mean_slope:.4f}")
            if sex_slopes.get("female") is not None:
                ann_lines.append(f"h\u00b2\u2640 = {sex_slopes['female']:.4f}")
            if sex_slopes.get("male") is not None:
                ann_lines.append(f"h\u00b2\u2642 = {sex_slopes['male']:.4f}")
            ann_lines.append(f"r = {mean_r:.4f}")
            ax.text(
                0.05,
                0.95,
                "\n".join(ann_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            if row == 0:
                ax.set_title(f"Gen {gen}")
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nOffspring Liability")
            if row == 1:
                ax.set_xlabel("Midparent Liability")

    # Legend on the last axes
    from matplotlib.lines import Line2D

    has_any_expected = params is not None and any(params.get(f"A{t}") is not None for t in [1, 2])
    legend_handles = [
        Line2D([0], [0], color=COLOR_AFFECTED, linewidth=1.2, label="Observed h\u00b2"),
        Line2D([0], [0], color=COLOR_FEMALE, linewidth=1.2, label="Daughters"),
        Line2D([0], [0], color=COLOR_MALE, linewidth=1.2, label="Sons"),
    ]
    if has_any_expected:
        legend_handles.append(
            Line2D([0], [0], color=COLOR_UNAFFECTED, linestyle="--", linewidth=1.0, label="Expected (A)")
        )
    axes[0, -1].legend(handles=legend_handles, loc="lower right", fontsize=8)

    finalize_plot(output_path, subsample_note=subsample_note, scenario=scenario)


def plot_tetrachoric_by_sex(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    params: dict[str, Any] | None = None,
) -> None:
    """Tetrachoric correlations for same-sex pairs: 2 rows (traits) x 2 cols (F/M).

    Same marker conventions as :func:`plot_tetrachoric_sibling`. Each trait row
    shares its y-axis across the female and male panels so cross-sex magnitude
    differences are directly comparable; the two trait rows are independent.
    """
    pair_types = PAIR_TYPES
    sex_labels = [("female", "Female"), ("male", "Male")]
    n_reps = max(len(all_stats), 1)

    if not any(s.get("tetrachoric_by_sex") for s in all_stats):
        save_placeholder_plot(output_path, "No sex-stratified tetrachoric data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)

    has_parametric_any = bool(params) and any(params.get(f"A{t}") is not None for t in (1, 2))

    for row_idx, trait_num in enumerate([1, 2]):
        trait_key = f"trait{trait_num}"
        row_states: list[dict] = []

        for col_idx, (sex_key, sex_display) in enumerate(sex_labels):
            ax = axes[row_idx, col_idx]

            observed: dict[str, list[float]] = {pt: [] for pt in pair_types}
            n_pairs: dict[str, int] = dict.fromkeys(pair_types, 0)
            for s in all_stats:
                cell = s.get("tetrachoric_by_sex", {}).get(sex_key, {}).get(trait_key, {})
                for ptype in pair_types:
                    entry = cell.get(ptype, {})
                    r = entry.get("r")
                    if r is not None:
                        observed[ptype].append(float(r))
                    n_pairs[ptype] += int(entry.get("n_pairs", 0) or 0)

            liability = _mean_per_pair_type(
                all_stats,
                lambda s, pt, _sk=sex_key, _tk=trait_key: (
                    s.get("tetrachoric_by_sex", {}).get(_sk, {}).get(_tk, {}).get(pt, {}).get("liability_r")
                ),
                pair_types,
            )
            parametric = _parametric_per_pair_type(params, trait_num, pair_types)

            state = setup_pair_type_panel(
                ax,
                pair_types=pair_types,
                n_pairs_per_ptype=n_pairs,
                n_reps=n_reps,
                observed_per_rep=observed,
                liability_r=liability or None,
                parametric_r=parametric or None,
            )

            if col_idx == 0:
                ax.set_ylabel(f"Trait {trait_num}\nTetrachoric correlation", fontsize=12)
            if row_idx == 0:
                ax.set_title(f"{sex_display}", fontsize=13)
            row_states.append(state)

        # Shared y-axis within a trait row only — cross-trait magnitudes differ.
        finalize_pair_type_panels(row_states)

    fig.legend(
        handles=pair_type_legend_handles(
            has_observed_mean=True,
            has_liability=True,
            has_frailty=False,
            has_parametric=has_parametric_any,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        fontsize=10,
        frameon=False,
    )

    finalize_plot(output_path, scenario=scenario, tight_rect=[0, 0, 1, 0.96])
