"""Correlation-related phenotype plots.

Contains: plot_tetrachoric_sibling, plot_tetrachoric_by_generation,
plot_cross_trait_tetrachoric, plot_parent_offspring_liability,
plot_heritability_by_generation, plot_broad_heritability_by_generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim_ace.core.utils import PAIR_TYPES
from sim_ace.plotting.plot_utils import PAIR_COLORS, draw_colored_violins, finalize_plot, save_placeholder_plot

logger = logging.getLogger(__name__)

# Parametric expected liability correlations under the ACE model
_EXPECTED_R_COEFFICIENTS: dict[str, tuple[float, float]] = {
    # pair_type: (coefficient of A, coefficient of C)
    "MZ twin": (1.0, 1.0),
    "Full sib": (0.5, 1.0),
    "Maternal half sib": (0.25, 0.0),
    "Paternal half sib": (0.25, 0.0),
    "Mother-offspring": (0.5, 0.0),
    "Father-offspring": (0.5, 0.0),
    "1st cousin": (0.125, 0.0),
}


def _expected_liability_corr(A: float, C: float, pair_type: str) -> float | None:
    """Return parametric E[r] for the given pair type, or None if unknown."""
    coeffs = _EXPECTED_R_COEFFICIENTS.get(pair_type)
    if coeffs is None:
        return None
    return coeffs[0] * A + coeffs[1] * C


def plot_tetrachoric_sibling(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Plot tetrachoric correlations by relationship type, violin with rep dots."""
    pair_types = PAIR_TYPES
    pair_colors = PAIR_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col_idx, trait_num in enumerate([1, 2]):
        ax = axes[col_idx]
        key = f"trait{trait_num}"

        # Build long-format data for violin plot
        rows = []
        total_pairs = {}
        for ptype in pair_types:
            n_total = 0
            for s in all_stats:
                entry = s["tetrachoric"][key].get(ptype, {})
                r = entry.get("r")
                n_p = entry.get("n_pairs", 0)
                n_total += n_p
                if r is not None:
                    rows.append({"pair_type": ptype, "r": r})
            total_pairs[ptype] = n_total

        df_plot = pd.DataFrame(rows)

        if not df_plot.empty:
            datasets = [df_plot.loc[df_plot["pair_type"] == pt, "r"].values for pt in pair_types]
            colors = [pair_colors[pt] for pt in pair_types]
            draw_colored_violins(ax, datasets, list(range(len(pair_types))), colors)

            # Overlay per-rep dots
            for i, ptype in enumerate(pair_types):
                rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                if len(rep_vals):
                    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(rep_vals))
                    ax.scatter(
                        i + jitter,
                        rep_vals,
                        color="black",
                        s=15,
                        alpha=0.9,
                        zorder=5,
                    )

            # Collect liability + frailty reference values for annotation placement
            liab_ref = {}
            for _i, ptype in enumerate(pair_types):
                liab_vals = [s.get("liability_correlations", {}).get(key, {}).get(ptype) for s in all_stats]
                liab_vals = [v for v in liab_vals if v is not None]
                liab_ref[ptype] = np.mean(liab_vals) if liab_vals else -np.inf

            # Annotate mean N pairs per rep (above dots AND reference lines)
            n_reps = len(all_stats)
            for i, ptype in enumerate(pair_types):
                rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                if len(rep_vals):
                    top = max(rep_vals.max(), liab_ref[ptype])
                    ax.text(
                        i,
                        top + 0.04,
                        f"n={total_pairs[ptype] // n_reps:,}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

        # Liability correlation lines (averaged across reps)
        for i, ptype in enumerate(pair_types):
            liab_vals = [s.get("liability_correlations", {}).get(key, {}).get(ptype) for s in all_stats]
            liab_vals = [v for v in liab_vals if v is not None]
            if liab_vals:
                mean_liab_r = np.mean(liab_vals)
                ax.hlines(
                    mean_liab_r,
                    i - 0.35,
                    i + 0.35,
                    colors="black",
                    linestyles="dashed",
                    linewidth=2,
                    zorder=4,
                )

        # Uncensored frailty pairwise correlation lines (averaged across reps)
        has_uncens = any(s.get("frailty_corr_uncensored") for s in all_stats)
        if has_uncens:
            for i, ptype in enumerate(pair_types):
                uncens_vals = [
                    s.get("frailty_corr_uncensored", {}).get(key, {}).get(ptype, {}).get("r") for s in all_stats
                ]
                uncens_vals = [v for v in uncens_vals if v is not None]
                if uncens_vals:
                    mean_uncens_r = np.mean(uncens_vals)
                    ax.hlines(
                        mean_uncens_r,
                        i - 0.35,
                        i + 0.35,
                        colors="C2",
                        linestyles="dashdot",
                        linewidth=2,
                        zorder=5,
                    )

        # Parametric expected liability correlations
        has_parametric = False
        if params is not None:
            A = params.get(f"A{trait_num}")
            C = params.get(f"C{trait_num}")
            if A is not None and C is not None:
                for i, ptype in enumerate(pair_types):
                    exp_r = _expected_liability_corr(float(A), float(C), ptype)
                    if exp_r is not None:
                        has_parametric = True
                        ax.hlines(
                            exp_r,
                            i - 0.35,
                            i + 0.35,
                            colors="C3",
                            linestyles="dotted",
                            linewidth=2,
                            zorder=4,
                        )

        ax.set_xticks(range(len(pair_types)))
        ax.set_xticklabels(pair_types, fontsize=9, rotation=15, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("Tetrachoric Correlation")
        ax.set_title(f"Trait {trait_num}")
        ax.set_ylim(-0.1, 1.1)

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Liability r"),
        ]
        if has_uncens:
            legend_elements.append(
                Line2D([0], [0], color="C2", linestyle="-.", linewidth=2, label="Frailty r (uncensored)"),
            )
        if has_parametric:
            legend_elements.append(
                Line2D([0], [0], color="C3", linestyle=":", linewidth=2, label="Parametric E[r]"),
            )
        ax.legend(handles=legend_elements, loc="upper right")

    fig.suptitle(f"Tetrachoric Correlation [{scenario}]", fontsize=14)
    finalize_plot(output_path)


def plot_tetrachoric_by_generation(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    params: dict[str, Any] | None = None,
) -> None:
    """Plot tetrachoric correlations by relationship type, broken out by generation.

    2 rows (traits) x N cols (last 3 non-founder generations) grid.
    Each panel is a bar chart of 7 pair types with dashed liability reference lines.
    """
    # Determine which generations are available across reps
    gen_keys_sets = [set(s.get("tetrachoric_by_generation", {}).keys()) for s in all_stats]
    if not gen_keys_sets or not gen_keys_sets[0]:
        save_placeholder_plot(output_path, "No per-generation tetrachoric data")
        return

    # Intersection of available gens across all reps, sorted
    gen_keys = sorted(set.intersection(*gen_keys_sets))
    if not gen_keys:
        save_placeholder_plot(output_path, "No per-generation tetrachoric data")
        return

    pair_types = PAIR_TYPES
    pair_colors = PAIR_COLORS

    n_cols = len(gen_keys)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 5 * 2), squeeze=False)

    for row, trait_num in enumerate([1, 2]):
        trait_key = f"trait{trait_num}"

        for col, gen_key in enumerate(gen_keys):
            ax = axes[row, col]

            # Build long-format data for violin plot
            rows_data = []
            total_pairs = {}
            mean_liab_rs = []

            for ptype in pair_types:
                liab_rs = []
                n_total = 0

                for s in all_stats:
                    gen_data = s.get("tetrachoric_by_generation", {}).get(gen_key, {})
                    entry = gen_data.get(trait_key, {}).get(ptype, {})
                    r = entry.get("r")
                    n_p = entry.get("n_pairs", 0)
                    liab_r = entry.get("liability_r")
                    if r is not None:
                        rows_data.append({"pair_type": ptype, "r": r})
                    if liab_r is not None:
                        liab_rs.append(liab_r)
                    n_total += n_p

                total_pairs[ptype] = n_total
                mean_liab_rs.append(np.mean(liab_rs) if liab_rs else np.nan)

            df_plot = pd.DataFrame(rows_data)

            if not df_plot.empty:
                datasets = [df_plot.loc[df_plot["pair_type"] == pt, "r"].values for pt in pair_types]
                colors = [pair_colors[pt] for pt in pair_types]
                draw_colored_violins(ax, datasets, list(range(len(pair_types))), colors)

                # Overlay per-rep dots
                for i, ptype in enumerate(pair_types):
                    rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                    if len(rep_vals):
                        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(rep_vals))
                        ax.scatter(
                            i + jitter,
                            rep_vals,
                            color="black",
                            s=12,
                            alpha=0.9,
                            zorder=5,
                        )

                # Annotate mean N pairs per rep (above dots AND liability line)
                n_reps = len(all_stats)
                for i, ptype in enumerate(pair_types):
                    rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                    if len(rep_vals):
                        top = rep_vals.max()
                        # Also consider the liability reference line
                        liab_val = mean_liab_rs[i] if not np.isnan(mean_liab_rs[i]) else -np.inf
                        top = max(top, liab_val)
                        ax.text(
                            i,
                            top + 0.04,
                            f"n={total_pairs[ptype] // n_reps:,}",
                            ha="center",
                            va="bottom",
                            fontsize=6,
                        )

            # Liability correlation reference lines (dashed)
            for i, liab_r in enumerate(mean_liab_rs):
                if not np.isnan(liab_r):
                    ax.hlines(
                        liab_r,
                        i - 0.35,
                        i + 0.35,
                        colors="black",
                        linestyles="dashed",
                        linewidth=2,
                        zorder=4,
                    )

            # Parametric expected liability correlations
            if params is not None:
                A = params.get(f"A{trait_num}")
                C = params.get(f"C{trait_num}")
                if A is not None and C is not None:
                    for i, ptype in enumerate(pair_types):
                        exp_r = _expected_liability_corr(float(A), float(C), ptype)
                        if exp_r is not None:
                            ax.hlines(
                                exp_r,
                                i - 0.35,
                                i + 0.35,
                                colors="C3",
                                linestyles="dotted",
                                linewidth=2,
                                zorder=4,
                            )

            ax.set_xticks(range(len(pair_types)))
            ax.set_xticklabels(pair_types, fontsize=7, rotation=30, ha="right")
            ax.set_xlabel("")
            ax.set_ylim(-0.1, 1.1)

            if row == 0:
                # Extract gen number from key like "gen3"
                gen_num = gen_key.replace("gen", "")
                ax.set_title(f"Gen {gen_num}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nTetrachoric Correlation")

    from matplotlib.lines import Line2D

    has_parametric = params is not None and params.get("A1") is not None and params.get("C1") is not None
    legend_handles = [Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Liability r")]
    if has_parametric:
        legend_handles.append(Line2D([0], [0], color="C3", linestyle=":", linewidth=2, label="Parametric E[r]"))
    axes[0, -1].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
    )

    fig.suptitle(f"Tetrachoric Correlation by Generation [{scenario}]", fontsize=14)
    finalize_plot(output_path)


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
    pair_colors = PAIR_COLORS

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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
                ax_left.scatter(gen + jitter, r_val, color="C0", alpha=0.9, s=30, zorder=5)

        mean_rs = [np.mean(gen_data[g]) for g in generations]
        ax_left.plot(
            generations, mean_rs, color="C0", linewidth=2, marker="o", markersize=7, zorder=6, label="Per-gen mean"
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
                color="C2",
                linestyle="-.",
                linewidth=2,
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

    rows = []
    total_pairs: dict[str, int] = {}
    for ptype in pair_types:
        n_total = 0
        for s in all_stats:
            ct = s.get("cross_trait_tetrachoric", {})
            entry = ct.get("cross_person", {}).get(ptype, {})
            r = entry.get("r")
            n_p = entry.get("n_pairs", 0)
            n_total += n_p
            if r is not None:
                rows.append({"pair_type": ptype, "r": r})
        total_pairs[ptype] = n_total

    df_plot = pd.DataFrame(rows)

    if not df_plot.empty:
        datasets = [df_plot.loc[df_plot["pair_type"] == pt, "r"].values for pt in pair_types]
        colors = [pair_colors[pt] for pt in pair_types]
        draw_colored_violins(ax_right, datasets, list(range(len(pair_types))), colors)

        for i, ptype in enumerate(pair_types):
            rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
            if len(rep_vals):
                jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(rep_vals))
                ax_right.scatter(
                    i + jitter,
                    rep_vals,
                    color="black",
                    s=15,
                    alpha=0.9,
                    zorder=5,
                )

        # N annotations
        n_reps = len(all_stats)
        for i, ptype in enumerate(pair_types):
            rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
            if len(rep_vals):
                top = rep_vals.max()
                ax_right.text(
                    i,
                    top + 0.04,
                    f"n={total_pairs[ptype] // n_reps:,}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    else:
        ax_right.text(0.5, 0.5, "No cross-person data", ha="center", va="center", transform=ax_right.transAxes)

    ax_right.set_xticks(range(len(pair_types)))
    ax_right.set_xticklabels(pair_types, fontsize=9, rotation=15, ha="right")
    ax_right.set_xlabel("")
    ax_right.set_ylabel("Cross-trait tetrachoric r")
    ax_right.set_title("Cross-Person: personA.affected1 vs personB.affected2")

    fig.suptitle(f"Cross-Trait Tetrachoric Correlations [{scenario}]", fontsize=14)
    finalize_plot(output_path)


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
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)

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
            from sim_ace.plotting.plot_distributions import COLOR_FEMALE, COLOR_MALE

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
                from sim_ace.core.utils import fast_linregress

                mean_slope, mean_intercept, mean_r, mean_stderr, _mean_pvalue = fast_linregress(
                    midparent_liab, offspring_liab
                )
                mean_n = int(valid.sum())

            # Observed regression line
            x_line = np.array([midparent_liab.min(), midparent_liab.max()])
            ax.plot(x_line, mean_slope * x_line + mean_intercept, color="C3", linewidth=2)

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
                        color="C3",
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
                        color="C1",
                        linestyle="--",
                        linewidth=2,
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
        Line2D([0], [0], color="C3", linewidth=2, label="Observed h\u00b2"),
        Line2D([0], [0], color=COLOR_FEMALE, linewidth=1.5, label="Daughters"),
        Line2D([0], [0], color=COLOR_MALE, linewidth=1.5, label="Sons"),
    ]
    if has_any_expected:
        legend_handles.append(Line2D([0], [0], color="C1", linestyle="--", linewidth=2, label="Expected (A)"))
    axes[0, -1].legend(handles=legend_handles, loc="lower right", fontsize=8)

    fig.suptitle(f"Midparent-Offspring Liability Regression [{scenario}]", fontsize=14)
    finalize_plot(output_path, subsample_note=subsample_note)


def plot_heritability_by_generation(
    all_validations: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot narrow-sense heritability h² = Var(A)/(Var(A)+Var(C)+Var(E)) per generation.

    Uses per-generation variance components from validation.yaml.
    """
    # Extract per-generation data from each replicate
    per_gen_all = [v.get("per_generation", {}) for v in all_validations]
    if not per_gen_all or not per_gen_all[0]:
        save_placeholder_plot(output_path, "No per-generation data")
        return

    # Determine generations from first replicate
    gen_keys = sorted(per_gen_all[0].keys(), key=lambda k: int(k.split("_")[1]))
    generations = [int(k.split("_")[1]) for k in gen_keys]

    # Get configured heritability (expected value) from first replicate's parameters
    params = all_validations[0].get("parameters", {})
    expected_h2 = {1: params.get("A1", None), 2: params.get("A2", None)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, trait_num in enumerate([1, 2]):
        ax = axes[col]
        a_key = f"A{trait_num}_var"
        c_key = f"C{trait_num}_var"
        e_key = f"E{trait_num}_var"

        # Compute h² for each replicate and generation
        h2_per_rep = []
        for pg in per_gen_all:
            rep_h2 = []
            for gk in gen_keys:
                gs = pg.get(gk, {})
                a_var = gs.get(a_key, 0)
                c_var = gs.get(c_key, 0)
                e_var = gs.get(e_key, 0)
                total = a_var + c_var + e_var
                rep_h2.append(a_var / total if total > 0 else np.nan)
            h2_per_rep.append(rep_h2)

        h2_arr = np.array(h2_per_rep)  # shape (n_reps, n_gens)

        # Plot per-replicate dots
        for rep_idx in range(h2_arr.shape[0]):
            jitter = np.random.default_rng(42 + rep_idx).uniform(-0.08, 0.08, len(generations))
            ax.scatter(
                np.array(generations) + jitter,
                h2_arr[rep_idx],
                color="C0",
                alpha=0.9,
                s=25,
                zorder=5,
            )

        # Expected heritability reference line
        exp = expected_h2.get(trait_num)
        if exp is not None:
            ax.axhline(
                y=exp, color="C1", linestyle="--", linewidth=2, alpha=0.7, label=f"Parametric A{trait_num} = {exp}"
            )
            ax.legend(loc="lower left", fontsize=9)

        ax.set_xlabel("Generation")
        ax.set_ylabel("h\u00b2 = Var(A) / Var(L)")
        ax.set_title(f"Trait {trait_num}")
        ax.set_xticks(generations)
        ax.set_ylim(0, 1)

    fig.suptitle(f"Realized Narrow-Sense Liability-Scale Heritability by Generation [{scenario}]", fontsize=14)
    finalize_plot(output_path)


def plot_broad_heritability_by_generation(
    all_validations: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot broad-sense heritability H² = (Var(A)+Var(C))/(Var(A)+Var(C)+Var(E)) per generation."""
    per_gen_all = [v.get("per_generation", {}) for v in all_validations]
    if not per_gen_all or not per_gen_all[0]:
        save_placeholder_plot(output_path, "No per-generation data")
        return

    gen_keys = sorted(per_gen_all[0].keys(), key=lambda k: int(k.split("_")[1]))
    generations = [int(k.split("_")[1]) for k in gen_keys]

    params = all_validations[0].get("parameters", {})
    expected_H2 = {}
    for t in [1, 2]:
        a = params.get(f"A{t}")
        c = params.get(f"C{t}")
        if a is not None and c is not None:
            expected_H2[t] = a + c

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, trait_num in enumerate([1, 2]):
        ax = axes[col]
        a_key = f"A{trait_num}_var"
        c_key = f"C{trait_num}_var"
        e_key = f"E{trait_num}_var"

        H2_per_rep = []
        for pg in per_gen_all:
            rep_H2 = []
            for gk in gen_keys:
                gs = pg.get(gk, {})
                a_var = gs.get(a_key, 0)
                c_var = gs.get(c_key, 0)
                e_var = gs.get(e_key, 0)
                total = a_var + c_var + e_var
                rep_H2.append((a_var + c_var) / total if total > 0 else np.nan)
            H2_per_rep.append(rep_H2)

        H2_arr = np.array(H2_per_rep)

        for rep_idx in range(H2_arr.shape[0]):
            jitter = np.random.default_rng(42 + rep_idx).uniform(-0.08, 0.08, len(generations))
            ax.scatter(
                np.array(generations) + jitter,
                H2_arr[rep_idx],
                color="C0",
                alpha=0.9,
                s=25,
                zorder=5,
            )

        exp = expected_H2.get(trait_num)
        if exp is not None:
            ax.axhline(
                y=exp,
                color="C1",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label=f"Parametric A{trait_num}+C{trait_num} = {exp}",
            )
            ax.legend(loc="lower left", fontsize=9)

        ax.set_xlabel("Generation")
        ax.set_ylabel("(Var(A)+Var(C)) / Var(L)")
        ax.set_title(f"Trait {trait_num}")
        ax.set_xticks(generations)
        ax.set_ylim(0, 1)

    fig.suptitle(f"Realized Additive Genetic and Shared Environment by Generation [{scenario}]", fontsize=14)
    finalize_plot(output_path)


def plot_tetrachoric_by_sex(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    params: dict[str, Any] | None = None,
) -> None:
    """Tetrachoric correlations for same-sex pairs: 2 rows (traits) x 2 cols (F/M)."""
    pair_types = PAIR_TYPES
    pair_colors = PAIR_COLORS

    sex_labels = [("female", "Female"), ("male", "Male")]

    # Check if data exists
    has_data = any(s.get("tetrachoric_by_sex") for s in all_stats)
    if not has_data:
        save_placeholder_plot(output_path, "No sex-stratified tetrachoric data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)

    for col_idx, (sex_key, sex_display) in enumerate(sex_labels):
        for row_idx, trait_num in enumerate([1, 2]):
            ax = axes[row_idx, col_idx]
            key = f"trait{trait_num}"

            rows = []
            total_pairs: dict[str, int] = {}
            for ptype in pair_types:
                n_total = 0
                for s in all_stats:
                    entry = s.get("tetrachoric_by_sex", {}).get(sex_key, {}).get(key, {}).get(ptype, {})
                    r = entry.get("r")
                    n_p = entry.get("n_pairs", 0)
                    n_total += n_p
                    if r is not None:
                        rows.append({"pair_type": ptype, "r": r})
                total_pairs[ptype] = n_total

            df_plot = pd.DataFrame(rows)

            if not df_plot.empty:
                datasets = [df_plot.loc[df_plot["pair_type"] == pt, "r"].values for pt in pair_types]
                colors = [pair_colors[pt] for pt in pair_types]
                draw_colored_violins(ax, datasets, list(range(len(pair_types))), colors)

                # Per-rep dots
                for i, ptype in enumerate(pair_types):
                    rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                    if len(rep_vals):
                        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(rep_vals))
                        ax.scatter(i + jitter, rep_vals, color="black", s=15, alpha=0.9, zorder=5)

                # Liability reference lines
                for i, ptype in enumerate(pair_types):
                    liab_vals = [
                        s.get("tetrachoric_by_sex", {}).get(sex_key, {}).get(key, {}).get(ptype, {}).get("liability_r")
                        for s in all_stats
                    ]
                    liab_vals = [v for v in liab_vals if v is not None]
                    if liab_vals:
                        ax.hlines(
                            np.mean(liab_vals),
                            i - 0.35,
                            i + 0.35,
                            colors="black",
                            linestyles="dashed",
                            linewidth=2,
                            zorder=4,
                        )

                # Parametric expected
                if params is not None:
                    A = params.get(f"A{trait_num}")
                    C = params.get(f"C{trait_num}")
                    if A is not None and C is not None:
                        for i, ptype in enumerate(pair_types):
                            exp_r = _expected_liability_corr(float(A), float(C), ptype)
                            if exp_r is not None:
                                ax.hlines(
                                    exp_r,
                                    i - 0.35,
                                    i + 0.35,
                                    colors="C3",
                                    linestyles="dotted",
                                    linewidth=2,
                                    zorder=4,
                                )

                # N pairs annotation
                n_reps = len(all_stats)
                for i, ptype in enumerate(pair_types):
                    rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                    if len(rep_vals):
                        ax.text(
                            i,
                            ax.get_ylim()[1] - 0.02,
                            f"n={total_pairs[ptype] // n_reps:,}",
                            ha="center",
                            va="top",
                            fontsize=7,
                        )

            ax.set_xticks(range(len(pair_types)))
            ax.set_xticklabels(pair_types, fontsize=8, rotation=15, ha="right")
            ax.set_ylabel("Tetrachoric r")
            ax.set_ylim(-0.1, 1.1)
            if row_idx == 0:
                ax.set_title(f"{sex_display} — Trait {trait_num}")
            else:
                ax.set_title(f"Trait {trait_num}")

    fig.suptitle(f"Tetrachoric Correlation by Sex (same-sex pairs) [{scenario}]", fontsize=14)
    finalize_plot(output_path)


def plot_heritability_by_sex_generation(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    params: dict[str, Any] | None = None,
) -> None:
    """Plot PO-regression heritability by offspring sex and generation.

    1x2 panel (one per trait). Each panel shows per-rep h² dots in two
    series: daughters (green) and sons (blue).
    """
    from sim_ace.plotting.plot_distributions import COLOR_FEMALE, COLOR_MALE

    has_data = any(s.get("parent_offspring_corr_by_sex") for s in all_stats)
    if not has_data:
        save_placeholder_plot(output_path, "No sex-stratified PO regression data")
        return

    # Discover generations from data
    gen_set: set[int] = set()
    for s in all_stats:
        po_sex = s.get("parent_offspring_corr_by_sex", {})
        for sex_key in ["female", "male"]:
            for trait_key in ["trait1", "trait2"]:
                for gk in po_sex.get(sex_key, {}).get(trait_key, {}):
                    gen_set.add(int(gk.replace("gen", "")))
    if not gen_set:
        save_placeholder_plot(output_path, "No generation data in PO sex stats")
        return
    generations = sorted(gen_set)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, trait_num in enumerate([1, 2]):
        ax = axes[col]
        trait_key = f"trait{trait_num}"

        for sex_key, sex_display, color in [
            ("female", "Daughters", COLOR_FEMALE),
            ("male", "Sons", COLOR_MALE),
        ]:
            for rep_idx, s in enumerate(all_stats):
                po_data = s.get("parent_offspring_corr_by_sex", {}).get(sex_key, {}).get(trait_key, {})
                h2_vals = []
                gen_vals = []
                for gen in generations:
                    entry = po_data.get(f"gen{gen}", {})
                    slope = entry.get("slope")
                    if slope is not None:
                        h2_vals.append(slope)
                        gen_vals.append(gen)
                if h2_vals:
                    jitter = np.random.default_rng(42 + rep_idx).uniform(-0.08, 0.08, len(gen_vals))
                    ax.scatter(
                        np.array(gen_vals) + jitter,
                        h2_vals,
                        color=color,
                        alpha=0.8,
                        s=25,
                        zorder=5,
                        label=sex_display if rep_idx == 0 else None,
                    )

        # Parametric expected h²
        if params is not None:
            exp = params.get(f"A{trait_num}")
            if exp is not None:
                ax.axhline(
                    y=exp,
                    color="C1",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"Parametric A{trait_num} = {exp}",
                )

        ax.set_xlabel("Generation")
        ax.set_ylabel("h\u00b2 (PO regression slope)")
        ax.set_title(f"Trait {trait_num}")
        ax.set_xticks(generations)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left", fontsize=9)

    fig.suptitle(f"PO-Regression Heritability by Offspring Sex [{scenario}]", fontsize=14)
    finalize_plot(output_path)
