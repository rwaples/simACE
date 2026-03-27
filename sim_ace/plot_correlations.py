"""Correlation-related phenotype plots.

Contains: plot_tetrachoric_sibling, plot_tetrachoric_by_generation,
plot_cross_trait_tetrachoric, plot_parent_offspring_liability,
plot_heritability_by_generation, plot_broad_heritability_by_generation,
plot_cross_trait_frailty_by_generation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim_ace.utils import PAIR_COLORS, PAIR_TYPES, draw_colored_violins, finalize_plot, save_placeholder_plot

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
) -> None:
    """2 x 3 scatter grid: midparent vs offspring liability by generation."""
    from scipy.stats import linregress

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

            ax.plot(midparent_liab, offspring_liab, "o", ms=2, mew=0, alpha=0.3, rasterized=True)

            # Collect pre-computed stats (averaged across reps)
            r_vals, r2_vals, slope_vals, intercept_vals, n_vals = [], [], [], [], []
            for s in all_stats:
                po = s.get("parent_offspring_corr", {}).get(f"trait{trait_num}", {}).get(f"gen{gen}", {})
                if po and po.get("r") is not None:
                    r_vals.append(po["r"])
                    r2_vals.append(po.get("r2", po["r"] ** 2))
                    slope_vals.append(po["slope"])
                    intercept_vals.append(po["intercept"])
                    n_vals.append(po["n_pairs"])

            if r_vals:
                mean_r = np.mean(r_vals)
                mean_r2 = np.mean(r2_vals)
                mean_slope = np.mean(slope_vals)
                mean_intercept = np.mean(intercept_vals)
                mean_n = int(np.mean(n_vals))
            else:
                reg = linregress(midparent_liab, offspring_liab)
                mean_r = float(reg.rvalue)
                mean_r2 = float(reg.rvalue**2)
                mean_slope = float(reg.slope)
                mean_intercept = float(reg.intercept)
                mean_n = int(valid.sum())

            # Regression line from pre-computed stats
            x_line = np.array([midparent_liab.min(), midparent_liab.max()])
            ax.plot(x_line, mean_slope * x_line + mean_intercept, color="C3", linewidth=2)

            ax.text(
                0.05,
                0.95,
                f"r = {mean_r:.4f}\nR\u00b2 = {mean_r2:.4f}\nn = {mean_n}",
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


def plot_cross_trait_frailty_by_generation(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot per-generation cross-trait frailty correlation estimates.

    Shows per-rep per-generation r as dots, mean line across generations,
    and reference lines for oracle (uncensored), stratified IVW, and naive
    pooled estimates.
    """
    # Collect per-generation data from each replicate
    gen_data: dict[int, list[tuple[float, float]]] = {}  # gen -> [(r, se), ...]
    oracle_rs: list[float] = []
    strat_rs: list[float] = []
    naive_rs: list[float] = []

    for s in all_stats:
        ct_strat = s.get("frailty_cross_trait_stratified", {})
        ct_unc = s.get("frailty_cross_trait_uncensored", {})
        ct_cens = s.get("frailty_cross_trait", {})

        if ct_unc and ct_unc.get("r") is not None:
            oracle_rs.append(ct_unc["r"])
        if ct_strat and ct_strat.get("r") is not None:
            strat_rs.append(ct_strat["r"])
        if ct_cens and ct_cens.get("r") is not None:
            naive_rs.append(ct_cens["r"])

        per_gen = ct_strat.get("per_generation", {}) if ct_strat else {}
        for gk, gv in per_gen.items():
            gen_num = int(gk.replace("gen", ""))
            r_g = gv.get("r")
            se_g = gv.get("se")
            if r_g is not None:
                gen_data.setdefault(gen_num, []).append((r_g, se_g if se_g is not None else 0.0))

    if not gen_data:
        save_placeholder_plot(
            output_path,
            "No per-generation cross-trait data\n\n(requires extra_tetrachoric: True in scenario config)",
            figsize=(8, 5),
        )
        return

    generations = sorted(gen_data.keys())
    _fig, ax = plt.subplots(figsize=(8, 5))

    # Per-replicate dots with jitter
    for gen in generations:
        reps = gen_data[gen]
        for rep_idx, (r_g, se_g) in enumerate(reps):
            jitter = np.random.default_rng(42 + rep_idx).uniform(-0.08, 0.08)
            ax.scatter(
                gen + jitter,
                r_g,
                color="C0",
                alpha=0.9,
                s=30,
                zorder=5,
            )
            if se_g > 0:
                ax.errorbar(
                    gen + jitter,
                    r_g,
                    yerr=1.96 * se_g,
                    color="C0",
                    alpha=0.4,
                    fmt="none",
                    capsize=2,
                    zorder=4,
                )

    # Mean line across generations
    mean_rs = [np.mean([r for r, _ in gen_data[g]]) for g in generations]
    ax.plot(
        generations, mean_rs, color="C0", linewidth=2, marker="o", markersize=7, zorder=6, label="Per-generation mean"
    )

    # Reference lines
    if oracle_rs:
        mean_oracle = np.mean(oracle_rs)
        ax.axhline(
            y=mean_oracle,
            color="C2",
            linestyle="-.",
            linewidth=2,
            alpha=0.7,
            label=f"Uncensored oracle = {mean_oracle:.3f}",
        )

    if strat_rs:
        mean_strat = np.mean(strat_rs)
        ax.axhline(
            y=mean_strat, color="C1", linestyle="--", linewidth=2, alpha=0.7, label=f"Stratified IVW = {mean_strat:.3f}"
        )

    if naive_rs:
        mean_naive = np.mean(naive_rs)
        ax.axhline(
            y=mean_naive, color="C3", linestyle=":", linewidth=2, alpha=0.7, label=f"Naive pooled = {mean_naive:.3f}"
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Cross-trait liability correlation (r)")
    ax.set_title(f"Cross-Trait Correlation by Generation [{scenario}]", fontsize=13)
    ax.set_xticks(generations)
    ax.legend(loc="best", fontsize=9)

    finalize_plot(output_path)
