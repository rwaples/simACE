"""Correlation-related phenotype plots.

Contains: plot_tetrachoric_sibling, plot_tetrachoric_by_generation,
plot_parent_offspring_liability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import logging
logger = logging.getLogger(__name__)


def plot_tetrachoric_sibling(all_stats: list[dict[str, Any]], output_path: str | Path, scenario: str) -> None:
    """Plot tetrachoric correlations by relationship type, violin with rep dots."""
    pair_types = ["MZ twin", "Full sib", "Mother-offspring", "Father-offspring", "Maternal half sib", "Paternal half sib", "1st cousin"]
    pair_colors = {"MZ twin": "C0", "Full sib": "C1", "Mother-offspring": "C3", "Father-offspring": "C5", "Maternal half sib": "C2", "Paternal half sib": "C6", "1st cousin": "C4"}

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
            sns.violinplot(
                data=df_plot, x="pair_type", y="r", hue="pair_type", ax=ax,
                order=pair_types, palette=pair_colors, legend=False,
                inner=None, cut=0, alpha=0.6, zorder=3,
            )

            # Overlay per-rep dots
            for i, ptype in enumerate(pair_types):
                rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                if len(rep_vals):
                    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(rep_vals))
                    ax.scatter(
                        i + jitter, rep_vals, color="black", s=15,
                        alpha=0.6, zorder=5,
                    )

            # Annotate mean N pairs per rep
            n_reps = len(all_stats)
            for i, ptype in enumerate(pair_types):
                rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                if len(rep_vals):
                    top = rep_vals.max()
                    ax.text(
                        i, top + 0.03,
                        f"N={total_pairs[ptype] // n_reps}", ha="center", va="bottom", fontsize=8,
                    )

        # Liability correlation lines (averaged across reps)
        for i, ptype in enumerate(pair_types):
            liab_vals = [
                s.get("liability_correlations", {}).get(key, {}).get(ptype)
                for s in all_stats
            ]
            liab_vals = [v for v in liab_vals if v is not None]
            if liab_vals:
                mean_liab_r = np.mean(liab_vals)
                ax.hlines(
                    mean_liab_r, i - 0.35, i + 0.35, colors="black",
                    linestyles="dashed", linewidth=2, zorder=4,
                )

        # Uncensored Weibull pairwise correlation lines (averaged across reps)
        has_uncens = any(s.get("weibull_corr_uncensored") for s in all_stats)
        if has_uncens:
            for i, ptype in enumerate(pair_types):
                uncens_vals = [
                    s.get("weibull_corr_uncensored", {}).get(key, {}).get(ptype, {}).get("r")
                    for s in all_stats
                ]
                uncens_vals = [v for v in uncens_vals if v is not None]
                if uncens_vals:
                    mean_uncens_r = np.mean(uncens_vals)
                    ax.hlines(
                        mean_uncens_r, i - 0.35, i + 0.35, colors="C2",
                        linestyles="dashdot", linewidth=2, zorder=5,
                    )

        ax.set_xticks(range(len(pair_types)))
        ax.set_xticklabels(pair_types, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Tetrachoric Correlation")
        ax.set_title(f"Trait {trait_num}")
        ax.set_ylim(-0.1, 1.1)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Liability r"),
        ]
        if has_uncens:
            legend_elements.append(
                Line2D([0], [0], color="C2", linestyle="-.", linewidth=2, label="Weibull r (uncensored)"),
            )
        ax.legend(handles=legend_elements, loc="upper right")

    fig.suptitle(
        f"Tetrachoric Correlation (Weibull) [{scenario}]", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_tetrachoric_by_generation(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot tetrachoric correlations by relationship type, broken out by generation.

    2 rows (traits) x N cols (last 3 non-founder generations) grid.
    Each panel is a bar chart of 7 pair types with dashed liability reference lines.
    """
    # Determine which generations are available across reps
    gen_keys_sets = [
        set(s.get("tetrachoric_by_generation", {}).keys()) for s in all_stats
    ]
    if not gen_keys_sets or not gen_keys_sets[0]:
        # No generation data — create placeholder
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No per-generation tetrachoric data",
                ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    # Intersection of available gens across all reps, sorted
    gen_keys = sorted(set.intersection(*gen_keys_sets))
    if not gen_keys:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No per-generation tetrachoric data",
                ha="center", va="center", transform=ax.transAxes)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    pair_types = [
        "MZ twin", "Full sib", "Mother-offspring", "Father-offspring",
        "Maternal half sib", "Paternal half sib", "1st cousin",
    ]
    pair_colors = {"MZ twin": "C0", "Full sib": "C1", "Mother-offspring": "C3", "Father-offspring": "C5", "Maternal half sib": "C2", "Paternal half sib": "C6", "1st cousin": "C4"}

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
                sns.violinplot(
                    data=df_plot, x="pair_type", y="r", hue="pair_type", ax=ax,
                    order=pair_types, palette=pair_colors, legend=False,
                    inner=None, cut=0, alpha=0.6, zorder=3,
                )

                # Overlay per-rep dots
                for i, ptype in enumerate(pair_types):
                    rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                    if len(rep_vals):
                        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(rep_vals))
                        ax.scatter(
                            i + jitter, rep_vals, color="black", s=12,
                            alpha=0.6, zorder=5,
                        )

                # Annotate mean N pairs per rep
                n_reps = len(all_stats)
                for i, ptype in enumerate(pair_types):
                    rep_vals = df_plot.loc[df_plot["pair_type"] == ptype, "r"].values
                    if len(rep_vals):
                        top = rep_vals.max()
                        ax.text(
                            i, top + 0.03,
                            f"N={total_pairs[ptype] // n_reps}", ha="center", va="bottom", fontsize=7,
                        )

            # Liability correlation reference lines (dashed)
            for i, liab_r in enumerate(mean_liab_rs):
                if not np.isnan(liab_r):
                    ax.hlines(
                        liab_r, i - 0.35, i + 0.35, colors="black",
                        linestyles="dashed", linewidth=2, zorder=4,
                    )

            ax.set_xticks(range(len(pair_types)))
            ax.set_xticklabels(pair_types, fontsize=7, rotation=30, ha="right")
            ax.set_ylim(-0.1, 1.1)

            if row == 0:
                # Extract gen number from key like "gen3"
                gen_num = gen_key.replace("gen", "")
                ax.set_title(f"Gen {gen_num}", fontsize=12)
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nTetrachoric Correlation")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Liability r"),
    ]
    axes[0, -1].legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.suptitle(
        f"Tetrachoric Correlation by Generation (Weibull) [{scenario}]", fontsize=14
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_parent_offspring_liability(
    df_samples: pd.DataFrame,
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """2 x 3 scatter grid: midparent vs offspring liability by generation."""
    from scipy.stats import linregress

    if "generation" not in df_samples.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No generation data", ha="center", va="center",
                transform=ax.transAxes)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    max_gen = int(df_samples["generation"].max())
    # Last 3 non-founder generations (gen 0 = founders)
    plot_gens = list(range(max(1, max_gen - 2), max_gen + 1))

    # Build id -> row lookup within df_samples
    ids_arr = df_samples["id"].values.astype(np.int64)
    max_id = int(ids_arr.max()) + 1
    id_to_row = np.full(max_id, -1, dtype=np.int32)
    id_to_row[ids_arr] = np.arange(len(df_samples), dtype=np.int32)

    n_cols = len(plot_gens)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)

    for row, trait_num in enumerate([1, 2]):
        liability = df_samples[f"liability{trait_num}"].values

        for col, gen in enumerate(plot_gens):
            ax = axes[row, col]
            gen_idx = np.where(df_samples["generation"].values == gen)[0]

            mother_ids = df_samples["mother"].values[gen_idx].astype(np.int64)
            father_ids = df_samples["father"].values[gen_idx].astype(np.int64)

            has_m = (mother_ids >= 0) & (mother_ids < max_id)
            has_f = (father_ids >= 0) & (father_ids < max_id)

            m_rows = np.full(len(gen_idx), -1, dtype=np.int32)
            f_rows = np.full(len(gen_idx), -1, dtype=np.int32)
            m_rows[has_m] = id_to_row[mother_ids[has_m]]
            f_rows[has_f] = id_to_row[father_ids[has_f]]

            valid = (m_rows >= 0) & (f_rows >= 0)

            if valid.sum() < 2:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                        transform=ax.transAxes)
                if row == 0:
                    ax.set_title(f"Gen {gen}")
                continue

            offspring_liab = liability[gen_idx[valid]]
            midparent_liab = (liability[m_rows[valid]] + liability[f_rows[valid]]) / 2.0

            ax.scatter(midparent_liab, offspring_liab, alpha=0.15, s=3, rasterized=True)

            # Regression line
            reg = linregress(midparent_liab, offspring_liab)
            x_line = np.array([midparent_liab.min(), midparent_liab.max()])
            ax.plot(x_line, reg.slope * x_line + reg.intercept, color="C3", linewidth=2)

            # Annotation from pre-computed stats (averaged across reps)
            r_vals = []
            n_vals = []
            for s in all_stats:
                po = s.get("parent_offspring_corr", {}).get(
                    f"trait{trait_num}", {}
                ).get(f"gen{gen}", {})
                if po and po.get("r") is not None:
                    r_vals.append(po["r"])
                    n_vals.append(po["n_pairs"])

            if r_vals:
                mean_r = np.mean(r_vals)
                mean_n = int(np.mean(n_vals))
            else:
                mean_r = float(np.corrcoef(midparent_liab, offspring_liab)[0, 1])
                mean_n = int(valid.sum())

            ax.text(
                0.05, 0.95, f"r = {mean_r:.3f}\nn = {mean_n}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

            if row == 0:
                ax.set_title(f"Gen {gen}")
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nOffspring Liability")
            if row == 1:
                ax.set_xlabel("Midparent Liability")

    fig.suptitle(f"Parent-Offspring Liability Correlation [{scenario}]", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
