"""Liability-related phenotype plots.

Contains: plot_liability_joint, plot_liability_joint_affected,
plot_liability_violin, plot_liability_violin_by_generation, plot_joint_affection,
plot_censoring_confusion, plot_censoring_cascade.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sim_ace.stats import tetrachoric_corr
from sim_ace.utils import save_placeholder_plot, finalize_plot

import logging
logger = logging.getLogger(__name__)


def _plot_joint_grid(
    df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "",
    color_by_affected: bool = False,
) -> None:
    """Internal: 2x2 grid of jointplots for cross-trait correlations."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    panels = [
        ("liability1", "liability2", "Liability"),
        ("A1", "A2", "A (Additive genetic)"),
        ("C1", "C2", "C (Common environment)"),
        ("E1", "E2", "E (Unique environment)"),
    ]
    panels = [
        (x, y, t)
        for x, y, t in panels
        if x in df_samples.columns and y in df_samples.columns
    ]

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"Cross-Trait Correlations [{scenario}]", fontsize=14, y=1.01)
    outer = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    if color_by_affected:
        affected = df_samples["affected1"].values.astype(bool)

    for idx, (xcol, ycol, title) in enumerate(panels):
        inner = GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[idx],
            width_ratios=[4, 1], height_ratios=[1, 4],
            hspace=0.05, wspace=0.05,
        )
        ax_joint = fig.add_subplot(inner[1, 0])
        ax_marg_x = fig.add_subplot(inner[0, 0], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(inner[1, 1], sharey=ax_joint)

        x, y = df_samples[xcol].values, df_samples[ycol].values

        if color_by_affected:
            bins_x = np.linspace(x.min(), x.max(), 51)
            bins_y = np.linspace(y.min(), y.max(), 51)
            for mask, color, alpha, label in [
                (~affected, "C0", 0.2, "Unaffected"),
                (affected, "C3", 0.5, "Affected (T1)"),
            ]:
                ax_joint.scatter(
                    x[mask], y[mask], c=color, alpha=alpha, s=3, rasterized=True, label=label,
                )
            ax_marg_x.hist(x[~affected], bins=bins_x.tolist(), edgecolor="none", alpha=0.5, color="C0")
            ax_marg_x.hist(x[affected], bins=bins_x.tolist(), edgecolor="none", alpha=0.7, color="C3")
            ax_marg_y.hist(
                y[~affected], bins=bins_y.tolist(), orientation="horizontal", edgecolor="none", alpha=0.5, color="C0",
            )
            ax_marg_y.hist(
                y[affected], bins=bins_y.tolist(), orientation="horizontal", edgecolor="none", alpha=0.7, color="C3",
            )
        else:
            ax_joint.scatter(x, y, alpha=0.3, s=3, rasterized=True)
            ax_marg_x.hist(x, bins=50, edgecolor="none", alpha=0.7)
            ax_marg_y.hist(
                y, bins=50, orientation="horizontal", edgecolor="none", alpha=0.7
            )

        r = np.corrcoef(x, y)[0, 1]
        ax_joint.text(
            0.05, 0.95, f"r = {r:.4f}",
            transform=ax_joint.transAxes, va="top", fontsize=11,
        )
        ax_joint.set_xlabel(f"{title} (Trait 1)")
        ax_joint.set_ylabel(f"{title} (Trait 2)")

        ax_marg_x.set_title(f"{title}: Trait 1 vs Trait 2", fontsize=11)
        ax_marg_x.tick_params(labelbottom=False, labelleft=False)
        ax_marg_x.set_ylabel("")
        ax_marg_y.tick_params(labelleft=False, labelbottom=False)
        ax_marg_y.set_xlabel("")

        ax_corner = fig.add_subplot(inner[0, 1])
        ax_corner.axis("off")

    if color_by_affected:
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=8, label="Unaffected"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", markersize=8, label="Affected (T1)"),
        ]
        fig.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)

    finalize_plot(output_path)


def plot_liability_joint(df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "") -> None:
    """2x2 grid of jointplots: Liability, A, C, E (trait 1 vs trait 2)."""
    _plot_joint_grid(df_samples, output_path, scenario, color_by_affected=False)


def plot_liability_joint_affected(df_samples: pd.DataFrame, output_path: str | Path, scenario: str = "") -> None:
    """2x2 grid of jointplots colored by affected status."""
    _plot_joint_grid(df_samples, output_path, scenario, color_by_affected=True)


def plot_liability_violin(df_samples: pd.DataFrame, all_stats: list[dict[str, Any]], output_path: str | Path, scenario: str = "") -> None:
    """Split violin plot of liability by trait, split on affected status."""
    violin_data = pd.concat(
        [
            pd.DataFrame({
                "Trait": "Trait 1",
                "Liability": df_samples["liability1"],
                "Affected": df_samples["affected1"],
            }),
            pd.DataFrame({
                "Trait": "Trait 2",
                "Liability": df_samples["liability2"],
                "Affected": df_samples["affected2"],
            }),
        ],
        ignore_index=True,
    )

    # Use pre-computed prevalence averaged across reps
    prev1 = np.mean([s["prevalence"]["trait1"] for s in all_stats])
    prev2 = np.mean([s["prevalence"]["trait2"] for s in all_stats])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.violinplot(
        data=violin_data, x="Trait", y="Liability", hue="Affected",
        split=True, cut=0, ax=ax,
    )
    ax.set_title(
        f"Liability by Affected Status [{scenario}]"
    )

    # Annotate mean liability for each trait x affected/unaffected group
    for i, trait_num in enumerate([1, 2]):
        liab = df_samples[f"liability{trait_num}"].values
        aff = df_samples[f"affected{trait_num}"].values.astype(bool)
        if aff.any():
            mean_aff = liab[aff].mean()
            ax.plot(i + 0.05, mean_aff, "D", color="black", markersize=6, zorder=5)
            ax.text(
                i + 0.12, mean_aff, f"\u03bc={mean_aff:.2f}",
                ha="left", va="center", fontsize=9, fontweight="bold",
            )
        if (~aff).any():
            mean_unaff = liab[~aff].mean()
            ax.plot(i - 0.05, mean_unaff, "D", color="black", markersize=6, zorder=5)
            ax.text(
                i - 0.12, mean_unaff, f"\u03bc={mean_unaff:.2f}",
                ha="right", va="center", fontsize=9, fontweight="bold",
            )

    ax.text(
        0, ax.get_ylim()[0], f"Prevalence: {prev1:.1%}",
        ha="center", va="top", fontsize=10, fontstyle="italic",
    )
    ax.text(
        1, ax.get_ylim()[0], f"Prevalence: {prev2:.1%}",
        ha="center", va="top", fontsize=10, fontstyle="italic",
    )
    finalize_plot(output_path)


def plot_liability_violin_by_generation(df_samples: pd.DataFrame, all_stats: list[dict[str, Any]], output_path: str | Path, scenario: str = "") -> None:
    """Split violin of liability by affected status, one column per generation."""
    if "generation" not in df_samples.columns:
        save_placeholder_plot(output_path, "No generation data")
        return

    gens = sorted(df_samples["generation"].unique())
    n_gens = len(gens)

    fig, axes = plt.subplots(2, n_gens, figsize=(4 * n_gens, 8), squeeze=False)

    for row, trait_num in enumerate([1, 2]):
        liab_col = f"liability{trait_num}"
        aff_col = f"affected{trait_num}"

        for col, gen in enumerate(gens):
            ax = axes[row, col]
            gen_mask = df_samples["generation"] == gen
            df_gen = df_samples.loc[gen_mask]

            violin_data = pd.DataFrame({
                "Trait": f"Trait {trait_num}",
                "Liability": df_gen[liab_col].values,
                "Affected": df_gen[aff_col].values,
            })

            if len(violin_data) > 1:
                sns.violinplot(
                    data=violin_data, x="Trait", y="Liability",
                    hue="Affected", split=True,
                    legend=(row == 0 and col == n_gens - 1),
                    ax=ax, cut=0,
                )

                # Annotate means
                liab = df_gen[liab_col].values
                aff = df_gen[aff_col].values.astype(bool)
                if aff.any():
                    mu = liab[aff].mean()
                    ax.plot(0.05, mu, "D", color="black", markersize=5, zorder=5)
                    ax.text(0.12, mu, f"\u03bc={mu:.2f}",
                            ha="left", va="center", fontsize=8, fontweight="bold")
                if (~aff).any():
                    mu = liab[~aff].mean()
                    ax.plot(-0.05, mu, "D", color="black", markersize=5, zorder=5)
                    ax.text(-0.12, mu, f"\u03bc={mu:.2f}",
                            ha="right", va="center", fontsize=8, fontweight="bold")

                # Prevalence annotation
                obs_prev = aff.mean()
                ax.set_xlabel(f"prev: {obs_prev:.1%}", fontsize=8)

            if row == 0:
                label = f"Gen {gen}"
                if col == 0:
                    label += " (oldest)"
                elif col == n_gens - 1:
                    label += " (youngest)"
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"Trait {trait_num}\nLiability", fontsize=10)
            else:
                ax.set_ylabel("")

    fig.suptitle(
        f"Liability by Affected Status per Generation [{scenario}]",
        fontsize=14,
    )
    finalize_plot(output_path)


def plot_censoring_confusion(
    df_samples: pd.DataFrame,
    censor_age: float,
    output_path: str | Path,
    scenario: str = "",
    gen_censoring: dict[int, list[float]] | None = None,
) -> None:
    """Per-trait 2x2 confusion matrix: true affected vs. observed affected.

    True affected  = t{i} < censor_age  (raw event time before global censor age).
    Observed affected = affected{i}     (after generation censoring + death censoring).

    Only includes individuals from phenotyped generations (observation window width > 0).
    """
    df = df_samples.copy()

    # Filter to phenotyped generations if gen_censoring is provided
    if gen_censoring is not None and "generation" in df.columns:
        active_gens = {
            int(g) for g, (lo, hi) in gen_censoring.items() if hi > lo
        }
        if active_gens:
            df = df[df["generation"].isin(active_gens)]

    if len(df) == 0:
        save_placeholder_plot(output_path, "No phenotyped individuals")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Censoring Confusion Matrix [{scenario}]", fontsize=14)

    for col, trait in enumerate([1, 2]):
        ax = axes[col]
        t_col = f"t{trait}"
        a_col = f"affected{trait}"

        if t_col not in df.columns or a_col not in df.columns:
            ax.text(0.5, 0.5, f"Missing {t_col} or {a_col}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Trait {trait}")
            continue

        true_aff = df[t_col].values < censor_age
        obs_aff = df[a_col].values.astype(bool)
        n = len(df)

        # Confusion matrix: rows = True (Yes/No), cols = Observed (Yes/No)
        tp = int(np.sum(true_aff & obs_aff))
        fn = int(np.sum(true_aff & ~obs_aff))
        fp = int(np.sum(~true_aff & obs_aff))
        tn = int(np.sum(~true_aff & ~obs_aff))

        props = np.array([
            [tp / n, fn / n],
            [fp / n, tn / n],
        ])
        labels = np.array([
            [f"{tp / n:.2f}\n(n={tp})", f"{fn / n:.2f}\n(n={fn})"],
            [f"{fp / n:.2f}\n(n={fp})", f"{tn / n:.2f}\n(n={tn})"],
        ])

        sns.heatmap(
            props, annot=labels, fmt="", cmap="Blues", ax=ax,
            xticklabels=["Observed Yes", "Observed No"],
            yticklabels=["True Yes", "True No"],
            vmin=0, vmax=max(props.max(), 0.01),
            cbar_kws={"label": "Proportion"},
        )

        # Sensitivity and specificity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        metrics = f"Sensitivity: {sensitivity:.3f}   Specificity: {specificity:.3f}"
        ax.set_title(f"Trait {trait}\n{metrics}", fontsize=11)

    finalize_plot(output_path)


def plot_censoring_cascade(
    df_samples: pd.DataFrame,
    censor_age: float,
    output_path: str | Path,
    scenario: str = "",
    gen_censoring: dict[int, list[float]] | None = None,
) -> None:
    """Per-trait stacked bar chart decomposing true cases by censoring fate per generation.

    For each generation, true affected individuals (event time < censor_age) are
    partitioned into four mutually exclusive categories:
      - Left-truncated: event before observation window start
      - Right-censored: event after observation window end
      - Death-censored: event in window but death occurs before event
      - Observed (TP): event in window and observed

    Total bar height = true affected count. Sensitivity annotated per generation.
    """
    df = df_samples.copy()

    if len(df) == 0 or "generation" not in df.columns:
        save_placeholder_plot(output_path, "No data available")
        return

    # Build per-generation observation windows
    gens = sorted(df["generation"].unique())
    windows: dict[int, tuple[float, float]] = {}
    for g in gens:
        if gen_censoring is not None and int(g) in gen_censoring:
            lo, hi = gen_censoring[int(g)]
        else:
            lo, hi = 0.0, censor_age
        if hi > lo:
            windows[int(g)] = (lo, hi)

    if not windows:
        save_placeholder_plot(output_path, "No non-degenerate observation windows")
        return

    active_gens = sorted(windows.keys())

    # Colors
    color_observed = "#4CAF50"
    color_death = "#E57373"
    color_right = "#FFB74D"
    color_left = "#FF9800"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Censoring Cascade [{scenario}]", fontsize=14)

    for col, trait in enumerate([1, 2]):
        ax = axes[col]
        t_col = f"t{trait}"
        a_col = f"affected{trait}"

        if t_col not in df.columns or a_col not in df.columns:
            ax.text(0.5, 0.5, f"Missing {t_col} or {a_col}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Trait {trait}")
            continue

        has_death = "death_age" in df.columns

        counts_observed = []
        counts_death = []
        counts_right = []
        counts_left = []
        sensitivities = []
        x_labels = []

        for g in active_gens:
            lo, hi = windows[g]
            gen_mask = df["generation"] == g
            df_g = df.loc[gen_mask]
            t = df_g[t_col].values
            true_affected = t < censor_age

            n_true = int(true_affected.sum())
            if n_true == 0:
                counts_observed.append(0)
                counts_death.append(0)
                counts_right.append(0)
                counts_left.append(0)
                sensitivities.append(float("nan"))
                x_labels.append(f"Gen {g}\n[{lo:.0f}, {hi:.0f}]")
                continue

            left_trunc = true_affected & (t < lo)
            right_cens = true_affected & (t > hi)
            in_window = true_affected & (t >= lo) & (t <= hi)

            if has_death:
                death_age = df_g["death_age"].values
                death_cens = in_window & (death_age < t)
                observed = in_window & (death_age >= t)
            else:
                death_cens = np.zeros_like(in_window)
                observed = in_window

            n_obs = int(observed.sum())
            n_death = int(death_cens.sum())
            n_right = int(right_cens.sum())
            n_left = int(left_trunc.sum())

            counts_observed.append(n_obs)
            counts_death.append(n_death)
            counts_right.append(n_right)
            counts_left.append(n_left)
            sensitivities.append(n_obs / n_true if n_true > 0 else float("nan"))
            x_labels.append(f"Gen {g}\n[{lo:.0f}, {hi:.0f}]")

        x = np.arange(len(active_gens))
        bar_width = 0.6

        # Stacked bars: observed (bottom) -> death -> right -> left (top)
        bottom = np.zeros(len(active_gens))
        bars_obs = np.array(counts_observed, dtype=float)
        bars_death = np.array(counts_death, dtype=float)
        bars_right = np.array(counts_right, dtype=float)
        bars_left = np.array(counts_left, dtype=float)

        ax.bar(x, bars_obs, bar_width, bottom=bottom, color=color_observed, label="Observed (TP)")
        bottom += bars_obs
        ax.bar(x, bars_death, bar_width, bottom=bottom, color=color_death, label="Death-censored")
        bottom += bars_death
        ax.bar(x, bars_right, bar_width, bottom=bottom, color=color_right, label="Right-censored")
        bottom += bars_right
        ax.bar(x, bars_left, bar_width, bottom=bottom, color=color_left, label="Left-truncated")
        bottom += bars_left

        # Annotate segments (skip if < 3% of bar height)
        for i in range(len(active_gens)):
            total = bottom[i]
            if total == 0:
                continue
            # Annotate each segment
            cum = 0.0
            for count, clr in [
                (bars_obs[i], color_observed),
                (bars_death[i], color_death),
                (bars_right[i], color_right),
                (bars_left[i], color_left),
            ]:
                if count > 0 and count / total >= 0.03:
                    mid = cum + count / 2
                    ax.text(x[i], mid, f"{int(count)}", ha="center", va="center",
                            fontsize=8, fontweight="bold")
                cum += count

        # Sensitivity annotation above each bar
        for i, sens in enumerate(sensitivities):
            if not np.isnan(sens):
                ax.text(x[i], bottom[i] + max(bottom) * 0.02,
                        f"sens={sens:.2f}", ha="center", va="bottom", fontsize=8)

        # Overall sensitivity
        total_obs = sum(counts_observed)
        total_true = sum(counts_observed) + sum(counts_death) + sum(counts_right) + sum(counts_left)
        overall_sens = total_obs / total_true if total_true > 0 else float("nan")
        ax.set_title(f"Trait {trait}  (overall sensitivity: {overall_sens:.3f})", fontsize=11)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel("True affected count")

    # Shared legend above the subplots, below the suptitle
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.95), framealpha=0.9)

    finalize_plot(output_path, tight_rect=[0, 0, 1, 0.93])


def plot_joint_affection(
    df_samples: pd.DataFrame,
    output_path: str | Path,
    scenario: str = "",
    all_stats: list[dict[str, Any]] | None = None,
) -> None:
    """2x2 heatmap of joint affection status (trait1 x trait2)."""
    a1 = df_samples["affected1"].values.astype(bool)
    a2 = df_samples["affected2"].values.astype(bool)
    n = len(df_samples)

    props = {
        "both": np.sum(a1 & a2) / n,
        "trait1_only": np.sum(a1 & ~a2) / n,
        "trait2_only": np.sum(~a1 & a2) / n,
        "neither": np.sum(~a1 & ~a2) / n,
    }
    counts = {
        "both": int(np.sum(a1 & a2)),
        "trait1_only": int(np.sum(a1 & ~a2)),
        "trait2_only": int(np.sum(~a1 & a2)),
        "neither": int(np.sum(~a1 & ~a2)),
    }

    matrix = np.array([
        [props["both"], props["trait1_only"]],
        [props["trait2_only"], props["neither"]],
    ])
    labels = np.array([
        [f"{props['both']:.2f}\n(n={counts['both']})",
         f"{props['trait1_only']:.2f}\n(n={counts['trait1_only']})"],
        [f"{props['trait2_only']:.2f}\n(n={counts['trait2_only']})",
         f"{props['neither']:.2f}\n(n={counts['neither']})"],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        matrix, annot=labels, fmt="", cmap="Blues", ax=ax,
        xticklabels=["Affected", "Unaffected"],
        yticklabels=["Affected", "Unaffected"],
        vmin=0, vmax=max(matrix.max(), 0.01),
        cbar_kws={"label": "Proportion"},
    )
    # Cross-trait tetrachoric correlation
    r_tet = tetrachoric_corr(a1, a2)
    r_label = f"r_tet = {r_tet:.3f}" if not np.isnan(r_tet) else "r_tet = N/A"

    # Cross-trait correlation (averaged across reps)
    frailty_parts = []
    if all_stats:
        for s in all_stats:
            ct_unc = s.get("frailty_cross_trait_uncensored", {})
            ct_strat = s.get("frailty_cross_trait_stratified", {})
            ct_cens = s.get("frailty_cross_trait", {})
            r_uncens = ct_unc.get("r") if ct_unc else None
            r_strat = ct_strat.get("r") if ct_strat else None
            r_cens = ct_cens.get("r") if ct_cens else None
            if r_uncens is not None:
                frailty_parts.append((r_uncens, r_strat, r_cens))

    if frailty_parts:
        mean_uncens = np.mean([p[0] for p in frailty_parts])
        r_label += f"  |  r_frailty = {mean_uncens:.3f}"
        strat_vals = [p[1] for p in frailty_parts if p[1] is not None]
        if strat_vals:
            mean_strat = np.mean(strat_vals)
            r_label += f" (stratified: {mean_strat:.3f})"
        cens_vals = [p[2] for p in frailty_parts if p[2] is not None]
        if cens_vals:
            mean_cens = np.mean(cens_vals)
            r_label += f" (naive: {mean_cens:.3f})"

    ax.set_xlabel("Trait 1")
    ax.set_ylabel("Trait 2")
    ax.set_title(f"Joint Affected Status [{scenario}]\n{r_label}", fontsize=14)
    finalize_plot(output_path)
