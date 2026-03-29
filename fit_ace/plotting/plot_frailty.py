"""Cross-trait frailty correlation plots.

Moved from sim_ace.plotting.plot_correlations to fit_ace.plotting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from sim_ace.plotting.plot_utils import finalize_plot, save_placeholder_plot

if TYPE_CHECKING:
    from pathlib import Path


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
            "No per-generation cross-trait data\n\n(requires frailty phenotype model)",
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
