"""Assortative-mating additive-variance equilibrium plot for the atlas.

Shows the simulated additive genetic variance ``Var(A)`` per generation against
the infinitesimal-model assortative-mating recursion and its equilibrium ``a²``
(see :mod:`simace.simulation.am_equilibrium`). Under phenotypic assortative
mating ``Var(A)`` inflates across generations to the Bulmer equilibrium, which
equals the assortative-mating-only ``a²`` of Herzig et al. (2026, *Theor. Popul.
Biol.* 170:26–35, doi:10.1016/j.tpb.2026.06.003).
"""

from __future__ import annotations

__all__ = ["plot_am_equilibrium"]

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simace.plotting.plot_style import (
    COLOR_EXPECTED,
    COLOR_OBSERVED,
    COLOR_TRUE,
    COLOR_UNAFFECTED,
    enable_value_gridlines,
)
from simace.plotting.plot_utils import finalize_plot, save_placeholder_plot
from simace.simulation.am_equilibrium import am_equilibrium_variance, am_variance_trajectory

logger = logging.getLogger(__name__)

_CITATION = "Equilibrium theory: Herzig et al. (2026) Theor. Popul. Biol. 170:26-35, doi:10.1016/j.tpb.2026.06.003"


def plot_am_equilibrium(
    all_views: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot per-generation ``Var(A)`` vs the AM recursion and equilibrium ``a²``.

    1×2 figure (one panel per trait). Per-replicate observed ``Var(A)`` dots and
    their mean line are overlaid with the deterministic AM variance recursion
    (orange dashed) and the closed-form equilibrium ``a²`` (grey dotted). When a
    trait has no assortative mating, only the flat configured ``A`` reference is
    shown. Per-generation (dict-valued) AM/C/E suppresses the theory overlay
    (the equilibrium is then ill-defined).

    Args:
        all_views: Per-replicate report views, each with ``per_generation`` and
            ``parameters`` (as produced by ``plotting_report_view``).
        output_path: Image path to write.
        scenario: Scenario label for the figure subtitle.
    """
    per_gen_all = [v.get("per_generation", {}) for v in all_views]
    if not per_gen_all or not per_gen_all[0]:
        save_placeholder_plot(output_path, "No per-generation data")
        return

    sim = all_views[0].get("sim")
    if sim is not None and sim.mating_model != "standard":
        save_placeholder_plot(output_path, "No assortative mating (Wright-Fisher model)")
        return

    if sim is None or not sim.assortment.any_active:
        save_placeholder_plot(output_path, "No assortative mating configured")
        return
    schedule = sim.assortment.schedule

    gen_keys = sorted(per_gen_all[0].keys(), key=lambda k: int(k.split("_")[1]))
    generations = [int(k.split("_")[1]) for k in gen_keys]
    x = np.array(generations, dtype=float)

    # Map each recorded generation label g (1..G_ped) to its reproduce-step
    # count from founders: g + burnin. The final recorded generation therefore
    # sits at exactly G_sim steps.
    g_sim, burnin = sim.span.G_sim, sim.span.burnin

    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, t in enumerate([1, 2]):
        ax = axes[col]
        a_key = f"A{t}_var"

        # Per-replicate observed Var(A_t) trajectories.
        obs_per_rep = [[pg.get(gk, {}).get(a_key, np.nan) for gk in gen_keys] for pg in per_gen_all]
        obs_arr = np.array(obs_per_rep, dtype=float)

        for rep_idx in range(obs_arr.shape[0]):
            values = obs_arr[rep_idx]
            finite = np.isfinite(values)
            if not finite.any():
                continue
            jitter = np.random.default_rng(42 + rep_idx).uniform(-0.05, 0.05, len(generations))
            ax.scatter(x[finite] + jitter[finite], values[finite], color=COLOR_OBSERVED, alpha=0.8, s=25, zorder=5)

        if np.isfinite(obs_arr).any():
            mean_values = np.nanmean(obs_arr, axis=0)
            finite_mean = np.isfinite(mean_values)
            if finite_mean.any():
                ax.plot(
                    x[finite_mean],
                    mean_values[finite_mean],
                    color=COLOR_OBSERVED,
                    linewidth=1.4,
                    label="Observed Var(A) (mean)",
                )

        r_ho = schedule[t - 1][0]
        A_base = float(sim.vc.A[t - 1])
        per_gen_env = not sim.vc.scalar_E(t) or not sim.assortment.constant(t)

        if r_ho and A_base > 0 and not per_gen_env:
            V_env = float(sim.vc.C[t - 1]) + sim.vc.E_per_gen(t, 1)[0]
            traj = am_variance_trajectory(A_base, V_env, r_ho, g_sim)
            steps = np.array([g + burnin for g in generations], dtype=int)
            predicted = traj[steps]
            ax.plot(
                x,
                predicted,
                color=COLOR_EXPECTED,
                linestyle="--",
                linewidth=1.3,
                label="AM recursion (theory)",
            )
            a2 = am_equilibrium_variance(A_base, V_env, r_ho)
            if np.isfinite(a2):
                ax.axhline(
                    y=a2,
                    color=COLOR_TRUE,
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.9,
                    label=f"Equilibrium a² = {a2:.3f}",
                )
            ax.axhline(y=A_base, color=COLOR_UNAFFECTED, linestyle="-", linewidth=0.8, alpha=0.5)
            ax.set_title(f"Trait {t}  (AM r={r_ho:g})")
        else:
            if A_base > 0:
                ax.axhline(
                    y=A_base,
                    color=COLOR_UNAFFECTED,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label=f"Configured A{t} = {A_base:g}",
                )
            note = "per-gen params: theory overlay omitted" if per_gen_env else "no assortative mating"
            ax.set_title(f"Trait {t}  ({note})")

        ax.set_xlabel("Generation")
        ax.set_ylabel("Var(A)")
        ax.set_xticks(generations)
        ax.legend(loc="best", fontsize=8)
        enable_value_gridlines(ax)

    _fig.text(0.01, 0.005, _CITATION, fontsize=7, color="0.5", ha="left", va="bottom")
    finalize_plot(output_path, scenario=scenario)
