"""Time-varying E figures: per-generation components, cohort FS correlation and Falconer, prevalence drift."""

from __future__ import annotations

__all__ = [
    "compare_cohort_falconer",
    "compare_cohort_fs_correlations",
    "compare_components_by_generation",
    "compare_prevalence_drift",
]

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from simace.analysis.stats.naive_h2 import falconer_h2
from simace.analysis.stats.pedigree_estimates import load_pedigree_estimates, load_pedigree_estimates_per_generation
from simace.core.parquet import load_parquet
from simace.core.relationships import reference_corr
from simace.plotting.compare._draw import draw_envelope_line, padded_range, rep_envelope, save, scenario_color
from simace.plotting.compare.estimates import load_per_gen_prevalence
from simace.plotting.plot_style import apply_nature_style, enable_value_gridlines

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from simace.analysis.stats.pedigree_estimates import CohortEstimates
    from simace.plotting.compare._draw import Envelope
    from simace.plotting.compare._inputs import PrevalenceSeries, ScenarioReps
    from simace.simulation.params import SimulationParams


def compare_components_by_generation(
    scenarios: Sequence[ScenarioReps],
    output_path: Path,
    trait: int = 1,
    show_generations: tuple[int, ...] = (1, 5, 9),
    n_bins: int = 50,
) -> None:
    """Per-generation distributions of A and total liability, one column per scenario.

    Two rows × *n_scenarios* columns. Top row shows the additive genetic
    component A; bottom row shows total liability (A + C + E). Within each
    panel, distributions for *show_generations* are overlaid (lighter →
    darker shade for earlier → later gens) so the reader can see the
    cohort evolution of the per-individual distribution directly. The A
    panels should look essentially identical across both gens and
    scenarios (A is fixed by config); the liability panels widen,
    narrow, or stay put depending on the per-gen E schedule.

    Args:
        scenarios: one column each; each rep's recorded pedigree is read.
        output_path: image path to save.
        trait: 1 or 2.
        show_generations: which generations to overlay within each panel.
            Default (1, 5, 9) bookends + middle for a 10-gen pedigree.
        n_bins: histogram bin count, shared across panels for comparability.
    """
    n_scen = len(scenarios)
    apply_nature_style()
    a_key = f"A{trait}"
    l_key = f"liability{trait}"

    # Pool reps within each (scenario, generation) cell.
    per_scen_a: list[dict[int, np.ndarray]] = []
    per_scen_l: list[dict[int, np.ndarray]] = []
    for scenario in scenarios:
        a_by_gen: dict[int, list[np.ndarray]] = {g: [] for g in show_generations}
        l_by_gen: dict[int, list[np.ndarray]] = {g: [] for g in show_generations}
        for path in scenario.pedigrees:
            df = load_parquet(path, columns=["generation", a_key, l_key])
            gen_arr = df["generation"].to_numpy()
            a_arr = df[a_key].to_numpy()
            l_arr = df[l_key].to_numpy()
            for g in show_generations:
                gen_mask = gen_arr == g
                if not gen_mask.any():
                    continue
                a_by_gen[g].append(a_arr[gen_mask])
                l_by_gen[g].append(l_arr[gen_mask])
        per_scen_a.append({g: np.concatenate(v) if v else np.array([]) for g, v in a_by_gen.items()})
        per_scen_l.append({g: np.concatenate(v) if v else np.array([]) for g, v in l_by_gen.items()})

    # Shared x-range across both rows + every panel for cross-comparability.
    lo, hi = padded_range(np.concatenate([arr for d in per_scen_a + per_scen_l for arr in d.values() if arr.size]))
    bins = np.linspace(lo, hi, n_bins + 1)

    # Lighter → darker shade for earlier → later gens, using a perceptually
    # uniform colormap so the gen ordering is unambiguous in the legend.
    cmap = plt.get_cmap("viridis")
    n_g = len(show_generations)
    gen_colors = [cmap(0.15 + 0.7 * (i / max(1, n_g - 1))) for i in range(n_g)]

    fig, axes = plt.subplots(2, n_scen, figsize=(3.2 * n_scen, 5.5), sharex=True, sharey="row")
    if n_scen == 1:
        axes = axes.reshape(2, 1)

    for col, (scenario, a_dict, l_dict) in enumerate(zip(scenarios, per_scen_a, per_scen_l, strict=True)):
        for row_idx, (ax, vals_dict) in enumerate(((axes[0, col], a_dict), (axes[1, col], l_dict))):
            for g_idx, g in enumerate(show_generations):
                vals = vals_dict.get(g, np.array([]))
                if vals.size == 0:
                    continue
                color = gen_colors[g_idx]
                sd = float(np.std(vals, ddof=1))
                ax.hist(
                    vals,
                    bins=bins,
                    density=True,
                    histtype="step",
                    color=color,
                    linewidth=1.6,
                    label=f"gen {g}  (sd = {sd:.3f})",
                )
            if row_idx == 0:
                ax.set_title(scenario.label, fontsize=10)
            enable_value_gridlines(ax)

        axes[0, col].legend(loc="upper left", fontsize=7, frameon=False)
        axes[1, col].legend(loc="upper left", fontsize=7, frameon=False)

    axes[0, 0].set_ylabel(f"Density — A{trait}")
    axes[1, 0].set_ylabel(f"Density — liability {trait}")
    for ax in axes[1, :]:
        ax.set_xlabel(f"Value (trait {trait})")

    fig.suptitle("Per-individual A and total liability by generation")
    save(fig, output_path, rect=(0, 0, 1, 0.96))


def _per_gen_envelope(
    per_rep_pergen: list[dict[int, CohortEstimates]],
    gens: list[int],
    get_val: Callable[[CohortEstimates], float],
) -> list[Envelope]:
    """One envelope per generation of a per-rep cohort value; reps without the generation are skipped."""
    return [rep_envelope(get_val(d[g]) for d in per_rep_pergen if g in d) for g in gens]


def compare_cohort_fs_correlations(
    scenarios: Sequence[ScenarioReps],
    output_path: Path,
    trait: int = 1,
    reference: SimulationParams | None = None,
    min_generation: int = 1,
) -> None:
    """Per-generation FS liability correlation, one line per scenario.

    Within each generation cohort, computes Pearson ``r`` on liability between
    full-sib pairs (both members in the same generation; twins already
    excluded by the extractor).  Plots ``r_FS(g)`` as a line per scenario
    with min/max envelope across replicates.

    Args:
        scenarios: the scenarios to overlay; each rep's recorded pedigree is read.
        output_path: image path to save.
        trait: 1 or 2.
        reference: parameters of the first scenario
            (:meth:`ScenarioReps.parameters`), for the FS reference line
            (:func:`~simace.core.relationships.reference_corr`). Omitted if
            ``None`` or when that scenario has no analytic FS value.
        min_generation: smallest generation to include (founders have no FS
            pairs by construction; default 1).
    """
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for scen_idx, scenario in enumerate(scenarios):
        per_rep = [load_pedigree_estimates_per_generation(p, trait=trait) for p in scenario.pedigrees]
        # Union of generations seen across reps, restricted to >= min_generation.
        gens = sorted({g for d in per_rep for g in d if g >= min_generation})
        envelopes = _per_gen_envelope(per_rep, gens, lambda cell: cell.r_fs)
        draw_envelope_line(ax, gens, envelopes, scenario_color(scen_idx), scenario.label)

    ref = (
        None
        if reference is None
        else reference_corr("FS", reference.assortment, reference.c_transmission, reference.vc, trait)
    )
    if ref is not None:
        ax.axhline(
            y=ref,
            linestyle="--",
            color="#888888",
            linewidth=1,
            alpha=0.8,
            label=f"expected FS r ({scenarios[0].label}) = {ref:.3f}",
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel(f"Within-cohort r(FS, liability {trait})")
    ax.set_title("Per-cohort full-sib liability correlation")
    enable_value_gridlines(ax)
    ax.legend(loc="best", fontsize=9, frameon=False)

    save(fig, output_path)


def compare_cohort_falconer(
    scenarios: Sequence[ScenarioReps],
    output_path: Path,
    trait: int = 1,
    min_generation: int = 1,
) -> None:
    """Per-generation realized h² vs per-gen Falconer vs pooled Falconer.

    Side-by-side panels (one per scenario, shared y-axis):
      - per-gen realized h² (variance ratio in the cohort)
      - per-gen Falconer = 2(r_MZ - r_FS) computed within the cohort
      - horizontal dashed line: pooled-across-gens Falconer (computed on
        the full pedigree).

    Args:
        scenarios: one panel each; each rep's recorded pedigree is read.
        output_path: image path to save.
        trait: 1 or 2.
        min_generation: smallest generation to include in per-gen lines
            (founders have no FS pairs).  Pooled estimate always uses every
            individual.
    """
    n_scen = len(scenarios)
    apply_nature_style()
    fig, axes = plt.subplots(1, n_scen, figsize=(3.4 * n_scen, 4.5), sharey=True)
    if n_scen == 1:
        axes = np.array([axes])

    for scenario, ax in zip(scenarios, axes, strict=True):
        per_rep_pergen = [load_pedigree_estimates_per_generation(p, trait=trait) for p in scenario.pedigrees]
        per_rep_pooled = [load_pedigree_estimates(p, trait=trait, min_generation=None) for p in scenario.pedigrees]
        gens = sorted({g for d in per_rep_pergen for g in d if g >= min_generation})

        truth = _per_gen_envelope(per_rep_pergen, gens, lambda d: d.realized_h2)
        falconer = _per_gen_envelope(per_rep_pergen, gens, lambda d: falconer_h2(d.r_mz, d.r_fs))
        draw_envelope_line(ax, gens, truth, scenario_color(0), "Realized h²")
        draw_envelope_line(ax, gens, falconer, scenario_color(1), "Per-cohort Falconer", marker="s")

        pooled_mean = rep_envelope(falconer_h2(est.r["MZ"], est.r["FS"]) for est in per_rep_pooled).mean
        if np.isfinite(pooled_mean):
            ax.axhline(
                y=pooled_mean,
                linestyle="--",
                color="#444444",
                linewidth=1.2,
                alpha=0.9,
                label=f"Pooled Falconer = {pooled_mean:.3f}",
            )

        ax.set_title(scenario.label)
        ax.set_xlabel("Generation")
        enable_value_gridlines(ax)
        ax.legend(loc="best", fontsize=8, frameon=False)

    axes[0].set_ylabel(f"h² (trait {trait})")
    fig.suptitle("Per-cohort vs pooled-across-gens naive Falconer")
    save(fig, output_path, rect=(0, 0, 1, 0.96))


def compare_prevalence_drift(
    series: Sequence[PrevalenceSeries],
    output_path: Path,
    trait: int = 1,
    target_prevalence: float | None = None,
) -> None:
    """Per-generation observed prevalence, one panel per scenario and one line per variant.

    Built for comparing ``standardize`` modes across E trajectories: each
    series is one mode (e.g. ``standardize=global``, ``none``,
    ``per_generation``) run over every trajectory (e.g. e_flat /
    e_rise_mild / e_rise_steep / e_fall_steep).

    Args:
        series: the lines of every panel, in legend order; every series
            runs the same scenarios, by label and in panel order. Each rep's
            ``report.yaml`` is read.
        output_path: image path to save.
        trait: 1 or 2.
        target_prevalence: input K to draw as dashed reference (typically
            0.1).  Omitted if ``None``.

    Raises:
        ValueError: There is no series, or the series run different scenarios.
    """
    if not series:
        raise ValueError("compare_prevalence_drift needs at least one series")
    panel_labels = [s.label for s in series[0].scenarios]
    for line in series[1:]:
        if [s.label for s in line.scenarios] != panel_labels:
            raise ValueError(
                f"series {line.label!r} runs {[s.label for s in line.scenarios]}, "
                f"but {series[0].label!r} runs {panel_labels}"
            )

    n_traj = len(panel_labels)
    apply_nature_style()
    fig, axes = plt.subplots(1, n_traj, figsize=(3.4 * n_traj, 4.5), sharey=True)
    if n_traj == 1:
        axes = np.array([axes])

    for traj_idx, (ax, traj_label) in enumerate(zip(axes, panel_labels, strict=True)):
        for line_idx, line in enumerate(series):
            per_gen = load_per_gen_prevalence(line.scenarios[traj_idx].reports, trait=trait)
            gens = sorted(per_gen.keys())
            draw_envelope_line(ax, gens, [rep_envelope(per_gen[g]) for g in gens], scenario_color(line_idx), line.label)

        if target_prevalence is not None:
            ax.axhline(
                y=target_prevalence,
                linestyle="--",
                color="#888888",
                linewidth=1,
                alpha=0.8,
            )

        ax.set_title(traj_label)
        ax.set_xlabel("Generation")
        enable_value_gridlines(ax)
        ax.legend(loc="best", fontsize=8, frameon=False)

    axes[0].set_ylabel(f"Observed prevalence (trait {trait})")
    fig.suptitle("Per-generation prevalence: " + " vs ".join(line.label for line in series))
    save(fig, output_path, rect=(0, 0, 1, 0.96))
