"""Heritability plots for the per-scenario atlas.

Groups the narrow-sense (``Var(A)/Var(L)``), broad-sense
(``(Var(A)+Var(C))/Var(L)``), sex-stratified midparent-offspring, and
observed-scale (phi-Falconer with Dempster-Lerner lift) heritability pages.
"""

from __future__ import annotations

__all__ = [
    "plot_broad_heritability_by_generation",
    "plot_heritability_by_generation",
    "plot_heritability_by_sex_generation",
    "plot_observed_heritability",
]

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from simace.plotting.plot_style import (
    COLOR_FEMALE,
    COLOR_MALE,
    COLOR_OBSERVED,
    COLOR_UNAFFECTED,
    apply_nature_style,
    enable_value_gridlines,
)
from simace.plotting.plot_utils import finalize_plot, save_placeholder_plot

logger = logging.getLogger(__name__)


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

    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))

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
                color=COLOR_OBSERVED,
                alpha=0.9,
                s=25,
                zorder=5,
            )

        # Expected heritability reference line
        exp = expected_h2.get(trait_num)
        if exp is not None:
            ax.axhline(
                y=exp,
                color=COLOR_UNAFFECTED,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"Parametric A{trait_num} = {exp}",
            )
            ax.legend(loc="lower left", fontsize=9)

        ax.set_xlabel("Generation")
        ax.set_ylabel("h² = Var(A) / Var(L)")
        ax.set_title(f"Trait {trait_num}")
        ax.set_xticks(generations)
        ax.set_ylim(0, 1)

        enable_value_gridlines(ax)

    finalize_plot(output_path, scenario=scenario)


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

    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))

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
                color=COLOR_OBSERVED,
                alpha=0.9,
                s=25,
                zorder=5,
            )

        exp = expected_H2.get(trait_num)
        if exp is not None:
            ax.axhline(
                y=exp,
                color=COLOR_UNAFFECTED,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"Parametric A{trait_num}+C{trait_num} = {exp}",
            )
            ax.legend(loc="lower left", fontsize=9)

        ax.set_xlabel("Generation")
        ax.set_ylabel("(Var(A)+Var(C)) / Var(L)")
        ax.set_title(f"Trait {trait_num}")
        ax.set_xticks(generations)
        ax.set_ylim(0, 1)

        enable_value_gridlines(ax)

    finalize_plot(output_path, scenario=scenario)


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

    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))

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
                    color=COLOR_UNAFFECTED,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label=f"Parametric A{trait_num} = {exp}",
                )

        ax.set_xlabel("Generation")
        ax.set_ylabel("h² (PO regression slope)")
        ax.set_title(f"Trait {trait_num}")
        ax.set_xticks(generations)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower left", fontsize=9)

    for ax in axes:
        enable_value_gridlines(ax)

    finalize_plot(output_path, scenario=scenario)


# ---------------------------------------------------------------------------
# Observed-scale h² from binary affected status
# ---------------------------------------------------------------------------

_OBSERVED_ESTIMATOR_LABELS: tuple[tuple[str, str], ...] = (
    ("falconer", "Falconer\n2(r_MZ − r_FS)"),  # noqa: RUF001
    ("sibs", "Sibs\n2·r_FS"),
    ("po", "PO slope\n(binary)"),
    ("hs", "Half-sibs\n4·r̄_HS"),
    ("cousins", "Cousins\n8·r_1C"),
)


def _dempster_lerner_factor(K: float) -> float:
    """K(1−K) / z(K)² where z(K) = φ(Φ⁻¹(1−K)).

    Converts observed-scale h² to liability-scale h² under LTM.
    """
    K = float(np.clip(K, 1e-3, 1.0 - 1e-3))
    z = float(norm.pdf(norm.ppf(1.0 - K)))
    if z <= 0:
        return float("nan")
    return K * (1.0 - K) / (z * z)


def plot_observed_heritability(
    all_stats: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
    params: dict[str, Any] | None = None,
) -> None:
    """Observed-scale h² and its Dempster-Lerner liability-scale back-transform.

    2x2 grid: rows = traits, columns = scale (observed | D-L).  At each of five
    x-positions (Falconer, Sibs-only, Midparent PO on binary, Half-sibs,
    Cousins) per-rep dots are scattered with small jitter.

    - Left column (observed): ``h²_obs = 2(r_MZ − r_FS)`` etc., computed from
      Pearson r on the binary affected indicator (phi coefficient).  Dotted
      grey reference at ``A · z(K̄)² / (K̄·(1−K̄))`` marks the LTM expectation at
      mean observed prevalence K̄.
    - Right column (liability via D-L): each observed-scale estimate
      multiplied by ``K(1−K)/z(K)²`` per rep.  Fixed y-range ``(0, 1)``.  A
      small in-figure text annotation flags that the D-L correction assumes a
      threshold-normal (LTM) mapping from liability to affected status and is
      biased under non-threshold phenotype models (e.g. pure frailty).

    Args:
        all_stats: per-rep ``phenotype_stats.yaml`` dicts.  Must contain
            ``observed_h2_estimators`` and ``prevalence``.
        output_path: image path to save.
        scenario: scenario label (for subtitle).
        params: optional dict with ``A1``/``A2`` for the observed-scale LTM
            reference line.
    """
    # Aggregate per-rep data.
    per_trait: dict[int, dict[str, list[float]]] = {
        1: {k: [] for k, _ in _OBSERVED_ESTIMATOR_LABELS} | {"K": [], "dl": []},
        2: {k: [] for k, _ in _OBSERVED_ESTIMATOR_LABELS} | {"K": [], "dl": []},
    }

    for s in all_stats:
        est = s.get("observed_h2_estimators") or {}
        prev = s.get("prevalence") or {}
        for t in (1, 2):
            K = prev.get(f"trait{t}")
            if K is None or not (1e-3 <= float(K) <= 1.0 - 1e-3):
                continue
            dl = _dempster_lerner_factor(float(K))
            per_trait[t]["K"].append(float(K))
            per_trait[t]["dl"].append(dl)
            trait_est = est.get(f"trait{t}") or {}
            for est_key, _label in _OBSERVED_ESTIMATOR_LABELS:
                v = trait_est.get(est_key)
                per_trait[t][est_key].append(float(v) if v is not None else float("nan"))

    # Placeholder if no rep had any usable estimator across either trait.
    def _all_nan(trait_dict: dict[str, list[float]]) -> bool:
        return all(all(not np.isfinite(x) for x in trait_dict[est_key]) for est_key, _ in _OBSERVED_ESTIMATOR_LABELS)

    if not per_trait[1]["K"] and not per_trait[2]["K"]:
        save_placeholder_plot(output_path, "Observed h² not computable (K out of range or pair counts too small)")
        return
    if _all_nan(per_trait[1]) and _all_nan(per_trait[2]):
        save_placeholder_plot(output_path, "Observed h² not computable (K out of range or pair counts too small)")
        return

    apply_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)

    x_positions = np.arange(len(_OBSERVED_ESTIMATOR_LABELS), dtype=float)
    est_keys = [k for k, _ in _OBSERVED_ESTIMATOR_LABELS]
    tick_labels = [lab for _, lab in _OBSERVED_ESTIMATOR_LABELS]

    for row, trait_num in enumerate([1, 2]):
        trait_data = per_trait[trait_num]
        K_vals = trait_data["K"]
        dl_vals = trait_data["dl"]
        ax_obs = axes[row, 0]
        ax_dl = axes[row, 1]
        n_reps = len(K_vals)

        # Per-rep dots on each column.
        for rep_idx in range(n_reps):
            jitter = np.random.default_rng(42 + rep_idx).uniform(-0.12, 0.12, len(est_keys))
            obs_vals = np.array([trait_data[k][rep_idx] for k in est_keys], dtype=float)
            ax_obs.scatter(
                x_positions + jitter,
                obs_vals,
                color=COLOR_OBSERVED,
                alpha=0.9,
                s=25,
                zorder=5,
            )
            dl_scaled = obs_vals * dl_vals[rep_idx]
            ax_dl.scatter(
                x_positions + jitter,
                dl_scaled,
                color=COLOR_OBSERVED,
                alpha=0.9,
                s=25,
                zorder=5,
            )

        # Observed-column reference: A * mean dl_inv
        if params is not None and K_vals:
            A = params.get(f"A{trait_num}")
            if A is not None:
                K_bar = float(np.mean(K_vals))
                z_bar = float(norm.pdf(norm.ppf(1.0 - K_bar)))
                if z_bar > 0 and 0 < K_bar < 1:
                    expected_obs = float(A) * (z_bar * z_bar) / (K_bar * (1.0 - K_bar))
                    ax_obs.axhline(
                        y=expected_obs,
                        color=COLOR_UNAFFECTED,
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.9,
                        label=f"LTM expected at K̄={K_bar:.3f}: {expected_obs:.3f}",
                    )
                    ax_obs.legend(loc="best", fontsize=8, frameon=False)

        # Axis cosmetics.
        for ax, title, ylabel in (
            (ax_obs, f"Trait {trait_num} — observed scale", "h² (observed)"),
            (ax_dl, f"Trait {trait_num} — liability scale (D-L)", "h² (liability)"),
        ):
            ax.set_xticks(x_positions)
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            enable_value_gridlines(ax)
        ax_dl.set_ylim(0.0, 1.0)

        # Mean-K annotation (top-right) per row.
        if K_vals:
            ax_obs.text(
                0.98,
                0.97,
                f"K̄ = {np.mean(K_vals):.3f}",
                transform=ax_obs.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, linewidth=0),
            )

    # D-L column caveat annotation (figure-level, above top-right axes).
    axes[0, 1].text(
        0.5,
        1.12,
        "D-L assumes LTM (see caption)",
        transform=axes[0, 1].transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        style="italic",
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    finalize_plot(output_path, scenario=scenario)
