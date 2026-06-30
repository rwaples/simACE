"""Heritability plots for the per-scenario atlas.

Groups realized A/C variance proportions, broad-sense
(``(Var(A)+Var(C))/Var(L)``), sex-stratified midparent-offspring, and
observed-scale (phi-Falconer with Dempster-Lerner lift) heritability plots.
"""

from __future__ import annotations

__all__ = [
    "plot_broad_heritability_by_generation",
    "plot_ge_covariance_by_generation",
    "plot_heritability_by_generation",
    "plot_heritability_by_sex_generation",
    "plot_observed_heritability",
    "plot_snp_like_heritability_by_generation",
]

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from simace.plotting.plot_style import (
    COLOR_EXPECTED,
    COLOR_FEMALE,
    COLOR_MALE,
    COLOR_OBSERVED,
    COLOR_UNAFFECTED,
    apply_nature_style,
    enable_value_gridlines,
)
from simace.plotting.plot_utils import finalize_plot, save_placeholder_plot

logger = logging.getLogger(__name__)


def _component_expected_values(value: Any, generations: list[int]) -> np.ndarray | None:
    """Return configured component values aligned to generation labels."""
    if value is None:
        return None
    if isinstance(value, dict):
        vals = []
        for gen in generations:
            raw = value.get(gen, value.get(str(gen), np.nan))
            vals.append(float(raw) if raw is not None else np.nan)
        arr = np.array(vals, dtype=float)
        return arr if np.isfinite(arr).any() else None
    return np.full(len(generations), float(value), dtype=float)


def _finite_float(value: Any) -> float:
    """Return value as float, or NaN for missing/non-numeric values."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _generation_keys(per_gen_all: list[dict[str, Any]]) -> list[str]:
    """Return sorted generation keys from the first non-empty replicate."""
    first = next((pg for pg in per_gen_all if pg), None)
    if not first:
        return []
    return sorted(first.keys(), key=lambda k: int(k.split("_")[1]))


def _derive_ge_h2_metrics(
    var_a: Any,
    var_liability: Any,
    cov_a_non_genetic: Any,
    n: Any = None,
) -> dict[str, float]:
    """Derive GE-covariance and SNP-like h² plotting metrics.

    Inputs are per-generation primitives computed with population denominators
    (ddof=0). ``var_non_genetic`` is derived from
    ``Var(L) = Var(A) + Var(C+E) + 2 Cov(A,C+E)`` so the plots only require the
    stored primitives.
    """
    a = _finite_float(var_a)
    var_l = _finite_float(var_liability)
    cov = _finite_float(cov_a_non_genetic)
    n_float = _finite_float(n)

    finite_core = np.isfinite([a, var_l, cov]).all()
    if not finite_core:
        return {
            "ge_cov_fraction": float("nan"),
            "h2_realized_A": float("nan"),
            "h2_snp_like": float("nan"),
            "var_non_genetic": float("nan"),
            "null_sd": float("nan"),
        }

    var_u = var_l - a - 2.0 * cov
    if var_u < 0 and np.isclose(var_u, 0.0, atol=1e-12):
        var_u = 0.0

    ge_cov_fraction = 2.0 * cov / var_l if var_l != 0 else float("nan")
    h2_realized = a / var_l if var_l != 0 else float("nan")
    h2_snp_like = ((a + cov) ** 2) / (a * var_l) if a > 0 and var_l > 0 else float("nan")

    null_sd = float("nan")
    if n_float > 1 and a >= 0 and var_u >= 0 and var_l > 0:
        null_sd = abs(2.0 * np.sqrt(a * var_u) / var_l) / np.sqrt(n_float - 1.0)

    return {
        "ge_cov_fraction": float(ge_cov_fraction),
        "h2_realized_A": float(h2_realized),
        "h2_snp_like": float(h2_snp_like),
        "var_non_genetic": float(var_u),
        "null_sd": float(null_sd),
    }


def _metric_array(
    per_gen_all: list[dict[str, Any]],
    gen_keys: list[str],
    trait_num: int,
    metric: str,
) -> np.ndarray:
    """Build a replicate × generation array for a derived GE/h² metric."""
    rows = []
    for pg in per_gen_all:
        row = []
        for gk in gen_keys:
            gs = pg.get(gk, {})
            derived = _derive_ge_h2_metrics(
                gs.get(f"A{trait_num}_var"),
                gs.get(f"liability{trait_num}_variance"),
                gs.get(f"A{trait_num}_cov_non_genetic"),
                gs.get("n"),
            )
            row.append(derived[metric])
        rows.append(row)
    return np.array(rows, dtype=float)


def _plot_replicate_points(
    ax: plt.Axes,
    x: np.ndarray,
    values_by_rep: np.ndarray,
    *,
    color: str,
    offset: float = 0.0,
    jitter_width: float = 0.025,
    alpha: float = 0.8,
) -> None:
    """Scatter finite per-replicate generation values with deterministic jitter."""
    for rep_idx in range(values_by_rep.shape[0]):
        values = values_by_rep[rep_idx]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        jitter = np.random.default_rng(42 + rep_idx).uniform(-jitter_width, jitter_width, len(x))
        ax.scatter(
            x[finite] + offset + jitter[finite],
            values[finite],
            color=color,
            alpha=alpha,
            s=25,
            zorder=5,
        )


def _plot_mean_line(ax: plt.Axes, x: np.ndarray, values_by_rep: np.ndarray, *, color: str, label: str) -> None:
    """Plot a replicate mean line when at least one finite value exists."""
    if not np.isfinite(values_by_rep).any():
        return
    mean_values = np.nanmean(values_by_rep, axis=0)
    finite_mean = np.isfinite(mean_values)
    if finite_mean.any():
        ax.plot(x[finite_mean], mean_values[finite_mean], color=color, linewidth=1.4, label=label)


def _finite_values(*arrays: np.ndarray) -> np.ndarray:
    """Flatten finite values from one or more arrays into one vector."""
    finite_parts = [arr[np.isfinite(arr)] for arr in arrays]
    non_empty = [part for part in finite_parts if part.size]
    return np.concatenate(non_empty) if non_empty else np.array([], dtype=float)


def _add_legend_if_labeled(ax: plt.Axes) -> None:
    """Add a legend only when at least one labeled artist exists."""
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)


def plot_heritability_by_generation(
    all_views: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot realized A and C variance proportions per generation.

    The A series is the narrow-sense heritability h². The C series is shown
    separately rather than folded into a broad-sense ``A + C`` panel.
    """
    per_gen_all = [v.get("per_generation", {}) for v in all_views]
    if not per_gen_all or not per_gen_all[0]:
        save_placeholder_plot(output_path, "No per-generation data")
        return

    gen_keys = sorted(per_gen_all[0].keys(), key=lambda k: int(k.split("_")[1]))
    generations = [int(k.split("_")[1]) for k in gen_keys]
    x = np.array(generations, dtype=float)

    params = all_views[0].get("parameters", {})
    component_specs = (
        ("A", "Additive genetic (A)", COLOR_OBSERVED, -0.06),
        ("C", "Common environment (C)", COLOR_EXPECTED, 0.06),
    )

    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, trait_num in enumerate([1, 2]):
        ax = axes[col]
        e_key = f"E{trait_num}_var"

        for component, label, color, offset in component_specs:
            component_key = f"{component}{trait_num}_var"
            component_per_rep = []
            for pg in per_gen_all:
                rep_values = []
                for gk in gen_keys:
                    gs = pg.get(gk, {})
                    a_var = gs.get(f"A{trait_num}_var", 0)
                    c_var = gs.get(f"C{trait_num}_var", 0)
                    e_var = gs.get(e_key, 0)
                    total = a_var + c_var + e_var
                    rep_values.append(gs.get(component_key, 0) / total if total > 0 else np.nan)
                component_per_rep.append(rep_values)

            component_arr = np.array(component_per_rep, dtype=float)
            for rep_idx in range(component_arr.shape[0]):
                values = component_arr[rep_idx]
                finite = np.isfinite(values)
                if not finite.any():
                    continue
                jitter = np.random.default_rng(42 + rep_idx).uniform(-0.025, 0.025, len(generations))
                ax.scatter(
                    x[finite] + offset + jitter[finite],
                    values[finite],
                    color=color,
                    alpha=0.8,
                    s=25,
                    zorder=5,
                )

            if np.isfinite(component_arr).any():
                mean_values = np.nanmean(component_arr, axis=0)
                finite_mean = np.isfinite(mean_values)
                if finite_mean.any():
                    ax.plot(x[finite_mean], mean_values[finite_mean], color=color, linewidth=1.4, label=f"Mean {label}")

            expected = _component_expected_values(params.get(f"{component}{trait_num}"), generations)
            if expected is not None:
                finite_expected = np.isfinite(expected)
                if finite_expected.any():
                    ax.plot(
                        x[finite_expected],
                        expected[finite_expected],
                        color=color,
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.7,
                        label=f"Configured {component}{trait_num}",
                    )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Variance proportion / Var(L)")
        ax.set_title(f"Trait {trait_num}")
        ax.set_xticks(generations)
        ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=8)

        enable_value_gridlines(ax)

    finalize_plot(output_path, scenario=scenario)


def plot_ge_covariance_by_generation(
    all_views: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot realized 2 Cov(A, C+E) / Var(L) by generation.

    Under the current simACE model, C and E are drawn independently of A each
    generation, so this diagnostic should fluctuate around zero on the
    recorded pedigree.
    """
    per_gen_all = [v.get("per_generation", {}) for v in all_views]
    gen_keys = _generation_keys(per_gen_all)
    if not gen_keys:
        save_placeholder_plot(output_path, "No per-generation data")
        return

    generations = [int(k.split("_")[1]) for k in gen_keys]
    x = np.array(generations, dtype=float)

    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, trait_num in enumerate([1, 2]):
        ax = axes[col]
        ge_arr = _metric_array(per_gen_all, gen_keys, trait_num, "ge_cov_fraction")
        null_sd_arr = _metric_array(per_gen_all, gen_keys, trait_num, "null_sd")

        _plot_replicate_points(ax, x, ge_arr, color=COLOR_OBSERVED)
        _plot_mean_line(ax, x, ge_arr, color=COLOR_OBSERVED, label="Mean")

        if np.isfinite(null_sd_arr).any():
            band = 1.96 * np.nanmean(null_sd_arr, axis=0)
            finite_band = np.isfinite(band)
            if finite_band.any():
                ax.fill_between(
                    x[finite_band],
                    -band[finite_band],
                    band[finite_band],
                    color=COLOR_EXPECTED,
                    alpha=0.18,
                    label="Approx. null ±1.96 SD",
                )

        ax.axhline(0.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("Generation")
        ax.set_ylabel("2 Cov(A, C+E) / Var(L)")
        ax.set_title(f"Trait {trait_num}")
        ax.set_xticks(generations)
        finite_values = _finite_values(ge_arr, 1.96 * null_sd_arr)
        ylim = max(0.05, float(np.nanmax(np.abs(finite_values))) * 1.15) if finite_values.size else 0.05
        ax.set_ylim(-ylim, ylim)
        _add_legend_if_labeled(ax)
        enable_value_gridlines(ax)

    finalize_plot(output_path, scenario=scenario)


def plot_snp_like_heritability_by_generation(
    all_views: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot realized Var(A)/Var(L) against the SNP-like h² target."""
    per_gen_all = [v.get("per_generation", {}) for v in all_views]
    gen_keys = _generation_keys(per_gen_all)
    if not gen_keys:
        save_placeholder_plot(output_path, "No per-generation data")
        return

    generations = [int(k.split("_")[1]) for k in gen_keys]
    x = np.array(generations, dtype=float)

    _fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for col, trait_num in enumerate([1, 2]):
        ax = axes[col]
        realized_arr = _metric_array(per_gen_all, gen_keys, trait_num, "h2_realized_A")
        snp_like_arr = _metric_array(per_gen_all, gen_keys, trait_num, "h2_snp_like")

        for label, arr, color, offset in (
            ("Var(A) / Var(L)", realized_arr, COLOR_OBSERVED, -0.04),
            ("SNP-like h² target", snp_like_arr, COLOR_EXPECTED, 0.04),
        ):
            _plot_replicate_points(ax, x, arr, color=color, offset=offset, jitter_width=0.02, alpha=0.75)
            _plot_mean_line(ax, x, arr, color=color, label=label)

        ax.set_xlabel("Generation")
        ax.set_ylabel("h²-like fraction")
        ax.set_title(f"Trait {trait_num}")
        ax.set_xticks(generations)
        finite_values = _finite_values(realized_arr, snp_like_arr)
        if finite_values.size and np.nanmax(finite_values) > 1.02:
            ax.set_ylim(0.0, float(np.nanmax(finite_values)) * 1.05)
        else:
            ax.set_ylim(0.0, 1.0)
        _add_legend_if_labeled(ax)
        enable_value_gridlines(ax)

    finalize_plot(output_path, scenario=scenario)


def plot_broad_heritability_by_generation(
    all_views: list[dict[str, Any]],
    output_path: str | Path,
    scenario: str = "",
) -> None:
    """Plot broad-sense heritability H² = (Var(A)+Var(C))/(Var(A)+Var(C)+Var(E)) per generation."""
    per_gen_all = [v.get("per_generation", {}) for v in all_views]
    if not per_gen_all or not per_gen_all[0]:
        save_placeholder_plot(output_path, "No per-generation data")
        return

    gen_keys = sorted(per_gen_all[0].keys(), key=lambda k: int(k.split("_")[1]))
    generations = [int(k.split("_")[1]) for k in gen_keys]

    params = all_views[0].get("parameters", {})
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
        all_stats: per-replicate stats report dicts. Must contain
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
    fig.text(
        0.5,
        0.5,
        "UNDER CONSTRUCTION",
        ha="center",
        va="center",
        rotation=25,
        fontsize=46,
        fontweight="bold",
        color="0.25",
        alpha=0.18,
        zorder=1000,
        transform=fig.transFigure,
    )
    finalize_plot(output_path, scenario=scenario)
