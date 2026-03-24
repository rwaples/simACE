"""EPIMIGHT heritability bias analysis plots.

Reads the consolidated epimight_bias_summary.tsv and produces diagnostic
plots quantifying how prevalence, censoring, shared environment, and
phenotype model affect EPIMIGHT h² estimates.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm

from sim_ace import setup_logging
from sim_ace.plot_atlas import assemble_atlas

logger = logging.getLogger(__name__)

# Consistent kind ordering and colours
KIND_ORDER = ["PO", "FS", "HS", "mHS", "pHS", "Av", "1G", "1C"]
KIND_COLORS = {k: f"C{i}" for i, k in enumerate(KIND_ORDER)}

CENSOR_ORDER = ["none", "death_only", "window_only", "both"]
CENSOR_LABELS = {
    "none": "No censoring",
    "death_only": "Death only",
    "window_only": "Window only",
    "both": "Death + Window",
}

EPIMIGHT_BIAS_CAPTIONS: dict[str, str] = {
    "epimight_bias_vs_prevalence": (
        "EPIMIGHT h² vs prevalence by censoring level\n"
        "Each line is a relationship kind. Dashed grey = liability h² (0.50). "
        "Attenuation increases with prevalence due to c2 cohort dilution."
    ),
    "epimight_bias_by_censoring": (
        "Attenuation ratio by censoring level\n"
        "EPIMIGHT h² / liability h² grouped by censoring at each prevalence. "
        "Values near 1.0 = less bias. "
        "Censoring adds only minor additional downward bias beyond the prevalence effect."
    ),
    "epimight_bias_heatmap": (
        "Relative bias heatmap\n"
        "(EPIMIGHT - true) / true across all scenario x kind combinations. "
        "Deeper blue = stronger downward bias. "
        "Scenarios ordered by prevalence (columns) and kind (rows)."
    ),
    "epimight_c_effect": (
        "Effect of shared environment (C)\n"
        "Paired bars: C=0 vs C=0.2 at each prevalence, per kind. "
        "FS and HS estimates are inflated by C because siblings share environment. "
        "PO, pHS, and more distant kinds are unaffected."
    ),
    "epimight_model_comparison": (
        "Phenotype model comparison (K=0.10, C=0, no censoring)\n"
        "adult_ltm (deterministic threshold) vs weibull (stochastic frailty) "
        "vs cure_frailty (hybrid) vs adult_cox (proportional hazards). "
        "Stochastic models show severe additional attenuation."
    ),
    "epimight_forest": (
        "Forest plot: FS kind across all 28 scenarios\n"
        "Point estimates with 95% CIs. "
        "Dashed line = liability h². Dotted line = LTM Falconer h²."
    ),
    "epimight_attenuation_summary": (
        "Attenuation ratio vs prevalence (no censoring, C=0)\n"
        "EPIMIGHT / liability h² stratified by relationship kind. "
        "Shows how the fraction of true h² recovered by EPIMIGHT "
        "decreases with disease prevalence. PO is most robust; 1C is most attenuated."
    ),
    "epimight_dilution_ratio": (
        "c2 cohort dilution: mechanism of prevalence bias\n"
        "Left: analytical dilution ratio (fraction of ideal genetic enrichment preserved "
        "in the c2 cohort) vs prevalence, by kind. Dots = empirical values from simulation. "
        "Kinds with more relatives (1C, N=13) dilute fastest. "
        "Right: fraction of the population in the c2 cohort. "
        "When c2 approaches 100%, having an affected relative is uninformative."
    ),
    "epimight_corrected_h2": (
        "Raw vs dilution-corrected EPIMIGHT h² (no censoring, C=0)\n"
        "Left: raw estimates showing prevalence-dependent attenuation. "
        "Right: h² after analytical dilution correction for c2 cohort dilution "
        "(at high prevalence, most people have affected relatives, "
        "diluting the genetic enrichment). "
        "1C at K=0.40 is NA (dilution < 5%, correction unreliable)."
    ),
}


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


def _log_pct_xaxis(ax: plt.Axes) -> None:
    """Set log-scale x-axis with percentage tick labels at tested prevalences."""
    ax.set_xscale("log")
    ticks = [0.01, 0.05, 0.10, 0.20, 0.40]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x * 100:g}%"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


# ---------------------------------------------------------------------------
# 1. Bias vs prevalence (main figure)
# ---------------------------------------------------------------------------


def plot_epimight_bias_vs_prevalence(df: pd.DataFrame, output_path: Path) -> None:
    """h² vs prevalence, one panel per censoring level, lines by kind."""
    nocensor = df[(df["C"] == 0) & (df["phenotype_model"] == "adult_ltm")]
    censors = [c for c in CENSOR_ORDER if c in nocensor["censor_label"].values]
    n_panels = len(censors)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax, cl in zip(axes, censors):
        sub = nocensor[nocensor["censor_label"] == cl]
        for kind in KIND_ORDER:
            ks = sub[sub["kind"] == kind].sort_values("prevalence")
            if ks.empty:
                continue
            ax.fill_between(
                ks["prevalence"], ks["h2_l95"], ks["h2_u95"],
                color=KIND_COLORS[kind], alpha=0.15,
            )
            ax.plot(ks["prevalence"], ks["h2_epimight"], "o-", color=KIND_COLORS[kind], label=kind, markersize=4)

        # Liability reference
        h2_true = sub["h2_true_liability"].iloc[0] if not sub.empty else 0.5
        ax.axhline(h2_true, ls="--", color="0.4", linewidth=1, label="Liability h²")

        _log_pct_xaxis(ax)
        ax.set_xlabel("Prevalence")
        ax.set_ylabel("h² estimate")
        ax.set_title(CENSOR_LABELS.get(cl, cl))
        ax.set_ylim(bottom=-0.05)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("EPIMIGHT h² vs Prevalence by Censoring Level", fontsize=12)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Bias by censoring
# ---------------------------------------------------------------------------


def plot_epimight_bias_by_censoring(df: pd.DataFrame, output_path: Path) -> None:
    """Attenuation ratio grouped by censoring at each prevalence."""
    sub = df[(df["C"] == 0) & (df["phenotype_model"] == "adult_ltm")]
    prevs = sorted(sub["prevalence"].dropna().unique())
    n_panels = len(prevs)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, prev in zip(axes, prevs):
        psub = sub[sub["prevalence"] == prev]
        censors_present = [c for c in CENSOR_ORDER if c in psub["censor_label"].values]
        x = np.arange(len(censors_present))
        width = 0.8 / len(KIND_ORDER)

        for i, kind in enumerate(KIND_ORDER):
            ksub = psub[psub["kind"] == kind]
            vals = [ksub.loc[ksub["censor_label"] == c, "attenuation_ratio"].values for c in censors_present]
            vals = [v[0] if len(v) > 0 else np.nan for v in vals]
            ax.bar(x + i * width, vals, width, color=KIND_COLORS[kind], label=kind if prev == prevs[0] else "")

        ax.set_xticks(x + width * len(KIND_ORDER) / 2)
        ax.set_xticklabels([CENSOR_LABELS.get(c, c) for c in censors_present], fontsize=7, rotation=30, ha="right")
        ax.set_title(f"K = {prev}")
        ax.axhline(1.0, ls="--", color="0.4", linewidth=0.8)
        ax.set_ylabel("Attenuation ratio" if prev == prevs[0] else "")

    axes[0].legend(fontsize=7, ncol=2, loc="upper left")
    fig.suptitle("Attenuation Ratio by Censoring Level", fontsize=12)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Bias heatmap
# ---------------------------------------------------------------------------


def plot_epimight_bias_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap of relative bias: kinds × scenarios."""
    sub = df[(df["C"] == 0) & (df["phenotype_model"] == "adult_ltm")].copy()
    if sub.empty:
        return

    sub["label"] = sub.apply(lambda r: f"K={r['prevalence']:.2f}\n{r['censor_label']}", axis=1)
    # Order: by prevalence then censor
    sub["_sort"] = sub["prevalence"] * 10 + sub["censor_label"].map({c: i for i, c in enumerate(CENSOR_ORDER)}).fillna(
        0
    )
    sub = sub.sort_values("_sort")

    pivot = sub.pivot_table(index="kind", columns="label", values="rel_bias_liability", aggfunc="first")
    # Reorder kinds
    pivot = pivot.reindex([k for k in KIND_ORDER if k in pivot.index])
    # Reorder columns by sorted labels
    col_order = sub.drop_duplicates("label").sort_values("_sort")["label"].tolist()
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.7), 5))
    vlim = max(abs(np.nanmin(pivot.values)), abs(np.nanmax(pivot.values)), 0.5)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v < -0.5 else "black")

    fig.colorbar(im, ax=ax, label="Relative bias")
    ax.set_title("EPIMIGHT Relative Bias (vs Liability h²)")
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 4. C effect
# ---------------------------------------------------------------------------


def plot_epimight_c_effect(df: pd.DataFrame, output_path: Path) -> None:
    """Paired comparison: C=0 vs C=0.2 at each prevalence."""
    nocensor = df[(df["censor_label"] == "none") & (df["phenotype_model"] == "adult_ltm")]
    c0 = nocensor[nocensor["C"] == 0]
    c02 = nocensor[nocensor["C"] == 0.2]

    if c0.empty or c02.empty:
        return

    prevs = sorted(c0["prevalence"].dropna().unique())
    n_panels = len(prevs)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, prev in zip(axes, prevs):
        x = np.arange(len(KIND_ORDER))
        width = 0.35

        vals_c0 = []
        vals_c02 = []
        for kind in KIND_ORDER:
            v0 = c0.loc[(c0["prevalence"] == prev) & (c0["kind"] == kind), "h2_epimight"]
            v2 = c02.loc[(c02["prevalence"] == prev) & (c02["kind"] == kind), "h2_epimight"]
            vals_c0.append(v0.values[0] if len(v0) > 0 else np.nan)
            vals_c02.append(v2.values[0] if len(v2) > 0 else np.nan)

        ax.bar(x - width / 2, vals_c0, width, label="C = 0", color="steelblue")
        ax.bar(x + width / 2, vals_c02, width, label="C = 0.2", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(KIND_ORDER, fontsize=8)
        ax.set_title(f"K = {prev}")
        ax.set_ylabel("EPIMIGHT h²" if prev == prevs[0] else "")

        h2_true = c0.loc[c0["prevalence"] == prev, "h2_true_liability"]
        if not h2_true.empty:
            ax.axhline(h2_true.iloc[0], ls="--", color="0.4", linewidth=0.8)

    axes[0].legend(fontsize=8)
    fig.suptitle("Effect of Shared Environment (C) on EPIMIGHT h²", fontsize=12)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 5. Model comparison
# ---------------------------------------------------------------------------


def plot_epimight_model_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart comparing 4 phenotype models at baseline."""
    baseline = df[(df["prevalence"] == 0.10) & (df["C"] == 0) & (df["censor_label"] == "none")]
    models = baseline["phenotype_model"].unique()
    if len(models) == 0:
        return

    model_order = ["adult_ltm", "weibull", "cure_frailty", "adult_cox"]
    models = [m for m in model_order if m in models]
    model_colors = {"adult_ltm": "C0", "weibull": "C1", "cure_frailty": "C2", "adult_cox": "C3"}

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(KIND_ORDER))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        msub = baseline[baseline["phenotype_model"] == model]
        vals = []
        errs = []
        for kind in KIND_ORDER:
            ks = msub[msub["kind"] == kind]
            vals.append(ks["h2_epimight"].values[0] if len(ks) > 0 else np.nan)
            errs.append(ks["h2_se"].values[0] if len(ks) > 0 else 0)
        ax.bar(x + i * width, vals, width, yerr=errs, label=model, color=model_colors.get(model, f"C{i}"), capsize=2)

    h2_true = baseline["h2_true_liability"].iloc[0] if not baseline.empty else 0.5
    ax.axhline(h2_true, ls="--", color="0.4", linewidth=0.8, label="Liability h²")

    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels(KIND_ORDER)
    ax.set_ylabel("EPIMIGHT h²")
    ax.set_title("Model Comparison (K=0.10, C=0, No Censoring)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 6. Forest plot
# ---------------------------------------------------------------------------


def plot_epimight_forest(df: pd.DataFrame, output_path: Path, kind: str = "FS") -> None:
    """Forest plot for one kind across all scenarios."""
    sub = df[df["kind"] == kind].copy()
    if sub.empty:
        return

    sub = sub.sort_values(["phenotype_model", "C", "prevalence", "censor_label"])
    sub = sub.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(6, len(sub) * 0.3)))
    y = np.arange(len(sub))

    ax.errorbar(
        sub["h2_epimight"], y, xerr=1.96 * sub["h2_se"], fmt="o", color="C0", markersize=4, capsize=2, linewidth=1
    )

    # Reference lines
    if sub["h2_true_liability"].notna().any():
        ax.axvline(sub["h2_true_liability"].iloc[0], ls="--", color="0.4", label="Liability h²")

    ltm_median = sub["h2_ltm_falconer"].median()
    if not np.isnan(ltm_median):
        ax.axvline(ltm_median, ls=":", color="C1", label="LTM Falconer (median)")

    ax.set_yticks(y)
    ax.set_yticklabels(sub["scenario"], fontsize=6)
    ax.set_xlabel(f"EPIMIGHT h² ({kind})")
    ax.set_title(f"Forest Plot — {kind}")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 7. Attenuation summary
# ---------------------------------------------------------------------------


def plot_epimight_attenuation_summary(df: pd.DataFrame, output_path: Path) -> None:
    """Attenuation ratio vs prevalence, by kind, no-censoring C=0 only."""
    sub = df[(df["censor_label"] == "none") & (df["C"] == 0) & (df["phenotype_model"] == "adult_ltm")]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for kind in KIND_ORDER:
        ks = sub[sub["kind"] == kind].sort_values("prevalence")
        if ks.empty:
            continue
        ax.plot(ks["prevalence"], ks["attenuation_ratio"], "o-", color=KIND_COLORS[kind], label=kind, markersize=5)

    ax.axhline(1.0, ls="--", color="0.4", linewidth=0.8)
    _log_pct_xaxis(ax)
    ax.set_xlabel("Prevalence")
    ax.set_ylabel("Attenuation ratio (EPIMIGHT / liability h²)")
    ax.set_title("EPIMIGHT Attenuation vs Prevalence (No Censoring, C=0)")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(bottom=-0.05, top=1.15)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Dilution correction
# ---------------------------------------------------------------------------

# Kinship coefficients (f) for each EPIMIGHT kind
_KINSHIP = {
    "PO": 0.25,
    "FS": 0.25,
    "HS": 0.125,
    "mHS": 0.125,
    "pHS": 0.125,
    "1C": 0.0625,
    "Av": 0.125,
    "1G": 0.125,
}

# Mean number of relatives per kind (from default pedigree structure).
# In practice these should be read from the data; these are representative.
_DEFAULT_N_REL = {
    "PO": 2,
    "FS": 2,
    "HS": 3,
    "mHS": 2,
    "pHS": 2,
    "Av": 3,
    "1G": 4,
    "1C": 13,
}

# Minimum dilution ratio below which correction is unreliable
_MIN_DILUTION = 0.05


def _analytical_dilution(K: float, h2: float, kinship: float, n_rel: int) -> float:
    """Compute the c2 cohort dilution ratio analytically.

    Returns the ratio of mean liability in the any-relative c2 cohort
    to the theoretical mean under single-relative conditioning.
    """
    rho = 2 * kinship * h2
    if rho < 1e-9 or K < 1e-9 or K > 1 - 1e-9 or n_rel < 1:
        return 1.0
    threshold = norm.ppf(1 - K)
    z = norm.pdf(threshold)
    mean_single = rho * z / K
    if mean_single < 1e-9:
        return 1.0
    sd_cond = np.sqrt(max(1 - rho**2, 1e-12))

    def integrand_none(L: float) -> float:
        p_unaff = norm.cdf((threshold - rho * L) / sd_cond)
        return L * p_unaff**n_rel * norm.pdf(L)

    def integrand_p_none(L: float) -> float:
        p_unaff = norm.cdf((threshold - rho * L) / sd_cond)
        return p_unaff**n_rel * norm.pdf(L)

    E_L_none, _ = quad(integrand_none, -6, 6)
    P_none, _ = quad(integrand_p_none, -6, 6)
    P_any = 1 - P_none
    if P_any < 1e-9:
        return _MIN_DILUTION
    mean_any = -E_L_none / P_any
    ratio = mean_any / mean_single
    return max(ratio, _MIN_DILUTION)


def compute_dilution_corrected_h2(
    df: pd.DataFrame,
    n_rel: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Add dilution-corrected h² to the bias summary DataFrame.

    Adds columns: ``dilution_ratio``, ``h2_corrected``.
    Uses iterative correction: initial h² estimate -> dilution -> corrected h²
    -> re-compute dilution -> converge (typically 2-3 iterations).
    """
    if n_rel is None:
        n_rel = _DEFAULT_N_REL
    df = df.copy().reset_index(drop=True)
    h2_corr = np.full(len(df), np.nan)
    dilution_arr = np.full(len(df), np.nan)

    for i in range(len(df)):
        row = df.iloc[i]
        kind = row["kind"]
        K = row.get("prevalence", np.nan)
        h2_raw = row["h2_epimight"]
        if np.isnan(K) or np.isnan(h2_raw) or kind not in _KINSHIP:
            continue
        kinship = _KINSHIP[kind]
        nr = n_rel.get(kind, 2)

        # Iterative correction: start with h2_raw as liability h² estimate
        h2_est = max(h2_raw, 0.01)
        d = 1.0
        for _ in range(5):
            d = _analytical_dilution(K, h2_est, kinship, nr)
            h2_new = h2_raw / d if d > _MIN_DILUTION else np.nan
            if h2_new is None or np.isnan(h2_new):
                break
            if abs(h2_new - h2_est) < 0.001:
                h2_est = h2_new
                break
            h2_est = np.clip(h2_new, 0.01, 2.0)

        dilution_arr[i] = d
        h2_corr[i] = h2_est if d >= _MIN_DILUTION else np.nan

    df["dilution_ratio"] = dilution_arr
    df["h2_corrected"] = h2_corr
    return df


# ---------------------------------------------------------------------------
# 8. Corrected h² vs prevalence
# ---------------------------------------------------------------------------


def plot_epimight_corrected_h2(df: pd.DataFrame, output_path: Path) -> None:
    """h² before and after dilution correction, by prevalence and kind."""
    sub = df[(df["C"] == 0) & (df["phenotype_model"] == "adult_ltm") & (df["censor_label"] == "none")]
    if "h2_corrected" not in sub.columns:
        sub = compute_dilution_corrected_h2(sub)
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, col, title in [
        (axes[0], "h2_epimight", "Raw EPIMIGHT h²"),
        (axes[1], "h2_corrected", "Dilution-corrected h²"),
    ]:
        for kind in KIND_ORDER:
            ks = sub[sub["kind"] == kind].sort_values("prevalence")
            if ks.empty:
                continue
            vals = ks[col].values
            ax.plot(ks["prevalence"], vals, "o-", color=KIND_COLORS[kind], label=kind, markersize=5)

        h2_true = sub["h2_true_liability"].iloc[0] if not sub.empty else 0.5
        ax.axhline(h2_true, ls="--", color="0.4", linewidth=1, label="Liability h²")
        _log_pct_xaxis(ax)
        ax.set_xlabel("Prevalence")
        ax.set_ylabel("h² estimate")
        ax.set_title(title)
        ax.set_ylim(-0.05, 0.7)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle("EPIMIGHT h²: Raw vs Dilution-Corrected (No Censoring, C=0)", fontsize=12)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 9. Dilution ratio vs prevalence
# ---------------------------------------------------------------------------


def plot_epimight_dilution_ratio(df: pd.DataFrame, output_path: Path) -> None:
    """Analytical dilution ratio vs prevalence, by kind."""
    # Compute dilution over a fine prevalence grid for smooth curves
    K_grid = np.geomspace(0.005, 0.50, 50)
    h2_assumed = 0.50

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: analytical curves
    ax = axes[0]
    for kind in KIND_ORDER:
        kinship = _KINSHIP.get(kind)
        nr = _DEFAULT_N_REL.get(kind)
        if kinship is None or nr is None:
            continue
        dilutions = [_analytical_dilution(K, h2_assumed, kinship, nr) for K in K_grid]
        ax.plot(K_grid, dilutions, "-", color=KIND_COLORS[kind], label=f"{kind} (N={nr})", linewidth=1.5)

    # Overlay empirical points from the data
    sub = df[(df["C"] == 0) & (df["phenotype_model"] == "adult_ltm") & (df["censor_label"] == "none")]
    if "dilution_ratio" in sub.columns:
        for kind in KIND_ORDER:
            ks = sub[sub["kind"] == kind].sort_values("prevalence")
            if ks.empty or ks["dilution_ratio"].isna().all():
                continue
            ax.scatter(
                ks["prevalence"],
                ks["dilution_ratio"],
                color=KIND_COLORS[kind],
                s=30,
                zorder=5,
                edgecolors="k",
                linewidths=0.5,
            )

    ax.axhline(1.0, ls="--", color="0.4", linewidth=0.8)
    ax.axhline(_MIN_DILUTION, ls=":", color="red", linewidth=0.8, label=f"Min reliable ({_MIN_DILUTION})")
    _log_pct_xaxis(ax)
    ax.set_xlabel("Prevalence")
    ax.set_ylabel("Dilution ratio")
    ax.set_title("Analytical dilution ratio")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7, ncol=2, loc="lower left")

    # Right panel: c2 cohort fraction
    ax2 = axes[1]
    for kind in KIND_ORDER:
        kinship = _KINSHIP.get(kind)
        nr = _DEFAULT_N_REL.get(kind)
        if kinship is None or nr is None:
            continue
        # Approximate P(>=1 affected relative) = 1 - (1-K)^N
        fracs = [1 - (1 - K) ** nr for K in K_grid]
        ax2.plot(K_grid, fracs, "-", color=KIND_COLORS[kind], label=f"{kind} (N={nr})", linewidth=1.5)

    ax2.axhline(0.5, ls=":", color="0.5", linewidth=0.8, label="50% in c2")
    _log_pct_xaxis(ax2)
    ax2.set_xlabel("Prevalence")
    ax2.set_ylabel("Fraction of population in c2")
    ax2.set_title("c2 cohort saturation")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7, ncol=2, loc="upper left")

    fig.suptitle("c2 Cohort Dilution: Mechanism of Prevalence Bias", fontsize=12)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_PLOT_BASENAMES = [
    "epimight_bias_vs_prevalence",
    "epimight_bias_by_censoring",
    "epimight_bias_heatmap",
    "epimight_c_effect",
    "epimight_model_comparison",
    "epimight_forest",
    "epimight_attenuation_summary",
    "epimight_dilution_ratio",
    "epimight_corrected_h2",
]

# Plots that only apply to the standard (all-relatives) analysis
_DILUTION_CORRECTION_PLOTS = {"epimight_dilution_ratio", "epimight_corrected_h2"}

_PLOT_FUNCS = {
    "epimight_bias_vs_prevalence": plot_epimight_bias_vs_prevalence,
    "epimight_bias_by_censoring": plot_epimight_bias_by_censoring,
    "epimight_bias_heatmap": plot_epimight_bias_heatmap,
    "epimight_c_effect": plot_epimight_c_effect,
    "epimight_model_comparison": plot_epimight_model_comparison,
    "epimight_forest": plot_epimight_forest,
    "epimight_attenuation_summary": plot_epimight_attenuation_summary,
    "epimight_dilution_ratio": plot_epimight_dilution_ratio,
    "epimight_corrected_h2": plot_epimight_corrected_h2,
}


def generate_all_plots(
    df: pd.DataFrame,
    plots_dir: Path,
    include_dilution_correction: bool = True,
) -> list[Path]:
    """Generate all bias plots and return paths in atlas order."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    if include_dilution_correction and "h2_corrected" not in df.columns:
        df = compute_dilution_corrected_h2(df)
    paths = []
    for name in _PLOT_BASENAMES:
        if not include_dilution_correction and name in _DILUTION_CORRECTION_PLOTS:
            continue
        path = plots_dir / f"{name}.png"
        func = _PLOT_FUNCS[name]
        try:
            func(df, path)
            if path.exists():
                paths.append(path)
        except Exception:
            logger.exception("Failed to generate %s", name)
    return paths


def assemble_epimight_bias_atlas(
    tsv_path: str | Path,
    output_path: str | Path,
    include_dilution_correction: bool = True,
) -> None:
    """Read bias summary, generate all plots, assemble into atlas PDF."""
    tsv_path = Path(tsv_path)
    output_path = Path(output_path)
    plots_dir = output_path.parent

    df = pd.read_csv(tsv_path, sep="\t")
    logger.info("Loaded %d rows from %s", len(df), tsv_path)

    paths = generate_all_plots(df, plots_dir, include_dilution_correction=include_dilution_correction)
    assemble_atlas(paths, EPIMIGHT_BIAS_CAPTIONS, output_path)
    logger.info("Assembled EPIMIGHT bias atlas: %s", output_path)


def main(tsv_path: str, output_path: str, include_dilution_correction: bool = True) -> None:
    assemble_epimight_bias_atlas(tsv_path, output_path, include_dilution_correction=include_dilution_correction)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Generate EPIMIGHT bias analysis plots")
    parser.add_argument("--tsv", required=True, help="Path to epimight_bias_summary.tsv")
    parser.add_argument("--output", required=True, help="Output atlas PDF path")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    setup_logging(log_file=args.log_file)
    main(args.tsv, args.output)


if __name__ == "__main__":
    cli()
