"""PA-FGRS diagnostic atlas: scatter, metrics, distributions, calibration."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)

TRAITS = ["trait1", "trait2"]
CIP_SOURCES = ["empirical", "true"]
H2_SOURCES = ["true", "estimated"]

MAX_SCATTER_POINTS = 100_000


def _load_variant_data(base_dir: str) -> dict[str, pd.DataFrame]:
    """Load score data keyed by variant tag.

    Reads ``scores.parquet`` (wide format) and unpacks each variant
    into a DataFrame with columns: id, est, var, true_A, affected,
    generation, n_relatives.
    """
    combined = pd.read_parquet(Path(base_dir) / "scores.parquet")
    data = {}
    for t in TRAITS:
        trait_num = t[-1]  # "1" or "2"
        for c in CIP_SOURCES:
            for h in H2_SOURCES:
                tag = f"{t}_{c}_{h}"
                est_col = f"est_{tag}"
                if est_col not in combined.columns:
                    continue
                data[tag] = pd.DataFrame(
                    {
                        "id": combined["id"],
                        "est": combined[est_col],
                        "var": combined[f"var_{tag}"],
                        "true_A": combined[f"true_A{trait_num}"],
                        "affected": combined[f"affected{trait_num}"],
                        "generation": combined["generation"],
                        "n_relatives": combined[f"nrel_{tag}"],
                    }
                )
    return data


def _load_metrics(base_dir: str) -> pd.DataFrame:
    """Load ``metrics.tsv`` (wide) and melt to long format.

    Returns DataFrame with columns: trait, cip_source, h2_source, metric, value.
    """
    wide = pd.read_csv(Path(base_dir) / "metrics.tsv", sep="\t")
    id_cols = ["trait", "cip_source", "h2_source"]
    value_cols = [c for c in wide.columns if c not in id_cols]
    return wide.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric")


def _subsample(df: pd.DataFrame, max_n: int = MAX_SCATTER_POINTS, seed: int = 42) -> tuple[pd.DataFrame, str]:
    """Subsample a DataFrame for plotting; return (df, note)."""
    if len(df) <= max_n:
        return df, ""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_n, replace=False)
    return df.iloc[idx], f"(showing {max_n:,} of {len(df):,})"


def _page_scatter(pdf: PdfPages, data: dict[str, pd.DataFrame], trait: str) -> None:
    """Scatter: est vs true_A, subsampled and rasterized for large N."""
    variants = [(c, h) for c in CIP_SOURCES for h in H2_SOURCES]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"PA-FGRS est vs true A — {trait}", fontsize=14)

    for ax, (cip, h2src) in zip(axes.flat, variants, strict=False):
        tag = f"{trait}_{cip}_{h2src}"
        if tag not in data:
            ax.set_visible(False)
            continue
        df_full = data[tag]
        df, note = _subsample(df_full)
        aff = df["affected"].values.astype(bool)
        ax.scatter(
            df["true_A"][~aff],
            df["est"][~aff],
            s=1,
            alpha=0.15,
            c="steelblue",
            label="Control",
            rasterized=True,
        )
        ax.scatter(
            df["true_A"][aff],
            df["est"][aff],
            s=1,
            alpha=0.4,
            c="firebrick",
            label="Case",
            rasterized=True,
        )
        # Correlation computed on FULL data
        if len(df_full) > 2:
            from sim_ace.core.utils import fast_pearsonr

            r, p = fast_pearsonr(df_full["est"].values, df_full["true_A"].values)
            p_str = "p<.001" if p < 0.001 else f"p={p:.3f}"
            title = f"CIP={cip}, h2={h2src}  (r={r:.3f}, {p_str}, n={len(df_full):,})"
        else:
            title = f"CIP={cip}, h2={h2src}  (r=N/A)"
        if note:
            title += f"\n{note}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("True A")
        ax.set_ylabel("PA-FGRS est")
        lo = min(df["true_A"].min(), df["est"].min())
        hi = max(df["true_A"].max(), df["est"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.3)
        ax.legend(markerscale=5, fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig, dpi=300)
    plt.close(fig)


def _page_metrics_bars(pdf: PdfPages, metrics_df: pd.DataFrame, trait: str) -> None:
    """Bar charts of r, R², AUC, bias across parameter combos."""
    sub = metrics_df[metrics_df["trait"] == trait]
    if sub.empty:
        return

    metric_names = ["r", "r2", "auc", "bias"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(f"PA-FGRS Metrics — {trait}", fontsize=14)

    for ax, metric in zip(axes, metric_names, strict=False):
        vals = sub[sub["metric"] == metric]
        if vals.empty:
            ax.set_visible(False)
            continue
        labels = [f"{row['cip_source']}\n{row['h2_source']}" for _, row in vals.iterrows()]
        colors = ["#4c72b0" if "true" in lab else "#dd8452" for lab in labels]
        bars = ax.bar(range(len(vals)), vals["value"].values, color=colors)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(metric, fontsize=11)
        for bar, v in zip(bars, vals["value"].values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_score_distributions(pdf: PdfPages, data: dict[str, pd.DataFrame], trait: str) -> None:
    """Histogram of est for cases vs controls."""
    from scipy.stats import ks_2samp

    variants = [(c, h) for c in CIP_SOURCES for h in H2_SOURCES]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"PA-FGRS Score Distributions — {trait}", fontsize=14)

    for ax, (cip, h2src) in zip(axes.flat, variants, strict=False):
        tag = f"{trait}_{cip}_{h2src}"
        if tag not in data:
            ax.set_visible(False)
            continue
        df = data[tag]
        aff = df["affected"].values.astype(bool)
        est = df["est"].values
        bins = np.linspace(np.percentile(est, 1), np.percentile(est, 99), 80)
        ax.hist(est[~aff], bins=bins, alpha=0.6, density=True, label="Control", color="steelblue")
        ax.hist(est[aff], bins=bins, alpha=0.6, density=True, label="Case", color="firebrick")
        ks, ks_p = ks_2samp(est[aff], est[~aff]) if aff.sum() > 1 else (0, 1)
        ks_p_str = "p<.001" if ks_p < 0.001 else f"p={ks_p:.3f}"
        ax.set_title(f"CIP={cip}, h2={h2src}  (KS={ks:.3f}, {ks_p_str})", fontsize=10)
        ax.set_xlabel("PA-FGRS est")
        ax.legend(fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_calibration(pdf: PdfPages, data: dict[str, pd.DataFrame], trait: str) -> None:
    """Variance calibration: reported var vs actual (est - true_A)² binned."""
    variants = [(c, h) for c in CIP_SOURCES for h in H2_SOURCES]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Variance Calibration — {trait}", fontsize=14)

    for ax, (cip, h2src) in zip(axes.flat, variants, strict=False):
        tag = f"{trait}_{cip}_{h2src}"
        if tag not in data:
            ax.set_visible(False)
            continue
        df = data[tag]
        error_sq = (df["est"].values - df["true_A"].values) ** 2
        reported_var = df["var"].values
        n_bins = 20
        bin_edges = np.linspace(reported_var.min(), reported_var.max(), n_bins + 1)
        bin_centers, bin_means = [], []
        for b in range(n_bins):
            mask = (reported_var >= bin_edges[b]) & (reported_var < bin_edges[b + 1])
            if mask.sum() > 5:
                bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                bin_means.append(error_sq[mask].mean())
        if bin_centers:
            ax.scatter(bin_centers, bin_means, c="steelblue", s=20)
        lo = 0
        hi = max(max(bin_centers, default=0), max(bin_means, default=0)) * 1.1
        if hi > lo:
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.5, label="Perfect calibration")
        ax.set_title(f"CIP={cip}, h2={h2src}", fontsize=10)
        ax.set_xlabel("Reported Var")
        ax.set_ylabel("Actual MSE")
        ax.legend(fontsize=8)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _page_generation_breakdown(pdf: PdfPages, data: dict[str, pd.DataFrame], trait: str) -> None:
    """r(est, true_A) broken down by generation."""
    tag = f"{trait}_true_true"
    if tag not in data:
        tag = f"{trait}_empirical_true"
    if tag not in data:
        return

    df = data[tag]
    gens = sorted(df["generation"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    r_vals = []
    for g in gens:
        sub = df[df["generation"] == g]
        if len(sub) > 5 and sub["est"].std() > 1e-10:
            from sim_ace.core._numba_utils import _pearsonr_core

            r = float(_pearsonr_core(sub["est"].values, sub["true_A"].values))
        else:
            r = 0.0
        r_vals.append(r)

    ax.bar(range(len(gens)), r_vals, color="steelblue")
    ax.set_xticks(range(len(gens)))
    ax.set_xticklabels([f"Gen {g}" for g in gens])
    ax.set_ylabel("r(est, true A)")
    ax.set_title(f"PA-FGRS accuracy by generation — {trait} (CIP=true, h2=true)")
    for i, v in enumerate(r_vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def generate_atlas(base_dir: str, output_path: str) -> None:
    """Generate the full PA-FGRS diagnostic atlas PDF."""
    data = _load_variant_data(base_dir)
    metrics_df = _load_metrics(base_dir)

    if not data:
        logger.warning("No PA-FGRS score files found in %s", base_dir)
        return

    n_total = sum(len(df) for df in data.values())
    logger.info("Atlas data: %d variants, %d total rows", len(data), n_total)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        for trait in TRAITS:
            if not any(k.startswith(trait) for k in data):
                continue
            _page_scatter(pdf, data, trait)
            _page_metrics_bars(pdf, metrics_df, trait)
            _page_score_distributions(pdf, data, trait)
            _page_calibration(pdf, data, trait)
            _page_generation_breakdown(pdf, data, trait)

    logger.info("Wrote PA-FGRS atlas: %s (%d pages)", output_path, 5 * len(TRAITS))
