"""Four-page PDF visualizing the top-X PCs of the expected A-matrix (2φ).

Page 1: scree plot (eigenvalue on log-y) with a secondary axis for
cumulative variance explained.

Page 2: histogram of PC1 (the mean-relatedness / connectivity axis).

Pages 3–4: PC2 vs PC3 and PC4 vs PC5 scatter, colored by trait-1
affected status (cases red, controls grey — matching the main
scenario atlas).  Only individuals with phenotype data are shown.
"""

from __future__ import annotations

__all__ = ["plot_grm_pcs"]

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from sim_ace.plotting.plot_style import COLOR_AFFECTED, COLOR_UNAFFECTED
from sim_ace.plotting.plot_utils import save_placeholder_plot

logger = logging.getLogger(__name__)


def plot_grm_pcs(
    pcs_df: pd.DataFrame,
    eigenvalues_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    output_path: str | Path,
    scenario: str = "",
    rep: str = "",
) -> None:
    """Render scree + PC1 histogram + PC2vs3 + PC4vs5 scatters to a 4-page PDF.

    Parameters
    ----------
    pcs_df : wide PC table with columns ``id, PC1, PC2, ...``.
    eigenvalues_df : TSV schema with columns
        ``rank, eigenvalue, variance_explained, cumulative_variance_explained``.
    phenotype_df : must contain ``id`` and ``affected1``.  Only
        individuals present in this table are shown on the scatter pages.
    output_path : path to the output PDF.
    scenario, rep : labels for the plot titles.
    """
    finite = eigenvalues_df.dropna(subset=["eigenvalue"]).copy()
    k = len(finite)

    if k < 1:
        save_placeholder_plot(
            output_path,
            f"GRM PCs: k={k} — no PCs computed.\nScenario: {scenario}, rep: {rep}.",
        )
        logger.warning("plot_grm_pcs: k=%d, wrote placeholder to %s", k, output_path)
        return

    merged = pcs_df.merge(phenotype_df[["id", "affected1"]], on="id", how="inner")

    def _title(base: str, n: int) -> str:
        parts = [base, f"N={n}"]
        if scenario:
            parts.append(scenario)
        if rep:
            parts.append(f"rep{rep}")
        return " — ".join(parts)

    def _scatter_page(pdf_, x_col: str, y_col: str) -> None:
        if x_col not in merged.columns or y_col not in merged.columns:
            return
        x = merged[x_col].to_numpy()
        y = merged[y_col].to_numpy()
        if not (np.any(np.isfinite(x)) and np.any(np.isfinite(y))):
            return
        aff = merged["affected1"].to_numpy().astype(bool)
        n_aff = int(aff.sum())
        n_un = int((~aff).sum())

        fig, ax = plt.subplots(figsize=(7, 6))
        # Plot unaffected first so affected cases are drawn on top
        ax.scatter(
            x[~aff], y[~aff],
            c=COLOR_UNAFFECTED, s=5, alpha=0.4, linewidths=0,
            label=f"unaffected (n={n_un})",
        )
        ax.scatter(
            x[aff], y[aff],
            c=COLOR_AFFECTED, s=8, alpha=0.8, linewidths=0,
            label=f"affected (n={n_aff})",
        )
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.axvline(0, color="0.7", lw=0.5)
        ax.legend(loc="best", frameon=False, fontsize=9)
        ax.set_title(_title(f"GRM {x_col} vs {y_col}", len(merged)))
        fig.tight_layout()
        pdf_.savefig(fig)
        plt.close(fig)

    with PdfPages(output_path) as pdf:
        # Page 1: scree + cumulative VE
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ranks = finite["rank"].to_numpy()
        eigs = finite["eigenvalue"].to_numpy()
        cum = finite["cumulative_variance_explained"].to_numpy()

        ax1.plot(ranks, eigs, marker="o", color="tab:blue", lw=1.5, ms=4, label="eigenvalue")
        ax1.set_yscale("log")
        ax1.set_xlabel("PC rank")
        ax1.set_ylabel("eigenvalue (log scale)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, which="both", alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(ranks, cum, marker="s", color="tab:orange", lw=1.5, ms=4, label="cumulative VE")
        ax2.set_ylabel("cumulative variance explained", color="tab:orange")
        ax2.set_ylim(0.0, min(1.0, max(cum.max() * 1.05, 0.01)))
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.set_title(_title(f"GRM scree (k={k})", len(pcs_df)))
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: PC1 histogram
        fig, ax = plt.subplots(figsize=(7, 5))
        pc1 = merged["PC1"].to_numpy()
        pc1 = pc1[np.isfinite(pc1)]
        ax.hist(pc1, bins=50, color="tab:blue", alpha=0.8, edgecolor="black", lw=0.3)
        ax.axvline(float(pc1.mean()), color="black", lw=0.8, ls="--", label=f"mean = {pc1.mean():.3g}")
        ax.set_xlabel("PC1 score")
        ax.set_ylabel("count")
        ax.set_title(_title("GRM PC1 distribution", len(merged)))
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Pages 3–4: PC2 vs PC3, PC4 vs PC5
        _scatter_page(pdf, "PC2", "PC3")
        _scatter_page(pdf, "PC4", "PC5")

    logger.info("plot_grm_pcs: wrote %s", output_path)
