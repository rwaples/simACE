"""EPIMIGHT plot atlas: visualize heritability and genetic correlation across relationship kinds.

Discovers TSV output files from guide-yob.R, generates comparison plots across
relationship kinds, and assembles them into a multi-page PDF atlas.

Usage:
    python epimight/plot_epimight.py results/epimight/base/baseline100K/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from sim_ace.plot_atlas import assemble_atlas

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KIND_ORDER: list[str] = ["PO", "FS", "HS", "mHS", "pHS", "Av", "1G", "1C"]

KIND_COLORS: dict[str, str] = {
    "PO": "C0", "FS": "C1", "HS": "C2", "mHS": "C3",
    "pHS": "C4", "Av": "C5", "1G": "C6", "1C": "C7",
}

KIND_LABELS: dict[str, str] = {
    "PO": "Parent-Offspring", "FS": "Full Sibling", "HS": "Half Sibling",
    "mHS": "Maternal HS", "pHS": "Paternal HS", "Av": "Avuncular",
    "1G": "Grandparent-GC", "1C": "1st Cousin",
}

_PRIMARY_KINDS = ["PO", "FS", "HS"]

_PLOT_BASENAMES = [
    "cif_base",
    "cif_primary_d1", "cif_primary_d2",
    "cif_secondary_d1", "cif_secondary_d2",
    "h2_time_d1", "h2_time_d2",
    "h2_bar_d1", "h2_bar_d2",
    "gc_bar", "summary_table",
]

EPIMIGHT_CAPTIONS: dict[str, str] = {
    "cif_base": (
        "Figure 1: Base population CIF.\n\n"
        "Cumulative incidence in the full population (c1) for both disorders, "
        "one line per birth year. Shared y-axis. Color encodes birth year (grayscale)."
    ),
    "cif_primary_d1": (
        "Figure 2a: CIF \u2014 Disorder 1 (PO, FS, HS).\n\n"
        "Four panels: c1 base population, then exposed cohorts for parent-offspring, "
        "full sibling, and half sibling. Solid grey lines = c1 base, dashed colored "
        "lines = exposed cohort (c2). Lines colored by birth year."
    ),
    "cif_primary_d2": (
        "Figure 2b: CIF \u2014 Disorder 2 (PO, FS, HS).\n\n"
        "Same layout as Figure 2a but for disorder 2 (c3 exposed cohort)."
    ),
    "cif_secondary_d1": (
        "Figure 3a: CIF \u2014 Disorder 1 (remaining kinds).\n\n"
        "Panels for maternal HS, paternal HS, avuncular, grandparent-GC, and 1st cousin. "
        "Solid grey = c1 base, dashed colored = c2 exposed cohort."
    ),
    "cif_secondary_d2": (
        "Figure 3b: CIF \u2014 Disorder 2 (remaining kinds).\n\n"
        "Same layout as Figure 3a but for disorder 2."
    ),
    "h2_time_d1": (
        "Figure 4: Heritability over follow-up \u2014 Disorder 1.\n\n"
        "Heritability (h\u00b2) estimated at each follow-up time point for disorder 1, "
        "one panel per birth cohort. Each colored line represents a different relationship "
        "kind. Shaded bands show 95% CIs. Horizontal dashed line marks the true h\u00b2."
    ),
    "h2_time_d2": (
        "Figure 5: Heritability over follow-up \u2014 Disorder 2.\n\n"
        "Same as Figure 4 but for disorder 2."
    ),
    "h2_bar_d1": (
        "Figure 6a: Heritability at maximum follow-up \u2014 Disorder 1.\n\n"
        "One panel per relationship kind. Bars show h\u00b2 at maximum follow-up "
        "for each birth cohort. Error bars show 95% CIs. "
        "Horizontal dashed line marks the true h\u00b2."
    ),
    "h2_bar_d2": (
        "Figure 6b: Heritability at maximum follow-up \u2014 Disorder 2.\n\n"
        "Same layout as Figure 6a but for disorder 2."
    ),
    "gc_bar": (
        "Figure 7: Genetic correlation at maximum follow-up.\n\n"
        "Bar chart of the genetic correlation estimate (\u03c1g) at maximum follow-up "
        "across relationship kinds, grouped by birth cohort. Error bars show 95% CIs. "
        "Horizontal dashed line marks the true genetic correlation."
    ),
    "summary_table": (
        "Figure 8: Summary comparison table.\n\n"
        "Per-kind summary: exposed cohort sizes (c2 for d1, c3 for d2), median h\u00b2 "
        "at maximum follow-up for each disorder, and median genetic correlation (\u03c1g). "
        "True parameter values shown in the last row."
    ),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def discover_kinds(tsv_dir: Path) -> list[str]:
    """Discover which relationship kinds have TSV output."""
    found = set()
    for p in tsv_dir.glob("h2_d1_*.tsv"):
        kind = p.stem.replace("h2_d1_", "")
        found.add(kind)
    ordered = [k for k in KIND_ORDER if k in found]
    extras = sorted(found - set(KIND_ORDER))
    return ordered + extras


def _load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Missing TSV: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_cif(tsv_dir: Path, disorder: str, cohort: str, kind: str) -> pd.DataFrame:
    return _load_tsv(tsv_dir / f"cif_{disorder}_{cohort}_{kind}.tsv")


def load_h2(tsv_dir: Path, disorder: str, kind: str) -> pd.DataFrame:
    return _load_tsv(tsv_dir / f"h2_{disorder}_{kind}.tsv")


def load_gc(tsv_dir: Path, kind: str) -> pd.DataFrame:
    return _load_tsv(tsv_dir / f"gc_full_{kind}.tsv")


def load_true_params(scenario_dir: Path) -> dict | None:
    path = scenario_dir / "true_parameters.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_cohort_sizes(scenario_dir: Path, kind: str) -> dict[str, int]:
    """Parse cohort sizes from results_{kind}.md.

    Returns dict like {"c1": 600000, "c2": 190962, "c3": 377841}.
    """
    path = scenario_dir / f"results_{kind}.md"
    sizes = {}
    if not path.exists():
        return sizes
    text = path.read_text()
    for m in re.finditer(r"\|\s*(c\d)\s*\|[^|]*\|\s*(\d+)\s*\|", text):
        sizes[m.group(1)] = int(m.group(2))
    return sizes


def tmax_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the row at maximum time per born_at_year."""
    if df.empty:
        return df
    idx = df.groupby("born_at_year")["time"].idxmax()
    return df.loc[idx].sort_values("born_at_year")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _make_panel_grid(n_panels: int, n_cols: int = 3, **subplot_kw):
    """Create a multi-panel subplot grid and return (fig, axes, n_rows, n_cols)."""
    n_cols = min(n_panels, n_cols)
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig_kw = {"figsize": (6 * n_cols, 4.5 * n_rows), "sharey": True, "squeeze": False}
    fig_kw.update(subplot_kw)
    fig, axes = plt.subplots(n_rows, n_cols, **fig_kw)
    return fig, axes, n_rows, n_cols


def _hide_unused(axes, n_used: int, n_rows: int, n_cols: int) -> None:
    """Hide axes beyond *n_used* in a (n_rows x n_cols) grid."""
    for idx in range(n_used, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def _plot_c1_on_ax(ax, c1: pd.DataFrame, years: list, norm, cmap):
    """Plot c1 base CIF curves on a single axes."""
    for year in years:
        c1y = c1[c1["born_at_year"] == year].sort_values("time")
        if c1y.empty:
            continue
        ax.plot(c1y["time"], c1y["estimate"], color=cmap(norm(year)),
                linewidth=0.8, alpha=0.5)


def _add_cif_colorbar_and_legend(fig, axes, norm, cmap_exposed, exposed_cohort):
    """Add horizontal colorbar and legend below the figure."""

    sm = cm.ScalarMappable(cmap=cmap_exposed, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, -0.06, 0.5, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Birth year")
    cbar.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    legend_elements = [
        Line2D([0], [0], color="0.4", linewidth=1.2, label="c1 (base)"),
        Line2D([0], [0], color="0.4", linewidth=1.2, linestyle="--",
               label=f"{exposed_cohort} (exposed)"),
    ]
    fig.legend(handles=legend_elements, loc="lower right", ncol=2,
               fontsize=10, frameon=True, bbox_to_anchor=(0.95, -0.07))


def plot_cif_base(
    tsv_dir: Path, kinds: list[str], output_path: Path,
) -> None:
    """Two-panel plot of c1 base population CIF (D1 left, D2 right), shared y-axis."""
    c1_d1 = load_cif(tsv_dir, "d1", "c1", kinds[0])
    c1_d2 = load_cif(tsv_dir, "d2", "c1", kinds[0])
    if c1_d1.empty and c1_d2.empty:
        return

    ref = c1_d1 if not c1_d1.empty else c1_d2
    years = sorted(ref["born_at_year"].unique())
    norm = Normalize(vmin=min(years), vmax=max(years))
    # Truncate Greys so lightest color is ~0.7 gray, not white
    cmap = LinearSegmentedColormap.from_list(
        "Greys_trunc", cm.Greys(np.linspace(0.25, 1.0, 256))
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, c1, label in [(ax1, c1_d1, "D1"), (ax2, c1_d2, "D2")]:
        if c1.empty:
            continue
        for year in years:
            c1y = c1[c1["born_at_year"] == year].sort_values("time")
            if c1y.empty:
                continue
            ax.plot(c1y["time"], c1y["estimate"], color=cmap(norm(year)),
                    linewidth=1.0)
        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Follow-up time (age)")

    ax1.set_ylabel("Cumulative Incidence")

    fig.suptitle("Base Population CIF", fontsize=14)
    plt.tight_layout()

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.3, -0.06, 0.4, 0.025])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Birth year")
    cbar.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_cif_panels(
    tsv_dir: Path, panel_kinds: list[str], disorder: str,
    output_path: Path, title: str, n_cols: int = 2,
) -> None:
    """CIF panels: first panel is c1-only, remaining panels show c1 + exposed."""

    exposed_cohort = "c2" if disorder == "d1" else "c3"
    c1 = load_cif(tsv_dir, disorder, "c1", panel_kinds[0] if panel_kinds else "PO")
    if c1.empty:
        return

    years = sorted(c1["born_at_year"].unique())
    norm = Normalize(vmin=min(years), vmax=max(years))
    cmap_base = LinearSegmentedColormap.from_list(
        "Greys_trunc", cm.Greys(np.linspace(0.25, 1.0, 256))
    )
    # Use both sides of vanimo but skip the dark center and light ends
    _vanimo_colors = np.vstack([
        cm.vanimo(np.linspace(0.15, 0.35, 128)),
        cm.vanimo(np.linspace(0.65, 0.85, 128)),
    ])
    cmap_exposed = LinearSegmentedColormap.from_list("vanimo_clamp", _vanimo_colors)

    # panels: c1-base + one per kind
    n_panels = 1 + len(panel_kinds)
    fig, axes, n_rows, n_cols = _make_panel_grid(n_panels, n_cols=n_cols)

    # Panel 0: c1-only (grayscale)
    ax0 = axes[0, 0]
    _plot_c1_on_ax(ax0, c1, years, norm, cmap_base)
    ax0.set_title("c1 (base)", fontsize=11)
    ax0.set_xlabel("Follow-up time (age)", fontsize=9)
    ax0.set_ylabel("Cumulative Incidence")

    # Panels 1..N: c1 grey + exposed dashed colored
    for idx, kind in enumerate(panel_kinds):
        panel_idx = idx + 1
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row, col]

        _plot_c1_on_ax(ax, c1, years, norm, cmap_base)

        c_exp = load_cif(tsv_dir, disorder, exposed_cohort, kind)
        if not c_exp.empty:
            for year in years:
                c_ey = c_exp[c_exp["born_at_year"] == year].sort_values("time")
                if c_ey.empty:
                    continue
                ax.plot(c_ey["time"], c_ey["estimate"],
                        color=cmap_exposed(norm(year)),
                        linewidth=0.8, alpha=0.7, linestyle="--")

        ax.set_title(KIND_LABELS.get(kind, kind), fontsize=11)
        ax.set_xlabel("Follow-up time (age)", fontsize=9)
        if col == 0:
            ax.set_ylabel("Cumulative Incidence")

    _hide_unused(axes, n_panels, n_rows, n_cols)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    _add_cif_colorbar_and_legend(fig, axes, norm, cmap_exposed, exposed_cohort)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cif_primary(
    tsv_dir: Path, kinds: list[str], disorder: str, output_path: Path,
) -> None:
    """CIF panels for primary kinds (PO, FS, HS) with c1 reference panel."""
    primary = [k for k in _PRIMARY_KINDS if k in kinds]
    if not primary:
        return
    d_label = disorder.upper()
    _plot_cif_panels(tsv_dir, primary, disorder, output_path,
                     f"CIF: Base + Primary Kinds \u2014 {d_label}")


def plot_cif_secondary(
    tsv_dir: Path, kinds: list[str], disorder: str, output_path: Path,
) -> None:
    """CIF panels for remaining kinds after PO/FS/HS."""
    secondary = [k for k in kinds if k not in _PRIMARY_KINDS]
    if not secondary:
        return
    d_label = disorder.upper()
    _plot_cif_panels(tsv_dir, secondary, disorder, output_path,
                     f"CIF: Secondary Kinds \u2014 {d_label}", n_cols=3)


def plot_h2_by_time(
    tsv_dir: Path, kinds: list[str], disorder: str,
    true_h2: float | None, output_path: Path,
) -> None:
    """h2 vs follow-up time, one panel per kind. Faint lines per birth year, bold median."""
    fig, axes, n_rows, n_cols = _make_panel_grid(len(kinds))

    for idx, kind in enumerate(kinds):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        color = KIND_COLORS.get(kind, "black")

        h2 = load_h2(tsv_dir, disorder, kind)
        if h2.empty:
            continue

        years = sorted(h2["born_at_year"].unique())

        # Faint line per birth year
        for year in years:
            h2y = h2[h2["born_at_year"] == year].sort_values("time")
            if h2y.empty:
                continue
            ax.plot(h2y["time"], h2y["h2"], color=color, linewidth=0.5, alpha=0.25)

        # Bold median line across birth years
        times = sorted(h2["time"].unique())
        median_h2 = []
        for t in times:
            vals = h2.loc[h2["time"] == t, "h2"].dropna()
            median_h2.append(vals.median() if len(vals) > 0 else np.nan)
        ax.plot(times, median_h2, color=color, linewidth=2.5,
                label=f"Median ({KIND_LABELS.get(kind, kind)})")

        if true_h2 is not None:
            ax.axhline(true_h2, color="black", linestyle="--", linewidth=1.5,
                        alpha=0.7, label=f"True h\u00b2 = {true_h2:.3f}")

        ax.set_title(KIND_LABELS.get(kind, kind), fontsize=11)
        ax.set_xlabel("Follow-up time (age)", fontsize=9)
        if col == 0:
            ax.set_ylabel("h\u00b2")
        ax.legend(fontsize=7, loc="best")

    _hide_unused(axes, len(kinds), n_rows, n_cols)

    fig.suptitle(f"Heritability over Follow-up \u2014 {disorder.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_bar_panels(
    kinds: list[str],
    load_fn,
    value_col: str,
    true_value: float | None,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Bar chart at tmax, one panel per relationship kind.

    Args:
        load_fn: callable(kind) -> DataFrame with columns [born_at_year, time, value_col, l95, u95].
        value_col: column name for the bar values (e.g. "h2" or "rhog").
    """
    fig, axes, n_rows, n_cols = _make_panel_grid(len(kinds))

    for idx, kind in enumerate(kinds):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        color = KIND_COLORS.get(kind, "black")

        df = load_fn(kind)
        if df.empty:
            continue
        tm = tmax_rows(df)
        years = tm["born_at_year"].values
        vals = tm[value_col].values
        err_lo = np.maximum(0, vals - tm["l95"].values)
        err_hi = np.maximum(0, tm["u95"].values - vals)

        ax.bar(range(len(years)), vals, color=color, alpha=0.8, zorder=3)
        ax.errorbar(range(len(years)), vals, yerr=[err_lo, err_hi],
                    fmt="none", color="black", capsize=2, linewidth=0.6, zorder=4)

        if true_value is not None:
            ax.axhline(true_value, color="black", linestyle="--", linewidth=1.5,
                        alpha=0.7, label=f"True = {true_value:.3f}")

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels([str(int(y)) for y in years], fontsize=6, rotation=45)
        ax.set_xlabel("Birth cohort", fontsize=9)
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(KIND_LABELS.get(kind, kind), fontsize=11)
        ax.legend(fontsize=7, loc="best")

    _hide_unused(axes, len(kinds), n_rows, n_cols)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_h2_bar(
    tsv_dir: Path, kinds: list[str], disorder: str,
    true_h2: float | None, output_path: Path,
) -> None:
    """Bar chart of h2 at tmax, one panel per relationship kind."""
    _plot_bar_panels(
        kinds,
        load_fn=lambda kind: load_h2(tsv_dir, disorder, kind),
        value_col="h2",
        true_value=true_h2,
        ylabel="h\u00b2",
        title=f"Heritability at Maximum Follow-up \u2014 {disorder.upper()}",
        output_path=output_path,
    )


def plot_gc_bar(
    tsv_dir: Path, kinds: list[str], true_params: dict | None, output_path: Path,
) -> None:
    """Bar chart of rhog at tmax, one panel per relationship kind.

    Note: the GC TSV ``h2_l95``/``h2_u95`` columns are the 95% CIs for
    ``rhog`` (= l95/h2_comb and u95/h2_comb from EPIMIGHT).
    Values outside [-1, 1] are clamped and marked with arrows.
    """
    true_gc = true_params.get("genetic_correlation_true") if true_params else None
    ylim = (-1.05, 1.05)

    fig, axes, n_rows, n_cols = _make_panel_grid(len(kinds))

    for idx, kind in enumerate(kinds):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        color = KIND_COLORS.get(kind, "black")

        gc = load_gc(tsv_dir, kind)
        if gc.empty:
            continue
        tm = tmax_rows(gc)
        years = tm["born_at_year"].values
        vals = tm["rhog"].values

        # Error bars from rhog CIs (h2_l95 / h2_u95 in the GC TSV)
        err_lo = np.clip(vals - tm["h2_l95"].values, 0, None)
        err_hi = np.clip(tm["h2_u95"].values - vals, 0, None)

        # Clamp bars to [-1, 1] for display
        clamped = np.clip(vals, -1, 1)
        ax.bar(range(len(years)), clamped, color=color, alpha=0.8, zorder=3)
        ax.errorbar(range(len(years)), clamped,
                    yerr=[np.minimum(err_lo, clamped - ylim[0]),
                          np.minimum(err_hi, ylim[1] - clamped)],
                    fmt="none", color="black", capsize=2, linewidth=0.6,
                    zorder=4, clip_on=True)

        # Mark out-of-range values with arrows at the clamp boundary
        for i, v in enumerate(vals):
            if v > 1:
                ax.annotate("\u2191", xy=(i, 1), ha="center", va="bottom",
                            fontsize=7, color="red", fontweight="bold")
            elif v < -1:
                ax.annotate("\u2193", xy=(i, -1), ha="center", va="top",
                            fontsize=7, color="red", fontweight="bold")

        if true_gc is not None:
            ax.axhline(true_gc, color="black", linestyle="--", linewidth=1.5,
                        alpha=0.7, label=f"True = {true_gc:.3f}")

        ax.set_ylim(ylim)
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels([str(int(y)) for y in years], fontsize=6, rotation=45)
        ax.set_xlabel("Birth cohort", fontsize=9)
        if col == 0:
            ax.set_ylabel("Genetic correlation (\u03c1g)")
        ax.set_title(KIND_LABELS.get(kind, kind), fontsize=11)
        ax.legend(fontsize=7, loc="best")

    _hide_unused(axes, len(kinds), n_rows, n_cols)

    fig.suptitle("Genetic Correlation at Maximum Follow-up", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_table(
    tsv_dir: Path, kinds: list[str], true_params: dict | None, output_path: Path,
) -> None:
    """Render a summary comparison table as a figure."""
    columns = ["Kind", "c2 N", "c3 N", "h\u00b2 d1", "h\u00b2 d2", "\u03c1g"]
    table_data = []
    scenario_dir = tsv_dir.parent

    for kind in kinds:
        label = KIND_LABELS.get(kind, kind)

        # Cohort sizes from results_{kind}.md
        sizes = load_cohort_sizes(scenario_dir, kind)
        c2_n = f"{sizes['c2']:,}" if "c2" in sizes else "N/A"
        c3_n = f"{sizes['c3']:,}" if "c3" in sizes else "N/A"

        # h2 d1 median at tmax
        h2_d1 = load_h2(tsv_dir, "d1", kind)
        h2_d1_val = f"{tmax_rows(h2_d1)['h2'].median():.4f}" if not h2_d1.empty else "N/A"

        # h2 d2 median at tmax
        h2_d2 = load_h2(tsv_dir, "d2", kind)
        h2_d2_val = f"{tmax_rows(h2_d2)['h2'].median():.4f}" if not h2_d2.empty else "N/A"

        # rhog median at tmax
        gc = load_gc(tsv_dir, kind)
        gc_val = f"{tmax_rows(gc)['rhog'].median():.4f}" if not gc.empty else "N/A"

        table_data.append([label, str(c2_n), str(c3_n), h2_d1_val, h2_d2_val, gc_val])

    # True row
    if true_params:
        table_data.append([
            "True",
            "\u2014", "\u2014",
            f"{true_params['h2_trait1_true']:.4f}",
            f"{true_params['h2_trait2_true']:.4f}",
            f"{true_params['genetic_correlation_true']:.4f}",
        ])

    n_rows = len(table_data)
    fig_h = max(3, 1.2 + 0.45 * n_rows)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Right-align numeric columns (all except Kind)
    for i in range(n_rows + 1):  # +1 for header
        for j in range(1, len(columns)):
            table[i, j].get_text().set_ha("right")
            table[i, j].PAD = 0.05

    # Style header row
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Style True row
    if true_params:
        for j in range(len(columns)):
            cell = table[n_rows, j]
            cell.set_facecolor("#E2EFDA")
            cell.set_text_props(fontweight="bold")

    # Color kind cells
    for i, kind in enumerate(kinds):
        color = KIND_COLORS.get(kind, "black")
        rgba = to_rgba(color, alpha=0.15)
        for j in range(len(columns)):
            table[i + 1, j].set_facecolor(rgba)

    fig.suptitle("Summary Comparison Across Relationship Kinds", fontsize=14, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Atlas assembly
# ---------------------------------------------------------------------------


def assemble_epimight_atlas(scenario_dir: str | Path) -> None:
    """Discover EPIMIGHT TSV output, generate plots, and assemble atlas PDF."""
    scenario_dir = Path(scenario_dir)
    tsv_dir = scenario_dir / "tsv"
    plots_dir = scenario_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="colorblind")

    kinds = discover_kinds(tsv_dir)
    if not kinds:
        print(f"No EPIMIGHT TSV output found in {tsv_dir}")
        return

    print(f"Discovered kinds: {kinds}")

    true_params = load_true_params(scenario_dir)

    # Generate individual plots
    plot_cif_base(tsv_dir, kinds, plots_dir / "cif_base.png")
    for d in ("d1", "d2"):
        plot_cif_primary(tsv_dir, kinds, d, plots_dir / f"cif_primary_{d}.png")
        plot_cif_secondary(tsv_dir, kinds, d, plots_dir / f"cif_secondary_{d}.png")

    true_h2_d1 = true_params.get("h2_trait1_true") if true_params else None
    true_h2_d2 = true_params.get("h2_trait2_true") if true_params else None
    plot_h2_by_time(tsv_dir, kinds, "d1", true_h2_d1, plots_dir / "h2_time_d1.png")
    plot_h2_by_time(tsv_dir, kinds, "d2", true_h2_d2, plots_dir / "h2_time_d2.png")

    plot_h2_bar(tsv_dir, kinds, "d1", true_h2_d1, plots_dir / "h2_bar_d1.png")
    plot_h2_bar(tsv_dir, kinds, "d2", true_h2_d2, plots_dir / "h2_bar_d2.png")
    plot_gc_bar(tsv_dir, kinds, true_params, plots_dir / "gc_bar.png")
    plot_summary_table(tsv_dir, kinds, true_params, plots_dir / "summary_table.png")

    # Assemble atlas
    plot_paths = [plots_dir / f"{name}.png" for name in _PLOT_BASENAMES]
    section_breaks = {
        0: ("Cumulative Incidence", "CIF curves across relationship kinds"),
        5: ("Heritability", "h\u00b2 estimates by follow-up time and at maximum follow-up"),
        9: ("Genetic Correlation", "Cross-trait genetic correlation at maximum follow-up"),
        10: ("Summary", "Comparison across relationship kinds"),
    }

    atlas_path = plots_dir / "atlas.pdf"
    assemble_atlas(plot_paths, EPIMIGHT_CAPTIONS, atlas_path,
                   section_breaks=section_breaks)
    print(f"Atlas saved to {atlas_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate EPIMIGHT plot atlas comparing relationship kinds"
    )
    parser.add_argument(
        "scenario_dir",
        help="Path to EPIMIGHT results directory (contains tsv/ and true_parameters.json)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    assemble_epimight_atlas(args.scenario_dir)


if __name__ == "__main__":
    main()
