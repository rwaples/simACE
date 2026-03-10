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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

_PLOT_BASENAMES = [
    "cif_d1", "cif_d2", "h2_time_d1", "h2_time_d2",
    "h2_bar", "gc_bar", "summary_table",
]

EPIMIGHT_CAPTIONS: dict[str, str] = {
    "cif_d1": (
        "Figure 1: Cumulative incidence comparison \u2014 Disorder 1.\n\n"
        "CIF curves for disorder 1 across relationship kinds, one panel per birth cohort. "
        "Grey solid line shows the base population (c1). Dashed colored lines show the "
        "exposed cohort (c2: individuals with a diagnosed relative) for each relationship kind. "
        "Shaded bands are 95% confidence intervals."
    ),
    "cif_d2": (
        "Figure 2: Cumulative incidence comparison \u2014 Disorder 2.\n\n"
        "CIF curves for disorder 2 across relationship kinds, one panel per birth cohort. "
        "Grey solid line shows the base population (c1). Dashed colored lines show the "
        "exposed cohort (c3: individuals with a relative diagnosed with disorder 2) for each "
        "relationship kind. Shaded bands are 95% confidence intervals."
    ),
    "h2_time_d1": (
        "Figure 3: Heritability over follow-up \u2014 Disorder 1.\n\n"
        "Heritability (h\u00b2) estimated at each follow-up time point for disorder 1, "
        "one panel per birth cohort. Each colored line represents a different relationship "
        "kind. Shaded bands show 95% CIs. Horizontal dashed line marks the true h\u00b2."
    ),
    "h2_time_d2": (
        "Figure 4: Heritability over follow-up \u2014 Disorder 2.\n\n"
        "Same as Figure 3 but for disorder 2."
    ),
    "h2_bar": (
        "Figure 5: Heritability at maximum follow-up.\n\n"
        "Bar chart comparing h\u00b2 at maximum follow-up across relationship kinds, "
        "grouped by birth cohort. Left panel: disorder 1, right panel: disorder 2. "
        "Error bars show 95% CIs. Horizontal dashed line marks the true h\u00b2."
    ),
    "gc_bar": (
        "Figure 6: Genetic correlation at maximum follow-up.\n\n"
        "Bar chart of the genetic correlation estimate (\u03c1g) at maximum follow-up "
        "across relationship kinds, grouped by birth cohort. Error bars show 95% CIs. "
        "Horizontal dashed line marks the true genetic correlation."
    ),
    "summary_table": (
        "Figure 7: Summary comparison table.\n\n"
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


def tmax_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return the row at maximum time per born_at_year."""
    if df.empty:
        return df
    idx = df.groupby("born_at_year")["time"].idxmax()
    return df.loc[idx].sort_values("born_at_year")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_cif_comparison(
    tsv_dir: Path, kinds: list[str], disorder: str, output_path: Path,
) -> None:
    """CIF curves overlaid across relationship kinds, one panel per born_at_year."""
    exposed_cohort = "c2" if disorder == "d1" else "c3"

    # Load c1 from first kind (identical across kinds)
    c1 = load_cif(tsv_dir, disorder, "c1", kinds[0])
    if c1.empty:
        return

    years = sorted(c1["born_at_year"].unique())
    n_cols = len(years)
    fig, axes = plt.subplots(1, n_cols, figsize=(min(5 * n_cols, 16), 5), squeeze=False)

    for col, year in enumerate(years):
        ax = axes[0, col]

        # c1 reference (grey)
        c1y = c1[c1["born_at_year"] == year].sort_values("time")
        if not c1y.empty:
            ax.plot(c1y["time"], c1y["estimate"], color="0.5", linewidth=1.5, label="c1 (base)")
            ax.fill_between(c1y["time"], c1y["l95"], c1y["u95"], color="0.5", alpha=0.1)

        # Exposed cohort per kind
        for kind in kinds:
            c_exp = load_cif(tsv_dir, disorder, exposed_cohort, kind)
            if c_exp.empty:
                continue
            c_ey = c_exp[c_exp["born_at_year"] == year].sort_values("time")
            if c_ey.empty:
                continue
            color = KIND_COLORS.get(kind, "black")
            label = KIND_LABELS.get(kind, kind)
            ax.plot(c_ey["time"], c_ey["estimate"], color=color, linestyle="--",
                    linewidth=1.5, label=f"{exposed_cohort} ({label})")
            ax.fill_between(c_ey["time"], c_ey["l95"], c_ey["u95"],
                            color=color, alpha=0.1)

        ax.set_title(f"born {year}")
        ax.set_xlabel("Follow-up time")
        if col == 0:
            ax.set_ylabel("Cumulative Incidence")

    # Single legend on last panel
    axes[0, -1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"CIF Comparison \u2014 {disorder.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_h2_by_time(
    tsv_dir: Path, kinds: list[str], disorder: str,
    true_h2: float | None, output_path: Path,
) -> None:
    """h2 vs follow-up time, one panel per born_at_year, lines colored by kind."""
    # Determine born_at_year values from first available kind
    ref = load_h2(tsv_dir, disorder, kinds[0])
    if ref.empty:
        return

    years = sorted(ref["born_at_year"].unique())
    n_cols = len(years)
    fig, axes = plt.subplots(1, n_cols, figsize=(min(5 * n_cols, 16), 5), squeeze=False)

    for col, year in enumerate(years):
        ax = axes[0, col]

        for kind in kinds:
            h2 = load_h2(tsv_dir, disorder, kind)
            if h2.empty:
                continue
            h2y = h2[h2["born_at_year"] == year].sort_values("time")
            if h2y.empty:
                continue
            color = KIND_COLORS.get(kind, "black")
            label = KIND_LABELS.get(kind, kind)
            ax.plot(h2y["time"], h2y["h2"], color=color, linewidth=1.5, label=label)
            ax.fill_between(h2y["time"], h2y["l95"], h2y["u95"],
                            color=color, alpha=0.1)

        if true_h2 is not None:
            ax.axhline(true_h2, color="black", linestyle="--", linewidth=1.5,
                        alpha=0.7, label=f"True h\u00b2 = {true_h2:.3f}")

        ax.set_title(f"born {year}")
        ax.set_xlabel("Follow-up time")
        if col == 0:
            ax.set_ylabel("h\u00b2")

    axes[0, -1].legend(fontsize=7, loc="best")
    fig.suptitle(f"Heritability over Follow-up \u2014 {disorder.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_h2_bar(
    tsv_dir: Path, kinds: list[str], true_params: dict | None, output_path: Path,
) -> None:
    """Bar chart of h2 at tmax, grouped by born_at_year, two panels (d1, d2)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col, disorder in enumerate(["d1", "d2"]):
        ax = axes[col]
        true_key = f"h2_trait{col + 1}_true"
        true_h2 = true_params.get(true_key) if true_params else None

        # Collect tmax data per kind
        rows = []
        for kind in kinds:
            h2 = load_h2(tsv_dir, disorder, kind)
            if h2.empty:
                continue
            tm = tmax_rows(h2)
            for _, r in tm.iterrows():
                rows.append({
                    "kind": kind, "born_at_year": int(r["born_at_year"]),
                    "h2": r["h2"], "l95": r["l95"], "u95": r["u95"],
                })

        if not rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        df = pd.DataFrame(rows)
        years = sorted(df["born_at_year"].unique())
        present_kinds = [k for k in kinds if k in df["kind"].values]
        n_kinds = len(present_kinds)
        bar_width = 0.8 / max(n_kinds, 1)

        for i, kind in enumerate(present_kinds):
            dk = df[df["kind"] == kind]
            x_positions = []
            h2_vals = []
            err_lo = []
            err_hi = []
            for j, year in enumerate(years):
                row = dk[dk["born_at_year"] == year]
                if row.empty:
                    continue
                row = row.iloc[0]
                x_positions.append(j + (i - n_kinds / 2 + 0.5) * bar_width)
                h2_vals.append(row["h2"])
                err_lo.append(max(0, row["h2"] - row["l95"]))
                err_hi.append(max(0, row["u95"] - row["h2"]))

            color = KIND_COLORS.get(kind, "black")
            label = KIND_LABELS.get(kind, kind)
            ax.bar(x_positions, h2_vals, width=bar_width, color=color,
                   label=label, alpha=0.8, zorder=3)
            ax.errorbar(x_positions, h2_vals, yerr=[err_lo, err_hi],
                        fmt="none", color="black", capsize=3, linewidth=0.8, zorder=4)

        if true_h2 is not None:
            ax.axhline(true_h2, color="black", linestyle="--", linewidth=1.5,
                        alpha=0.7, label=f"True = {true_h2:.3f}")

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels([str(y) for y in years])
        ax.set_xlabel("Birth cohort")
        ax.set_ylabel("h\u00b2")
        ax.set_title(f"Disorder {col + 1}")

    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Heritability at Maximum Follow-up", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_gc_bar(
    tsv_dir: Path, kinds: list[str], true_params: dict | None, output_path: Path,
) -> None:
    """Bar chart of rhog at tmax across kinds, grouped by born_at_year."""
    true_gc = true_params.get("genetic_correlation_true") if true_params else None

    rows = []
    for kind in kinds:
        gc = load_gc(tsv_dir, kind)
        if gc.empty:
            continue
        tm = tmax_rows(gc)
        for _, r in tm.iterrows():
            rows.append({
                "kind": kind, "born_at_year": int(r["born_at_year"]),
                "rhog": r["rhog"], "l95": r["l95"], "u95": r["u95"],
            })

    fig, ax = plt.subplots(figsize=(10, 6))

    if not rows:
        ax.text(0.5, 0.5, "No genetic correlation data", ha="center",
                va="center", transform=ax.transAxes)
    else:
        df = pd.DataFrame(rows)
        years = sorted(df["born_at_year"].unique())
        present_kinds = [k for k in kinds if k in df["kind"].values]
        n_kinds = len(present_kinds)
        bar_width = 0.8 / max(n_kinds, 1)

        for i, kind in enumerate(present_kinds):
            dk = df[df["kind"] == kind]
            x_positions = []
            rhog_vals = []
            err_lo = []
            err_hi = []
            for j, year in enumerate(years):
                row = dk[dk["born_at_year"] == year]
                if row.empty:
                    continue
                row = row.iloc[0]
                x_positions.append(j + (i - n_kinds / 2 + 0.5) * bar_width)
                rhog_vals.append(row["rhog"])
                err_lo.append(max(0, row["rhog"] - row["l95"]))
                err_hi.append(max(0, row["u95"] - row["rhog"]))

            color = KIND_COLORS.get(kind, "black")
            label = KIND_LABELS.get(kind, kind)
            ax.bar(x_positions, rhog_vals, width=bar_width, color=color,
                   label=label, alpha=0.8, zorder=3)
            ax.errorbar(x_positions, rhog_vals, yerr=[err_lo, err_hi],
                        fmt="none", color="black", capsize=3, linewidth=0.8, zorder=4)

        if true_gc is not None:
            ax.axhline(true_gc, color="black", linestyle="--", linewidth=1.5,
                        alpha=0.7, label=f"True = {true_gc:.3f}")

        ax.set_xticks(range(len(years)))
        ax.set_xticklabels([str(y) for y in years])
        ax.set_xlabel("Birth cohort")
        ax.set_ylabel("Genetic correlation (\u03c1g)")
        ax.legend(fontsize=8, loc="best")

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

    for kind in kinds:
        label = KIND_LABELS.get(kind, kind)

        # c2 size from cif_d1_c2 (max cases at tmax)
        c2 = load_cif(tsv_dir, "d1", "c2", kind)
        c2_n = int(c2["cases"].max()) if not c2.empty and "cases" in c2.columns else "N/A"

        # c3 size from cif_d2_c3
        c3 = load_cif(tsv_dir, "d2", "c3", kind)
        c3_n = int(c3["cases"].max()) if not c3.empty and "cases" in c3.columns else "N/A"

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
        # Convert matplotlib color to rgba for cell background tint
        from matplotlib.colors import to_rgba
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
    plot_cif_comparison(tsv_dir, kinds, "d1", plots_dir / "cif_d1.png")
    plot_cif_comparison(tsv_dir, kinds, "d2", plots_dir / "cif_d2.png")

    true_h2_d1 = true_params.get("h2_trait1_true") if true_params else None
    true_h2_d2 = true_params.get("h2_trait2_true") if true_params else None
    plot_h2_by_time(tsv_dir, kinds, "d1", true_h2_d1, plots_dir / "h2_time_d1.png")
    plot_h2_by_time(tsv_dir, kinds, "d2", true_h2_d2, plots_dir / "h2_time_d2.png")

    plot_h2_bar(tsv_dir, kinds, true_params, plots_dir / "h2_bar.png")
    plot_gc_bar(tsv_dir, kinds, true_params, plots_dir / "gc_bar.png")
    plot_summary_table(tsv_dir, kinds, true_params, plots_dir / "summary_table.png")

    # Assemble atlas
    plot_paths = [plots_dir / f"{name}.png" for name in _PLOT_BASENAMES]
    section_breaks = {
        0: ("Cumulative Incidence", "CIF curves across relationship kinds"),
        2: ("Heritability", "h\u00b2 estimates by follow-up time and at maximum follow-up"),
        5: ("Genetic Correlation", "Cross-trait genetic correlation at maximum follow-up"),
        6: ("Summary", "Comparison across relationship kinds"),
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
