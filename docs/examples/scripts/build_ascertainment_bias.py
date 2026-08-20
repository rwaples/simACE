#!/usr/bin/env python3
"""Build figures for docs/examples/ascertainment-bias.md.

Run from the repository root after generating the four ascertainment example
scenarios, for example:

    snakemake --cores 4 results/examples/ascertainment_uniform50k/rep1/report.yaml
    python docs/examples/scripts/build_ascertainment_bias.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from simace.core.parquet import load_parquet

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results" / "examples"
OUT_DIR = REPO_ROOT / "docs" / "images" / "examples" / "ascertainment"
CONFIGURED_PREVALENCE = 0.10
UNIFORM_SCENARIO = "ascertainment_uniform50k"
RELATIONSHIP_TYPES = ("PO", "FS", "MHS", "PHS", "GP")


@dataclass(frozen=True)
class Scenario:
    """One docs-example ascertainment scenario."""

    name: str
    label: str
    dropout_rate: float
    case_ratio: float


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("ascertainment_uniform50k", "Uniform\n50K", 0.0, 1.0),
    Scenario("ascertainment_dropout30_50k", "Dropout\n30%", 0.3, 1.0),
    Scenario("ascertainment_case5x_50k", "Case\n5x", 0.0, 5.0),
    Scenario("ascertainment_dropout30_case5x_50k", "Dropout 30%\n+ case 5x", 0.3, 5.0),
)


COLORS = {
    "trait": "#4C78A8",
    "pedigree": "#72B7B2",
    "full": "#B0B0B0",
    "case": "#F58518",
    "neutral": "#4C78A8",
}


def _require(path: Path) -> Path:
    if path.exists():
        return path
    targets = " ".join(f"results/examples/{scenario.name}/rep1/report.yaml" for scenario in SCENARIOS)
    raise FileNotFoundError(
        f"Required file is missing: {path}\nGenerate the example outputs first, e.g.:\n  snakemake --cores 4 {targets}"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _relationship_counts(stats: dict[str, Any]) -> dict[str, int]:
    # v2 report (ADR 0008): within-sample relationship pair counts live under
    # observed.analysis_sample.
    counts = stats.get("observed", {}).get("analysis_sample", {}).get("relationship_pair_counts", {})
    if not isinstance(counts, dict):
        return {}
    out = {str(k): int(v) for k, v in counts.items()}
    out["PO"] = out.get("MO", 0) + out.get("FO", 0)
    return out


def load_metrics() -> pl.DataFrame:
    """Load per-scenario values used by all three figures."""
    rows: list[dict[str, Any]] = []
    for idx, scenario in enumerate(SCENARIOS):
        rep_dir = RESULTS_ROOT / scenario.name / "rep1"
        trait_path = _require(rep_dir / "trait.parquet")
        pedigree_path = _require(rep_dir / "pedigree.parquet")
        full_pedigree_path = _require(rep_dir / "pedigree.full.parquet")
        stats_path = _require(rep_dir / "report.yaml")

        trait = load_parquet(trait_path, columns=["affected1"])
        pedigree = load_parquet(pedigree_path, columns=["id"])
        full_pedigree = load_parquet(full_pedigree_path, columns=["id"])
        stats = _read_yaml(stats_path)
        rel_counts = _relationship_counts(stats)

        row: dict[str, Any] = {
            "scenario": scenario.name,
            "label": scenario.label,
            "order": idx,
            "dropout_rate": scenario.dropout_rate,
            "case_ratio": scenario.case_ratio,
            "affected_fraction": float(trait["affected1"].mean()),
            "trait_rows": len(trait),
            "pedigree_rows": len(pedigree),
            "full_pedigree_rows": len(full_pedigree),
            "closure_ratio": float(len(pedigree) / len(trait)) if len(trait) else np.nan,
        }
        for rel in RELATIONSHIP_TYPES:
            row[f"rel_{rel}"] = rel_counts.get(rel, 0)
        rows.append(row)
    return pl.DataFrame(rows).sort("order")


def _style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_case_fraction(df: pl.DataFrame) -> None:
    """Plot sampled Trait 1 affected fraction by scenario."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    colors = [COLORS["case"] if r > 1 else COLORS["neutral"] for r in df["case_ratio"]]
    bars = ax.bar(df["label"], df["affected_fraction"], color=colors, width=0.68)

    ax.axhline(
        CONFIGURED_PREVALENCE,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        label="configured prevalence K = 0.10",
    )
    baseline = float(df.filter(pl.col("scenario") == UNIFORM_SCENARIO)["affected_fraction"][0])
    ax.axhline(
        baseline,
        color="#4C78A8",
        linestyle=":",
        linewidth=1.1,
        label=f"realized uniform sample = {baseline:.3f}",
    )

    for bar, value in zip(bars, df["affected_fraction"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.01,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_title("Case weighting enriches the sampled trait table")
    ax.set_ylabel("Trait 1 affected fraction in trait.parquet")
    ax.set_ylim(0, max(0.45, float(df["affected_fraction"].max()) + 0.06))
    ax.legend(frameon=False, loc="upper left")
    _style_axes(ax)
    fig.savefig(OUT_DIR / "case_fraction.png", dpi=200)
    plt.close(fig)


def plot_sample_sizes(df: pl.DataFrame) -> None:
    """Plot trait rows, pedigree rows, and ancestor-closure expansion."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    x = np.arange(len(df))
    width = 0.25

    axes[0].bar(x - width, df["trait_rows"], width, label="sampled trait rows", color=COLORS["trait"])
    axes[0].bar(x, df["pedigree_rows"], width, label="ancestor-closure pedigree rows", color=COLORS["pedigree"])
    axes[0].bar(x + width, df["full_pedigree_rows"], width, label="source pedigree rows", color=COLORS["full"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["label"], rotation=0)
    axes[0].set_ylabel("Rows")
    axes[0].set_title("A fixed 50K trait sample still has a pedigree closure")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].yaxis.set_major_formatter(lambda val, _pos: f"{val / 1000:.0f}K")
    _style_axes(axes[0])

    bars = axes[1].bar(df["label"], df["closure_ratio"], color=COLORS["pedigree"], width=0.68)
    for bar, value in zip(bars, df["closure_ratio"], strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.2f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[1].set_ylabel("pedigree rows / trait rows")
    axes[1].set_title("Closure expansion varies after dropout")
    axes[1].set_ylim(0, max(1.2, float(df["closure_ratio"].max()) + 0.25))
    _style_axes(axes[1])

    fig.savefig(OUT_DIR / "sample_sizes.png", dpi=200)
    plt.close(fig)


def plot_relationship_pairs(df: pl.DataFrame) -> None:
    """Plot relationship-pair counts relative to uniform 50K sampling."""
    baseline = df.filter(pl.col("scenario") == UNIFORM_SCENARIO).row(0, named=True)
    rows = []
    for row in df.iter_rows(named=True):
        for rel in RELATIONSHIP_TYPES:
            denom = int(baseline[f"rel_{rel}"])
            value = int(row[f"rel_{rel}"])
            rows.append(
                {
                    "scenario": row["scenario"],
                    "label": row["label"],
                    "relationship": rel,
                    "relative_count": value / denom if denom else np.nan,
                }
            )
    rel_df = pl.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    x = np.arange(len(RELATIONSHIP_TYPES))
    width = 0.2
    plotted = [s for s in SCENARIOS if s.name != UNIFORM_SCENARIO]
    offsets = np.linspace(-width, width, len(plotted))
    colors = ["#72B7B2", "#F58518", "#E45756"]

    for offset, scenario, color in zip(offsets, plotted, colors, strict=True):
        vals = [
            float(
                rel_df.filter((pl.col("scenario") == scenario.name) & (pl.col("relationship") == rel))[
                    "relative_count"
                ][0]
            )
            for rel in RELATIONSHIP_TYPES
        ]
        ax.bar(x + offset, vals, width, label=scenario.label.replace("\n", " "), color=color)

    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1.0)
    ax.text(
        len(RELATIONSHIP_TYPES) - 0.45,
        1.02,
        "uniform 50K baseline",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(RELATIONSHIP_TYPES)
    ax.set_ylabel("Pair count relative to uniform 50K")
    ax.set_title("Ascertainment changes relationship evidence at fixed sample size")
    ax.set_ylim(0, max(1.25, float(rel_df["relative_count"].max()) + 0.15))
    ax.legend(frameon=False, ncol=1, loc="upper left")
    _style_axes(ax)
    fig.savefig(OUT_DIR / "relationship_pairs.png", dpi=200)
    plt.close(fig)


def main() -> None:
    """Build all ascertainment-bias documentation figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    plot_case_fraction(metrics)
    plot_sample_sizes(metrics)
    plot_relationship_pairs(metrics)
    print(f"Wrote figures to {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
