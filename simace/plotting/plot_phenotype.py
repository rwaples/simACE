"""Plot phenotype distributions from pre-computed per-replicate reports.

Reads the curated v2 report.yaml, the companion plot_payload.yaml (dense
incidence/censoring arrays), and plotting_sample.parquet files (one per
replicate). The report and payload are recombined into the flat plotting view
before plotting. No full trait parquet loading needed.
"""

from __future__ import annotations

__all__: list[str] = []

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from simace.core.yaml_io import load_yaml
from simace.plotting.plot_correlations import (
    plot_cross_trait_tetrachoric,
    plot_parent_offspring_liability,
    plot_tetrachoric_by_generation,
    plot_tetrachoric_by_sex,
    plot_tetrachoric_sibling,
)
from simace.plotting.plot_distributions import (
    plot_censoring_windows,
    plot_cumulative_incidence,
    plot_cumulative_incidence_aj,
    plot_cumulative_incidence_aj_by_sex,
    plot_cumulative_incidence_aj_by_sex_generation,
    plot_cumulative_incidence_by_sex,
    plot_cumulative_incidence_by_sex_generation,
    plot_death_age_distribution,
    plot_family_structure,
    plot_trait_phenotype,
    plot_trait_regression,
)
from simace.plotting.plot_heritability import (
    plot_heritability_by_generation,
    plot_heritability_by_sex_generation,
    plot_observed_heritability,
)
from simace.plotting.plot_liability import (
    plot_censoring_cascade,
    plot_censoring_confusion,
    plot_joint_affection,
    plot_liability_components_by_generation,
    plot_liability_joint,
    plot_liability_joint_affected,
    plot_liability_joint_affected_t2,
    plot_liability_violin,
    plot_liability_violin_by_generation,
    plot_liability_violin_by_sex_generation,
    plot_mate_correlation,
)
from simace.plotting.plot_pedigree_counts import plot_pedigree_relationship_counts
from simace.plotting.plot_utils import save_placeholder_plot
from simace.plotting.stats_report import plotting_report_views

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

MAX_PLOT_POINTS = 200_000


@dataclass(frozen=True)
class RenderContext:
    """Shared inputs handed to every phenotype plot renderer."""

    all_stats: list[dict]
    df_samples: pd.DataFrame
    scenario: str
    censor_age: float
    subsample_note: str
    params: dict  # resolved scenario parameters; drives reference lines
    gen_censoring: dict[int, list[float]] | None
    max_degree: int


@dataclass(frozen=True)
class PlotRenderSpec:
    """One phenotype plot: its output basename and how to render it.

    ``render`` receives the shared :class:`RenderContext` and the full output
    path (``<basename>.<ext>``) and writes the figure. The set of basenames is
    kept in lockstep with
    :data:`simace.plotting.atlas_manifest.PHENOTYPE_ATLAS` by the
    renderer-coverage test in ``tests/plotting/test_atlas_manifest.py``.
    """

    basename: str
    render: Callable[[RenderContext, Path], None]


def _render_censoring(ctx: RenderContext, path: Path) -> None:
    """Per-generation censoring windows, or a placeholder when none configured."""
    if ctx.gen_censoring is not None:
        plot_censoring_windows(ctx.all_stats, path, ctx.scenario, gen_censoring=ctx.gen_censoring)
    else:
        save_placeholder_plot(path, "No censoring windows configured")


# Ordered registry binding each phenotype basename to its renderer. Adding a
# plot means adding a PlotEntry to PHENOTYPE_ATLAS *and* a spec here; the
# renderer-coverage test fails if the two basename sets diverge.
PHENOTYPE_RENDERERS: tuple[PlotRenderSpec, ...] = (
    # Pedigree relationship pair counts
    PlotRenderSpec(
        "pedigree_counts.ped",
        lambda ctx, p: plot_pedigree_relationship_counts(
            ctx.all_stats,
            p,
            ctx.scenario,
            stats_key="pair_counts_ped",
            generations_label="G_ped",
            max_degree=ctx.max_degree,
        ),
    ),
    PlotRenderSpec(
        "pedigree_counts",
        lambda ctx, p: plot_pedigree_relationship_counts(
            ctx.all_stats,
            p,
            ctx.scenario,
            generations_label="G_pheno",
            max_degree=ctx.max_degree,
        ),
    ),
    # Family structure
    PlotRenderSpec(
        "family_structure",
        lambda ctx, p: plot_family_structure(ctx.all_stats, p, ctx.scenario),
    ),
    PlotRenderSpec(
        "mate_correlation",
        lambda ctx, p: plot_mate_correlation(ctx.all_stats, p, ctx.scenario, params=ctx.params),
    ),
    # Distribution plots
    PlotRenderSpec(
        "mortality",
        lambda ctx, p: plot_death_age_distribution(
            ctx.all_stats,
            ctx.censor_age,
            p,
            ctx.scenario,
            df_samples=ctx.df_samples,
            subsample_note=ctx.subsample_note,
        ),
    ),
    PlotRenderSpec(
        "age_at_onset_death",
        lambda ctx, p: plot_trait_phenotype(ctx.df_samples, p, ctx.scenario, subsample_note=ctx.subsample_note),
    ),
    PlotRenderSpec(
        "liability_vs_aoo",
        lambda ctx, p: plot_trait_regression(
            ctx.df_samples, ctx.all_stats, p, ctx.scenario, subsample_note=ctx.subsample_note
        ),
    ),
    # Liability plots
    PlotRenderSpec(
        "cross_trait",
        lambda ctx, p: plot_liability_joint(ctx.df_samples, p, ctx.scenario, subsample_note=ctx.subsample_note),
    ),
    PlotRenderSpec(
        "cross_trait.phenotype",
        lambda ctx, p: plot_liability_joint_affected(
            ctx.df_samples, p, ctx.scenario, subsample_note=ctx.subsample_note
        ),
    ),
    PlotRenderSpec(
        "cross_trait.phenotype.t2",
        lambda ctx, p: plot_liability_joint_affected_t2(
            ctx.df_samples, p, ctx.scenario, subsample_note=ctx.subsample_note
        ),
    ),
    PlotRenderSpec(
        "liability_violin.phenotype",
        lambda ctx, p: plot_liability_violin(
            ctx.df_samples, ctx.all_stats, p, ctx.scenario, subsample_note=ctx.subsample_note
        ),
    ),
    PlotRenderSpec(
        "liability_violin.phenotype.by_generation",
        lambda ctx, p: plot_liability_violin_by_generation(
            ctx.df_samples, ctx.all_stats, p, ctx.scenario, subsample_note=ctx.subsample_note
        ),
    ),
    PlotRenderSpec(
        "liability_violin.phenotype.by_sex.by_generation",
        lambda ctx, p: plot_liability_violin_by_sex_generation(
            ctx.df_samples, ctx.all_stats, p, ctx.scenario, subsample_note=ctx.subsample_note
        ),
    ),
    # Genetic selection by generation
    PlotRenderSpec(
        "liability_components.by_generation",
        lambda ctx, p: plot_liability_components_by_generation(
            ctx.df_samples, p, ctx.scenario, subsample_note=ctx.subsample_note
        ),
    ),
    # Survival / incidence plots
    PlotRenderSpec(
        "cumulative_incidence.phenotype",
        lambda ctx, p: plot_cumulative_incidence(ctx.all_stats, ctx.censor_age, p, ctx.scenario),
    ),
    PlotRenderSpec(
        "cumulative_incidence.by_sex",
        lambda ctx, p: plot_cumulative_incidence_by_sex(ctx.all_stats, p, ctx.scenario),
    ),
    PlotRenderSpec(
        "cumulative_incidence.by_sex.by_generation",
        lambda ctx, p: plot_cumulative_incidence_by_sex_generation(ctx.all_stats, p, ctx.scenario),
    ),
    PlotRenderSpec(
        "cumulative_incidence_aj.phenotype",
        lambda ctx, p: plot_cumulative_incidence_aj(ctx.all_stats, ctx.censor_age, p, ctx.scenario),
    ),
    PlotRenderSpec(
        "cumulative_incidence_aj.by_sex",
        lambda ctx, p: plot_cumulative_incidence_aj_by_sex(ctx.all_stats, p, ctx.scenario),
    ),
    PlotRenderSpec(
        "cumulative_incidence_aj.by_sex.by_generation",
        lambda ctx, p: plot_cumulative_incidence_aj_by_sex_generation(ctx.all_stats, p, ctx.scenario),
    ),
    PlotRenderSpec(
        "joint_affected.phenotype",
        lambda ctx, p: plot_joint_affection(ctx.all_stats, p, ctx.scenario),
    ),
    # Censoring
    PlotRenderSpec("censoring", _render_censoring),
    PlotRenderSpec(
        "censoring_confusion",
        lambda ctx, p: plot_censoring_confusion(ctx.all_stats, p, ctx.scenario),
    ),
    PlotRenderSpec(
        "censoring_cascade",
        lambda ctx, p: plot_censoring_cascade(ctx.all_stats, p, ctx.scenario),
    ),
    # Correlation plots
    PlotRenderSpec(
        "tetrachoric.phenotype",
        lambda ctx, p: plot_tetrachoric_sibling(ctx.all_stats, p, ctx.scenario, params=ctx.params),
    ),
    PlotRenderSpec(
        "tetrachoric.phenotype.by_sex",
        lambda ctx, p: plot_tetrachoric_by_sex(ctx.all_stats, p, ctx.scenario, params=ctx.params),
    ),
    PlotRenderSpec(
        "tetrachoric.phenotype.by_generation",
        lambda ctx, p: plot_tetrachoric_by_generation(ctx.all_stats, p, ctx.scenario, params=ctx.params),
    ),
    PlotRenderSpec(
        "cross_trait_tetrachoric",
        lambda ctx, p: plot_cross_trait_tetrachoric(ctx.all_stats, p, ctx.scenario),
    ),
    PlotRenderSpec(
        "parent_offspring_liability.by_generation",
        lambda ctx, p: plot_parent_offspring_liability(
            ctx.df_samples,
            ctx.all_stats,
            p,
            ctx.scenario,
            subsample_note=ctx.subsample_note,
            params=ctx.params,
        ),
    ),
    # Per-generation heritability
    PlotRenderSpec(
        "heritability.by_generation",
        lambda ctx, p: plot_heritability_by_generation(ctx.all_stats, p, ctx.scenario),
    ),
    # PO-regression heritability by sex
    PlotRenderSpec(
        "heritability.by_sex.by_generation",
        lambda ctx, p: plot_heritability_by_sex_generation(ctx.all_stats, p, ctx.scenario, params=ctx.params),
    ),
    # Observed-scale heritability
    PlotRenderSpec(
        "observed_h2",
        lambda ctx, p: plot_observed_heritability(ctx.all_stats, p, ctx.scenario, params=ctx.params),
    ),
)


def main(
    report_paths: list[str],
    plot_payload_paths: list[str],
    sample_paths: list[str],
    output_dir: str,
    censor_age: float,
    gen_censoring: dict[int, list[float]] | None = None,
    plot_ext: str = "png",
    max_degree: int = 2,
) -> None:
    """Generate all phenotype plots from pre-computed combined reports."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario = out_dir.parent.name
    from simace.plotting.plot_style import apply_nature_style

    apply_nature_style()

    reports = [load_yaml(p) for p in report_paths]
    payloads = [load_yaml(p) for p in plot_payload_paths]
    all_stats = plotting_report_views(reports, payloads)

    df_samples = pd.concat([pd.read_parquet(p) for p in sample_paths], ignore_index=True)
    subsample_note = ""
    if len(df_samples) > MAX_PLOT_POINTS:
        original_n = len(df_samples)
        df_samples = df_samples.sample(n=MAX_PLOT_POINTS, random_state=42).reset_index(drop=True)
        subsample_note = f"Plotting sample: {MAX_PLOT_POINTS:,} of {original_n:,} shown"

    # Resolved scenario parameters (from inputs.parameters) reconstructed into
    # the flat view; needed by the correlation/heritability reference lines.
    validation_params = all_stats[0].get("parameters", {})

    ctx = RenderContext(
        all_stats=all_stats,
        df_samples=df_samples,
        scenario=scenario,
        censor_age=censor_age,
        subsample_note=subsample_note,
        params=validation_params,
        gen_censoring=gen_censoring,
        max_degree=max_degree,
    )
    for spec in PHENOTYPE_RENDERERS:
        spec.render(ctx, out_dir / f"{spec.basename}.{plot_ext}")

    logger.info("Phenotype plots saved to %s", out_dir)


def cli() -> None:
    """Command-line interface for generating phenotype plots."""
    from simace.core.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Plot phenotype distributions")
    add_logging_args(parser)
    parser.add_argument("--report", nargs="+", required=True, help="report.yaml paths")
    parser.add_argument("--plot-payload", nargs="+", required=True, help="plot_payload.yaml paths")
    parser.add_argument("--samples", nargs="+", required=True, help="Sample parquet paths")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--censor-age", type=float, required=True, help="Maximum follow-up age")
    parser.add_argument("--gen-censoring", type=str, default=None, help="Per-generation censoring windows as JSON dict")
    parser.add_argument(
        "--plot-format", choices=["png", "pdf"], default="png", help="Output plot format (default: png)"
    )
    args = parser.parse_args()

    init_logging(args)

    gen_censoring = None
    if args.gen_censoring:
        gen_censoring = {int(k): v for k, v in json.loads(args.gen_censoring).items()}

    main(
        args.report,
        args.plot_payload,
        args.samples,
        args.output_dir,
        args.censor_age,
        gen_censoring=gen_censoring,
        plot_ext=args.plot_format,
    )
