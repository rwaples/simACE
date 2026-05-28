"""Plot phenotype distributions from pre-computed per-replicate reports.

Reads the curated v2 report.yaml, the companion plot_payload.yaml (dense
incidence/censoring arrays), and plotting_sample.parquet files (one per
replicate). The report and payload are recombined into the flat plotting view
before plotting. No full trait parquet loading needed.
"""

__all__: list[str] = []

import argparse
import json
import logging
from pathlib import Path

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
    plot_broad_heritability_by_generation,
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

logger = logging.getLogger(__name__)

MAX_PLOT_POINTS = 200_000


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
        subsample_note = f"Subsampled: {MAX_PLOT_POINTS:,} of {original_n:,} individuals shown"

    ext = plot_ext

    # Resolved scenario parameters (from inputs.parameters) reconstructed into
    # the flat view; needed by the correlation/heritability reference lines.
    validation_params = all_stats[0].get("parameters", {})

    # Pedigree relationship pair counts
    plot_pedigree_relationship_counts(
        all_stats,
        out_dir / f"pedigree_counts.ped.{ext}",
        scenario,
        stats_key="pair_counts_ped",
        generations_label="G_ped",
        max_degree=max_degree,
    )
    plot_pedigree_relationship_counts(
        all_stats,
        out_dir / f"pedigree_counts.{ext}",
        scenario,
        generations_label="G_pheno",
        max_degree=max_degree,
    )

    # Family structure (offspring and mate distributions)
    plot_family_structure(
        all_stats,
        out_dir / f"family_structure.{ext}",
        scenario,
    )

    # Mate correlation heatmap
    plot_mate_correlation(
        all_stats,
        out_dir / f"mate_correlation.{ext}",
        scenario,
        params=validation_params,
    )

    # Distribution plots
    plot_death_age_distribution(
        all_stats,
        censor_age,
        out_dir / f"mortality.{ext}",
        scenario,
    )
    plot_trait_phenotype(
        df_samples,
        out_dir / f"age_at_onset_death.{ext}",
        scenario,
        subsample_note=subsample_note,
    )
    plot_trait_regression(
        df_samples,
        all_stats,
        out_dir / f"liability_vs_aoo.{ext}",
        scenario,
        subsample_note=subsample_note,
    )

    # Liability plots
    plot_liability_joint(
        df_samples,
        out_dir / f"cross_trait.{ext}",
        scenario,
        subsample_note=subsample_note,
    )
    plot_liability_joint_affected(
        df_samples,
        out_dir / f"cross_trait.phenotype.{ext}",
        scenario,
        subsample_note=subsample_note,
    )
    plot_liability_joint_affected_t2(
        df_samples,
        out_dir / f"cross_trait.phenotype.t2.{ext}",
        scenario,
        subsample_note=subsample_note,
    )
    plot_liability_violin(
        df_samples,
        all_stats,
        out_dir / f"liability_violin.phenotype.{ext}",
        scenario,
        subsample_note=subsample_note,
    )
    plot_liability_violin_by_generation(
        df_samples,
        all_stats,
        out_dir / f"liability_violin.phenotype.by_generation.{ext}",
        scenario,
        subsample_note=subsample_note,
    )
    plot_liability_violin_by_sex_generation(
        df_samples,
        all_stats,
        out_dir / f"liability_violin.phenotype.by_sex.by_generation.{ext}",
        scenario,
        subsample_note=subsample_note,
    )

    # Genetic selection by generation
    plot_liability_components_by_generation(
        df_samples,
        out_dir / f"liability_components.by_generation.{ext}",
        scenario,
        subsample_note=subsample_note,
    )

    # Survival / incidence plots
    plot_cumulative_incidence(
        all_stats,
        censor_age,
        out_dir / f"cumulative_incidence.phenotype.{ext}",
        scenario,
    )
    plot_cumulative_incidence_by_sex(
        all_stats,
        out_dir / f"cumulative_incidence.by_sex.{ext}",
        scenario,
    )
    plot_cumulative_incidence_by_sex_generation(
        all_stats,
        out_dir / f"cumulative_incidence.by_sex.by_generation.{ext}",
        scenario,
    )
    plot_cumulative_incidence_aj(
        all_stats,
        censor_age,
        out_dir / f"cumulative_incidence_aj.phenotype.{ext}",
        scenario,
    )
    plot_cumulative_incidence_aj_by_sex(
        all_stats,
        out_dir / f"cumulative_incidence_aj.by_sex.{ext}",
        scenario,
    )
    plot_cumulative_incidence_aj_by_sex_generation(
        all_stats,
        out_dir / f"cumulative_incidence_aj.by_sex.by_generation.{ext}",
        scenario,
    )
    plot_joint_affection(
        all_stats,
        out_dir / f"joint_affected.phenotype.{ext}",
        scenario,
    )

    # Censoring
    if gen_censoring is not None:
        plot_censoring_windows(
            all_stats,
            out_dir / f"censoring.{ext}",
            scenario,
            gen_censoring=gen_censoring,
        )
    else:
        save_placeholder_plot(out_dir / f"censoring.{ext}", "No censoring windows configured")

    plot_censoring_confusion(
        all_stats,
        out_dir / f"censoring_confusion.{ext}",
        scenario,
    )
    plot_censoring_cascade(
        all_stats,
        out_dir / f"censoring_cascade.{ext}",
        scenario,
    )

    # Correlation plots
    plot_tetrachoric_sibling(
        all_stats,
        out_dir / f"tetrachoric.phenotype.{ext}",
        scenario,
        params=validation_params,
    )
    plot_tetrachoric_by_sex(
        all_stats,
        out_dir / f"tetrachoric.phenotype.by_sex.{ext}",
        scenario,
        params=validation_params,
    )
    plot_tetrachoric_by_generation(
        all_stats,
        out_dir / f"tetrachoric.phenotype.by_generation.{ext}",
        scenario,
        params=validation_params,
    )
    plot_cross_trait_tetrachoric(
        all_stats,
        out_dir / f"cross_trait_tetrachoric.{ext}",
        scenario,
    )
    plot_parent_offspring_liability(
        df_samples,
        all_stats,
        out_dir / f"parent_offspring_liability.by_generation.{ext}",
        scenario,
        subsample_note=subsample_note,
        params=validation_params,
    )
    # Per-generation heritability (from truth.realized_by_generation, exposed as
    # `per_generation` in the view). Both helpers fall back to a placeholder
    # when no per-generation data is present.
    plot_heritability_by_generation(
        all_stats,
        out_dir / f"heritability.by_generation.{ext}",
        scenario,
    )
    plot_broad_heritability_by_generation(
        all_stats,
        out_dir / f"additive_shared.by_generation.{ext}",
        scenario,
    )

    # PO-regression heritability by sex
    plot_heritability_by_sex_generation(
        all_stats,
        out_dir / f"heritability.by_sex.by_generation.{ext}",
        scenario,
        params=validation_params,
    )

    # Observed-scale heritability from binary affected status + Dempster-Lerner lift
    plot_observed_heritability(
        all_stats,
        out_dir / f"observed_h2.{ext}",
        scenario,
        params=validation_params,
    )

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
