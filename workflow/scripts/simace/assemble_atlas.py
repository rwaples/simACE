"""Assemble scenario plot atlas - Snakemake wrapper with CLI fallback."""

import logging
from pathlib import Path

import yaml

from simace import _snakemake_tag, setup_logging
from simace.plotting.atlas_manifest import build_phenotype_atlas
from simace.plotting.plot_atlas import assemble_atlas

logger = logging.getLogger(__name__)


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    p = snakemake.params

    phenotype_paths = [Path(x) for x in snakemake.input.phenotype]
    output_path = Path(snakemake.output[0])
    plot_dir = phenotype_paths[0].parent

    with open(snakemake.input.params_yaml, encoding="utf-8") as fh:
        scenario_params = yaml.safe_load(fh)

    # Merge in config-level parameters not present in params.yaml
    extra_keys = [
        "scenario",
        "replicates",
        "folder",
        "beta1",
        "beta_sex1",
        "phenotype_model1",
        "phenotype_params1",
        "beta2",
        "beta_sex2",
        "phenotype_model2",
        "phenotype_params2",
        "standardize",
        "censor_age",
        "gen_censoring",
        "death_scale",
        "death_rho",
        "G_pheno",
        "N_sample",
        "pedigree_dropout_rate",
        "case_ascertainment_ratio",
        "max_degree",
        "plot_format",
    ]
    for key in extra_keys:
        val = getattr(p, key, None)
        if val is not None:
            scenario_params[key] = val

    plot_ext = scenario_params.get("plot_format", "png")
    items = build_phenotype_atlas(scenario_params)

    # Load per-rep phenotype stats for Table 1
    all_stats = []
    for stats_path in snakemake.input.stats:
        with open(stats_path, encoding="utf-8") as fh:
            all_stats.append(yaml.safe_load(fh))

    assemble_atlas(
        items,
        plot_dir,
        output_path,
        plot_ext=plot_ext,
        scenario_params=scenario_params,
        stats_data=all_stats,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        import argparse

        from simace.core.cli_base import add_logging_args, init_logging

        parser = argparse.ArgumentParser(description="Assemble scenario plot atlas")
        add_logging_args(parser)
        parser.add_argument("--plot-dir", required=True, help="Directory containing the plot PNGs")
        parser.add_argument("--params-yaml", default=None, help="Scenario params.yaml for title page")
        parser.add_argument("--stats", nargs="*", default=[], help="phenotype_stats.yaml paths (one per rep)")
        parser.add_argument("--scenario", default="unknown", help="Scenario name")
        parser.add_argument("--output", required=True, help="Output PDF path")
        parser.add_argument("--plot-ext", default="png", help="Plot file extension (default: png)")
        args = parser.parse_args()
        init_logging(args)

        scenario_params = None
        if args.params_yaml:
            with open(args.params_yaml, encoding="utf-8") as fh:
                scenario_params = yaml.safe_load(fh)
            scenario_params["scenario"] = args.scenario

        all_stats = []
        for sp in args.stats:
            with open(sp, encoding="utf-8") as fh:
                all_stats.append(yaml.safe_load(fh))

        items = build_phenotype_atlas(scenario_params)
        assemble_atlas(
            items,
            Path(args.plot_dir),
            Path(args.output),
            plot_ext=args.plot_ext,
            scenario_params=scenario_params,
            stats_data=all_stats or None,
        )
    else:
        _run_snakemake()
