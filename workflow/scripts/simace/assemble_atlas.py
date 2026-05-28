"""Assemble scenario plot atlas - Snakemake wrapper with CLI fallback."""

import logging
from pathlib import Path

from simace import _snakemake_tag, setup_logging
from simace.core.yaml_io import load_yaml
from simace.plotting.atlas_manifest import build_phenotype_atlas
from simace.plotting.plot_atlas import assemble_atlas
from simace.plotting.stats_report import merge_plot_payload, plotting_stats_views

logger = logging.getLogger(__name__)


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    p = snakemake.params

    phenotype_paths = [Path(x) for x in snakemake.input.phenotype]
    output_path = Path(snakemake.output[0])
    plot_dir = phenotype_paths[0].parent

    scenario_params = load_yaml(snakemake.input.params_yaml)

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
        "dropout_rate",
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

    # Load per-replicate combined reports for Table 1 (six stats groups read
    # via the view; the extra `validation` group is ignored). The dense
    # plot_payload arrays are merged back so Table 1 can derive onset quartiles.
    reports = [load_yaml(p) for p in snakemake.input.report]
    payloads = [load_yaml(p) for p in snakemake.input.plot_payload]
    all_stats = plotting_stats_views([merge_plot_payload(r, pl) for r, pl in zip(reports, payloads, strict=True)])

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
        parser.add_argument("--report", nargs="*", default=[], help="report.yaml paths (one per replicate)")
        parser.add_argument(
            "--plot-payload", nargs="*", default=[], help="plot_payload.yaml paths (one per replicate)"
        )
        parser.add_argument("--scenario", default="unknown", help="Scenario name")
        parser.add_argument("--output", required=True, help="Output PDF path")
        parser.add_argument("--plot-ext", default="png", help="Plot file extension (default: png)")
        args = parser.parse_args()
        init_logging(args)

        scenario_params = None
        if args.params_yaml:
            scenario_params = load_yaml(args.params_yaml)
            scenario_params["scenario"] = args.scenario

        reports = [load_yaml(rp) for rp in args.report]
        payloads = [load_yaml(pp) for pp in args.plot_payload] if args.plot_payload else [{}] * len(reports)
        all_stats = plotting_stats_views(
            [merge_plot_payload(r, pl) for r, pl in zip(reports, payloads, strict=True)]
        )

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
