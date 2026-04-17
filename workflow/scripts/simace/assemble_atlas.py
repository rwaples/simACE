"""Assemble scenario plot atlas - Snakemake wrapper with CLI fallback."""

import logging
from pathlib import Path

import yaml

from simace import _snakemake_tag, setup_logging
from simace.plotting.plot_atlas import (
    PHENOTYPE_CAPTIONS,
    SIMPLE_LTM_CAPTIONS,
    assemble_atlas,
    get_model_equation,
    get_model_family,
)

logger = logging.getLogger(__name__)


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))
    p = snakemake.params

    phenotype_paths = [Path(x) for x in snakemake.input.phenotype]
    simple_ltm_paths = []
    output_path = Path(snakemake.output[0])

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
        "prevalence1",
        "prevalence2",
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

    captions = {**PHENOTYPE_CAPTIONS, **SIMPLE_LTM_CAPTIONS}
    all_paths = phenotype_paths + simple_ltm_paths

    model_name, model_desc = get_model_family(scenario_params)
    model_equations = get_model_equation(scenario_params)
    section_breaks = {
        9: (
            f"{model_name} Phenotype",
            model_desc,
            model_equations,
        ),
        13: (
            "Age of Onset & Censoring",
            "Age-at-onset, mortality, cumulative incidence, and censoring analysis",
        ),
        21: (
            "Within-Trait Correlations",
            "Familial tetrachoric correlations",
        ),
        24: (
            "Cross-Trait Correlations",
            "Cross-trait correlation by generation and relationship type",
        ),
    }
    if simple_ltm_paths:
        section_breaks[len(phenotype_paths)] = (
            "Liability Threshold Phenotype",
            "Simple liability threshold on latent liability (no age-at-onset modeling)",
            [r"$\mathrm{affected} = \mathbf{1}(L > \Phi^{-1}(1-K)), \qquad L = A + C + E$"],
        )

    # Load per-rep phenotype stats for Table 1
    all_stats = []
    for stats_path in snakemake.input.stats:
        with open(stats_path, encoding="utf-8") as fh:
            all_stats.append(yaml.safe_load(fh))

    assemble_atlas(
        all_paths,
        captions,
        output_path,
        scenario_params=scenario_params,
        section_breaks=section_breaks,
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
        parser.add_argument("--plots", nargs="+", required=True, help="Plot image paths")
        parser.add_argument("--params-yaml", default=None, help="Scenario params.yaml for title page")
        parser.add_argument("--stats", nargs="*", default=[], help="phenotype_stats.yaml paths (one per rep)")
        parser.add_argument("--scenario", default="unknown", help="Scenario name")
        parser.add_argument("--output", required=True, help="Output PDF path")
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

        captions = {**PHENOTYPE_CAPTIONS, **SIMPLE_LTM_CAPTIONS}
        all_paths = [Path(x) for x in args.plots]
        assemble_atlas(
            all_paths,
            captions,
            Path(args.output),
            scenario_params=scenario_params,
            stats_data=all_stats or None,
        )
    else:
        _run_snakemake()
