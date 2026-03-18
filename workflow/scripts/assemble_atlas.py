"""Assemble scenario plot atlas - Snakemake wrapper with CLI fallback."""

import logging
from pathlib import Path

import yaml

from sim_ace import setup_logging
from sim_ace.plot_atlas import (
    PHENOTYPE_CAPTIONS,
    SIMPLE_LTM_CAPTIONS,
    assemble_atlas,
    get_model_family,
)

logger = logging.getLogger(__name__)


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    p = snakemake.params

    phenotype_paths = [Path(x) for x in snakemake.input.phenotype]
    simple_ltm_paths = [Path(x) for x in snakemake.input.simple_ltm]
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
        "skip_2nd_cousins",
        "plot_format",
    ]
    for key in extra_keys:
        val = getattr(p, key, None)
        if val is not None:
            scenario_params[key] = val

    captions = {**PHENOTYPE_CAPTIONS, **SIMPLE_LTM_CAPTIONS}
    all_paths = phenotype_paths + simple_ltm_paths

    model_name, model_desc = get_model_family(scenario_params)
    section_breaks = {
        6: (
            f"{model_name} Phenotype",
            model_desc,
        ),
        8: (
            "Survival & Censoring",
            "Age-at-onset, mortality, cumulative incidence, and censoring analysis",
        ),
        17: (
            "Within-Trait Correlations",
            "Familial tetrachoric correlations and joint affected status",
        ),
        20: (
            "Cross-Trait Correlations",
            "Cross-trait correlation by generation and relationship type",
        ),
    }
    if simple_ltm_paths:
        section_breaks[len(phenotype_paths)] = (
            "Liability Threshold Phenotype",
            "Simple liability threshold on latent liability (no age-at-onset modeling)",
        )

    assemble_atlas(
        all_paths,
        captions,
        output_path,
        scenario_params=scenario_params,
        section_breaks=section_breaks,
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        import argparse

        from sim_ace.cli_base import add_logging_args, init_logging

        parser = argparse.ArgumentParser(description="Assemble scenario plot atlas")
        add_logging_args(parser)
        parser.add_argument("--plots", nargs="+", required=True, help="Plot image paths")
        parser.add_argument("--params-yaml", default=None, help="Scenario params.yaml for title page")
        parser.add_argument("--scenario", default="unknown", help="Scenario name")
        parser.add_argument("--output", required=True, help="Output PDF path")
        args = parser.parse_args()
        init_logging(args)

        scenario_params = None
        if args.params_yaml:
            with open(args.params_yaml, encoding="utf-8") as fh:
                scenario_params = yaml.safe_load(fh)
            scenario_params["scenario"] = args.scenario

        captions = {**PHENOTYPE_CAPTIONS, **SIMPLE_LTM_CAPTIONS}
        all_paths = [Path(x) for x in args.plots]
        assemble_atlas(
            all_paths,
            captions,
            Path(args.output),
            scenario_params=scenario_params,
        )
    else:
        _run_snakemake()
