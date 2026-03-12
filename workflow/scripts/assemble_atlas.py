"""Assemble scenario plot atlas - Snakemake wrapper with CLI fallback."""
from pathlib import Path

import yaml

from sim_ace import setup_logging
from sim_ace.plot_atlas import (
    assemble_atlas,
    PHENOTYPE_CAPTIONS,
    THRESHOLD_CAPTIONS,
)

import logging
logger = logging.getLogger(__name__)


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])   
    p = snakemake.params                        

    frailty_paths   = [Path(x) for x in snakemake.input.frailty]    
    threshold_paths = [Path(x) for x in snakemake.input.threshold]  
    output_path     = Path(snakemake.output[0])                      

    with open(snakemake.input.params_yaml, encoding="utf-8") as fh:  
        scenario_params = yaml.safe_load(fh)

    # Merge in config-level parameters not present in params.yaml
    extra_keys = [
        "scenario", "replicates", "folder",
        "beta1", "beta_sex1", "phenotype_model1", "phenotype_params1",
        "beta2", "beta_sex2", "phenotype_model2", "phenotype_params2",
        "standardize", "censor_age", "gen_censoring",
        "death_scale", "death_rho", "prevalence1", "prevalence2",
        "G_pheno", "plot_format",
    ]
    for key in extra_keys:
        val = getattr(p, key, None)
        if val is not None:
            scenario_params[key] = val

    captions  = {**PHENOTYPE_CAPTIONS, **THRESHOLD_CAPTIONS}
    all_paths = frailty_paths + threshold_paths

    section_breaks = {
        9: (
            "Frailty Phenotype",
            "Survival-time phenotyping, censoring, and correlation analysis",
        ),
    }
    if threshold_paths:
        section_breaks[len(frailty_paths)] = (
            "Liability Threshold Phenotype",
            "Binary affected status from liability threshold model",
        )

    assemble_atlas(
        all_paths, captions, output_path,
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

        captions  = {**PHENOTYPE_CAPTIONS, **THRESHOLD_CAPTIONS}
        all_paths = [Path(x) for x in args.plots]
        assemble_atlas(
            all_paths, captions, Path(args.output),
            scenario_params=scenario_params,
        )
    else:
        _run_snakemake()