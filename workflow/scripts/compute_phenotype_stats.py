"""Compute phenotype statistics - Snakemake wrapper with CLI fallback."""

from sim_ace import setup_logging
from sim_ace.stats import cli as _cli
from sim_ace.stats import main


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])
    p = snakemake.params

    gen_censoring_raw = p.get("gen_censoring", None)
    gen_censoring = {int(k): v for k, v in gen_censoring_raw.items()} if gen_censoring_raw else None

    from sim_ace.phenotype import _FRAILTY_MODELS

    pm1, pm2 = p.phenotype_model1, p.phenotype_model2
    frailty_params = {
        "trait1": {
            "beta": p.beta1,
            "hazard_model": pm1,
            "hazard_params": p.phenotype_params1,
        }
        if pm1 in _FRAILTY_MODELS
        else {},
        "trait2": {
            "beta": p.beta2,
            "hazard_model": pm2,
            "hazard_params": p.phenotype_params2,
        }
        if pm2 in _FRAILTY_MODELS
        else {},
    }

    main(
        snakemake.input.phenotype,
        p.censor_age,
        snakemake.output.stats,
        snakemake.output.samples,
        seed=p.seed,
        gen_censoring=gen_censoring,
        frailty_params=frailty_params,
        extra_tetrachoric=p.get("extra_tetrachoric", True),
        pedigree_path=snakemake.input.pedigree,
        skip_2nd_cousins=p.get("skip_2nd_cousins", True),
        case_ascertainment_ratio=p.get("case_ascertainment_ratio", 1.0),
    )


if __name__ == "__main__":
    try:
        snakemake
    except NameError:
        _cli()
    else:
        _run_snakemake()
