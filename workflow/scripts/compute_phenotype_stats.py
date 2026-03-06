"""Compute phenotype statistics - Snakemake wrapper with CLI fallback."""
from sim_ace import setup_logging
from sim_ace.stats import main, cli as _cli


def _run_snakemake():
    setup_logging(log_file=snakemake.log[0])   
    p = snakemake.params                        

    gen_censoring_raw = p.get("gen_censoring", None)
    gen_censoring = {int(k): v for k, v in gen_censoring_raw.items()} if gen_censoring_raw else None

    frailty_params = {
        "trait1": {
            "beta":          p.beta1,
            "hazard_model":  p.hazard_model1,
            "hazard_params": p.hazard_params1,
        },
        "trait2": {
            "beta":          p.beta2,
            "hazard_model":  p.hazard_model2,
            "hazard_params": p.hazard_params2,
        },
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
    )


if __name__ == "__main__":
    try:
        snakemake   
    except NameError:
        _cli()
    else:
        _run_snakemake()