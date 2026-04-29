# ---------------------------------------------------------------------------
# Phenotype simulation rules
# ---------------------------------------------------------------------------


rule phenotype:
    """Phenotype simulation with pluggable model."""
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        phenotype=temp("results/{folder}/{scenario}/rep{rep}/phenotype.raw.parquet"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        G_pheno=lambda w: get_param(config, w.scenario, "G_pheno"),
        standardize=lambda w: get_param(config, w.scenario, "standardize"),
        phenotype_model1=lambda w: get_param(config, w.scenario, "phenotype_model1"),
        phenotype_model2=lambda w: get_param(config, w.scenario, "phenotype_model2"),
        beta1=lambda w: get_param(config, w.scenario, "beta1"),
        beta_sex1=lambda w: get_param(config, w.scenario, "beta_sex1"),
        phenotype_params1=lambda w: get_param(config, w.scenario, "phenotype_params1"),
        beta2=lambda w: get_param(config, w.scenario, "beta2"),
        beta_sex2=lambda w: get_param(config, w.scenario, "beta_sex2"),
        phenotype_params2=lambda w: get_param(config, w.scenario, "phenotype_params2"),
    script:
        "../../scripts/simace/phenotype.py"


rule censor_weibull:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.raw.parquet",
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet",
    log:
        "logs/{folder}/{scenario}/rep{rep}/censor_weibull.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/censor_weibull.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        death_scale=lambda w: get_param(config, w.scenario, "death_scale"),
        death_rho=lambda w: get_param(config, w.scenario, "death_rho"),
        gen_censoring=lambda w: get_param(config, w.scenario, "gen_censoring"),
    script:
        "../../scripts/simace/censor.py"


rule phenotype_simple_ltm:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.simple_ltm.parquet",
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_simple_ltm.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_simple_ltm.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        # PR3: prevalence now lives inside per-trait phenotype_params (for
        # adult / cure_frailty); the threshold path falls back to a default
        # for traits whose main model doesn't carry one (frailty / first_passage).
        phenotype_params1=lambda w: get_param(config, w.scenario, "phenotype_params1"),
        phenotype_params2=lambda w: get_param(config, w.scenario, "phenotype_params2"),
        G_pheno=lambda w: get_param(config, w.scenario, "G_pheno"),
        standardize=lambda w: get_param(config, w.scenario, "standardize"),
    script:
        "../../scripts/simace/phenotype_threshold.py"
