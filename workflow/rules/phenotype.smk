# ---------------------------------------------------------------------------
# Phenotype simulation rules
# ---------------------------------------------------------------------------


rule phenotype_frailty:
    """Frailty phenotype simulation with pluggable baseline hazard."""
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet"
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.raw.parquet"
    params:
        seed            = lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        G_pheno         = lambda w: get_param(config, w.scenario, "G_pheno"),
        standardize     = lambda w: get_param(config, w.scenario, "standardize"),
        phenotype_model1  = lambda w: get_param(config, w.scenario, "phenotype_model1"),
        phenotype_model2  = lambda w: get_param(config, w.scenario, "phenotype_model2"),
        prevalence1       = lambda w: get_param(config, w.scenario, "prevalence1"),
        prevalence2       = lambda w: get_param(config, w.scenario, "prevalence2"),
        beta1             = lambda w: get_param(config, w.scenario, "beta1"),
        beta_sex1         = lambda w: get_param(config, w.scenario, "beta_sex1"),
        phenotype_params1 = lambda w: get_param(config, w.scenario, "phenotype_params1"),
        beta2             = lambda w: get_param(config, w.scenario, "beta2"),
        beta_sex2         = lambda w: get_param(config, w.scenario, "beta_sex2"),
        phenotype_params2 = lambda w: get_param(config, w.scenario, "phenotype_params2"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_frailty.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_frailty.tsv"
    resources:
        mem_mb  = 4000,
        runtime = 10
    threads: 1
    script:
        "../scripts/phenotype.py"


rule censor_weibull:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.raw.parquet"
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet"
    params:
        seed          = lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        censor_age    = lambda w: get_param(config, w.scenario, "censor_age"),
        death_scale   = lambda w: get_param(config, w.scenario, "death_scale"),
        death_rho     = lambda w: get_param(config, w.scenario, "death_rho"),
        gen_censoring = lambda w: get_param(config, w.scenario, "gen_censoring"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/censor_weibull.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/censor_weibull.tsv"
    resources:
        mem_mb  = 4000,
        runtime = 5
    threads: 1
    script:
        "../scripts/censor.py"


rule phenotype_threshold:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet"
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.parquet"
    params:
        prevalence1 = lambda w: get_param(config, w.scenario, "prevalence1"),
        prevalence2 = lambda w: get_param(config, w.scenario, "prevalence2"),
        G_pheno     = lambda w: get_param(config, w.scenario, "G_pheno"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_threshold.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_threshold.tsv"
    resources:
        mem_mb  = 2000,
        runtime = 5
    threads: 1
    script:
        "../scripts/phenotype_threshold.py"