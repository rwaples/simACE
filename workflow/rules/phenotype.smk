rule phenotype_weibull:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet"
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.weibull.parquet"
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        # Trait 1 phenotype parameters
        beta1=lambda w: get_param(config, w.scenario, "beta1"),
        scale1=lambda w: get_param(config, w.scenario, "scale1"),
        rho1=lambda w: get_param(config, w.scenario, "rho1"),
        # Trait 2 phenotype parameters
        beta2=lambda w: get_param(config, w.scenario, "beta2"),
        scale2=lambda w: get_param(config, w.scenario, "scale2"),
        rho2=lambda w: get_param(config, w.scenario, "rho2"),
        # Shared parameters
        standardize=lambda w: get_param(config, w.scenario, "standardize"),
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        death_scale=lambda w: get_param(config, w.scenario, "death_scale"),
        death_rho=lambda w: get_param(config, w.scenario, "death_rho"),
        gen_censoring=lambda w: get_param(config, w.scenario, "gen_censoring"),
        G_pheno=lambda w: get_param(config, w.scenario, "G_pheno"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_weibull.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_weibull.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "../scripts/phenotype.py"


rule phenotype_threshold:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet"
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.parquet"
    params:
        prevalence1=lambda w: get_param(config, w.scenario, "prevalence1"),
        prevalence2=lambda w: get_param(config, w.scenario, "prevalence2"),
        G_pheno=lambda w: get_param(config, w.scenario, "G_pheno"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_threshold.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_threshold.tsv"
    resources:
        mem_mb=2000,
        runtime=5
    threads: 1
    script:
        "../scripts/phenotype_threshold.py"
