# ---------------------------------------------------------------------------
# Phenotype simulation rules
#
# Both rules consume the *pre-ascertainment* pedigree via the shared
# _pre_ascertainment_pedigree_input helper (respects use_gene_drop).  trait.raw
# is temp; trait.full is durable (ADR 0008) so the Analyze stage can quantify
# ascertainment distortion against the full phenotyped population. The
# ascertainment stage produces the canonical post-stage trait.parquet that
# downstream consumers read.
# ---------------------------------------------------------------------------


rule phenotype:
    """Phenotype simulation with pluggable model."""
    input:
        pedigree=lambda w: _pre_ascertainment_pedigree_input(w, config),
    output:
        phenotype=temp("results/{folder}/{scenario}/rep{rep}/trait.raw.parquet"),
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
        phenotype="results/{folder}/{scenario}/rep{rep}/trait.raw.parquet",
        pedigree=lambda w: _pre_ascertainment_pedigree_input(w, config),
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/trait.full.parquet",
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
