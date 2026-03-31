rule sample_phenotype:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet",
    output:
        phenotype=temp("results/{folder}/{scenario}/rep{rep}/phenotype.sampled.parquet"),
    params:
        N_sample=lambda w: get_param(config, w.scenario, "N_sample"),
        case_ascertainment_ratio=lambda w: get_param(
            config, w.scenario, "case_ascertainment_ratio"
        ),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/sample_phenotype.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/sample_phenotype.tsv"
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_pheno"),
    threads: 1
    script:
        "../scripts/sample.py"


rule sample_simple_ltm:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.simple_ltm.parquet",
    output:
        phenotype=temp(
            "results/{folder}/{scenario}/rep{rep}/phenotype.simple_ltm.sampled.parquet"
        ),
    params:
        N_sample=lambda w: get_param(config, w.scenario, "N_sample"),
        case_ascertainment_ratio=lambda w: get_param(
            config, w.scenario, "case_ascertainment_ratio"
        ),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/sample_simple_ltm.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/sample_simple_ltm.tsv"
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_pheno"),
    threads: 1
    script:
        "../scripts/sample.py"
