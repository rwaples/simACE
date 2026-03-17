rule sample_frailty:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet"
    output:
        phenotype=temp("results/{folder}/{scenario}/rep{rep}/phenotype.sampled.parquet")
    params:
        N_sample = lambda w: get_param(config, w.scenario, "N_sample"),
        seed     = lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/sample_frailty.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/sample_frailty.tsv"
    resources:
        mem_mb  = lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime = lambda w: _scale_runtime(config, w.scenario, "G_pheno")
    threads: 1
    script:
        "../scripts/sample.py"


rule sample_threshold:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.parquet"
    output:
        phenotype=temp("results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.sampled.parquet")
    params:
        N_sample = lambda w: get_param(config, w.scenario, "N_sample"),
        seed     = lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/sample_threshold.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/sample_threshold.tsv"
    resources:
        mem_mb  = lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime = lambda w: _scale_runtime(config, w.scenario, "G_pheno")
    threads: 1
    script:
        "../scripts/sample.py"
