rule sample_frailty:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet"
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.sampled.parquet"
    params:
        N_sample = lambda w: get_param(config, w.scenario, "N_sample"),
        seed     = lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/sample_frailty.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/sample_frailty.tsv"
    resources:
        mem_mb  = 2000,
        runtime = 5
    threads: 1
    script:
        "../scripts/sample.py"


rule sample_threshold:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.parquet"
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.sampled.parquet"
    params:
        N_sample = lambda w: get_param(config, w.scenario, "N_sample"),
        seed     = lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/sample_threshold.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/sample_threshold.tsv"
    resources:
        mem_mb  = 2000,
        runtime = 5
    threads: 1
    script:
        "../scripts/sample.py"