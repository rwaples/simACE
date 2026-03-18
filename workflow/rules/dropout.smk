rule pedigree_dropout:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.full.parquet",
    output:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    params:
        dropout_rate=lambda w: get_param(config, w.scenario, "pedigree_dropout_rate"),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/dropout.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/dropout.tsv"
    resources:
        mem_mb  = lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime = 5
    threads: 1
    script:
        "../scripts/dropout.py"
