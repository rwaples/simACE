rule pedigree_dropout:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.full.parquet",
    output:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    log:
        "logs/{folder}/{scenario}/rep{rep}/dropout.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/dropout.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=5,
    params:
        dropout_rate=lambda w: get_param(config, w.scenario, "pedigree_dropout_rate"),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    script:
        "../../scripts/sim_ace/dropout.py"
