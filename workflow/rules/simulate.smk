rule simulate_pedigree_liability:
    output:
        pedigree=temp("results/{folder}/{scenario}/rep{rep}/pedigree.full.parquet"),
        params="results/{folder}/{scenario}/rep{rep}/params.yaml"
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        N=lambda w: get_param(config, w.scenario, "N"),
        G_ped=lambda w: get_param(config, w.scenario, "G_ped"),
        G_sim=lambda w: get_param(config, w.scenario, "G_sim"),
        mating_lambda=lambda w: get_param(config, w.scenario, "mating_lambda"),
        p_mztwin=lambda w: get_param(config, w.scenario, "p_mztwin"),
        rep=lambda w: int(w.rep),
        A1=lambda w: get_param(config, w.scenario, "A1"),
        C1=lambda w: get_param(config, w.scenario, "C1"),
        A2=lambda w: get_param(config, w.scenario, "A2"),
        C2=lambda w: get_param(config, w.scenario, "C2"),
        rA=lambda w: get_param(config, w.scenario, "rA"),
        rC=lambda w: get_param(config, w.scenario, "rC"),
        assort1=lambda w: get_param(config, w.scenario, "assort1"),
        assort2=lambda w: get_param(config, w.scenario, "assort2"),
        assort_matrix=lambda w: get_param(config, w.scenario, "assort_matrix"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/simulate.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/simulate.tsv"
    resources:
        mem_mb  = lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime = lambda w: _scale_runtime(config, w.scenario, "G_ped")
    threads: 4
    script:
        "../scripts/simulate.py"
