rule simulate_pedigree_liability:
    output:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.full.parquet",
    log:
        "logs/{folder}/{scenario}/rep{rep}/simulate.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/simulate.tsv"
    threads: 4
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        N=lambda w: get_param(config, w.scenario, "N"),
        G_ped=lambda w: get_param(config, w.scenario, "G_ped"),
        G_sim=lambda w: get_param(config, w.scenario, "G_sim"),
        mating_lambda=lambda w: get_param(config, w.scenario, "mating_lambda"),
        p_mztwin=lambda w: get_param(config, w.scenario, "p_mztwin"),
        A1=lambda w: get_param(config, w.scenario, "A1"),
        C1=lambda w: get_param(config, w.scenario, "C1"),
        E1=lambda w: get_param(config, w.scenario, "E1"),
        A2=lambda w: get_param(config, w.scenario, "A2"),
        C2=lambda w: get_param(config, w.scenario, "C2"),
        E2=lambda w: get_param(config, w.scenario, "E2"),
        rA=lambda w: get_param(config, w.scenario, "rA"),
        rC=lambda w: get_param(config, w.scenario, "rC"),
        rE=lambda w: get_param(config, w.scenario, "rE"),
        assort1=lambda w: get_param(config, w.scenario, "assort1"),
        assort2=lambda w: get_param(config, w.scenario, "assort2"),
        assort_matrix=lambda w: get_param(config, w.scenario, "assort_matrix"),
    script:
        "../../scripts/simace/simulate.py"


rule emit_params:
    output:
        params="results/{folder}/{scenario}/rep{rep}/params.yaml",
    log:
        "logs/{folder}/{scenario}/rep{rep}/emit_params.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/emit_params.tsv"
    threads: 1
    resources:
        mem_mb=200,
        runtime=1,
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        rep=lambda w: int(w.rep),
        A1=lambda w: get_param(config, w.scenario, "A1"),
        C1=lambda w: get_param(config, w.scenario, "C1"),
        E1=lambda w: get_param(config, w.scenario, "E1"),
        A2=lambda w: get_param(config, w.scenario, "A2"),
        C2=lambda w: get_param(config, w.scenario, "C2"),
        E2=lambda w: get_param(config, w.scenario, "E2"),
        rA=lambda w: get_param(config, w.scenario, "rA"),
        rC=lambda w: get_param(config, w.scenario, "rC"),
        rE=lambda w: get_param(config, w.scenario, "rE"),
        N=lambda w: get_param(config, w.scenario, "N"),
        G_ped=lambda w: get_param(config, w.scenario, "G_ped"),
        G_sim=lambda w: get_param(config, w.scenario, "G_sim"),
        mating_lambda=lambda w: get_param(config, w.scenario, "mating_lambda"),
        p_mztwin=lambda w: get_param(config, w.scenario, "p_mztwin"),
        assort1=lambda w: get_param(config, w.scenario, "assort1"),
        assort2=lambda w: get_param(config, w.scenario, "assort2"),
        assort_matrix=lambda w: get_param(config, w.scenario, "assort_matrix"),
    script:
        "../../scripts/simace/emit_params.py"
