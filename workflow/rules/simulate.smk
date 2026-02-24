rule simulate_pedigree_liability:
    output:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
        params="results/{folder}/{scenario}/rep{rep}/params.yaml"
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        N=lambda w: get_param(config, w.scenario, "N"),
        G_ped=lambda w: get_param(config, w.scenario, "G_ped"),
        G_sim=lambda w: get_param(config, w.scenario, "G_sim"),
        fam_size=lambda w: get_param(config, w.scenario, "fam_size"),
        p_mztwin=lambda w: get_param(config, w.scenario, "p_mztwin"),
        p_nonsocial_father=lambda w: get_param(config, w.scenario, "p_nonsocial_father"),
        rep=lambda w: int(w.rep),
        A1=lambda w: get_param(config, w.scenario, "A1"),
        C1=lambda w: get_param(config, w.scenario, "C1"),
        A2=lambda w: get_param(config, w.scenario, "A2"),
        C2=lambda w: get_param(config, w.scenario, "C2"),
        rA=lambda w: get_param(config, w.scenario, "rA"),
        rC=lambda w: get_param(config, w.scenario, "rC"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/simulate.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/simulate.tsv"
    resources:
        mem_mb=8000,
        runtime=10
    threads: 1
    script:
        "../scripts/simulate.py"
