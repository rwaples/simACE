# ---------------------------------------------------------------------------
# Simple LTM fit statistics + plots (fit-side)
# ---------------------------------------------------------------------------


rule stats_simple_ltm:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.simple_ltm.sampled.parquet",
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        stats="results/{folder}/{scenario}/rep{rep}/simple_ltm_stats.yaml",
        samples=temp("results/{folder}/{scenario}/rep{rep}/simple_ltm_samples.parquet"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/simple_ltm_stats.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/simple_ltm_stats.tsv"
    threads: 2
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_pheno"),
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        max_degree=lambda w: get_param(config, w.scenario, "max_degree"),
        case_ascertainment_ratio=lambda w: get_param(
            config, w.scenario, "case_ascertainment_ratio"
        ),
    script:
        "../../scripts/fit_ace/compute_simple_ltm_stats.py"


rule plot_simple_ltm:
    input:
        stats=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/simple_ltm_stats.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        samples=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/simple_ltm_samples.parquet",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
    output:
        expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=SIMPLE_LTM_PLOTS),
    log:
        "logs/{folder}/{scenario}/plot_simple_ltm.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/plot_simple_ltm.tsv"
    threads: 1
    resources:
        mem_mb=2000,
        runtime=5,
    params:
        prevalence1=lambda w: get_param(config, w.scenario, "prevalence1"),
        prevalence2=lambda w: get_param(config, w.scenario, "prevalence2"),
        A1=lambda w: get_param(config, w.scenario, "A1"),
        C1=lambda w: get_param(config, w.scenario, "C1"),
        A2=lambda w: get_param(config, w.scenario, "A2"),
        C2=lambda w: get_param(config, w.scenario, "C2"),
        plot_format=lambda w: config["defaults"].get("plot_format", "png"),
    script:
        "../../scripts/fit_ace/plot_simple_ltm.py"
