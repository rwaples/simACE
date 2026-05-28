# ---------------------------------------------------------------------------
# Combined Analyze stage: Validate + Stats in one job, one report (ADR 0006, 0007)
# ---------------------------------------------------------------------------


rule analyze:
    input:
        pedigree_full="results/{folder}/{scenario}/rep{rep}/pedigree.full.parquet",
        params="results/{folder}/{scenario}/rep{rep}/params.yaml",
        trait="results/{folder}/{scenario}/rep{rep}/trait.parquet",
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        report="results/{folder}/{scenario}/rep{rep}/report.yaml",
        plot_payload="results/{folder}/{scenario}/rep{rep}/plot_payload.yaml",
        samples=temp("results/{folder}/{scenario}/rep{rep}/plotting_sample.parquet"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/analyze.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/analyze.tsv"
    threads: 5
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        gen_censoring=lambda w: get_param(config, w.scenario, "gen_censoring"),
        max_degree=lambda w: get_param(config, w.scenario, "max_degree"),
        case_ascertainment_ratio=lambda w: get_param(
            config, w.scenario, "case_ascertainment_ratio"
        ),
    script:
        "../../scripts/simace/analyze.py"
