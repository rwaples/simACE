rule stats_weibull:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.weibull.parquet"
    output:
        stats="results/{folder}/{scenario}/rep{rep}/phenotype_stats.yaml",
        samples="results/{folder}/{scenario}/rep{rep}/phenotype_samples.parquet"
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        gen_censoring=lambda w: get_param(config, w.scenario, "gen_censoring"),
        beta1=lambda w: get_param(config, w.scenario, "beta1"),
        scale1=lambda w: get_param(config, w.scenario, "scale1"),
        rho1=lambda w: get_param(config, w.scenario, "rho1"),
        beta2=lambda w: get_param(config, w.scenario, "beta2"),
        scale2=lambda w: get_param(config, w.scenario, "scale2"),
        rho2=lambda w: get_param(config, w.scenario, "rho2"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_stats.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_stats.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "../scripts/compute_phenotype_stats.py"


rule plot_weibull:
    input:
        stats=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/phenotype_stats.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        samples=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/phenotype_samples.parquet",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
    params:
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        gen_censoring=lambda w: get_param(config, w.scenario, "gen_censoring"),
    output:
        expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=PHENOTYPE_PLOTS)
    log:
        "logs/{folder}/{scenario}/plot_phenotype.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/plot_phenotype.tsv"
    resources:
        mem_mb=2000,
        runtime=5
    threads: 1
    script:
        "../scripts/plot_phenotype.py"


rule stats_threshold:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.parquet"
    output:
        stats="results/{folder}/{scenario}/rep{rep}/threshold_stats.yaml",
        samples="results/{folder}/{scenario}/rep{rep}/threshold_samples.parquet"
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/threshold_stats.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/threshold_stats.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "../scripts/compute_threshold_stats.py"


rule plot_threshold:
    input:
        stats=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/threshold_stats.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        samples=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/threshold_samples.parquet",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
    params:
        prevalence1=lambda w: get_param(config, w.scenario, "prevalence1"),
        prevalence2=lambda w: get_param(config, w.scenario, "prevalence2"),
    output:
        expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=THRESHOLD_PLOTS)
    log:
        "logs/{folder}/{scenario}/plot_threshold.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/plot_threshold.tsv"
    resources:
        mem_mb=2000,
        runtime=5
    threads: 1
    script:
        "../scripts/plot_threshold.py"
