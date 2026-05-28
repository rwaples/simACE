# ---------------------------------------------------------------------------
# Statistics and plotting rules (sim-side)
# ---------------------------------------------------------------------------
#
# NOTE: report.yaml + plotting_sample.parquet are produced by the combined
# `analyze` rule (analyze.smk, ADR 0006/0007). report.yaml merges the former
# validation.yaml + stats_report.yaml (six stats groups + a `validation` group).
# The standalone simace-phenotype-stats CLI / script wrapper is retained for
# debugging.


rule plot_phenotype:
    input:
        report=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/report.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        plot_payload=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/plot_payload.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        samples=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/plotting_sample.parquet",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
    output:
        expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=PHENOTYPE_PLOTS),
    log:
        "logs/{folder}/{scenario}/plot_phenotype.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/plot_phenotype.tsv"
    threads: 1
    resources:
        mem_mb=2000,
        runtime=5,
    params:
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        gen_censoring=lambda w: get_param(config, w.scenario, "gen_censoring"),
        max_degree=lambda w: get_param(config, w.scenario, "max_degree"),
        plot_format=lambda w: config["defaults"].get("plot_format", "png"),
    script:
        "../../scripts/simace/plot_phenotype.py"


rule assemble_scenario_atlas:
    input:
        phenotype=expand(
            "results/{{folder}}/{{scenario}}/plots/{plot}", plot=PHENOTYPE_PLOTS
        ),
        params_yaml="results/{folder}/{scenario}/rep1/params.yaml",
        report=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/report.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        plot_payload=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/plot_payload.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
    output:
        "results/{folder}/{scenario}/plots/atlas.pdf",
    log:
        "logs/{folder}/{scenario}/assemble_atlas.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/assemble_atlas.tsv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    params:
        scenario=lambda w: w.scenario,
        replicates=lambda w: get_param(config, w.scenario, "replicates"),
        folder=lambda w: get_param(config, w.scenario, "folder"),
        standardize=lambda w: get_param(config, w.scenario, "standardize"),
        beta1=lambda w: get_param(config, w.scenario, "beta1"),
        beta_sex1=lambda w: get_param(config, w.scenario, "beta_sex1"),
        phenotype_model1=lambda w: get_param(config, w.scenario, "phenotype_model1"),
        phenotype_params1=lambda w: get_param(config, w.scenario, "phenotype_params1"),
        beta2=lambda w: get_param(config, w.scenario, "beta2"),
        beta_sex2=lambda w: get_param(config, w.scenario, "beta_sex2"),
        phenotype_model2=lambda w: get_param(config, w.scenario, "phenotype_model2"),
        phenotype_params2=lambda w: get_param(config, w.scenario, "phenotype_params2"),
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        gen_censoring=lambda w: get_param(config, w.scenario, "gen_censoring"),
        death_scale=lambda w: get_param(config, w.scenario, "death_scale"),
        death_rho=lambda w: get_param(config, w.scenario, "death_rho"),
        # PR3: prevalence is now inside phenotype_params{N}; the atlas's
        # plot_pipeline reads it from there directly.
        G_pheno=lambda w: get_param(config, w.scenario, "G_pheno"),
        N_sample=lambda w: get_param(config, w.scenario, "N_sample"),
        dropout_rate=lambda w: get_param(config, w.scenario, "dropout_rate"),
        case_ascertainment_ratio=lambda w: get_param(
            config, w.scenario, "case_ascertainment_ratio"
        ),
        max_degree=lambda w: get_param(config, w.scenario, "max_degree"),
        plot_format=lambda w: config["defaults"].get("plot_format", "png"),
    script:
        "../../scripts/simace/assemble_atlas.py"
