# ---------------------------------------------------------------------------
# Statistics and plotting rules
# ---------------------------------------------------------------------------


rule stats_phenotype:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.sampled.parquet",
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        stats="results/{folder}/{scenario}/rep{rep}/phenotype_stats.yaml",
        samples=temp("results/{folder}/{scenario}/rep{rep}/phenotype_samples.parquet"),
    params:
        seed             = lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        censor_age       = lambda w: get_param(config, w.scenario, "censor_age"),
        gen_censoring    = lambda w: get_param(config, w.scenario, "gen_censoring"),
        extra_tetrachoric = lambda w: get_param(config, w.scenario, "extra_tetrachoric"),
        skip_2nd_cousins  = lambda w: get_param(config, w.scenario, "skip_2nd_cousins"),
        beta1             = lambda w: get_param(config, w.scenario, "beta1"),
        phenotype_model1  = lambda w: get_param(config, w.scenario, "phenotype_model1"),
        phenotype_params1 = lambda w: get_param(config, w.scenario, "phenotype_params1"),
        beta2             = lambda w: get_param(config, w.scenario, "beta2"),
        phenotype_model2  = lambda w: get_param(config, w.scenario, "phenotype_model2"),
        phenotype_params2 = lambda w: get_param(config, w.scenario, "phenotype_params2"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_stats.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_stats.tsv"
    resources:
        mem_mb  = lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime = lambda w: _scale_runtime(config, w.scenario, "G_ped")
    threads: 1
    script:
        "../scripts/compute_phenotype_stats.py"


rule plot_phenotype:
    input:
        stats=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/phenotype_stats.yaml",
            folder=w.folder, scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        samples=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/phenotype_samples.parquet",
            folder=w.folder, scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        validations=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/validation.yaml",
            folder=w.folder, scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
    output:
        expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=PHENOTYPE_PLOTS),
    params:
        censor_age    = lambda w: get_param(config, w.scenario, "censor_age"),
        gen_censoring = lambda w: get_param(config, w.scenario, "gen_censoring"),
        plot_format   = lambda w: config["defaults"].get("plot_format", "png"),
    log:
        "logs/{folder}/{scenario}/plot_phenotype.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/plot_phenotype.tsv"
    resources:
        mem_mb  = 2000,
        runtime = 5
    threads: 1
    script:
        "../scripts/plot_phenotype.py"


rule stats_simple_ltm:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.simple_ltm.sampled.parquet",
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        stats="results/{folder}/{scenario}/rep{rep}/simple_ltm_stats.yaml",
        samples=temp("results/{folder}/{scenario}/rep{rep}/simple_ltm_samples.parquet"),
    params:
        seed             = lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        extra_tetrachoric = lambda w: get_param(config, w.scenario, "extra_tetrachoric"),
        skip_2nd_cousins  = lambda w: get_param(config, w.scenario, "skip_2nd_cousins"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/simple_ltm_stats.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/simple_ltm_stats.tsv"
    resources:
        mem_mb  = lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime = lambda w: _scale_runtime(config, w.scenario, "G_pheno")
    threads: 1
    script:
        "../scripts/compute_simple_ltm_stats.py"


rule plot_simple_ltm:
    input:
        stats=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/simple_ltm_stats.yaml",
            folder=w.folder, scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        samples=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/simple_ltm_samples.parquet",
            folder=w.folder, scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
    output:
        expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=SIMPLE_LTM_PLOTS),
    params:
        prevalence1 = lambda w: get_param(config, w.scenario, "prevalence1"),
        prevalence2 = lambda w: get_param(config, w.scenario, "prevalence2"),
        plot_format = lambda w: config["defaults"].get("plot_format", "png"),
    log:
        "logs/{folder}/{scenario}/plot_simple_ltm.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/plot_simple_ltm.tsv"
    resources:
        mem_mb  = 2000,
        runtime = 5
    threads: 1
    script:
        "../scripts/plot_simple_ltm.py"


rule assemble_scenario_atlas:
    input:
        phenotype   = expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=PHENOTYPE_PLOTS),
        simple_ltm = expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=SIMPLE_LTM_PLOTS),
        params_yaml = "results/{folder}/{scenario}/rep1/params.yaml",
    output:
        "results/{folder}/{scenario}/plots/atlas.pdf",
    params:
        scenario       = lambda w: w.scenario,
        replicates     = lambda w: get_param(config, w.scenario, "replicates"),
        folder         = lambda w: get_param(config, w.scenario, "folder"),
        standardize    = lambda w: get_param(config, w.scenario, "standardize"),
        beta1             = lambda w: get_param(config, w.scenario, "beta1"),
        beta_sex1         = lambda w: get_param(config, w.scenario, "beta_sex1"),
        phenotype_model1  = lambda w: get_param(config, w.scenario, "phenotype_model1"),
        phenotype_params1 = lambda w: get_param(config, w.scenario, "phenotype_params1"),
        beta2             = lambda w: get_param(config, w.scenario, "beta2"),
        beta_sex2         = lambda w: get_param(config, w.scenario, "beta_sex2"),
        phenotype_model2  = lambda w: get_param(config, w.scenario, "phenotype_model2"),
        phenotype_params2 = lambda w: get_param(config, w.scenario, "phenotype_params2"),
        censor_age     = lambda w: get_param(config, w.scenario, "censor_age"),
        gen_censoring  = lambda w: get_param(config, w.scenario, "gen_censoring"),
        death_scale    = lambda w: get_param(config, w.scenario, "death_scale"),
        death_rho      = lambda w: get_param(config, w.scenario, "death_rho"),
        prevalence1    = lambda w: get_param(config, w.scenario, "prevalence1"),
        prevalence2    = lambda w: get_param(config, w.scenario, "prevalence2"),
        G_pheno        = lambda w: get_param(config, w.scenario, "G_pheno"),
        N_sample       = lambda w: get_param(config, w.scenario, "N_sample"),
        plot_format    = lambda w: config["defaults"].get("plot_format", "png"),
    log:
        "logs/{folder}/{scenario}/assemble_atlas.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/assemble_atlas.tsv"
    resources:
        mem_mb  = 1000,
        runtime = 5
    threads: 1
    script:
        "../scripts/assemble_atlas.py"