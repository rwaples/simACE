# ---------------------------------------------------------------------------
# EPIMIGHT bias analysis
#
# Quantifies bias in EPIMIGHT h² estimates from:
#   - liability vs observed scale
#   - censoring and mortality
#   - prevalence / number of affected relatives
#   - shared environment (C)
#
# Three new rules:
#   1. ltm_falconer      – Falconer h² from simple LTM binary phenotype
#   2. epimight_bias_gather  – consolidate results across all bias scenarios
#   3. epimight_bias_plots   – generate bias analysis atlas PDF
# ---------------------------------------------------------------------------

EPIMIGHT_BIAS_SCENARIOS = [
    s for s in config["scenarios"]
    if get_folder(config, s) == "epimight_bias"
]


rule ltm_falconer:
    """Compute Falconer h² from simple LTM phenotype per EPIMIGHT kind."""
    input:
        simple_ltm="results/{folder}/{scenario}/rep{rep}/phenotype.simple_ltm.parquet",
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        json="results/{folder}/{scenario}/rep{rep}/ltm_falconer.json",
    params:
        kinds=lambda w: get_param(config, w.scenario, "epimight_kinds"),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/ltm_falconer.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/ltm_falconer.tsv"
    resources:
        mem_mb  = lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime = lambda w: _scale_runtime(config, w.scenario, "G_pheno")
    threads: 1
    script:
        "../scripts/ltm_falconer.py"


rule epimight_bias_gather:
    """Gather EPIMIGHT + LTM Falconer results across all bias scenarios."""
    input:
        epimight_atlases=[
            f"results/epimight_bias/{s}/rep1/epimight/plots/atlas.pdf"
            for s in EPIMIGHT_BIAS_SCENARIOS
        ],
        ltm_falconers=[
            f"results/epimight_bias/{s}/rep1/ltm_falconer.json"
            for s in EPIMIGHT_BIAS_SCENARIOS
        ],
    output:
        tsv="results/epimight_bias/epimight_bias_summary.tsv",
    params:
        results_dir="results/epimight_bias",
        scenarios=EPIMIGHT_BIAS_SCENARIOS,
    log:
        "logs/epimight_bias/epimight_bias_gather.log"
    benchmark:
        "benchmarks/epimight_bias/epimight_bias_gather.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "../scripts/epimight_bias_analysis.py"


rule epimight_bias_plots:
    """Generate EPIMIGHT bias analysis plots and atlas."""
    input:
        tsv="results/epimight_bias/epimight_bias_summary.tsv",
    output:
        atlas="results/epimight_bias/plots/epimight_bias_atlas.pdf",
    log:
        "logs/epimight_bias/epimight_bias_plots.log"
    benchmark:
        "benchmarks/epimight_bias/epimight_bias_plots.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "../scripts/plot_epimight_bias.py"


rule epimight_bias_all:
    """Run the complete EPIMIGHT bias analysis pipeline."""
    input:
        "results/epimight_bias/plots/epimight_bias_atlas.pdf",
