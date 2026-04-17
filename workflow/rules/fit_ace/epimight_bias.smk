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

EPIMIGHT_BIAS_FOLDERS = [
    f for f in get_all_folders(config) if f.startswith("epimight_bias")
]

EPIMIGHT_BIAS_SCENARIOS_BY_FOLDER = {
    f: get_scenarios_for_folder(config, f) for f in EPIMIGHT_BIAS_FOLDERS
}


rule ltm_falconer:
    """Compute Falconer h² from simple LTM phenotype per EPIMIGHT kind."""
    input:
        simple_ltm="results/{folder}/{scenario}/rep{rep}/phenotype.simple_ltm.parquet",
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        json="results/{folder}/{scenario}/rep{rep}/ltm_falconer.json",
    log:
        "logs/{folder}/{scenario}/rep{rep}/ltm_falconer.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/ltm_falconer.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario),
        runtime=lambda w: _scale_runtime(config, w.scenario),
    params:
        kinds=lambda w: get_param(config, w.scenario, "epimight_kinds"),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    script:
        "../../scripts/fit_ace/ltm_falconer.py"


rule epimight_bias_gather:
    """Gather EPIMIGHT + LTM Falconer results across all bias scenarios."""
    input:
        epimight_atlases=lambda w: [
            f"results/{w.folder}/{s}/rep1/epimight/plots/atlas.pdf"
            for s in EPIMIGHT_BIAS_SCENARIOS_BY_FOLDER[w.folder]
        ],
        ltm_falconers=lambda w: [
            f"results/{w.folder}/{s}/rep1/ltm_falconer.json"
            for s in EPIMIGHT_BIAS_SCENARIOS_BY_FOLDER[w.folder]
        ],
    output:
        tsv="results/{folder}/epimight_bias_summary.tsv",
    log:
        "logs/{folder}/epimight_bias_gather.log",
    benchmark:
        "benchmarks/{folder}/epimight_bias_gather.tsv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    params:
        results_dir=lambda w: f"results/{w.folder}",
        scenarios=lambda w: EPIMIGHT_BIAS_SCENARIOS_BY_FOLDER[w.folder],
    script:
        "../../scripts/fit_ace/epimight_bias_analysis.py"


rule epimight_bias_plots:
    """Generate EPIMIGHT bias analysis plots and atlas."""
    input:
        tsv="results/{folder}/epimight_bias_summary.tsv",
    output:
        atlas="results/{folder}/plots/epimight_bias_atlas.pdf",
    log:
        "logs/{folder}/epimight_bias_plots.log",
    benchmark:
        "benchmarks/{folder}/epimight_bias_plots.tsv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    script:
        "../../scripts/fit_ace/plot_epimight_bias.py"


rule epimight_bias_all:
    """Run the complete EPIMIGHT bias analysis pipeline for all bias folders."""
    input:
        [
            f"results/{folder}/plots/epimight_bias_atlas.pdf"
            for folder in EPIMIGHT_BIAS_FOLDERS
        ],
