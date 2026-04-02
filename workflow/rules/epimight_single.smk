# ---------------------------------------------------------------------------
# EPIMIGHT single-pair analysis
#
# Mirrors the standard EPIMIGHT pipeline but uses --single-pair relative
# selection to avoid c2 cohort dilution at high prevalence.
# Output goes to epimight_single/ subdirectory alongside existing epimight/.
# ---------------------------------------------------------------------------

EPIMIGHT_SINGLE_BIAS_SCENARIOS = [
    s for s in config["scenarios"] if get_folder(config, s) == "epimight_bias"
]


rule epimight_single_create_parquet:
    """Convert ACE phenotype to EPIMIGHT TTE format with single-pair selection."""
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet",
    output:
        t1="results/{folder}/{scenario}/rep{rep}/epimight_single/trait1.epimight_in.parquet",
        t2="results/{folder}/{scenario}/rep{rep}/epimight_single/trait2.epimight_in.parquet",
        truth="results/{folder}/{scenario}/rep{rep}/epimight_single/true_parameters.json",
    log:
        "logs/{folder}/{scenario}/rep{rep}/epimight_single_create_parquet.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/epimight_single_create_parquet.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_pheno"),
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    script:
        "../scripts/epimight_create_parquet_single.py"


rule epimight_single_guide_yob:
    """Run EPIMIGHT analysis for one kind on single-pair parquets."""
    input:
        t1="results/{folder}/{scenario}/rep{rep}/epimight_single/trait1.epimight_in.parquet",
        t2="results/{folder}/{scenario}/rep{rep}/epimight_single/trait2.epimight_in.parquet",
    output:
        h2_d1="results/{folder}/{scenario}/rep{rep}/epimight_single/tsv/h2_d1_{kind}.tsv",
        h2_d2="results/{folder}/{scenario}/rep{rep}/epimight_single/tsv/h2_d2_{kind}.tsv",
        gc="results/{folder}/{scenario}/rep{rep}/epimight_single/tsv/gc_full_{kind}.tsv",
        report="results/{folder}/{scenario}/rep{rep}/epimight_single/results_{kind}.md",
    log:
        "logs/{folder}/{scenario}/rep{rep}/epimight_single_guide_yob_{kind}.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/epimight_single_guide_yob_{kind}.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_pheno"),
    shell:
        "conda run -n epimight "
        "Rscript -e \"if (!requireNamespace('epimight', quietly=TRUE)) "
        "install.packages('fit_ace/epimight/EPIMIGHT/epimight', repos=NULL, type='source')\" "
        ">{log} 2>&1 && "
        "conda run -n epimight "
        "Rscript fit_ace/epimight/guide-yob.R "
        "results/{wildcards.folder}/{wildcards.scenario}/rep{wildcards.rep}/epimight_single "
        "{wildcards.kind} "
        ">>{log} 2>&1"


rule epimight_single_atlas:
    """Generate EPIMIGHT single-pair plot atlas."""
    input:
        tsv=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/epimight_single/tsv/h2_d1_{kind}.tsv",
            folder=w.folder,
            scenario=w.scenario,
            rep=w.rep,
            kind=get_param(config, w.scenario, "epimight_kinds"),
        ),
        truth="results/{folder}/{scenario}/rep{rep}/epimight_single/true_parameters.json",
    output:
        atlas="results/{folder}/{scenario}/rep{rep}/epimight_single/plots/atlas.pdf",
    log:
        "logs/{folder}/{scenario}/rep{rep}/epimight_single_atlas.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/epimight_single_atlas.tsv"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=10,
    script:
        "../scripts/epimight_atlas.py"


# ---------------------------------------------------------------------------
# Single-pair bias analysis (gather + plots across all bias scenarios)
# ---------------------------------------------------------------------------


rule epimight_single_bias_gather:
    """Gather single-pair EPIMIGHT results across all bias scenarios."""
    input:
        epimight_atlases=[
            f"results/epimight_bias/{s}/rep1/epimight_single/plots/atlas.pdf"
            for s in EPIMIGHT_SINGLE_BIAS_SCENARIOS
        ],
        ltm_falconers=[
            f"results/epimight_bias/{s}/rep1/ltm_falconer.json"
            for s in EPIMIGHT_SINGLE_BIAS_SCENARIOS
        ],
    output:
        tsv="results/epimight_bias/epimight_single_bias_summary.tsv",
    log:
        "logs/epimight_bias/epimight_single_bias_gather.log",
    benchmark:
        "benchmarks/epimight_bias/epimight_single_bias_gather.tsv"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=10,
    params:
        results_dir="results/epimight_bias",
        scenarios=EPIMIGHT_SINGLE_BIAS_SCENARIOS,
        subdir="epimight_single",
    script:
        "../scripts/epimight_bias_analysis.py"


rule epimight_single_bias_plots:
    """Generate single-pair bias analysis plots and atlas."""
    input:
        tsv="results/epimight_bias/epimight_single_bias_summary.tsv",
    output:
        atlas="results/epimight_bias/plots/epimight_single_bias_atlas.pdf",
    log:
        "logs/epimight_bias/epimight_single_bias_plots.log",
    benchmark:
        "benchmarks/epimight_bias/epimight_single_bias_plots.tsv"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=10,
    params:
        include_dilution_correction=False,
    script:
        "../scripts/plot_epimight_bias.py"


rule epimight_single_bias_all:
    """Run the complete single-pair EPIMIGHT bias analysis pipeline."""
    input:
        "results/epimight_bias/plots/epimight_single_bias_atlas.pdf",


# ---------------------------------------------------------------------------
# Large-N single-pair validation (epimight_bias_2M folder)
# ---------------------------------------------------------------------------

EPIMIGHT_SINGLE_2M_SCENARIOS = [
    s for s in config["scenarios"] if get_folder(config, s) == "epimight_bias_2M"
]


rule epimight_single_2M_gather:
    """Gather single-pair EPIMIGHT results for N=2M scenarios."""
    input:
        epimight_atlases=[
            f"results/epimight_bias_2M/{s}/rep1/epimight_single/plots/atlas.pdf"
            for s in EPIMIGHT_SINGLE_2M_SCENARIOS
        ],
        ltm_falconers=[
            f"results/epimight_bias_2M/{s}/rep1/ltm_falconer.json"
            for s in EPIMIGHT_SINGLE_2M_SCENARIOS
        ],
    output:
        tsv="results/epimight_bias_2M/epimight_single_bias_summary.tsv",
    log:
        "logs/epimight_bias_2M/epimight_single_bias_gather.log",
    benchmark:
        "benchmarks/epimight_bias_2M/epimight_single_bias_gather.tsv"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=10,
    params:
        results_dir="results/epimight_bias_2M",
        scenarios=EPIMIGHT_SINGLE_2M_SCENARIOS,
        subdir="epimight_single",
    script:
        "../scripts/epimight_bias_analysis.py"


rule epimight_single_2M_plots:
    """Generate single-pair bias plots for N=2M scenarios."""
    input:
        tsv="results/epimight_bias_2M/epimight_single_bias_summary.tsv",
    output:
        atlas="results/epimight_bias_2M/plots/epimight_single_bias_atlas.pdf",
    log:
        "logs/epimight_bias_2M/epimight_single_bias_plots.log",
    benchmark:
        "benchmarks/epimight_bias_2M/epimight_single_bias_plots.tsv"
    threads: 1
    resources:
        mem_mb=4000,
        runtime=10,
    params:
        include_dilution_correction=False,
    script:
        "../scripts/plot_epimight_bias.py"


rule epimight_single_2M_all:
    """Run the complete N=2M single-pair EPIMIGHT bias analysis."""
    input:
        "results/epimight_bias_2M/plots/epimight_single_bias_atlas.pdf",
