import platform

# NOTE: validation results are now produced by the combined `analyze` rule
# (analyze.smk, ADR 0006/0007) inside report.yaml's `validation` group; the
# standalone validation.yaml artifact is gone. The standalone simace-validate
# CLI / script wrapper is retained for early, ascertainment-independent
# debugging on the full pedigree, but no Snakemake rule invokes it.


rule gather_validation:
    input:
        validations=lambda w: get_folder_validations(config, w.folder),
    output:
        tsv="results/{folder}/validation_summary.tsv",
    log:
        "logs/{folder}/gather_validation.log",
    benchmark:
        "benchmarks/{folder}/gather_validation.tsv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    script:
        "../../scripts/simace/gather_validation.py"


# Windows patch
filtered_plots = VALIDATION_PLOTS.copy()
if platform.system() == "Windows":
    if "memory.png" in filtered_plots:
        filtered_plots.remove("memory.png")


rule plot_validation:
    input:
        tsv="results/{folder}/validation_summary.tsv",
    output:
        expand("results/{{folder}}/plots/{plot}", plot=filtered_plots),
        "results/{folder}/plots/atlas.pdf",
    log:
        "logs/{folder}/plot_validation.log",
    benchmark:
        "benchmarks/{folder}/plot_validation.tsv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    params:
        plot_format=config["defaults"].get("plot_format", "png"),
    script:
        "../../scripts/simace/plot_validation.py"
