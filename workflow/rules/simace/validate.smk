import platform

# NOTE: per-replicate scientific results are produced by the combined `analyze`
# rule (analyze.smk, ADR 0006/0007/0008) as the curated report.yaml; the
# folder-level report_summary.tsv is gathered from those reports. The standalone
# simace-validate CLI / script wrapper is retained for early,
# ascertainment-independent debugging on the full pedigree, but no rule invokes it.


rule gather_report_summary:
    input:
        reports=lambda w: get_folder_validations(config, w.folder),
    output:
        tsv="results/{folder}/report_summary.tsv",
    log:
        "logs/{folder}/gather_report_summary.log",
    benchmark:
        "benchmarks/{folder}/gather_report_summary.tsv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    script:
        "../../scripts/simace/gather_report_summary.py"


# Windows patch
filtered_plots = VALIDATION_PLOTS.copy()
if platform.system() == "Windows":
    if "memory.png" in filtered_plots:
        filtered_plots.remove("memory.png")


rule plot_validation:
    input:
        tsv="results/{folder}/report_summary.tsv",
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
