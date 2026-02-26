rule validate_pedigree_liability:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
        params="results/{folder}/{scenario}/rep{rep}/params.yaml"
    output:
        report="results/{folder}/{scenario}/rep{rep}/validation.yaml"
    log:
        "logs/{folder}/{scenario}/rep{rep}/validate.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/validate.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "../scripts/validate.py"


rule gather_validation:
    input:
        validations=lambda w: get_folder_validations(config, w.folder)
    output:
        tsv="results/{folder}/validation_summary.tsv"
    log:
        "logs/{folder}/gather_validation.log"
    benchmark:
        "benchmarks/{folder}/gather_validation.tsv"
    resources:
        mem_mb=1000,
        runtime=5
    threads: 1
    script:
        "../scripts/gather_validation.py"


rule plot_validation:
    input:
        tsv="results/{folder}/validation_summary.tsv"
    params:
        plot_format=config["defaults"].get("plot_format", "png"),
    output:
        expand("results/{{folder}}/plots/{plot}", plot=VALIDATION_PLOTS),
        "results/{folder}/plots/atlas.pdf"
    log:
        "logs/{folder}/plot_validation.log"
    benchmark:
        "benchmarks/{folder}/plot_validation.tsv"
    resources:
        mem_mb=1000,
        runtime=5
    threads: 1
    script:
        "../scripts/plot_validation.py"
