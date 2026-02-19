rule validate:
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
    output:
        "results/{folder}/plots/variance_components.png",
        "results/{folder}/plots/twin_rate.png",
        "results/{folder}/plots/correlations_A.png",
        "results/{folder}/plots/correlations_phenotype.png",
        "results/{folder}/plots/heritability_estimates.png",
        "results/{folder}/plots/half_sib_proportions.png",
        "results/{folder}/plots/cross_trait_correlations.png",
        "results/{folder}/plots/family_size.png",
        "results/{folder}/plots/summary_bias.png",
        "results/{folder}/plots/runtime.png",
        "results/{folder}/plots/memory.png"
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
