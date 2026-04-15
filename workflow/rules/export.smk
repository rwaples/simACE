# ---------------------------------------------------------------------------
# Export rules: convert simulation outputs to R-friendly formats
# ---------------------------------------------------------------------------


rule parquet_to_tsv:
    """Convert any .parquet file to .tsv.gz for use in R."""
    input:
        "{path}.parquet",
    output:
        "{path}.tsv.gz",
    log:
        "logs/{path}.parquet_to_tsv.log",
    threads: 1
    resources:
        mem_mb=2000,
        runtime=5,
    params:
        float_precision=config["defaults"].get("tsv_float_precision", 4),
    script:
        "../scripts/parquet_to_tsv.py"


rule parquet_to_tsv_uncompressed:
    """Convert any .parquet file to uncompressed .tsv for use in R."""
    input:
        "{path}.parquet",
    output:
        "{path}.tsv",
    log:
        "logs/{path}.parquet_to_tsv_uncompressed.log",
    threads: 1
    resources:
        mem_mb=2000,
        runtime=5,
    params:
        float_precision=config["defaults"].get("tsv_float_precision", 4),
        gzip=False,
    script:
        "../scripts/parquet_to_tsv.py"


# ---------------------------------------------------------------------------
# Tidy per-rep exports for external-tool consumption (R, GCTA, PLINK,
# sparseREML).  Three independent, on-demand rules — each is a leaf target
# requested by path, e.g.:
#
#   snakemake results/{folder}/{scenario}/rep{N}/exports/pairwise_relatedness.tsv
#   snakemake results/{folder}/{scenario}/rep{N}/exports/grm/sparse.grm.sp.bin
#
# No aggregator rule, no scenario-level sentinel — exports are opt-in and
# are not consumed by any downstream pipeline stage.
# ---------------------------------------------------------------------------


ruleorder: export_cumulative_incidence > parquet_to_tsv_uncompressed
ruleorder: export_pairwise_relatedness > parquet_to_tsv_uncompressed


rule export_cumulative_incidence:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet",
    output:
        "results/{folder}/{scenario}/rep{rep}/exports/cumulative_incidence.tsv",
    log:
        "logs/{folder}/{scenario}/rep{rep}/export_cumulative_incidence.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/export_cumulative_incidence.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_pheno"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_pheno"),
    params:
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
    script:
        "../scripts/export_cumulative_incidence.py"


rule export_pairwise_relatedness:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        "results/{folder}/{scenario}/rep{rep}/exports/pairwise_relatedness.tsv",
    log:
        "logs/{folder}/{scenario}/rep{rep}/export_pairwise_relatedness.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/export_pairwise_relatedness.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        min_kinship=lambda w: get_param(
            config, w.scenario, "export_pair_list_min_kinship"
        ),
    script:
        "../scripts/export_pairwise_relatedness.py"


rule export_sparse_grm:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        bin="results/{folder}/{scenario}/rep{rep}/exports/grm/sparse.grm.sp.bin",
        ids="results/{folder}/{scenario}/rep{rep}/exports/grm/sparse.grm.id",
    log:
        "logs/{folder}/{scenario}/rep{rep}/export_sparse_grm.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/export_sparse_grm.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        grm_threshold=lambda w: get_param(config, w.scenario, "export_grm_threshold"),
    script:
        "../scripts/export_sparse_grm.py"
