# ---------------------------------------------------------------------------
# Export rules: convert simulation outputs to R-friendly formats
# ---------------------------------------------------------------------------


rule parquet_to_tsv:
    """Convert any .parquet file to .tsv.gz for use in R."""
    input:
        "{path}.parquet"
    output:
        "{path}.tsv.gz"
    params:
        float_precision = config["defaults"].get("tsv_float_precision", 4),
    log:
        "logs/{path}.parquet_to_tsv.log"
    resources:
        mem_mb  = 2000,
        runtime = 5
    threads: 1
    script:
        "../scripts/parquet_to_tsv.py"
