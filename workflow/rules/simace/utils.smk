# ---------------------------------------------------------------------------
# Generic utility rules (parquet → tsv conversion)
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
        "../../scripts/simace/parquet_to_tsv.py"


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
        "../../scripts/simace/parquet_to_tsv.py"
