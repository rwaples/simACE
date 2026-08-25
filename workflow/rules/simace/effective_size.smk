# ---------------------------------------------------------------------------
# Effective population size (Ne) — opt-in target.
#
# These rules are NOT consumed by `stats.done`.  Run them explicitly:
#     snakemake results/{folder}/{scenario}/effective_size.done
# ---------------------------------------------------------------------------


rule effective_size_phenotype:
    """Per-rep Ne estimators on the observed-and-ancestors sub-pedigree.

    The post-ascertainment trait.parquet is the canonical input (it contains
    the ascertained subset). `pedigree.parquet` is the ancestor closure.

    `skip_ne_coancestry`: default True.  ne_coancestry is skipped (full
    sparse K not built) unless a scenario opts back in with False — the
    coancestry DP dominates this rule's memory and OOMs on large pedigrees.
    """
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
        phenotype="results/{folder}/{scenario}/rep{rep}/trait.parquet",
        params="results/{folder}/{scenario}/rep{rep}/params.yaml",
    output:
        stats="results/{folder}/{scenario}/rep{rep}/effective_size.yaml",
    log:
        "logs/{folder}/{scenario}/rep{rep}/effective_size.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/effective_size.tsv"
    threads: 1  # compute_all_ne is sequential numba DP; no internal parallelism.
    resources:
        # Streaming-θ DP (and the optional kinship matrix path) dominates RAM
        # at large N; _scale_mem_effective_size is the calibrated estimate
        # (α path or β path depending on skip_ne_coancestry).  The generic
        # _scale_mem floor of 4 GB underestimates by 2-3× at N=100K and
        # lets Snakemake over-parallelize into OOM.
        mem_mb=lambda w: _scale_mem_effective_size(config, w.scenario),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        skip_ne_coancestry=lambda w: get_param(config, w.scenario, "skip_ne_coancestry"),
    script:
        "../../scripts/simace/compute_effective_size.py"


rule effective_size_scenario:
    """Aggregate per-rep Ne yamls — opt-in target, NOT consumed by stats.done."""
    input:
        lambda w: [
            f"results/{w.folder}/{w.scenario}/rep{r}/effective_size.yaml"
            for r in range(1, get_param(config, w.scenario, "replicates") + 1)
        ],
    output:
        touch("results/{folder}/{scenario}/effective_size.done"),
