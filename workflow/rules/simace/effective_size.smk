# ---------------------------------------------------------------------------
# Effective population size (Ne) — opt-in target.
#
# These rules are NOT consumed by `stats.done`.  Run them explicitly:
#     snakemake results/{folder}/{scenario}/effective_size.done
# ---------------------------------------------------------------------------


rule effective_size_phenotype:
    """Per-rep Ne estimators on the observed-and-ancestors sub-pedigree.

The phenotype.sampled.parquet is `temp()` upstream — opting in after a
normal stats run will trigger a sample-step rebuild (cheap; phenotype
and earlier files are persistent).
"""
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.sampled.parquet",
        params="results/{folder}/{scenario}/rep{rep}/params.yaml",
    output:
        stats="results/{folder}/{scenario}/rep{rep}/effective_size.yaml",
    log:
        "logs/{folder}/{scenario}/rep{rep}/effective_size.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/effective_size.tsv"
    threads: 1  # compute_all_ne is sequential numba DP; no internal parallelism.
    resources:
        # Reuses stats_phenotype's scaling factor.  At very large N the sparse
        # kinship matrix may dominate — bump the G_ped multiplier in
        # _scale_mem if this OOMs on bigger scenarios.
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
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
