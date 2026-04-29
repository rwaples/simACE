def _dropout_pedigree_input(w):
    """Pick the parametric or gene-drop-augmented pedigree based on use_gene_drop."""
    if get_param(config, w.scenario, "use_gene_drop"):
        return (
            f"results/{w.folder}/{w.scenario}/rep{w.rep}/pedigree.full.tstrait.parquet"
        )
    return f"results/{w.folder}/{w.scenario}/rep{w.rep}/pedigree.full.parquet"


rule pedigree_dropout:
    input:
        pedigree=_dropout_pedigree_input,
    output:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    log:
        "logs/{folder}/{scenario}/rep{rep}/dropout.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/dropout.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=5,
    params:
        pedigree_dropout_rate=lambda w: get_param(config, w.scenario, "pedigree_dropout_rate"),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    script:
        "../../scripts/simace/dropout.py"
