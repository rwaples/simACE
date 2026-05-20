# ---------------------------------------------------------------------------
# Ascertainment stage (per ADR 0001)
#
# Unified dropout + case-weighted N_sample draw, applied after censor. Writes
# the canonical post-stage pedigree.parquet / trait.parquet / trait.simple_ltm.parquet
# outputs that both simACE-stats and fitACE consume.
# ---------------------------------------------------------------------------


rule ascertainment:
    """Unified ascertainment: dropout + case-weighted N_sample draw."""
    input:
        pedigree=lambda w: _pre_ascertainment_pedigree_input(w, config),
        trait="results/{folder}/{scenario}/rep{rep}/trait.full.parquet",
        trait_simple_ltm="results/{folder}/{scenario}/rep{rep}/trait.simple_ltm.full.parquet",
    output:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
        trait="results/{folder}/{scenario}/rep{rep}/trait.parquet",
        trait_simple_ltm="results/{folder}/{scenario}/rep{rep}/trait.simple_ltm.parquet",
    log:
        "logs/{folder}/{scenario}/rep{rep}/ascertainment.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/ascertainment.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        dropout_rate=lambda w: get_param(config, w.scenario, "dropout_rate"),
        case_ascertainment_ratio=lambda w: get_param(config, w.scenario, "case_ascertainment_ratio"),
        N_sample=lambda w: get_param(config, w.scenario, "N_sample"),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    script:
        "../../scripts/simace/ascertainment.py"
