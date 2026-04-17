# ---------------------------------------------------------------------------
# Opt-in per-rep exports for external fitting / analysis tools
# (R / GCTA / PLINK / sparseREML).
#
# Each rule is a leaf target — no scenario sentinel, no downstream consumers.
# Generic parquet → tsv conversion lives in sim_ace/utils.smk.
# ---------------------------------------------------------------------------


ruleorder: export_cumulative_incidence > parquet_to_tsv_uncompressed
ruleorder: export_pairwise_relatedness > parquet_to_tsv_uncompressed
ruleorder: export_inbreeding > parquet_to_tsv_uncompressed


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
        "../../scripts/fit_ace/export_cumulative_incidence.py"


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
        "../../scripts/fit_ace/export_pairwise_relatedness.py"


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
        "../../scripts/fit_ace/export_sparse_grm.py"


rule export_inbreeding:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        "results/{folder}/{scenario}/rep{rep}/exports/inbreeding.tsv",
    log:
        "logs/{folder}/{scenario}/rep{rep}/export_inbreeding.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/export_inbreeding.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    script:
        "../../scripts/fit_ace/export_inbreeding.py"


rule export_pgs:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        parquet="results/{folder}/{scenario}/rep{rep}/exports/pgs.parquet",
        meta="results/{folder}/{scenario}/rep{rep}/exports/pgs.meta.json",
    log:
        "logs/{folder}/{scenario}/rep{rep}/export_pgs.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/export_pgs.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped"),
    params:
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        pgs_r2=lambda w: get_param(config, w.scenario, "export_pgs_r2"),
        rA=lambda w: get_param(config, w.scenario, "rA"),
        A1=lambda w: get_param(config, w.scenario, "A1"),
        A2=lambda w: get_param(config, w.scenario, "A2"),
    script:
        "../../scripts/fit_ace/export_pgs.py"
