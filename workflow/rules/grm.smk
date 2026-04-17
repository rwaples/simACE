# ---------------------------------------------------------------------------
# GRM outputs per replicate
# ---------------------------------------------------------------------------
#
#   grm_matrix : pedigree → sparse GRM (2φ) written in the ACEGRM binary
#                format that ace_iter_reml consumes (also readable back in
#                Python via sim_ace.analysis.export_grm.read_sparse_grm_binary).
#   grm_pcs    : ACEGRM binary → top-n_pcs eigenvectors + eigenvalues TSV.
#   plot_grm_pcs: scree + PC1 histogram + PC2v3 / PC4v5 scatters.
#
# Splitting build and eigendecomposition lets the PCA (and future REML)
# consumers reuse the cached matrix, and exposes the familiar
# ``A.grm.sp.bin`` / ``A.grm.id`` file pair for external tools.
# ---------------------------------------------------------------------------


rule grm_matrix:
    """Build the sparse GRM (2φ) and export in ACEGRM binary format."""
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        grm_bin="results/{folder}/{scenario}/rep{rep}/grm/A.grm.sp.bin",
        grm_id="results/{folder}/{scenario}/rep{rep}/grm/A.grm.id",
    log:
        "logs/{folder}/{scenario}/rep{rep}/grm_matrix.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/grm_matrix.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario),
        runtime=lambda w: _scale_runtime(config, w.scenario),
    params:
        min_kinship=lambda w: get_param(config, w.scenario, "grm_min_kinship"),
    script:
        "../scripts/grm_matrix.py"


rule grm_pcs:
    """Top-n_pcs eigenvectors of the cached sparse GRM."""
    input:
        grm_bin="results/{folder}/{scenario}/rep{rep}/grm/A.grm.sp.bin",
        grm_id="results/{folder}/{scenario}/rep{rep}/grm/A.grm.id",
    output:
        pcs="results/{folder}/{scenario}/rep{rep}/grm/pcs.parquet",
        eigenvalues="results/{folder}/{scenario}/rep{rep}/grm/eigenvalues.tsv",
    log:
        "logs/{folder}/{scenario}/rep{rep}/grm_pcs.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/grm_pcs.tsv"
    threads: 1
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario),
        runtime=lambda w: _scale_runtime(config, w.scenario),
    params:
        n_pcs=lambda w: get_param(config, w.scenario, "grm_n_pcs"),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    script:
        "../scripts/grm_pcs.py"


rule plot_grm_pcs:
    """Plot scree + PC1 histogram + PC2v3 / PC4v5 scatters (affected-colored)."""
    input:
        pcs="results/{folder}/{scenario}/rep{rep}/grm/pcs.parquet",
        eigenvalues="results/{folder}/{scenario}/rep{rep}/grm/eigenvalues.tsv",
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet",
    output:
        plot="results/{folder}/{scenario}/rep{rep}/grm/plots/pcs.pdf",
    log:
        "logs/{folder}/{scenario}/rep{rep}/plot_grm_pcs.log",
    threads: 1
    resources:
        mem_mb=2000,
        runtime=5,
    script:
        "../scripts/plot_grm_pcs.py"
