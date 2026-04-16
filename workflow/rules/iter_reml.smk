# ---------------------------------------------------------------------------
# Iterative sparse REML (ace_iter_reml) — A+C+E variance-component fit
# scaling to large n via PCG-AI-REML with optional two-stage deflation.
# ---------------------------------------------------------------------------


# Scenarios × precisions consumed by the bench rollup.  Keep in sync with
# config/iter_reml_bench.yaml: fp32 is run on every scenario; fp64 is
# only run at 10k (both AM and no-AM) as an accuracy/speed spot-check.
_ITER_REML_BENCH_SCENARIOS = [
    "iter_reml_10k",
    "iter_reml_10k_noam",
    "iter_reml_50k",
    "iter_reml_50k_noam",
    "iter_reml_100k",
    "iter_reml_100k_noam",
]
_ITER_REML_BENCH_FP64_SCENARIOS = ["iter_reml_10k", "iter_reml_10k_noam"]


wildcard_constraints:
    precision="fp32|fp64",


rule iter_reml_fit:
    """Fit A+C+E variance components via the ace_iter_reml binary.

Phase 1 RHE-mc warm-start → Phase 2 PCG-AI-REML.  Outputs match
fit_ace.sparse_reml's TSV schemas so downstream consumers are
interchangeable.  ``precision`` wildcard selects the PETSc build
(fp32 = default ~1.26× faster / 1.47× lower RAM; fp64 = reference).
"""
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet",
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        vc="results/{folder}/{scenario}/rep{rep}/iter_reml_{precision}/fit.vc.tsv",
        cov="results/{folder}/{scenario}/rep{rep}/iter_reml_{precision}/fit.cov.tsv",
        iter_log="results/{folder}/{scenario}/rep{rep}/iter_reml_{precision}/fit.iter.tsv",
        bench="results/{folder}/{scenario}/rep{rep}/iter_reml_{precision}/fit.bench.tsv",
        meta="results/{folder}/{scenario}/rep{rep}/iter_reml_{precision}/fit.vc.tsv.meta",
        phase1="results/{folder}/{scenario}/rep{rep}/iter_reml_{precision}/fit.phase1.tsv",
    log:
        "logs/{folder}/{scenario}/rep{rep}/iter_reml_{precision}.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/iter_reml_{precision}.tsv"
    threads: 8
    resources:
        # Rough memory model: O(nnz(V)) sparse + O(N·m) probe workspace +
        # O(N·k) deflation sketch.  Scales linearly with N_ped.
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped") * 2,
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped") * 2,
    params:
        precision=lambda w: w.precision,
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
        trait=lambda w: int(get_param(config, w.scenario, "iter_reml_trait")),
        ndegree=lambda w: int(get_param(config, w.scenario, "iter_reml_ndegree")),
        grm_threshold=lambda w: float(
            get_param(config, w.scenario, "iter_reml_grm_threshold")
        ),
        phase1_probes=lambda w: int(
            get_param(config, w.scenario, "iter_reml_phase1_probes")
        ),
        phase2_probes=lambda w: int(
            get_param(config, w.scenario, "iter_reml_phase2_probes")
        ),
        max_iter=lambda w: int(get_param(config, w.scenario, "iter_reml_max_iter")),
        tol=lambda w: float(get_param(config, w.scenario, "iter_reml_tol")),
        pcg_tol=lambda w: float(get_param(config, w.scenario, "iter_reml_pcg_tol")),
        pcg_max_iter=lambda w: int(
            get_param(config, w.scenario, "iter_reml_pcg_max_iter")
        ),
        pc_type=lambda w: str(get_param(config, w.scenario, "iter_reml_pc_type")),
        deflation_k=lambda w: int(
            get_param(config, w.scenario, "iter_reml_deflation_k")
        ),
        trace_method=lambda w: str(
            get_param(config, w.scenario, "iter_reml_trace_method")
        ),
        hutchpp_sketch_size=lambda w: int(
            get_param(config, w.scenario, "iter_reml_hutchpp_sketch_size")
        ),
        compute_logdet=lambda w: bool(
            get_param(config, w.scenario, "iter_reml_compute_logdet")
        ),
        slq_lanczos_steps=lambda w: int(
            get_param(config, w.scenario, "iter_reml_slq_lanczos_steps")
        ),
        slq_probes=lambda w: int(
            get_param(config, w.scenario, "iter_reml_slq_probes")
        ),
    script:
        "../scripts/run_iter_reml.py"


def _iter_reml_bench_inputs(wildcards):
    """Return the full (scenario × precision) fit output set for the rollup."""
    folder = wildcards.folder
    fits = []
    for scen in _ITER_REML_BENCH_SCENARIOS:
        fits.append(f"results/{folder}/{scen}/rep1/iter_reml_fp32/fit.vc.tsv.meta")
    for scen in _ITER_REML_BENCH_FP64_SCENARIOS:
        fits.append(f"results/{folder}/{scen}/rep1/iter_reml_fp64/fit.vc.tsv.meta")
    return fits


rule iter_reml_bench_summary:
    """Aggregate per-scenario fit outputs into a single bench table.

One row per (scenario, precision) tuple with wall time, peak RSS
(from Snakemake benchmark.tsv), iteration count, average PCG iters,
variance components, and logLik.
"""
    input:
        _iter_reml_bench_inputs,
    output:
        "results/{folder}/bench_summary.tsv",
    log:
        "logs/{folder}/bench_summary.log",
    params:
        scenarios_fp32=_ITER_REML_BENCH_SCENARIOS,
        scenarios_fp64=_ITER_REML_BENCH_FP64_SCENARIOS,
    script:
        "../scripts/collect_iter_reml_bench.py"
