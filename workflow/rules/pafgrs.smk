# ---------------------------------------------------------------------------
# PA-FGRS scoring rules
# ---------------------------------------------------------------------------

PAFGRS_TRAITS = ["trait1", "trait2"]
PAFGRS_CIP_SOURCES = ["empirical", "true"]
PAFGRS_H2_SOURCES = ["true", "estimated"]
PAFGRS_RA_SOURCES = ["true", "estimated"]
PAFGRS_COV_MODELS = ["genetic", "genetic_c"]


def _pafgrs_score_outputs(folder, scenario, rep):
    """All score parquet + metric TSV paths for a single replicate."""
    base = f"results/{folder}/{scenario}/rep{rep}/pafgrs"
    out = []
    for t in PAFGRS_TRAITS:
        for c in PAFGRS_CIP_SOURCES:
            for h in PAFGRS_H2_SOURCES:
                out.append(f"{base}/scores_{t}_{c}_{h}.parquet")
                out.append(f"{base}/metrics_{t}_{c}_{h}.tsv")
    return out


rule pafgrs_score:
    """Run PA-FGRS scoring for all trait × CIP × h2 combinations."""
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet",
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        done=touch("results/{folder}/{scenario}/rep{rep}/pafgrs/score.done"),
    params:
        trait_nums=[1, 2],
        cip_sources=PAFGRS_CIP_SOURCES,
        h2_sources=PAFGRS_H2_SOURCES,
        ndegree=lambda w: get_param(config, w.scenario, "pafgrs_ndegree"),
        A1=lambda w: get_param(config, w.scenario, "A1"),
        A2=lambda w: get_param(config, w.scenario, "A2"),
        prevalence1=lambda w: get_param(config, w.scenario, "prevalence1"),
        prevalence2=lambda w: get_param(config, w.scenario, "prevalence2"),
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        beta1=lambda w: get_param(config, w.scenario, "beta1"),
        beta2=lambda w: get_param(config, w.scenario, "beta2"),
        phenotype_model1=lambda w: get_param(config, w.scenario, "phenotype_model1"),
        phenotype_model2=lambda w: get_param(config, w.scenario, "phenotype_model2"),
        phenotype_params1=lambda w: get_param(config, w.scenario, "phenotype_params1"),
        phenotype_params2=lambda w: get_param(config, w.scenario, "phenotype_params2"),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/pafgrs_score.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/pafgrs_score.tsv"
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped"),
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped") * 4,
    threads: 1
    script:
        "../scripts/pafgrs_score.py"


rule pafgrs_atlas:
    """Generate PA-FGRS diagnostic atlas PDF."""
    input:
        done="results/{folder}/{scenario}/rep{rep}/pafgrs/score.done",
    output:
        atlas="results/{folder}/{scenario}/rep{rep}/pafgrs/plots/atlas.pdf",
    params:
        plot_format=lambda w: config["defaults"].get("plot_format", "png"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/pafgrs_atlas.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/pafgrs_atlas.tsv"
    resources:
        mem_mb=2000,
        runtime=5,
    threads: 1
    script:
        "../scripts/pafgrs_atlas.py"


rule pafgrs_bivariate_score:
    """Run bivariate PA-FGRS scoring for all variant combinations."""
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.parquet",
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
    output:
        done=touch("results/{folder}/{scenario}/rep{rep}/pafgrs_bivariate/score.done"),
    params:
        cip_sources=PAFGRS_CIP_SOURCES,
        h2_sources=PAFGRS_H2_SOURCES,
        rA_sources=PAFGRS_RA_SOURCES,
        cov_models=PAFGRS_COV_MODELS,
        ndegree=lambda w: get_param(config, w.scenario, "pafgrs_ndegree"),
        A1=lambda w: get_param(config, w.scenario, "A1"),
        A2=lambda w: get_param(config, w.scenario, "A2"),
        C1=lambda w: get_param(config, w.scenario, "C1"),
        C2=lambda w: get_param(config, w.scenario, "C2"),
        rA=lambda w: get_param(config, w.scenario, "rA"),
        rC=lambda w: get_param(config, w.scenario, "rC"),
        prevalence1=lambda w: get_param(config, w.scenario, "prevalence1"),
        prevalence2=lambda w: get_param(config, w.scenario, "prevalence2"),
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        beta1=lambda w: get_param(config, w.scenario, "beta1"),
        beta2=lambda w: get_param(config, w.scenario, "beta2"),
        phenotype_model1=lambda w: get_param(config, w.scenario, "phenotype_model1"),
        phenotype_model2=lambda w: get_param(config, w.scenario, "phenotype_model2"),
        phenotype_params1=lambda w: get_param(config, w.scenario, "phenotype_params1"),
        phenotype_params2=lambda w: get_param(config, w.scenario, "phenotype_params2"),
        seed=lambda w: get_param(config, w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/pafgrs_bivariate_score.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/pafgrs_bivariate_score.tsv"
    resources:
        mem_mb=lambda w: _scale_mem(config, w.scenario, "G_ped") * 2,
        runtime=lambda w: _scale_runtime(config, w.scenario, "G_ped") * 8,
    threads: 1
    script:
        "../scripts/pafgrs_bivariate_score.py"


rule pafgrs_all:
    """Run PA-FGRS for all scenarios and replicates."""
    input:
        [
            f"results/{get_folder(config, s)}/{s}/rep{r}/pafgrs/score.done"
            for s in config["scenarios"]
            for r in range(1, get_param(config, s, "replicates") + 1)
        ],
