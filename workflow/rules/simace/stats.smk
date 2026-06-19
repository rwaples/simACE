# ---------------------------------------------------------------------------
# Statistics and plotting rules (sim-side)
# ---------------------------------------------------------------------------
#
# NOTE: report.yaml + plot_payload.yaml + plotting_sample.parquet are produced
# by the combined `analyze` rule (analyze.smk, ADR 0006/0007/0008). report.yaml
# is the curated v2 scientific report; plot_payload.yaml holds the dense plot
# arrays. The standalone simace-phenotype-stats CLI / script wrapper is retained
# for debugging.


rule plot_phenotype:
    input:
        report=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/report.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        plot_payload=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/plot_payload.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
        samples=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/plotting_sample.parquet",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(config, w.scenario, "replicates") + 1),
        ),
    output:
        expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=PHENOTYPE_PLOTS),
    log:
        "logs/{folder}/{scenario}/plot_phenotype.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/plot_phenotype.tsv"
    threads: 1
    resources:
        mem_mb=2000,
        runtime=5,
    params:
        censor_age=lambda w: get_param(config, w.scenario, "censor_age"),
        gen_censoring=lambda w: get_param(config, w.scenario, "gen_censoring"),
        max_degree=lambda w: get_param(config, w.scenario, "max_degree"),
        plot_format=lambda w: config["defaults"].get("plot_format", "png"),
    script:
        "../../scripts/simace/plot_phenotype.py"


# Shared input + param spec for the scenario atlas, consumed by both the
# default atlas.html rule and the on-demand atlas.pdf rule (ADR 0010). Defined
# here (not common.py) because `expand`, `PHENOTYPE_PLOTS`, and `config` are in
# scope in the rule namespace.
def _scenario_atlas_inputs(w):
    reps = range(1, get_param(config, w.scenario, "replicates") + 1)
    return {
        "phenotype": expand(
            "results/{folder}/{scenario}/plots/{plot}",
            folder=w.folder,
            scenario=w.scenario,
            plot=PHENOTYPE_PLOTS,
        ),
        "params_yaml": f"results/{w.folder}/{w.scenario}/rep1/params.yaml",
        "report": expand(
            "results/{folder}/{scenario}/rep{rep}/report.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=reps,
        ),
        "plot_payload": expand(
            "results/{folder}/{scenario}/rep{rep}/plot_payload.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=reps,
        ),
    }


# Config keys passed via the `meta` param and merged onto rep1/params.yaml in
# the script. `scenario` (a wildcard) and `plot_format` (a config default) are
# added separately below.  PR3: prevalence lives inside phenotype_params{N},
# which the atlas's plot_pipeline reads directly.
_SCENARIO_ATLAS_PARAM_KEYS = (
    "replicates",
    "folder",
    "standardize",
    "beta1",
    "beta_sex1",
    "phenotype_model1",
    "phenotype_params1",
    "beta2",
    "beta_sex2",
    "phenotype_model2",
    "phenotype_params2",
    "censor_age",
    "gen_censoring",
    "death_scale",
    "death_rho",
    "G_pheno",
    "N_sample",
    "dropout_rate",
    "case_ascertainment_ratio",
    "max_degree",
)


def _scenario_atlas_params(w):
    meta = {
        key: get_param(config, w.scenario, key) for key in _SCENARIO_ATLAS_PARAM_KEYS
    }
    meta["scenario"] = w.scenario
    meta["plot_format"] = config["defaults"].get("plot_format", "png")
    return meta


rule assemble_scenario_atlas:
    input:
        unpack(_scenario_atlas_inputs),
    output:
        "results/{folder}/{scenario}/plots/atlas.html",
    log:
        "logs/{folder}/{scenario}/assemble_atlas.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/assemble_atlas.tsv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    params:
        meta=_scenario_atlas_params,
    script:
        "../../scripts/simace/assemble_atlas.py"


rule assemble_scenario_atlas_pdf:
    input:
        unpack(_scenario_atlas_inputs),
    output:
        "results/{folder}/{scenario}/plots/atlas.pdf",
    log:
        "logs/{folder}/{scenario}/assemble_atlas_pdf.log",
    benchmark:
        "benchmarks/{folder}/{scenario}/assemble_atlas_pdf.tsv"
    threads: 1
    resources:
        mem_mb=1000,
        runtime=5,
    params:
        meta=_scenario_atlas_params,
    script:
        "../../scripts/simace/assemble_atlas.py"
