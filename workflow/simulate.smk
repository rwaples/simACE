configfile: "config/config.yaml"


def get_param(scenario, param):
    """Get parameter value, falling back to defaults if not specified in scenario."""
    scenario_config = config["scenarios"].get(scenario, {})
    if param in scenario_config:
        return scenario_config[param]
    return config["defaults"][param]


def get_folder(scenario):
    """Get the folder grouping for a scenario."""
    return get_param(scenario, "folder")


def get_scenarios_for_folder(folder):
    """Return scenario names assigned to the given folder."""
    return [s for s in config["scenarios"] if get_folder(s) == folder]


def get_all_folders():
    """Return sorted unique folder names across all scenarios."""
    return sorted(set(get_folder(s) for s in config["scenarios"]))


PHENOTYPE_PLOTS = [
    "mortality.png",
    "age_at_onset_death.png",
    "liability_vs_aoo.png",
    "cross_trait.png",
    "cross_trait.weibull.png",
    "liability_violin.weibull.png",
    "cumulative_incidence.weibull.png",
    "censoring.png",
    "joint_affected.weibull.png",
    "tetrachoric.weibull.png",
]

THRESHOLD_PLOTS = [
    "prevalence_by_generation.png",
    "liability_violin.threshold.png",
    "tetrachoric.threshold.png",
    "joint_affected.threshold.png",
    "cross_trait.threshold.png",
]


wildcard_constraints:
    folder="[a-zA-Z0-9_]+",
    scenario="[a-zA-Z0-9_]+",
    rep="\\d+"


def get_scenario_sim_outputs(scenario):
    """Generate simulation, validation, and plot outputs for a single scenario."""
    folder = get_folder(scenario)
    n_reps = get_param(scenario, "replicates")
    outputs = []
    for rep in range(1, n_reps + 1):
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/pedigree.parquet")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype.weibull.parquet")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.parquet")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/validation.yaml")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype_stats.yaml")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/threshold_stats.yaml")
    for plot in PHENOTYPE_PLOTS:
        outputs.append(f"results/{folder}/{scenario}/plots/{plot}")
    for plot in THRESHOLD_PLOTS:
        outputs.append(f"results/{folder}/{scenario}/plots/{plot}")
    return outputs


def get_folder_validations(folder):
    """Generate validation file paths for scenarios in a given folder."""
    validations = []
    for scenario in get_scenarios_for_folder(folder):
        n_reps = get_param(scenario, "replicates")
        for rep in range(1, n_reps + 1):
            validations.append(f"results/{folder}/{scenario}/rep{rep}/validation.yaml")
    return validations


def get_phenotype_plot_outputs():
    """Generate phenotype plot output paths across scenarios."""
    outputs = []
    for scenario in config["scenarios"].keys():
        folder = get_folder(scenario)
        for plot in PHENOTYPE_PLOTS:
            outputs.append(f"results/{folder}/{scenario}/plots/{plot}")
    return outputs


rule all:
    input:
        [f"results/{get_folder(s)}/{s}/scenario.done" for s in config["scenarios"]]


rule folder:
    """Build all scenario outputs for a single folder grouping."""
    input:
        lambda w: [f"results/{w.folder}/{s}/scenario.done"
                    for s in get_scenarios_for_folder(w.folder)]
    output:
        touch("results/{folder}/folder.done")


rule scenario:
    """Build all sim/validation/plot outputs for a single scenario."""
    input:
        lambda w: get_scenario_sim_outputs(w.scenario),
        lambda w: f"results/{get_folder(w.scenario)}/validation_summary.tsv",
        lambda w: f"results/{get_folder(w.scenario)}/plots/variance_components.png",
    output:
        touch("results/{folder}/{scenario}/scenario.done")


rule simulate:
    output:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
        params="results/{folder}/{scenario}/rep{rep}/params.yaml"
    params:
        seed=lambda w: get_param(w.scenario, "seed") + int(w.rep) - 1,
        N=lambda w: get_param(w.scenario, "N"),
        G_ped=lambda w: get_param(w.scenario, "G_ped"),
        G_sim=lambda w: get_param(w.scenario, "G_sim"),
        fam_size=lambda w: get_param(w.scenario, "fam_size"),
        p_mztwin=lambda w: get_param(w.scenario, "p_mztwin"),
        p_nonsocial_father=lambda w: get_param(w.scenario, "p_nonsocial_father"),
        rep=lambda w: int(w.rep),
        A1=lambda w: get_param(w.scenario, "A1"),
        C1=lambda w: get_param(w.scenario, "C1"),
        A2=lambda w: get_param(w.scenario, "A2"),
        C2=lambda w: get_param(w.scenario, "C2"),
        rA=lambda w: get_param(w.scenario, "rA"),
        rC=lambda w: get_param(w.scenario, "rC"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/simulate.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/simulate.tsv"
    resources:
        mem_mb=8000,
        runtime=10
    threads: 1
    script:
        "scripts/simulate.py"


rule phenotype_weibull:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet"
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.weibull.parquet"
    params:
        seed=lambda w: get_param(w.scenario, "seed") + int(w.rep) - 1,
        # Trait 1 phenotype parameters
        beta1=lambda w: get_param(w.scenario, "beta1"),
        rate1=lambda w: get_param(w.scenario, "rate1"),
        k1=lambda w: get_param(w.scenario, "k1"),
        # Trait 2 phenotype parameters
        beta2=lambda w: get_param(w.scenario, "beta2"),
        rate2=lambda w: get_param(w.scenario, "rate2"),
        k2=lambda w: get_param(w.scenario, "k2"),
        # Shared parameters
        standardize=lambda w: get_param(w.scenario, "standardize"),
        censor_age=lambda w: get_param(w.scenario, "censor_age"),
        death_rate=lambda w: get_param(w.scenario, "death_rate"),
        death_k=lambda w: get_param(w.scenario, "death_k"),
        young_gen_censoring=lambda w: get_param(w.scenario, "young_gen_censoring"),
        middle_gen_censoring=lambda w: get_param(w.scenario, "middle_gen_censoring"),
        old_gen_censoring=lambda w: get_param(w.scenario, "old_gen_censoring"),
        G_pheno=lambda w: get_param(w.scenario, "G_pheno"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_weibull.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_weibull.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "scripts/phenotype.py"


rule phenotype_threshold:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet"
    output:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.parquet"
    params:
        prevalence1=lambda w: get_param(w.scenario, "prevalence1"),
        prevalence2=lambda w: get_param(w.scenario, "prevalence2"),
        G_pheno=lambda w: get_param(w.scenario, "G_pheno"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_threshold.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_threshold.tsv"
    resources:
        mem_mb=2000,
        runtime=5
    threads: 1
    script:
        "scripts/phenotype_threshold.py"


rule validate:
    input:
        pedigree="results/{folder}/{scenario}/rep{rep}/pedigree.parquet",
        params="results/{folder}/{scenario}/rep{rep}/params.yaml"
    output:
        report="results/{folder}/{scenario}/rep{rep}/validation.yaml"
    log:
        "logs/{folder}/{scenario}/rep{rep}/validate.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/validate.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "scripts/validate.py"


rule phenotype_stats:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.weibull.parquet"
    output:
        stats="results/{folder}/{scenario}/rep{rep}/phenotype_stats.yaml",
        samples="results/{folder}/{scenario}/rep{rep}/phenotype_samples.parquet"
    params:
        seed=lambda w: get_param(w.scenario, "seed") + int(w.rep) - 1,
        censor_age=lambda w: get_param(w.scenario, "censor_age"),
        young_gen_censoring=lambda w: get_param(w.scenario, "young_gen_censoring"),
        middle_gen_censoring=lambda w: get_param(w.scenario, "middle_gen_censoring"),
        old_gen_censoring=lambda w: get_param(w.scenario, "old_gen_censoring"),
    log:
        "logs/{folder}/{scenario}/rep{rep}/phenotype_stats.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/phenotype_stats.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "scripts/compute_phenotype_stats.py"


rule gather_validation:
    input:
        validations=lambda w: get_folder_validations(w.folder)
    output:
        tsv="results/{folder}/validation_summary.tsv"
    log:
        "logs/{folder}/gather_validation.log"
    benchmark:
        "benchmarks/{folder}/gather_validation.tsv"
    resources:
        mem_mb=1000,
        runtime=5
    threads: 1
    script:
        "scripts/gather_validation.py"


rule plot_phenotype:
    input:
        stats=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/phenotype_stats.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(w.scenario, "replicates") + 1),
        ),
        samples=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/phenotype_samples.parquet",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(w.scenario, "replicates") + 1),
        ),
    params:
        censor_age=lambda w: get_param(w.scenario, "censor_age"),
        young_gen_censoring=lambda w: get_param(w.scenario, "young_gen_censoring"),
        middle_gen_censoring=lambda w: get_param(w.scenario, "middle_gen_censoring"),
        old_gen_censoring=lambda w: get_param(w.scenario, "old_gen_censoring"),
    output:
        expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=PHENOTYPE_PLOTS)
    log:
        "logs/{folder}/{scenario}/plot_phenotype.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/plot_phenotype.tsv"
    resources:
        mem_mb=2000,
        runtime=5
    threads: 1
    script:
        "scripts/plot_phenotype.py"


rule threshold_stats:
    input:
        phenotype="results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.parquet"
    output:
        stats="results/{folder}/{scenario}/rep{rep}/threshold_stats.yaml",
        samples="results/{folder}/{scenario}/rep{rep}/threshold_samples.parquet"
    params:
        seed=lambda w: get_param(w.scenario, "seed") + int(w.rep) - 1,
    log:
        "logs/{folder}/{scenario}/rep{rep}/threshold_stats.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/rep{rep}/threshold_stats.tsv"
    resources:
        mem_mb=4000,
        runtime=10
    threads: 1
    script:
        "scripts/compute_threshold_stats.py"


rule plot_threshold:
    input:
        stats=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/threshold_stats.yaml",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(w.scenario, "replicates") + 1),
        ),
        samples=lambda w: expand(
            "results/{folder}/{scenario}/rep{rep}/threshold_samples.parquet",
            folder=w.folder,
            scenario=w.scenario,
            rep=range(1, get_param(w.scenario, "replicates") + 1),
        ),
    params:
        prevalence1=lambda w: get_param(w.scenario, "prevalence1"),
        prevalence2=lambda w: get_param(w.scenario, "prevalence2"),
    output:
        expand("results/{{folder}}/{{scenario}}/plots/{plot}", plot=THRESHOLD_PLOTS)
    log:
        "logs/{folder}/{scenario}/plot_threshold.log"
    benchmark:
        "benchmarks/{folder}/{scenario}/plot_threshold.tsv"
    resources:
        mem_mb=2000,
        runtime=5
    threads: 1
    script:
        "scripts/plot_threshold.py"


rule plot_validation:
    input:
        tsv="results/{folder}/validation_summary.tsv"
    output:
        "results/{folder}/plots/variance_components.png",
        "results/{folder}/plots/twin_rate.png",
        "results/{folder}/plots/correlations_A.png",
        "results/{folder}/plots/correlations_phenotype.png",
        "results/{folder}/plots/heritability_estimates.png",
        "results/{folder}/plots/half_sib_proportions.png",
        "results/{folder}/plots/cross_trait_correlations.png",
        "results/{folder}/plots/family_size.png",
        "results/{folder}/plots/summary_bias.png",
        "results/{folder}/plots/runtime.png",
        "results/{folder}/plots/memory.png"
    log:
        "logs/{folder}/plot_validation.log"
    benchmark:
        "benchmarks/{folder}/plot_validation.tsv"
    resources:
        mem_mb=1000,
        runtime=5
    threads: 1
    script:
        "scripts/plot_validation.py"
