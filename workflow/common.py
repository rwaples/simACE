"""Shared helpers for the ACE Snakemake workflows."""


def get_param(config, scenario, param):
    """Get parameter value, falling back to defaults if not specified in scenario."""
    scenario_config = config["scenarios"].get(scenario, {})
    if param in scenario_config:
        return scenario_config[param]
    return config["defaults"][param]


def get_folder(config, scenario):
    """Get the folder grouping for a scenario."""
    return get_param(config, scenario, "folder")


def get_scenarios_for_folder(config, folder):
    """Return scenario names assigned to the given folder."""
    return [s for s in config["scenarios"] if get_folder(config, s) == folder]


def get_all_folders(config):
    """Return sorted unique folder names across all scenarios."""
    return sorted(set(get_folder(config, s) for s in config["scenarios"]))


PHENOTYPE_PLOTS = [
    "mortality.png",
    "age_at_onset_death.png",
    "liability_vs_aoo.png",
    "cross_trait.png",
    "cross_trait.weibull.png",
    "liability_violin.weibull.png",
    "liability_violin.weibull.by_generation.png",
    "cumulative_incidence.weibull.png",
    "censoring.png",
    "joint_affected.weibull.png",
    "tetrachoric.weibull.png",
    "tetrachoric.weibull.by_generation.png",
    "parent_offspring_liability.by_generation.png",
]

THRESHOLD_PLOTS = [
    "prevalence_by_generation.png",
    "liability_violin.threshold.png",
    "liability_violin.threshold.by_generation.png",
    "tetrachoric.threshold.png",
    "joint_affected.threshold.png",
    "cross_trait.threshold.png",
]


def get_scenario_sim_outputs(config, scenario):
    """Generate simulation, validation, and plot outputs for a single scenario."""
    folder = get_folder(config, scenario)
    n_reps = get_param(config, scenario, "replicates")
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


def get_folder_validations(config, folder):
    """Generate validation file paths for scenarios in a given folder."""
    validations = []
    for scenario in get_scenarios_for_folder(config, folder):
        n_reps = get_param(config, scenario, "replicates")
        for rep in range(1, n_reps + 1):
            validations.append(f"results/{folder}/{scenario}/rep{rep}/validation.yaml")
    return validations


