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


# -- Plot filename basenames (without extension) --

_PHENOTYPE_BASENAMES = [
    "mortality",
    "age_at_onset_death",
    "liability_vs_aoo",
    "cross_trait",
    "cross_trait.weibull",
    "liability_violin.weibull",
    "liability_violin.weibull.by_generation",
    "cumulative_incidence.weibull",
    "censoring",
    "joint_affected.weibull",
    "tetrachoric.weibull",
    "tetrachoric.weibull.by_generation",
    "parent_offspring_liability.by_generation",
]

_THRESHOLD_BASENAMES = [
    "prevalence_by_generation",
    "liability_violin.threshold",
    "liability_violin.threshold.by_generation",
    "tetrachoric.threshold",
    "joint_affected.threshold",
    "cross_trait.threshold",
]

_VALIDATION_BASENAMES = [
    "variance_components",
    "twin_rate",
    "correlations_A",
    "correlations_phenotype",
    "heritability_estimates",
    "half_sib_proportions",
    "cross_trait_correlations",
    "family_size",
    "summary_bias",
    "runtime",
    "memory",
]


def plot_filenames(basenames, ext="png"):
    """Return plot filenames by appending the given extension to each basename."""
    return [f"{name}.{ext}" for name in basenames]


def get_scenario_sim_outputs(config, scenario, plot_ext="png"):
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
    for plot in plot_filenames(_PHENOTYPE_BASENAMES, plot_ext):
        outputs.append(f"results/{folder}/{scenario}/plots/{plot}")
    for plot in plot_filenames(_THRESHOLD_BASENAMES, plot_ext):
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
