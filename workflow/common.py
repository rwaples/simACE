"""Shared helpers for the ACE Snakemake workflows."""


def get_param(config, scenario, param):
    """Get parameter value, falling back to defaults if not specified in scenario."""
    scenario_config = config["scenarios"].get(scenario, {})
    if param in scenario_config:
        return scenario_config[param]
    return config["defaults"][param]


def _scale_mem(config, scenario, gen_key="G_pheno", mb_per_1k=2, floor=4000):
    """Estimate mem_mb from population size: N × G × mb_per_1k/1000, with a floor."""
    n = get_param(config, scenario, "N")
    g = get_param(config, scenario, gen_key)
    return max(floor, int(n * g * mb_per_1k / 1000))


def _scale_runtime(config, scenario, gen_key="G_pheno", min_per_1M=5, floor=5):
    """Estimate runtime (minutes) from population size."""
    n = get_param(config, scenario, "N")
    g = get_param(config, scenario, gen_key)
    return max(floor, int(n * g * min_per_1M / 1_000_000))


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

# Ordered by narrative flow: liability structure -> frailty phenotype ->
# censoring -> familial correlations & heritability.
_PHENOTYPE_BASENAMES = [
    # Pedigree structure
    "pedigree_counts.ped",
    "pedigree_counts",
    # Liability structure
    "cross_trait",
    # Liability-scale heritability (pedigree + liability only)
    "parent_offspring_liability.by_generation",
    "heritability.by_generation",
    "broad_heritability.by_generation",
    # Liability by affected status
    "cross_trait.frailty",
    "liability_violin.frailty",
    "liability_violin.frailty.by_generation",
    # Frailty phenotype
    "liability_vs_aoo",
    "age_at_onset_death",
    "mortality",
    "cumulative_incidence.frailty",
    "cumulative_incidence.by_sex",
    "cumulative_incidence.by_sex.by_generation",
    # Censoring
    "censoring",
    "censoring_confusion",
    # Familial correlations
    "joint_affected.frailty",
    "cross_trait_frailty.by_generation",
    "tetrachoric.frailty",
    "tetrachoric.frailty.by_generation",
    "cross_trait_tetrachoric",
]

# Ordered to mirror frailty: prevalence -> liability -> correlations.
_THRESHOLD_BASENAMES = [
    "prevalence_by_generation",
    "cross_trait.threshold",
    "liability_violin.threshold",
    "liability_violin.threshold.by_generation",
    "joint_affected.threshold",
    "tetrachoric.threshold",
    "cross_trait_tetrachoric.threshold",
]

# Ordered: pedigree structure -> variance & heritability -> cross-trait ->
# summary -> benchmarks.
_VALIDATION_BASENAMES = [
    # Pedigree structure
    "family_size",
    "twin_rate",
    "half_sib_proportions",
    # Variance components & heritability
    "variance_components",
    "correlations_A",
    "correlations_phenotype",
    "heritability_estimates",
    # Cross-trait
    "cross_trait_correlations",
    # Summary
    "summary_bias",
    # Benchmarks
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
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype.parquet")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype.sampled.parquet")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.parquet")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype.liability_threshold.sampled.parquet")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/validation.yaml")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype_stats.yaml")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/threshold_stats.yaml")
    for plot in plot_filenames(_PHENOTYPE_BASENAMES, plot_ext):
        outputs.append(f"results/{folder}/{scenario}/plots/{plot}")
    for plot in plot_filenames(_THRESHOLD_BASENAMES, plot_ext):
        outputs.append(f"results/{folder}/{scenario}/plots/{plot}")
    outputs.append(f"results/{folder}/{scenario}/plots/atlas.pdf")
    return outputs


def get_folder_validations(config, folder):
    """Generate validation file paths for scenarios in a given folder."""
    validations = []
    for scenario in get_scenarios_for_folder(config, folder):
        n_reps = get_param(config, scenario, "replicates")
        for rep in range(1, n_reps + 1):
            validations.append(f"results/{folder}/{scenario}/rep{rep}/validation.yaml")
    return validations