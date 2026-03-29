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
    return sorted({get_folder(config, s) for s in config["scenarios"]})


# -- Plot filename basenames (without extension) --

# Ordered by narrative flow: liability structure -> phenotype ->
# censoring -> familial correlations & heritability.
_PHENOTYPE_BASENAMES = [
    # Pedigree structure
    "pedigree_counts.ped",
    "pedigree_counts",
    # Family structure
    "family_structure",
    # Mate correlation
    "mate_correlation",
    # Liability structure
    "cross_trait",
    # Liability-scale heritability (pedigree + liability only)
    "parent_offspring_liability.by_generation",
    "heritability.by_generation",
    "heritability.by_sex.by_generation",
    "additive_shared.by_generation",
    # Liability by affected status
    "liability_violin.phenotype",
    "liability_violin.phenotype.by_generation",
    "liability_violin.phenotype.by_sex.by_generation",
    # Age of onset & censoring
    "age_at_onset_death",
    "mortality",
    "cumulative_incidence.by_sex",
    "cumulative_incidence.by_sex.by_generation",
    "cumulative_incidence.phenotype",
    "censoring",
    "censoring_confusion",
    "censoring_cascade",
    "liability_vs_aoo",
    # Within-trait correlations
    "tetrachoric.phenotype",
    "tetrachoric.phenotype.by_sex",
    "tetrachoric.phenotype.by_generation",
    # Cross-trait correlations
    "cross_trait.phenotype",
    "cross_trait.phenotype.t2",
    "joint_affected.phenotype",
    "cross_trait_tetrachoric",
]

# Ordered to mirror phenotype: prevalence -> liability -> correlations.
_SIMPLE_LTM_BASENAMES = [
    "prevalence_by_generation",
    "cross_trait.simple_ltm",
    "liability_violin.simple_ltm",
    "liability_violin.simple_ltm.by_generation",
    "joint_affected.simple_ltm",
    "tetrachoric.simple_ltm",
    "cross_trait_tetrachoric.simple_ltm",
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
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype.simple_ltm.parquet")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/validation.yaml")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/phenotype_stats.yaml")
        outputs.append(f"results/{folder}/{scenario}/rep{rep}/simple_ltm_stats.yaml")
    outputs.extend(
        f"results/{folder}/{scenario}/plots/{plot}" for plot in plot_filenames(_PHENOTYPE_BASENAMES, plot_ext)
    )
    outputs.extend(
        f"results/{folder}/{scenario}/plots/{plot}" for plot in plot_filenames(_SIMPLE_LTM_BASENAMES, plot_ext)
    )
    outputs.append(f"results/{folder}/{scenario}/plots/atlas.pdf")
    return outputs


def get_folder_validations(config, folder):
    """Generate validation file paths for scenarios in a given folder."""
    validations = []
    for scenario in get_scenarios_for_folder(config, folder):
        n_reps = get_param(config, scenario, "replicates")
        validations.extend(f"results/{folder}/{scenario}/rep{rep}/validation.yaml" for rep in range(1, n_reps + 1))
    return validations
