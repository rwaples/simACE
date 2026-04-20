"""Snakemake-specific helpers for the simACE workflow.

Config resolution (flattening, validation, accessors) lives in
``simace.config`` so that fitACE can consume the same sim-side state without
duplicating YAML files.  This module only holds helpers that depend on
Snakemake-specific concerns: resource scaling, plot-filename lists, and
per-scenario/per-folder output collectors.
"""

from __future__ import annotations

from simace.config import (
    KNOWN_SIM_KEYS,
    flatten_hierarchical,
    get_all_folders,
    get_folder,
    get_param,
    get_scenarios_for_folder,
    resolve_defaults,
    resolve_scenarios,
)

# Re-export names used directly by Snakemake rule files and existing tests.
__all__ = [
    "KNOWN_SIM_KEYS",
    "_PHENOTYPE_BASENAMES",
    "_VALIDATION_BASENAMES",
    "_scale_mem",
    "_scale_runtime",
    "flatten_hierarchical",
    "get_all_folders",
    "get_folder",
    "get_folder_validations",
    "get_param",
    "get_scenario_sim_outputs",
    "get_scenarios_for_folder",
    "load_folder_configs",
    "plot_filenames",
    "resolve_defaults",
    "resolve_scenarios",
]


def load_folder_configs(config: dict, config_dir: str = "config") -> None:
    """Populate ``config['defaults']`` and ``config['scenarios']`` in place.

    Snakemake-facing wrapper over ``simace.config.resolve_defaults`` and
    ``resolve_scenarios``.  The input ``config`` is the dict Snakemake builds
    from ``configfile:`` (so ``config['defaults']`` is already present, in
    hierarchical YAML form); this function flattens it and loads scenario
    files alongside.

    Args:
        config: the mutable Snakemake config dict.
        config_dir: directory containing ``_default.yaml`` + per-folder YAMLs.
    """
    config["defaults"] = flatten_hierarchical(config["defaults"])
    config["scenarios"] = resolve_scenarios(config_dir, defaults=config["defaults"])


def _scale_mem(config: dict, scenario: str, gen_key: str = "G_pheno", mb_per_1k: int = 2, floor: int = 4000) -> int:
    """Estimate mem_mb from population size: N × G × mb_per_1k/1000, with a floor."""
    n = get_param(config, scenario, "N")
    g = get_param(config, scenario, gen_key)
    return max(floor, int(n * g * mb_per_1k / 1000))


def _scale_runtime(config: dict, scenario: str, gen_key: str = "G_pheno", min_per_1M: int = 5, floor: int = 5) -> int:
    """Estimate runtime (minutes) from population size."""
    n = get_param(config, scenario, "N")
    g = get_param(config, scenario, gen_key)
    return max(floor, int(n * g * min_per_1M / 1_000_000))


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
    # Genetic selection
    "liability_components.by_generation",
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


def plot_filenames(basenames: list[str], ext: str = "png") -> list[str]:
    """Return plot filenames by appending the given extension to each basename."""
    return [f"{name}.{ext}" for name in basenames]


def get_scenario_sim_outputs(config: dict, scenario: str, plot_ext: str = "png") -> list[str]:
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
    outputs.extend(
        f"results/{folder}/{scenario}/plots/{plot}" for plot in plot_filenames(_PHENOTYPE_BASENAMES, plot_ext)
    )
    outputs.append(f"results/{folder}/{scenario}/plots/atlas.pdf")
    return outputs


def get_folder_validations(config: dict, folder: str) -> list[str]:
    """Generate validation file paths for scenarios in a given folder."""
    validations = []
    for scenario in get_scenarios_for_folder(config, folder):
        n_reps = get_param(config, scenario, "replicates")
        validations.extend(f"results/{folder}/{scenario}/rep{rep}/validation.yaml" for rep in range(1, n_reps + 1))
    return validations
