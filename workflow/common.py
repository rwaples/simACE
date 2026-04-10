"""Shared helpers for the ACE Snakemake workflows."""

import glob
import os
import re

import yaml


def load_folder_configs(config, config_dir="config"):
    """Load per-folder scenario YAML files and merge into config['scenarios'].

    Each file ``config/{folder}.yaml`` contains bare scenario dicts (no wrapper
    key).  The folder name is inferred from the filename stem.  An explicit
    ``folder`` key inside a scenario overrides the inferred name.

    Raises ``ValueError`` on duplicate scenario names, unknown parameter keys,
    or invalid folder names.
    """
    config.setdefault("scenarios", {})
    valid_defaults = set(config["defaults"].keys())
    folder_pattern = re.compile(r"^[a-zA-Z0-9_]+$")

    for path in sorted(glob.glob(os.path.join(config_dir, "*.yaml"))):
        if os.path.basename(path) == "config.yaml":
            continue

        folder = os.path.splitext(os.path.basename(path))[0]
        if not folder_pattern.match(folder):
            raise ValueError(f"Invalid folder name '{folder}' from {path}. Must match [a-zA-Z0-9_]+")

        with open(path) as fh:
            scenarios = yaml.safe_load(fh)
        if scenarios is None:
            continue

        for name, params in scenarios.items():
            if name in config["scenarios"]:
                raise ValueError(f"Duplicate scenario '{name}': already defined, also found in {path}")
            unknown = set(params.keys()) - valid_defaults
            if unknown:
                raise ValueError(
                    f"Scenario '{name}' in {path} has unknown keys: "
                    f"{sorted(unknown)}. Valid keys: {sorted(valid_defaults)}"
                )
            if "folder" not in params:
                params["folder"] = folder
            config["scenarios"][name] = params

    _validate_phenotype_config(config)


_VALID_MODEL_FAMILIES = {"frailty", "cure_frailty", "adult", "first_passage"}
_VALID_DISTRIBUTIONS = {"weibull", "exponential", "gompertz", "lognormal", "loglogistic", "gamma"}
_VALID_METHODS = {"ltm", "cox"}


def _validate_phenotype_config(config):
    """Validate phenotype model configuration for all scenarios at DAG construction time."""
    for name, params in config.get("scenarios", {}).items():
        for trait_num in (1, 2):
            model_key = f"phenotype_model{trait_num}"
            params_key = f"phenotype_params{trait_num}"
            model = params.get(model_key, config["defaults"].get(model_key))
            pp = params.get(params_key, config["defaults"].get(params_key, {}))

            if model not in _VALID_MODEL_FAMILIES:
                raise ValueError(
                    f"Scenario '{name}': {model_key}={model!r} is not valid. "
                    f"Choose from: {sorted(_VALID_MODEL_FAMILIES)}"
                )

            if model in ("frailty", "cure_frailty"):
                if "distribution" not in pp:
                    raise ValueError(
                        f"Scenario '{name}': {params_key} for model '{model}' "
                        f"must include 'distribution' key"
                    )
                if pp["distribution"] not in _VALID_DISTRIBUTIONS:
                    raise ValueError(
                        f"Scenario '{name}': {params_key} distribution="
                        f"{pp['distribution']!r} invalid; "
                        f"valid: {sorted(_VALID_DISTRIBUTIONS)}"
                    )

            if model == "adult":
                if "method" not in pp:
                    raise ValueError(
                        f"Scenario '{name}': {params_key} for model 'adult' "
                        f"must include 'method' key (valid: {sorted(_VALID_METHODS)})"
                    )
                if pp["method"] not in _VALID_METHODS:
                    raise ValueError(
                        f"Scenario '{name}': {params_key} method="
                        f"{pp['method']!r} invalid; "
                        f"valid: {sorted(_VALID_METHODS)}"
                    )


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


def _epimight_mem(config, scenario, base_mb=6000, mb_per_K=50):
    """Estimate mem_mb for EPIMIGHT MI rules (guide_yob).

    Adds K × mb_per_K for MI resamples on top of the base population scaling.
    Floor of 6000 MB covers R process overhead observed in benchmarks.
    """
    k = get_param(config, scenario, "epimight_mi_K")
    return _scale_mem(config, scenario, mb_per_1k=2, floor=base_mb) + k * mb_per_K


def _epimight_runtime(config, scenario):
    """Estimate runtime (minutes) for EPIMIGHT R rules (slower than Python steps)."""
    return _scale_runtime(config, scenario, min_per_1M=10)


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
