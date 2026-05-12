"""Snakemake-specific helpers for the simACE workflow.

Config resolution (flattening, validation, accessors) lives in
``simace.config`` so that fitACE can consume the same sim-side state without
duplicating YAML files.  This module only holds helpers that depend on
Snakemake-specific concerns: resource scaling, plot-filename lists, and
per-scenario/per-folder output collectors.
"""

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
from simace.plotting.atlas_manifest import (
    effective_size_basenames,
    phenotype_basenames,
    validation_basenames,
)

# Re-export names used directly by Snakemake rule files and existing tests.
__all__ = [
    "KNOWN_SIM_KEYS",
    "_scale_mem",
    "_scale_mem_effective_size",
    "_scale_runtime",
    "effective_size_basenames",
    "flatten_hierarchical",
    "get_all_folders",
    "get_folder",
    "get_folder_validations",
    "get_param",
    "get_scenario_sim_outputs",
    "get_scenarios_for_folder",
    "load_folder_configs",
    "phenotype_basenames",
    "plot_filenames",
    "resolve_defaults",
    "resolve_scenarios",
    "validation_basenames",
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


def _scale_mem_effective_size(config: dict, scenario: str) -> int:
    """Memory budget for ``effective_size_phenotype`` (MB).

    Two regimes after the streaming-θ refactor (notes/ne_next_steps.plan.md
    Phase 2):

    ``skip_ne_coancestry: true`` (α path)
        All seven non-coancestry estimators run; no DP, no CSC.  Phase 1b
        profile at G_ped=6 (scripts/profile_no_k_path.py) measured peak
        RSS of 0.36 GB at N=100K per-gen and 1.62 GB at N=1M per-gen,
        yielding ``peak_mb ≈ 0.0016 · N + 320``.  The dominant
        contributor is ``_caballero_toro_accumulators``.  Scales linearly
        with ``(G_ped + 1) / 7`` for depths other than 6.

    ``skip_ne_coancestry: false`` (β path)
        Ne_C runs via streaming θ̄ (plan 3); no K materialized but DP
        scratch dominates.  Initial guess based on May 10 K-build
        numbers minus the ~3.3 GB CSC: ``peak_mb ≈ 0.085 · N + 320``.
        **Provisional** — recalibrate once Phase 5a's 100K + 500K
        benches land.

    A 50 % safety margin is applied so Snakemake throttles parallel jobs
    against ``--resources mem_mb``.  The β-path DP's ``_grow_global``
    geometric-doubling event briefly holds both the old half-size and
    new full-size buffers simultaneously, lifting transient RSS ~20 %
    above the steady-state working set; the May-12 baseline100K bench
    measured ``max_rss = 12.7 GB`` against an 8.8 GB steady-state
    estimate (1.44×).  1.5× covers that spike plus ~10 % rep-to-rep
    variance with ~2σ of headroom.

    Args:
        config: Snakemake config dict.
        scenario: Scenario name.

    Returns:
        Memory budget in MB; floors at 4 GB for tiny scenarios.
    """
    n = get_param(config, scenario, "N")
    g = get_param(config, scenario, "G_ped")
    skip = bool(get_param(config, scenario, "skip_ne_coancestry"))
    if skip:
        # α path — calibrated against Phase 1b profile.  No DP doubling
        # spike here, so 1.2 × is still enough headroom for the α path.
        per_indiv_mb = 0.0016 * ((g + 1) / 7.0)
        safety = 1.2
    else:
        # β path — streaming-θ DP scratch dominates.  PROVISIONAL.
        # 1.5 × accounts for the _grow_global doubling event.
        per_indiv_mb = 0.085 * (g / 6.0)
        safety = 1.5
    peak = per_indiv_mb * n + 320.0
    return max(4000, int(peak * safety))


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
        f"results/{folder}/{scenario}/plots/{plot}" for plot in plot_filenames(phenotype_basenames(), plot_ext)
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
