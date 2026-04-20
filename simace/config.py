"""Config resolution for simACE scenarios.

Loads ``config/_default.yaml`` and per-folder scenario YAML files, flattens
hierarchical sections (e.g. ``pedigree.trait1.A`` → ``A1``), and validates
phenotype model parameters.

Public API:
    - ``resolve_defaults(config_dir)`` — load defaults into a flat dict
    - ``resolve_scenarios(config_dir)`` — load every scenario file
    - ``flatten_hierarchical(d, mapping)`` — generic flattener (reused by fitACE)
    - ``get_param``, ``get_folder``, ``get_scenarios_for_folder``, ``get_all_folders``
    - ``KNOWN_SIM_KEYS`` — frozenset of the flat keys simACE owns

The fitACE sister repo imports ``KNOWN_SIM_KEYS`` and ``flatten_hierarchical`` to
load its own fit-domain overlays on top of sim-side scenarios.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Hierarchical config → flat internal key mapping (sim domain only)
# ---------------------------------------------------------------------------

_HIERARCHICAL_TO_FLAT: dict[tuple[str, ...], str] = {
    # pedigree section
    ("pedigree", "mating_lambda"): "mating_lambda",
    ("pedigree", "p_mztwin"): "p_mztwin",
    ("pedigree", "assort1"): "assort1",
    ("pedigree", "assort2"): "assort2",
    ("pedigree", "assort_matrix"): "assort_matrix",
    ("pedigree", "trait1", "A"): "A1",
    ("pedigree", "trait1", "C"): "C1",
    ("pedigree", "trait1", "E"): "E1",
    ("pedigree", "trait2", "A"): "A2",
    ("pedigree", "trait2", "C"): "C2",
    ("pedigree", "trait2", "E"): "E2",
    ("pedigree", "rA"): "rA",
    ("pedigree", "rC"): "rC",
    ("pedigree", "rE"): "rE",
    # phenotype section
    ("phenotype", "trait1", "model"): "phenotype_model1",
    ("phenotype", "trait1", "params"): "phenotype_params1",
    ("phenotype", "trait1", "beta"): "beta1",
    ("phenotype", "trait1", "beta_sex"): "beta_sex1",
    ("phenotype", "trait1", "prevalence"): "prevalence1",
    ("phenotype", "trait2", "model"): "phenotype_model2",
    ("phenotype", "trait2", "params"): "phenotype_params2",
    ("phenotype", "trait2", "beta"): "beta2",
    ("phenotype", "trait2", "beta_sex"): "beta_sex2",
    ("phenotype", "trait2", "prevalence"): "prevalence2",
    # censoring section
    ("censoring", "max_age"): "censor_age",
    ("censoring", "gen_censoring"): "gen_censoring",
    ("censoring", "death_scale"): "death_scale",
    ("censoring", "death_rho"): "death_rho",
    # sampling section
    ("sampling", "N_sample"): "N_sample",
    ("sampling", "case_ascertainment_ratio"): "case_ascertainment_ratio",
    ("sampling", "pedigree_dropout_rate"): "pedigree_dropout_rate",
    # analysis section
    ("analysis", "max_degree"): "max_degree",
    ("analysis", "estimate_inbreeding"): "estimate_inbreeding",
}

# Top-level flat keys (globals) that live in _default.yaml without a section.
_SIM_FLAT_GLOBALS: frozenset[str] = frozenset(
    {
        "seed",
        "replicates",
        "folder",
        "N",
        "G_ped",
        "G_pheno",
        "G_sim",
        "standardize",
        "plot_format",
    }
)

#: Flat keys owned by the sim domain.  Used by fitACE to reject sim keys that
#: leak into fit-only overlay files.
KNOWN_SIM_KEYS: frozenset[str] = frozenset(_HIERARCHICAL_TO_FLAT.values()) | _SIM_FLAT_GLOBALS


def _flatten_section(
    flat: dict,
    prefix: tuple[str, ...],
    d: dict,
    mapping: dict[tuple[str, ...], str],
    valid_prefixes: frozenset[tuple[str, ...]],
) -> None:
    """Recursively walk a section dict applying a path→flat mapping.

    Args:
        flat: accumulator dict, mutated in place.
        prefix: current path from the section root (e.g. ``("pedigree",)``).
        d: the sub-dict at this prefix.
        mapping: path-tuple → flat-key table for the domain.
        valid_prefixes: precomputed prefixes of every mapped path, so that
            intermediate nesting levels are accepted as traversal points.

    Raises:
        ValueError: if a path is neither mapped nor a valid prefix.
    """
    for key, value in d.items():
        path = (*prefix, key)
        if path in mapping:
            flat[mapping[path]] = value
        elif path in valid_prefixes and isinstance(value, dict):
            _flatten_section(flat, path, value, mapping, valid_prefixes)
        else:
            raise ValueError(f"Unknown hierarchical config key: {'.'.join(path)}")


def flatten_hierarchical(d: dict, mapping: dict[tuple[str, ...], str] | None = None) -> dict:
    """Flatten a hierarchical config dict to flat internal keys.

    Accepts both flat (legacy) and hierarchical formats. If no top-level key
    matches a section in ``mapping``, the input is returned unchanged.  Mixed
    flat+hierarchical for the same parameter raises ``ValueError``.

    Args:
        d: input dict (may contain both flat globals and hierarchical sections).
        mapping: path-tuple → flat-key mapping.  Defaults to the sim-domain
            mapping (``_HIERARCHICAL_TO_FLAT``).  fitACE passes its own mapping.

    Returns:
        A flat dict suitable for ``get_param`` lookups.

    Raises:
        ValueError: unknown nested keys, or a parameter supplied both flat and
            hierarchically.
    """
    if mapping is None:
        mapping = _HIERARCHICAL_TO_FLAT

    section_keys = frozenset(path[0] for path in mapping)
    valid_prefixes = frozenset(path[:i] for path in mapping for i in range(1, len(path)))

    if not any(k in section_keys for k in d):
        return d

    top_level: dict = {}
    section_flat: dict = {}
    for key, value in d.items():
        if key in section_keys:
            _flatten_section(section_flat, (key,), value, mapping, valid_prefixes)
        else:
            top_level[key] = value

    overlap = set(top_level) & set(section_flat)
    if overlap:
        raise ValueError(f"Config keys specified in both flat and hierarchical form: {sorted(overlap)}")

    return {**top_level, **section_flat}


# ---------------------------------------------------------------------------
# Phenotype model validation
# ---------------------------------------------------------------------------

_VALID_MODEL_FAMILIES: frozenset[str] = frozenset({"frailty", "cure_frailty", "adult", "first_passage"})
_VALID_DISTRIBUTIONS: frozenset[str] = frozenset(
    {"weibull", "exponential", "gompertz", "lognormal", "loglogistic", "gamma"}
)
_VALID_METHODS: frozenset[str] = frozenset({"ltm", "cox"})


def _validate_phenotype_config(config: dict) -> None:
    """Validate phenotype model configuration for all scenarios.

    Checked at DAG-construction time so that bad configs fail fast before
    any simulation jobs are dispatched.

    Args:
        config: the merged ``{"defaults": ..., "scenarios": ...}`` dict.

    Raises:
        ValueError: if a model family, distribution, or method is invalid for
            any scenario.
    """
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
                        f"Scenario '{name}': {params_key} for model '{model}' must include 'distribution' key"
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


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------

_FOLDER_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


def resolve_defaults(config_dir: Path | str) -> dict:
    """Load ``<config_dir>/_default.yaml`` and return a flat defaults dict.

    The YAML file is expected to have a top-level ``defaults:`` mapping that
    may mix flat and hierarchical keys; both forms are flattened.

    Args:
        config_dir: directory containing ``_default.yaml``.

    Returns:
        Flat defaults dict (flattened per the sim-domain mapping).
    """
    config_dir = Path(config_dir)
    with open(config_dir / "_default.yaml") as fh:
        raw = yaml.safe_load(fh)
    return flatten_hierarchical(raw["defaults"])


def resolve_scenarios(config_dir: Path | str, defaults: dict | None = None) -> dict[str, dict]:
    """Load every ``<config_dir>/{folder}.yaml`` scenario file.

    Files starting with ``_`` are skipped.  The folder name is inferred from
    the filename stem; an explicit ``folder`` key inside a scenario overrides
    the inferred name.  Scenario params are flattened and validated against
    ``defaults.keys()`` — unknown keys raise ``ValueError``.

    Args:
        config_dir: directory containing per-folder scenario YAMLs.
        defaults: flat defaults dict (typically from ``resolve_defaults``).
            If ``None``, is loaded from ``<config_dir>/_default.yaml``.

    Returns:
        ``{scenario_name: flat_params}``.

    Raises:
        ValueError: duplicate scenario names, invalid folder names, unknown
            keys in a scenario, or an invalid phenotype model config.
    """
    config_dir = Path(config_dir)
    if defaults is None:
        defaults = resolve_defaults(config_dir)
    valid_defaults = set(defaults.keys())

    scenarios: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(str(config_dir), "*.yaml"))):
        if os.path.basename(path).startswith("_"):
            continue

        folder = os.path.splitext(os.path.basename(path))[0]
        if not _FOLDER_PATTERN.match(folder):
            raise ValueError(f"Invalid folder name '{folder}' from {path}. Must match [a-zA-Z0-9_]+")

        with open(path) as fh:
            file_scenarios = yaml.safe_load(fh)
        if file_scenarios is None:
            continue

        for name, params in file_scenarios.items():
            if name in scenarios:
                raise ValueError(f"Duplicate scenario '{name}': already defined, also found in {path}")
            params = flatten_hierarchical(params)
            unknown = set(params.keys()) - valid_defaults
            if unknown:
                raise ValueError(
                    f"Scenario '{name}' in {path} has unknown keys: "
                    f"{sorted(unknown)}. Valid keys: {sorted(valid_defaults)}"
                )
            if "folder" not in params:
                params["folder"] = folder
            scenarios[name] = params

    _validate_phenotype_config({"defaults": defaults, "scenarios": scenarios})
    return scenarios


# ---------------------------------------------------------------------------
# Snakemake-friendly accessors
# ---------------------------------------------------------------------------


def get_param(config: dict, scenario: str, param: str):
    """Return ``param`` for ``scenario``, falling back to ``config['defaults']``.

    Args:
        config: dict with ``"defaults"`` and ``"scenarios"`` sub-dicts.
        scenario: scenario name (must exist in ``config['scenarios']`` if the
            param is not overridden there, the fall-through still works).
        param: flat parameter name.

    Returns:
        The parameter value (from the scenario if set, else from defaults).
    """
    scenario_config = config["scenarios"].get(scenario, {})
    if param in scenario_config:
        return scenario_config[param]
    return config["defaults"][param]


def get_folder(config: dict, scenario: str) -> str:
    """Return the folder grouping for a scenario."""
    return get_param(config, scenario, "folder")


def get_scenarios_for_folder(config: dict, folder: str) -> list[str]:
    """Return scenario names assigned to the given folder."""
    return [s for s in config["scenarios"] if get_folder(config, s) == folder]


def get_all_folders(config: dict) -> list[str]:
    """Return sorted unique folder names across all scenarios."""
    return sorted({get_folder(config, s) for s in config["scenarios"]})
