"""Config resolution for simACE scenarios.

Loads ``config/_default.yaml`` and per-folder scenario YAML files, flattens
hierarchical sections (e.g. ``pedigree.trait1.A`` → ``A1``), and validates
phenotype model parameters.

Public API:
    - ``resolve_defaults(config_dir)`` — load defaults into a flat dict
    - ``load_scenarios(config_dir)`` — load and parse every scenario into a :class:`Scenario`
    - ``resolve_scenarios(config_dir)`` — every scenario's own flat keys
    - ``flatten_hierarchical(d, mapping)`` — generic flattener (reused by fitACE)
    - ``get_param``, ``get_folder``, ``get_scenarios_for_folder``, ``get_all_folders``
    - ``KNOWN_SIM_KEYS`` — frozenset of the flat keys simACE owns

The fitACE sister repo imports ``KNOWN_SIM_KEYS`` and ``flatten_hierarchical`` to
load its own fit-domain overlays on top of sim-side scenarios.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from simace.core.relationships import C_TRANSMISSION_KEYS, transmission_key
from simace.core.standardize import coerce_standardize_mode
from simace.core.yaml_io import load_yaml
from simace.simulation.inbreeding_plan import INBREEDING_MODES
from simace.simulation.params import (
    LEGACY_DEFAULTS,
    VALID_MATING_MODELS,
    Assortment,
    CTransmission,
    Span,
    VarianceComponents,
    am_transmitted_share,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hierarchical config → flat internal key mapping (sim domain only)
# ---------------------------------------------------------------------------

_HIERARCHICAL_TO_FLAT: dict[tuple[str, ...], str] = {
    # pedigree section
    ("pedigree", "mating_model"): "mating_model",
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
    **{
        ("pedigree", f"trait{t}", "C_transmission", side): transmission_key(side, t)
        for t in (1, 2)
        for side in ("maternal", "paternal")
    },
    ("pedigree", "rA"): "rA",
    ("pedigree", "rC"): "rC",
    ("pedigree", "rE"): "rE",
    ("pedigree", "inbreeding", "mode"): "inbreeding_mode",
    ("pedigree", "inbreeding", "depth"): "inbreeding_depth",
    # phenotype section. Note: phenotype.trait{N}.prevalence is no longer a
    # top-level mapping — PR3 moved prevalence inside per-model params for
    # adult / cure_frailty (it is forbidden for frailty / first_passage).
    ("phenotype", "trait1", "model"): "phenotype_model1",
    ("phenotype", "trait1", "params"): "phenotype_params1",
    ("phenotype", "trait1", "beta"): "beta1",
    ("phenotype", "trait1", "beta_sex"): "beta_sex1",
    ("phenotype", "trait2", "model"): "phenotype_model2",
    ("phenotype", "trait2", "params"): "phenotype_params2",
    ("phenotype", "trait2", "beta"): "beta2",
    ("phenotype", "trait2", "beta_sex"): "beta_sex2",
    # censoring section
    ("censoring", "max_age"): "censor_age",
    ("censoring", "gen_censoring"): "gen_censoring",
    ("censoring", "death_scale"): "death_scale",
    ("censoring", "death_rho"): "death_rho",
    # ascertainment section (formerly "sampling" with pedigree_dropout_rate)
    ("ascertainment", "N_sample"): "N_sample",
    ("ascertainment", "case_ascertainment_ratio"): "case_ascertainment_ratio",
    ("ascertainment", "dropout_rate"): "dropout_rate",
    # analysis section
    ("analysis", "max_degree"): "max_degree",
    ("analysis", "estimate_inbreeding"): "estimate_inbreeding",
    ("analysis", "skip_ne_coancestry"): "skip_ne_coancestry",
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
        # Opaque dict consumed by fitACE's blended_phenotype rule. When non-null
        # the rule re-derives trait-1 case status from a per-generation blend of
        # L1 and L2 liabilities. See notes/heritability/epimight_h2_temporal.md.
        "blended_diagnosis",
    }
)

#: Flat keys whose ``null`` means something (no matrix, no per-generation
#: windows, ``G_sim = G_ped``, no blend). Any other null is a config error.
NULLABLE_KEYS: frozenset[str] = frozenset({"assort_matrix", "gen_censoring", "G_sim", "blended_diagnosis"})

#: Keys of the tskit gene-drop pipeline, which moved to simACE-validate's
#: tskit_pipeline/ (its scripts take them as flags).
_GENE_DROP_KEYS: frozenset[str] = frozenset({"use_gene_drop", "drop_from", "tstrait", "tskit_preprocess"})


def _reject_gene_drop_keys(params: dict, where: str) -> None:
    """Raise if ``params`` sets a key of the gene-drop pipeline simACE no longer runs.

    Checked on defaults and scenarios alike: an old ``_default.yaml`` would
    otherwise make the keys valid, and a scenario asking for gene drop would
    silently run the parametric pipeline.
    """
    if moved := sorted(key for key in params if key in _GENE_DROP_KEYS or key.startswith("tstrait_")):
        raise ValueError(
            f"{where} sets gene-drop keys {moved}. simACE no longer runs gene drop: "
            "the tskit pipeline moved to simACE-validate's tskit_pipeline/, whose scripts take these as flags."
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
        elif len(path) == 3 and path[0] == "phenotype" and path[1] in ("trait1", "trait2") and path[2] == "prevalence":
            # Removed in PR3: phenotype.trait{N}.prevalence moved inside
            # params: for adult / cure_frailty, deleted for frailty / first_passage.
            raise ValueError(
                f"Top-level {'.'.join(path)} is no longer supported. "
                f"Move it inside phenotype.{path[1]}.params (for adult / "
                f"cure_frailty) or remove it (for frailty / first_passage)."
            )
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
# Sim-domain type coercions applied after flattening
# ---------------------------------------------------------------------------


def _coerce_sim_types(flat: dict) -> dict:
    """Coerce known sim-domain values to their runtime types.

    YAML loads dict keys as their scalar type, so unquoted ``0:`` parses as
    ``int`` but a quoted ``'0':`` parses as ``str``.  Sim-domain code expects
    integer generation keys for ``gen_censoring`` and the per-generation
    dicts (``E1``, ``E2``, ``assort1``, ``assort2``);
    coerce here so wrappers and rules don't need to know about the YAML quirk.

    Idempotent: safe to call on values that already have the expected types.
    """
    gen_censoring = flat.get("gen_censoring")
    if gen_censoring:
        flat["gen_censoring"] = {int(k): v for k, v in gen_censoring.items()}
    for key in ("E1", "E2", "assort1", "assort2"):
        value = flat.get(key)
        if isinstance(value, dict):
            flat[key] = {int(gen): v for gen, v in value.items()}
    for key in C_TRANSMISSION_KEYS:
        if flat.get(key) is not None:
            flat[key] = float(flat[key])
    if "standardize" in flat:
        flat["standardize"] = coerce_standardize_mode(flat["standardize"])
    if isinstance(flat.get("inbreeding_mode"), bool):
        raise ValueError(
            f"pedigree.inbreeding.mode is the boolean {flat['inbreeding_mode']!r}: YAML reads a bare off as false. "
            'Quote it: mode: "off" (ADR 0033)'
        )
    depth = flat.get("inbreeding_depth")
    if depth is not None and (isinstance(depth, bool) or not isinstance(depth, int)):
        raise ValueError(f"pedigree.inbreeding.depth must be an integer, got {depth!r} (ADR 0033)")
    return flat


# ---------------------------------------------------------------------------
# The parsed scenario
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    """One scenario, parsed once at config load.

    ``explicit`` holds the keys the scenario file sets (with ``folder``),
    ``flat`` those under ``config/_default.yaml``'s, which sit over
    :data:`~simace.simulation.params.LEGACY_DEFAULTS`, and the records are
    parsed from ``flat``. Stage command lines and ``params.yaml`` read
    ``flat``; the config checks read the records. Building one runs every
    config-load check of the scenario (:meth:`parse`).
    """

    name: str
    explicit: Mapping[str, Any]
    flat: Mapping[str, Any]
    span: Span
    vc: VarianceComponents
    assortment: Assortment
    c_transmission: CTransmission

    @property
    def folder(self) -> str:
        """The results folder the scenario belongs to."""
        return self.flat["folder"]

    @property
    def seed(self) -> int:
        """The base seed; rep ``k`` runs on ``seed + k - 1``."""
        return int(self.flat["seed"])

    @property
    def replicates(self) -> int:
        """The number of reps."""
        return int(self.flat["replicates"])

    @classmethod
    def parse(cls, name: str, defaults: Mapping[str, Any], explicit: Mapping[str, Any]) -> Scenario:
        """Merge ``explicit`` over ``defaults``, parse the records, and run the scenario's config-load checks.

        In order:

        1. Reject any value of ``None`` outside :data:`NULLABLE_KEYS`. A
           stage command line drops a null flag, so the stage would run on
           its own argparse default rather than the config's.
        2. Validate ``mating_model``, and reject a per-generation map for
           ``A1``, ``A2``, ``C1`` or ``C2``: the simulator takes scalars only.
        3. Parse the records, which validate their own ranges, and check
           the C transmission coefficients against the scope bound and the
           configured C (:meth:`CTransmission.validate`, ADR 0025).
        4. Validate the inbreeding mode and depth (ADR 0033). The cost
           refusal is the simulate stage's, so a refused scenario fails only
           its reps.
        5. Reject Wright-Fisher scenarios that set a standard-model knob
           (``mating_lambda``, ``p_mztwin``, ``assort1``, ``assort2``,
           ``assort_matrix``) to a non-no-op value themselves. Values they
           inherit are ignored at runtime, so only the scenario's own keys
           are checked.
        6. Build each trait's phenotype model: ``PhenotypeModel.from_config``
           is the only copy of a model's rules, and the models import
           without numba.

        Raises:
            ValueError: naming the scenario and the first check it fails.
        """
        flat = MappingProxyType({**LEGACY_DEFAULTS, **defaults, **explicit})
        try:
            span, vc, assortment, ctm = _pedigree_records(flat, explicit)
            _check_phenotype(flat)
        except ValueError as exc:
            raise ValueError(f"Scenario '{name}': {exc}") from exc
        except KeyError as exc:
            raise ValueError(
                f"Scenario '{name}': {exc.args[0]} is set neither in the scenario nor in the defaults"
            ) from exc
        _warn_am_transmission(name, vc, assortment, ctm)
        return cls(name, MappingProxyType(dict(explicit)), flat, span, vc, assortment, ctm)


def _pedigree_records(
    flat: Mapping[str, Any], explicit: Mapping[str, Any]
) -> tuple[Span, VarianceComponents, Assortment, CTransmission]:
    """Checks 1-5 of :meth:`Scenario.parse`, returning the parsed records."""
    nulls = sorted(key for key, value in flat.items() if value is None and key not in NULLABLE_KEYS)
    if nulls:
        raise ValueError(
            f"{', '.join(f'{key} is null' for key in nulls)}. Only {sorted(NULLABLE_KEYS)} may be null; "
            "declare a value in the scenario or in config/_default.yaml."
        )
    mating_model = flat["mating_model"]
    if mating_model not in VALID_MATING_MODELS:
        raise ValueError(f"mating_model={mating_model!r} is not valid. Choose from: {sorted(VALID_MATING_MODELS)}")
    for key in ("A1", "A2", "C1", "C2"):
        if isinstance(flat[key], dict):
            raise ValueError(
                f"{key} must be a scalar; the simulator does not support a per-generation map for A or C "
                "(E1, E2, assort1 and assort2 take one)"
            )

    span = Span.from_mapping(flat)
    vc = VarianceComponents.from_mapping(flat)
    assortment = Assortment.from_mapping(flat, span)
    ctm = CTransmission.from_mapping(flat)
    ctm.validate(vc.C[0], vc.C[1], vc.rC)

    mode = flat["inbreeding_mode"]
    if mode not in INBREEDING_MODES:
        raise ValueError(f"inbreeding_mode={mode!r} is not valid. Choose from: {list(INBREEDING_MODES)}")
    if flat["inbreeding_depth"] < 1:
        raise ValueError(f"inbreeding_depth must be >= 1, got {flat['inbreeding_depth']!r}")

    if mating_model == "wright_fisher":
        _check_wright_fisher_overrides(explicit)
    return span, vc, assortment, ctm


def _check_wright_fisher_overrides(explicit: Mapping[str, Any]) -> None:
    """Check 5 of :meth:`Scenario.parse`: ``explicit`` is the scenario's own keys, so ``key in explicit`` means set."""
    if "mating_lambda" in explicit:
        raise ValueError(
            f"mating_lambda={explicit['mating_lambda']!r} conflicts with mating_model=wright_fisher "
            "(mating_lambda is a standard-model knob with no defensible WF value). Remove the "
            "mating_lambda override or switch to mating_model=standard."
        )
    if "p_mztwin" in explicit and float(explicit["p_mztwin"]) != 0:
        raise ValueError(
            f"p_mztwin={explicit['p_mztwin']!r} conflicts with mating_model=wright_fisher (no MZ twins "
            "under WF). Remove the p_mztwin override or set it to 0."
        )
    for key in ("assort1", "assort2"):
        if key in explicit and not _is_zero_assort(explicit[key]):
            raise ValueError(
                f"{key}={explicit[key]!r} conflicts with mating_model=wright_fisher (WF is uniform random "
                f"mating). Remove the {key} override or switch to mating_model=standard."
            )
    if explicit.get("assort_matrix") is not None:
        raise ValueError(
            f"assort_matrix={explicit['assort_matrix']!r} conflicts with mating_model=wright_fisher (WF is "
            "uniform random mating). Remove the assort_matrix override or switch to mating_model=standard."
        )


def _is_zero_assort(value: Any) -> bool:
    """Return True if ``value`` is a no-op assortative-mating setting.

    Accepts both scalar (numeric ``0``) and per-generation dict forms;
    a dict whose values are all ``0`` is also treated as a no-op.
    """
    if isinstance(value, dict):
        return all(float(v) == 0 for v in value.values())
    return float(value) == 0


def _check_phenotype(flat: Mapping[str, Any]) -> None:
    """Check 6 of :meth:`Scenario.parse`: an unknown model, or one rejecting ``phenotype_params{N}``, ``beta{N}`` or ``beta_sex{N}``."""
    from simace.phenotype.models import MODELS

    for trait_num in (1, 2):
        model_key = f"phenotype_model{trait_num}"
        model = flat.get(model_key)
        if model not in MODELS:
            raise ValueError(f"{model_key}={model!r} is not valid. Choose from: {sorted(MODELS)}")
        MODELS[model].from_config(dict(flat), trait_num)


def _warn_am_transmission(name: str, vc: VarianceComponents, assortment: Assortment, ctm: CTransmission) -> None:
    """Warn when assortative mating likely pushes a trait's transmitted C past its configured total (ADR 0025).

    Uses the first-generation mate C correlation (:func:`am_transmitted_share`)
    at the trait's largest scheduled assortment, so it is a heads-up, not a
    guarantee: the run itself stops once the excess is measurable.
    """
    for t in (1, 2):
        b_m, b_f = ctm.coefficients(t)
        assort = max(assortment.schedule[t - 1], default=0.0)
        V_P = vc.V_P(t)
        if V_P is None or b_m * b_f == 0.0 or assort <= 0.0:
            continue
        C = float(vc.C[t - 1])
        share = am_transmitted_share(b_m, b_f, assort, C, V_P)
        if share > 1.0:
            logger.warning(
                "Scenario '%s': trait %d transmits %.3g of C under assortative mating (b_m² + b_f² + 2 b_m b_f "
                "r_CC with first-generation r_CC = %.3g); above 1 the simulation will stop once the excess is "
                "measurable. Lower Ctm_maternal%d/Ctm_paternal%d or assort%d (ADR 0025)",
                name,
                t,
                share,
                assort * C / V_P,
                t,
                t,
                t,
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
    path = config_dir / "_default.yaml"
    raw = load_yaml(path)
    flat = _coerce_sim_types(flatten_hierarchical(raw["defaults"]))
    _reject_gene_drop_keys({**raw, **flat}, str(path))  # tskit_preprocess sat beside defaults:
    return flat


def load_scenarios(config_dir: Path | str, defaults: Mapping[str, Any] | None = None) -> dict[str, Scenario]:
    """Load and parse every ``<config_dir>/{folder}.yaml`` scenario file.

    Files starting with ``_`` are skipped.  The folder name is inferred from
    the filename stem; an explicit ``folder`` key inside a scenario overrides
    the inferred name.  Scenario keys are flattened and must be keys of
    ``defaults``; each scenario then passes :meth:`Scenario.parse`.

    Args:
        config_dir: directory containing per-folder scenario YAMLs.
        defaults: flat defaults dict (typically from ``resolve_defaults``).
            If ``None``, is loaded from ``<config_dir>/_default.yaml``.

    Returns:
        ``{scenario_name: Scenario}``.

    Raises:
        ValueError: duplicate scenario names, invalid folder names, unknown
            keys in a scenario, or a scenario failing a config-load check.
    """
    config_dir = Path(config_dir)
    if defaults is None:
        defaults = resolve_defaults(config_dir)
    valid_defaults = set(defaults.keys())

    explicit: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(str(config_dir), "*.yaml"))):
        if os.path.basename(path).startswith("_"):
            continue

        folder = os.path.splitext(os.path.basename(path))[0]
        if not _FOLDER_PATTERN.match(folder):
            raise ValueError(f"Invalid folder name '{folder}' from {path}. Must match [a-zA-Z0-9_]+")

        file_scenarios = load_yaml(path)
        if file_scenarios is None:
            continue

        for name, raw_params in file_scenarios.items():
            if name in explicit:
                raise ValueError(f"Duplicate scenario '{name}': already defined, also found in {path}")
            params = _coerce_sim_types(flatten_hierarchical(raw_params))
            _reject_gene_drop_keys(params, f"Scenario '{name}' in {path}")
            unknown = set(params.keys()) - valid_defaults
            if unknown:
                raise ValueError(
                    f"Scenario '{name}' in {path} has unknown keys: "
                    f"{sorted(unknown)}. Valid keys: {sorted(valid_defaults)}"
                )
            if "folder" not in params:
                params["folder"] = folder
            explicit[name] = params

    return {name: Scenario.parse(name, defaults, params) for name, params in explicit.items()}


def resolve_scenarios(config_dir: Path | str, defaults: dict | None = None) -> dict[str, dict]:
    """Load every scenario file (:func:`load_scenarios`) and return each scenario's own flat keys.

    Returns:
        ``{scenario_name: flat_params}``, the keys each scenario sets (with
        ``folder``), without the defaults.
    """
    return {name: dict(scenario.explicit) for name, scenario in load_scenarios(config_dir, defaults).items()}


# ---------------------------------------------------------------------------
# Accessors over a {"defaults", "scenarios"} config dict (fitACE's workflow uses them)
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


def seed_collisions(scenarios: Mapping[str, Scenario], names: list[str]) -> list[tuple[str, str, int]]:
    """Return ``(a, b, shared)`` for each pair of same-folder scenarios whose rep seeds collide by accident.

    Rep ``k`` of a scenario runs on ``seed + k - 1``, so two scenarios with
    base seeds closer than their ``replicates`` share seeds, and where their
    pedigree parameters agree those reps are the same pedigrees.  An equal base
    seed reads as a deliberate pairing (the same pedigrees under, say, another
    prevalence) and is not reported; an overlap from different base seeds is.
    Only pairs with at least one scenario in ``names`` are returned.

    Args:
        scenarios: the parsed scenarios, by name.
        names: the scenarios being run.
    """
    ranges = {name: (s.folder, s.seed, s.seed + s.replicates - 1) for name, s in scenarios.items()}
    targets = set(names)
    out = []
    ordered = sorted(ranges)
    for i, a in enumerate(ordered):
        fa, lo_a, hi_a = ranges[a]
        for b in ordered[i + 1 :]:
            fb, lo_b, hi_b = ranges[b]
            if fa != fb or lo_a == lo_b or not ({a, b} & targets):
                continue
            shared = min(hi_a, hi_b) - max(lo_a, lo_b) + 1
            if shared > 0:
                out.append((a, b, shared))
    return out
