"""The per-rep stage sequence ``simace run`` executes, as argv builders.

Each stage is one ``simace <stage>`` subprocess. Its argv is built from the
resolved flat scenario dict and the :class:`~simace.cli.layout.Layout`, so a
dry run prints exactly what a real run executes.
"""

from __future__ import annotations

__all__ = [
    "PARAMS_YAML_KEYS",
    "REP_OUTPUTS",
    "REP_PARAM_KEYS",
    "STAGES",
    "ResolvedRep",
    "Stage",
    "atlas_argv",
    "plot_argv",
]

import inspect
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import yaml

from simace.cli.layout import RepArtifact

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from simace.cli.layout import Layout


@dataclass(frozen=True)
class ResolvedRep:
    """One replicate of one scenario, with its resolved flat parameters."""

    folder: str
    scenario: str
    rep: int
    params: Mapping[str, Any]

    @property
    def seed(self) -> int:
        """The replicate seed, ``seed + rep - 1``."""
        return int(self.params["seed"]) + self.rep - 1

    def path(self, layout: Layout, artifact: RepArtifact) -> str:
        """Return one of this rep's artifact paths as a string."""
        return str(layout.rep(self.folder, self.scenario, self.rep, artifact))


@dataclass(frozen=True)
class Stage:
    """A per-rep stage: its subcommand, the config keys it reads, the files it publishes, and its argv."""

    name: str
    keys: tuple[str, ...]
    outputs: tuple[RepArtifact, ...]
    argv: Callable[[ResolvedRep, Layout], list[str]]


_SIMULATE_KEYS = (
    "N",
    "G_ped",
    "G_sim",
    "mating_model",
    "mating_lambda",
    "p_mztwin",
    "A1",
    "C1",
    "E1",
    "A2",
    "C2",
    "E2",
    "rA",
    "rC",
    "rE",
    "assort1",
    "assort2",
    "assort_matrix",
)
_PHENOTYPE_KEYS = (
    "G_pheno",
    "standardize",
    "phenotype_model1",
    "beta1",
    "beta_sex1",
    "phenotype_params1",
    "phenotype_model2",
    "beta2",
    "beta_sex2",
    "phenotype_params2",
)
_CENSOR_KEYS = ("censor_age", "death_scale", "death_rho", "gen_censoring")
_ASCERTAIN_KEYS = ("dropout_rate", "case_ascertainment_ratio", "N_sample")
_ANALYZE_KEYS = ("censor_age", "gen_censoring", "max_degree", "case_ascertainment_ratio")


def _params_yaml_keys() -> tuple[str, ...]:
    from simace.simulation.emit_params import emit_params

    return tuple(name for name in inspect.signature(emit_params).parameters if name not in ("seed", "rep"))


#: Config keys ``params.yaml`` echoes: every ``emit_params`` argument but the per-rep seed and rep.
PARAMS_YAML_KEYS: tuple[str, ...] = _params_yaml_keys()


def _json(value: Any) -> str:
    return json.dumps(value)


def _yaml(value: Any) -> str:
    return yaml.safe_dump(value, default_flow_style=True, width=float("inf"), sort_keys=False).strip()


def _flags(values: Mapping[str, Any], encode: Mapping[str, Callable[[Any], str]] | None = None) -> list[str]:
    """Turn ``{flat_key: value}`` into ``--flat-key value`` pairs, skipping ``None``.

    Flag names follow the stage CLIs' convention of the flat key with ``_``
    replaced by ``-``. Values print with ``str`` (exact for floats) unless
    ``encode`` names a structured encoder for that key.
    """
    encode = encode or {}
    argv: list[str] = []
    for key, value in values.items():
        if value is None:
            continue
        argv += [f"--{key.replace('_', '-')}", encode.get(key, str)(value)]
    return argv


def _pick(rep: ResolvedRep, *keys: str) -> dict[str, Any]:
    return {key: rep.params[key] for key in keys}


def _generation_map_or_scalar(value: Any) -> str:
    return _json(value) if isinstance(value, dict) else str(value)


def _simulate(rep: ResolvedRep, layout: Layout) -> list[str]:
    values = {"seed": rep.seed, **_pick(rep, *_SIMULATE_KEYS)}
    encode = {
        "E1": _generation_map_or_scalar,
        "E2": _generation_map_or_scalar,
        "assort1": _generation_map_or_scalar,
        "assort2": _generation_map_or_scalar,
        "assort_matrix": _json,
    }
    return [*_flags(values, encode), "--output-pedigree", rep.path(layout, RepArtifact.PEDIGREE_FULL)]


def _phenotype(rep: ResolvedRep, layout: Layout) -> list[str]:
    values = {"seed": rep.seed, **_pick(rep, *_PHENOTYPE_KEYS)}
    encode = {"phenotype_params1": _yaml, "phenotype_params2": _yaml}
    return [
        "--pedigree",
        rep.path(layout, RepArtifact.PEDIGREE_FULL),
        "--output",
        rep.path(layout, RepArtifact.TRAIT_RAW),
        *_flags(values, encode),
    ]


def _censor(rep: ResolvedRep, layout: Layout) -> list[str]:
    values = {"seed": rep.seed, **_pick(rep, *_CENSOR_KEYS)}
    return [
        "--phenotype",
        rep.path(layout, RepArtifact.TRAIT_RAW),
        "--pedigree",
        rep.path(layout, RepArtifact.PEDIGREE_FULL),
        "--output",
        rep.path(layout, RepArtifact.TRAIT_FULL),
        *_flags(values, {"gen_censoring": _json}),
    ]


def _ascertain(rep: ResolvedRep, layout: Layout) -> list[str]:
    values = {"seed": rep.seed, **_pick(rep, *_ASCERTAIN_KEYS)}
    return [
        "--pedigree",
        rep.path(layout, RepArtifact.PEDIGREE_FULL),
        "--trait",
        rep.path(layout, RepArtifact.TRAIT_FULL),
        "--out-pedigree",
        rep.path(layout, RepArtifact.PEDIGREE),
        "--out-trait",
        rep.path(layout, RepArtifact.TRAIT),
        *_flags(values),
    ]


def _analyze(rep: ResolvedRep, layout: Layout) -> list[str]:
    values = {
        "seed": rep.seed,
        **_pick(rep, *_ANALYZE_KEYS),
        "folder": rep.folder,
        "scenario": rep.scenario,
        "rep": rep.rep,
    }
    return [
        "--pedigree-full",
        rep.path(layout, RepArtifact.PEDIGREE_FULL),
        "--params",
        rep.path(layout, RepArtifact.PARAMS),
        "--trait-full",
        rep.path(layout, RepArtifact.TRAIT_FULL),
        "--trait",
        rep.path(layout, RepArtifact.TRAIT),
        "--pedigree",
        rep.path(layout, RepArtifact.PEDIGREE),
        "--report-output",
        rep.path(layout, RepArtifact.REPORT),
        "--plot-payload-output",
        rep.path(layout, RepArtifact.PLOT_PAYLOAD),
        "--samples-output",
        rep.path(layout, RepArtifact.PLOTTING_SAMPLE),
        *_flags(values, {"gen_censoring": _json}),
    ]


STAGES: tuple[Stage, ...] = (
    Stage("simulate", _SIMULATE_KEYS, (RepArtifact.PEDIGREE_FULL,), _simulate),
    Stage("phenotype", _PHENOTYPE_KEYS, (RepArtifact.TRAIT_RAW,), _phenotype),
    Stage("censor", _CENSOR_KEYS, (RepArtifact.TRAIT_FULL,), _censor),
    Stage("ascertain", _ASCERTAIN_KEYS, (RepArtifact.PEDIGREE, RepArtifact.TRAIT), _ascertain),
    Stage(
        "analyze",
        _ANALYZE_KEYS,
        (RepArtifact.REPORT, RepArtifact.PLOT_PAYLOAD, RepArtifact.PLOTTING_SAMPLE),
        _analyze,
    ),
)

#: Every config key a rep's outputs depend on. A rep is stale only when one of these changes.
REP_PARAM_KEYS: frozenset[str] = frozenset({"seed", *PARAMS_YAML_KEYS, *(k for s in STAGES for k in s.keys)})

#: Every file a rep's recompute writes, and so every file it clears first.
REP_OUTPUTS: tuple[RepArtifact, ...] = (
    RepArtifact.RUN_MANIFEST,
    RepArtifact.PARAMS,
    RepArtifact.TIMING,
    *(out for s in STAGES for out in s.outputs),
)

# Scenario keys the atlas title page needs that params.yaml does not carry.
# max_degree is deliberately absent: rep1/params.yaml owns the extraction depth.
_ATLAS_META_KEYS = (
    "replicates",
    "folder",
    "standardize",
    "beta1",
    "beta_sex1",
    "phenotype_model1",
    "phenotype_params1",
    "beta2",
    "beta_sex2",
    "phenotype_model2",
    "phenotype_params2",
    "censor_age",
    "gen_censoring",
    "death_scale",
    "death_rho",
    "G_pheno",
    "N_sample",
    "dropout_rate",
    "case_ascertainment_ratio",
)


def _each_rep(reps: list[ResolvedRep], layout: Layout, artifact: RepArtifact) -> list[str]:
    return [rep.path(layout, artifact) for rep in reps]


def plot_argv(reps: list[ResolvedRep], layout: Layout) -> list[str]:
    """Return the ``simace plot`` argv over every rep of one scenario."""
    first = reps[0]
    values = {**_pick(first, "censor_age", "gen_censoring"), "plot_format": first.params["plot_format"]}
    return [
        "--report",
        *_each_rep(reps, layout, RepArtifact.REPORT),
        "--plot-payload",
        *_each_rep(reps, layout, RepArtifact.PLOT_PAYLOAD),
        "--samples",
        *_each_rep(reps, layout, RepArtifact.PLOTTING_SAMPLE),
        "--output-dir",
        str(layout.scenario_plots(first.folder, first.scenario)),
        *_flags(values, {"gen_censoring": _json}),
    ]


def atlas_argv(reps: list[ResolvedRep], layout: Layout, atlas_format: str) -> list[str]:
    """Return the ``simace atlas`` argv over every rep of one scenario."""
    first = reps[0]
    meta = {
        **_pick(first, *_ATLAS_META_KEYS),
        "scenario": first.scenario,
        "plot_format": first.params["plot_format"],
    }
    plots = layout.scenario_plots(first.folder, first.scenario)
    return [
        "--plot-dir",
        str(plots),
        "--params",
        first.path(layout, RepArtifact.PARAMS),
        "--meta",
        _yaml(meta),
        "--report",
        *_each_rep(reps, layout, RepArtifact.REPORT),
        "--plot-payload",
        *_each_rep(reps, layout, RepArtifact.PLOT_PAYLOAD),
        "--output",
        str(plots / f"atlas.{atlas_format}"),
    ]
