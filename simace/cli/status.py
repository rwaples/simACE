"""Scenario lookup and the on-disk state of a scenario's reps and plots.

Shared by ``simace run`` (what to compute), ``simace ls`` / ``show`` (what to
report), and ``simace gather`` (which reps to collect).
"""

from __future__ import annotations

__all__ = [
    "ScenarioError",
    "built_stages",
    "check_runnable",
    "expected_manifest",
    "load_scenario",
    "rep_outputs",
    "rep_ranges",
    "resolve_all",
    "scenario_plots_status",
    "stage_names",
    "status_on_disk",
]

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from simace.cli.layout import RepArtifact
from simace.cli.manifest import (
    Manifest,
    PlotsState,
    PlotsStatus,
    RepState,
    RepStatus,
    manifest_params,
    plots_status,
    read_manifest,
    recorded_stages,
    rep_status,
)
from simace.cli.stages import REP_LAYOUT, STAGES, rep_param_keys

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from simace.cli.layout import Layout
    from simace.cli.stages import ResolvedRep, Stage


class ScenarioError(Exception):
    """A scenario that cannot be run: unknown, or outside what ``run`` supports."""


def resolve_all(config_dir: Path) -> dict[str, dict[str, Any]]:
    """Return every scenario's resolved flat parameters (defaults merged in)."""
    from simace.config import resolve_defaults, resolve_scenarios

    defaults = resolve_defaults(config_dir)
    return {name: {**defaults, **params} for name, params in resolve_scenarios(config_dir, defaults).items()}


def check_runnable(name: str, params: dict[str, Any]) -> None:
    """Raise :class:`ScenarioError` if ``simace run`` cannot run this scenario."""
    if params.get("use_gene_drop") or params.get("drop_from") is not None:
        raise ScenarioError(
            f"scenario {name!r} uses gene drop (use_gene_drop / drop_from); "
            "`simace run` does not run it. Use the scripts in scripts/gene_drop/."
        )


def load_scenario(config_dir: Path, name: str, *, require_runnable: bool = True) -> dict[str, Any]:
    """Return the resolved flat parameters of one scenario.

    Raises:
        ScenarioError: the scenario is unknown, or uses gene drop when
            ``require_runnable`` is true.
    """
    scenarios = resolve_all(config_dir)
    if name not in scenarios:
        known = "\n  ".join(sorted(scenarios))
        raise ScenarioError(f"unknown scenario {name!r}; known scenarios:\n  {known}")
    if require_runnable:
        check_runnable(name, scenarios[name])
    return scenarios[name]


def stage_names(stages: Sequence[Stage] = STAGES) -> list[str]:
    """Return the names of ``stages``, in order."""
    return [stage.name for stage in stages]


def expected_manifest(rep: ResolvedRep, stages: Sequence[Stage] = STAGES) -> Manifest:
    """Return the manifest a rep built through ``stages`` would carry under the current parameters."""
    return Manifest(
        scenario=rep.scenario,
        rep=rep.rep,
        seed=rep.seed,
        resolved=manifest_params(rep.params, rep_param_keys(stages)),
        stages=stage_names(stages),
        layout=REP_LAYOUT,
    )


def built_stages(rep: ResolvedRep, layout: Layout) -> tuple[Stage, ...]:
    """Return the stages ``rep``'s ``run.yaml`` records when they begin the chain, else the whole chain."""
    return _built(read_manifest(rep.path(layout, RepArtifact.RUN_MANIFEST)))


def _built(recorded: Any) -> tuple[Stage, ...]:
    """Return the stages a loaded ``run.yaml`` records when they begin the chain, else the whole chain.

    Returning the whole chain for any other list lets :func:`rep_status`
    report the difference in ``stages``, which makes the rep stale.
    """
    names = recorded_stages(recorded)
    if names and names == stage_names()[: len(names)]:
        return STAGES[: len(names)]
    return STAGES


def status_on_disk(rep: ResolvedRep, layout: Layout, until: int = len(STAGES)) -> RepStatus:
    """Return one rep's state from its ``run.yaml`` and the outputs it declares, under the current parameters.

    The rep is checked against the stages its ``run.yaml`` records. One
    complete through fewer than the first ``until`` stages is partial.
    """
    recorded = read_manifest(rep.path(layout, RepArtifact.RUN_MANIFEST))
    if recorded is None:
        return RepStatus(RepState.ABSENT)
    built = _built(recorded)
    status = rep_status(recorded, expected_manifest(rep, built), rep_outputs(rep, layout, built))
    if status.state is RepState.COMPLETE and len(built) < until:
        return replace(status, state=RepState.PARTIAL, reasons=(f"built through {built[-1].name}",))
    return status


def scenario_plots_status(reps: list[ResolvedRep], layout: Layout) -> PlotsStatus:
    """Return whether a scenario's plots and atlas were built from its reps as they stand now."""
    if not reps:
        return PlotsStatus(PlotsState.ABSENT)
    folder, scenario = reps[0].folder, reps[0].scenario
    manifests = {f"rep{rep.rep}": rep.path(layout, RepArtifact.RUN_MANIFEST) for rep in reps}
    not_complete = [f"rep{rep.rep}" for rep in reps if status_on_disk(rep, layout).state is not RepState.COMPLETE]
    return plots_status(layout.scenario_plots_manifest(folder, scenario), manifests, not_complete)


def rep_outputs(rep: ResolvedRep, layout: Layout, stages: Sequence[Stage] = STAGES) -> list[Path]:
    """Return every output a rep built through ``stages`` must have, fingerprinted in its ``run.yaml``."""
    artifacts = (RepArtifact.PARAMS, RepArtifact.TIMING, *(out for stage in stages for out in stage.outputs))
    return [rep.path(layout, a) for a in artifacts]


def rep_ranges(reps: list[int]) -> str:
    """Render sorted rep numbers as ``rep2`` or ``reps 1-3, 7``."""
    runs: list[list[int]] = []
    for rep in reps:
        if runs and rep == runs[-1][-1] + 1:
            runs[-1].append(rep)
        else:
            runs.append([rep])
    text = ", ".join(f"{r[0]}-{r[-1]}" if len(r) > 1 else str(r[0]) for r in runs)
    return f"rep{text}" if len(reps) == 1 else f"reps {text}"
