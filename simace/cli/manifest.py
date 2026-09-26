"""The per-rep ``run.yaml`` manifest: proof that a rep finished, and with what.

A rep is complete only when ``run.yaml`` exists. ``simace run`` writes it
after every stage of the rep exits 0, and deletes it before recomputing.
Its ``resolved`` dict holds the values of the config keys the rep's stages
and ``params.yaml`` read, which is how a later run tells a finished rep from
a stale one. Only keys the current code reads are compared. The simace
version that built the rep is recorded but never makes it stale; ``simace ls``
shows it when it differs from the running version.
"""

from __future__ import annotations

__all__ = ["Manifest", "RepState", "RepStatus", "manifest_params", "rep_status", "write_manifest"]

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import yaml

import simace
from simace.core.publish import publish
from simace.core.yaml_io import dump_yaml, load_yaml, to_native

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path


def manifest_params(resolved: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Return ``resolved`` restricted to ``keys``, in YAML-native form.

    ``keys`` are the config keys a rep's outputs depend on, so changing any
    other key (a fitACE-only setting, a new default) leaves finished reps
    complete.
    """
    kept = {key: resolved[key] for key in sorted(keys)}
    return yaml.safe_load(yaml.safe_dump(to_native(kept)))


@dataclass(frozen=True)
class Manifest:
    """What a finished rep was computed from."""

    scenario: str
    rep: int
    seed: int
    resolved: dict[str, Any]
    stages: list[str]


class RepState(StrEnum):
    """Whether a rep's outputs are usable as they stand."""

    COMPLETE = "complete"
    STALE = "stale"
    ABSENT = "absent"


@dataclass(frozen=True)
class RepStatus:
    """A rep's state, the keys that differ when it is stale, and the simace version that built it."""

    state: RepState
    differing: tuple[str, ...] = ()
    simace_version: str | None = None


def write_manifest(path: Path, manifest: Manifest) -> None:
    """Publish ``run.yaml`` atomically."""
    body = {
        "simace_version": simace.__version__,
        "scenario": manifest.scenario,
        "rep": manifest.rep,
        "seed": manifest.seed,
        "resolved": manifest.resolved,
        "stages": manifest.stages,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    with publish(path) as (tmp,):
        dump_yaml(body, tmp)


def rep_status(path: Path, expected: Manifest) -> RepStatus:
    """Compare the ``run.yaml`` at ``path`` with what the rep would be built from now."""
    if not path.exists():
        return RepStatus(RepState.ABSENT)
    recorded = load_yaml(path)
    if not isinstance(recorded, dict) or not isinstance(recorded.get("resolved"), dict):
        return RepStatus(RepState.STALE, ("run.yaml is not a manifest",))
    old, new = recorded["resolved"], expected.resolved
    differing = sorted(key for key in new if old.get(key, _MISSING) != new[key])
    if recorded.get("stages") != expected.stages:
        differing.append("stages")
    built_by = recorded.get("simace_version")
    if differing:
        return RepStatus(RepState.STALE, tuple(differing), built_by)
    return RepStatus(RepState.COMPLETE, simace_version=built_by)


_MISSING = object()
