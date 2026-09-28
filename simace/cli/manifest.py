"""The per-rep ``run.yaml`` manifest: proof that a rep finished, and with what.

``simace run`` writes ``run.yaml`` after every stage of the rep exits 0, and
deletes it before recomputing. A rep is complete only when its ``run.yaml``
names this scenario, rep, and seed, records the current values of the config
keys the rep's stages and ``params.yaml`` read, lists the current stages, and
every output the rep declares exists. Only keys the current code reads are
compared. The simace version that built the rep is recorded but never makes
it stale; ``simace ls`` shows it when it differs from the running version.
"""

from __future__ import annotations

__all__ = ["Manifest", "RepState", "RepStatus", "manifest_params", "rep_status", "source_ref", "write_manifest"]

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

import simace
from simace.core.publish import publish
from simace.core.yaml_io import dump_yaml, load_yaml, to_native

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


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
    INCOMPLETE = "incomplete"
    ABSENT = "absent"


@dataclass(frozen=True)
class RepStatus:
    """A rep's state, why (the differing keys, or the missing outputs), and the simace version that built it.

    ``changes`` maps each differing key of a stale rep to its recorded and
    current values (``_MISSING`` when the manifest lacks the key).
    """

    state: RepState
    reasons: tuple[str, ...] = ()
    simace_version: str | None = None
    changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    source: str | None = None

    def describe(self) -> str:
        """Return the reasons, with ``key: old -> new`` for each recorded change."""
        return ", ".join(
            f"{key}: {_short(self.changes[key][0])} -> {_short(self.changes[key][1])}" if key in self.changes else key
            for key in self.reasons
        )


_VALUE_WIDTH = 40


def _short(value: Any) -> str:
    if value is _MISSING:
        return "(absent)"
    if isinstance(value, dict | list):
        text = yaml.safe_dump(value, default_flow_style=True, width=float("inf")).strip()
    else:
        text = str(value)
    return text if len(text) <= _VALUE_WIDTH else text[: _VALUE_WIDTH - 1] + "…"


@cache
def source_ref() -> str | None:
    """Return ``git describe --tags --always --dirty`` for the checkout simace is imported from, or None.

    None for a wheel install or without git. Computed once per process.
    """
    try:
        done = subprocess.run(
            ["git", "-C", str(Path(simace.__file__).parent), "describe", "--tags", "--always", "--dirty"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


def write_manifest(path: Path, manifest: Manifest) -> None:
    """Publish ``run.yaml`` atomically."""
    body = {
        "simace_version": simace.__version__,
        "source": source_ref(),
        "scenario": manifest.scenario,
        "rep": manifest.rep,
        "seed": manifest.seed,
        "resolved": manifest.resolved,
        "stages": manifest.stages,
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    with publish(path) as (tmp,):
        dump_yaml(body, tmp)


def rep_status(path: Path, expected: Manifest, outputs: Iterable[Path]) -> RepStatus:
    """Compare the ``run.yaml`` at ``path`` and the rep's ``outputs`` with what the rep would be built from now.

    A manifest that disagrees with ``expected`` makes the rep stale, which
    ``simace run`` refuses without ``--force``. A matching manifest with an
    output missing makes it incomplete, which ``simace run`` recomputes.
    """
    if not path.exists():
        return RepStatus(RepState.ABSENT)
    recorded = load_yaml(path)
    if not isinstance(recorded, dict) or not isinstance(recorded.get("resolved"), dict):
        return RepStatus(RepState.STALE, ("run.yaml is not a manifest",))
    identity = {"scenario": expected.scenario, "rep": expected.rep, "seed": expected.seed}
    changes = {
        key: (recorded.get(key, _MISSING), value) for key, value in identity.items() if recorded.get(key) != value
    }
    old, new = recorded["resolved"], expected.resolved
    changes.update(
        {key: (old.get(key, _MISSING), new[key]) for key in sorted(new) if old.get(key, _MISSING) != new[key]}
    )
    if recorded.get("stages") != expected.stages:
        changes["stages"] = (recorded.get("stages", _MISSING), expected.stages)
    built_by, source = recorded.get("simace_version"), recorded.get("source")
    if changes:
        return RepStatus(RepState.STALE, tuple(changes), built_by, changes, source)
    missing = tuple(output.name for output in outputs if not output.exists())
    if missing:
        return RepStatus(RepState.INCOMPLETE, missing, built_by, source=source)
    return RepStatus(RepState.COMPLETE, simace_version=built_by, source=source)


_MISSING = object()
