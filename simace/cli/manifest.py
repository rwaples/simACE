"""The per-rep ``run.yaml`` manifest, proof that a rep finished and with what, and the scenario ``plots/plots.yaml``.

``simace run`` writes ``run.yaml`` after every stage of the rep exits 0, and
deletes it before recomputing. A rep is complete only when its ``run.yaml``
names this scenario, rep, and seed, records the current values of the config
keys the rep's stages and ``params.yaml`` read, lists the current stages and
results layout (a rep from before ADR 0021 has none, so it reads stale), and
every output the rep declares exists with the size and mtime the manifest
recorded (a stage rerun by hand rewrites its output and so makes the rep
incomplete). Only keys the current code reads are compared. The simace version that built the rep is recorded but never makes
it stale; ``simace ls`` shows it when it differs from the running version.

``plots/plots.yaml`` is written after a scenario's plot and atlas stages
exit 0. It fingerprints the ``run.yaml`` of every rep the plots were built
from and the atlas files it produced, so ``simace ls`` and the run summary
can say whether the plots are current, stale (a rep recomputed or not
plotted, an atlas gone), or absent.
"""

from __future__ import annotations

__all__ = [
    "Fingerprint",
    "Manifest",
    "PlotsState",
    "PlotsStatus",
    "RepState",
    "RepStatus",
    "changed_outputs",
    "fingerprints",
    "manifest_params",
    "plots_status",
    "read_manifest",
    "recorded_stages",
    "rep_status",
    "source_ref",
    "write_manifest",
    "write_plots_manifest",
]

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
from simace.core.yaml_io import dump_yaml, load_yaml, to_native, yaml_loader

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


def manifest_params(resolved: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    """Return ``resolved`` restricted to ``keys``, in YAML-native form.

    ``keys`` are the config keys a rep's outputs depend on, so changing any
    other key (a fitACE-only setting, a new default) leaves finished reps
    complete.
    """
    kept = {key: resolved[key] for key in sorted(keys)}
    return yaml.load(yaml.dump(to_native(kept), Dumper=_SAFE_DUMPER), Loader=yaml_loader())


# libyaml's dumper when present: this round trip runs once per rep status check.
_SAFE_DUMPER: type = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


#: A file's size and nanosecond mtime, enough to tell an atomically replaced output from the one recorded.
Fingerprint = dict[str, int]


def fingerprints(paths: Iterable[Path]) -> dict[str, Fingerprint]:
    """Return ``{basename: {"size", "mtime_ns"}}`` for every path; each must exist."""
    out = {}
    for path in paths:
        st = path.stat()
        out[path.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    return out


def changed_outputs(recorded: Mapping[str, Any], paths: Iterable[Path]) -> tuple[str, ...]:
    """Return the basenames of ``paths`` whose current fingerprint differs from ``recorded`` (absent counts)."""
    changed = []
    for path in paths:
        try:
            st = path.stat()
        except FileNotFoundError:
            changed.append(path.name)
            continue
        if recorded.get(path.name) != {"size": st.st_size, "mtime_ns": st.st_mtime_ns}:
            changed.append(path.name)
    return tuple(changed)


@dataclass(frozen=True)
class Manifest:
    """What a finished rep was computed from."""

    scenario: str
    rep: int
    seed: int
    resolved: dict[str, Any]
    stages: list[str]
    layout: int


class RepState(StrEnum):
    """Whether a rep's outputs are usable as they stand.

    ``PARTIAL`` is a rep complete through fewer stages than were asked for
    (``simace run --until``); the missing stages can resume from it.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
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


def write_manifest(path: Path, manifest: Manifest, outputs: Iterable[Path]) -> None:
    """Publish ``run.yaml`` atomically, recording the fingerprint of every output the rep declares."""
    body = {
        "simace_version": simace.__version__,
        "source": source_ref(),
        "scenario": manifest.scenario,
        "rep": manifest.rep,
        "seed": manifest.seed,
        "resolved": manifest.resolved,
        "stages": manifest.stages,
        "layout": manifest.layout,
        "outputs": fingerprints(outputs),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    with publish(path) as (tmp,):
        dump_yaml(body, tmp)


def read_manifest(path: Path) -> Any:
    """Return the loaded ``run.yaml`` at ``path``, or None when it does not exist."""
    return load_yaml(path) if path.exists() else None


def recorded_stages(recorded: Any) -> list[str] | None:
    """Return the stage names a loaded ``run.yaml`` records, or None when there is no such list."""
    stages = recorded.get("stages") if isinstance(recorded, dict) else None
    return stages if isinstance(stages, list) else None


def rep_status(recorded: Any, expected: Manifest, outputs: Iterable[Path]) -> RepStatus:
    """Compare a loaded ``run.yaml`` and the rep's ``outputs`` with what the rep would be built from now.

    A manifest that disagrees with ``expected``, or that records no output
    fingerprints, makes the rep stale, which ``simace run`` refuses without
    ``--force``. A matching manifest with an output missing or rewritten
    since the manifest makes it incomplete, which ``simace run`` recomputes.
    """
    if recorded is None:
        return RepStatus(RepState.ABSENT)
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
    for key, value in (("stages", expected.stages), ("layout", expected.layout)):
        if recorded.get(key) != value:
            changes[key] = (recorded.get(key, _MISSING), value)
    if not isinstance(recorded.get("outputs"), dict):
        changes["outputs"] = (_MISSING, "fingerprints (run.yaml predates them)")
    built_by, source = recorded.get("simace_version"), recorded.get("source")
    if changes:
        return RepStatus(RepState.STALE, tuple(changes), built_by, changes, source)
    outputs = list(outputs)
    missing = tuple(output.name for output in outputs if not output.exists())
    if missing:
        return RepStatus(RepState.INCOMPLETE, tuple(f"{name} missing" for name in missing), built_by, source=source)
    changed = changed_outputs(recorded["outputs"], outputs)
    if changed:
        return RepStatus(RepState.INCOMPLETE, tuple(f"{name} changed" for name in changed), built_by, source=source)
    return RepStatus(RepState.COMPLETE, simace_version=built_by, source=source)


_MISSING = object()


class PlotsState(StrEnum):
    """Whether a scenario's plots and atlas were built from its complete reps as they stand."""

    CURRENT = "current"
    STALE = "stale"
    ABSENT = "absent"


@dataclass(frozen=True)
class PlotsStatus:
    """A scenario's plot state and why it is not current."""

    state: PlotsState
    reasons: tuple[str, ...] = ()

    def describe(self) -> str:
        """Return ``current``, ``absent``, or ``stale (why, why)``."""
        return f"{self.state} ({', '.join(self.reasons)})" if self.reasons else str(self.state)


def write_plots_manifest(path: Path, rep_manifests: Mapping[str, Path], outputs: Iterable[Path]) -> None:
    """Publish ``plots.yaml``: the fingerprint of each rep's ``run.yaml`` and of each atlas the pass wrote."""
    body = {
        "simace_version": simace.__version__,
        "source": source_ref(),
        "reps": {label: fingerprints([manifest])[manifest.name] for label, manifest in rep_manifests.items()},
        "outputs": fingerprints(outputs),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    with publish(path) as (tmp,):
        dump_yaml(body, tmp)


def plots_status(path: Path, rep_manifests: Mapping[str, Path], not_complete: Iterable[str] = ()) -> PlotsStatus:
    """Compare ``plots.yaml`` at ``path`` with the reps' ``run.yaml`` files now.

    ``rep_manifests`` maps every configured rep's label to its ``run.yaml``;
    ``not_complete`` names the reps that are not complete now, which make the
    plots stale even when their manifest is unchanged (an output rewritten by
    hand). A rep recomputed since, not plotted, or dropped from the
    configuration, or an atlas file changed or gone, also makes them stale.
    """
    if not path.exists():
        return PlotsStatus(PlotsState.ABSENT)
    recorded = load_yaml(path)
    if not isinstance(recorded, dict) or not isinstance(recorded.get("reps"), dict):
        return PlotsStatus(PlotsState.STALE, ("plots.yaml is not a manifest",))
    reasons = [f"{label} not complete" for label in not_complete]
    for label, manifest in rep_manifests.items():
        if label in not_complete:
            continue
        if label not in recorded["reps"]:
            reasons.append(f"{label} not plotted")
        elif changed_outputs({manifest.name: recorded["reps"][label]}, [manifest]):
            reasons.append(f"{label} recomputed since")
    reasons.extend(f"{label} no longer configured" for label in recorded["reps"] if label not in rep_manifests)
    outputs = recorded.get("outputs") if isinstance(recorded.get("outputs"), dict) else {}
    for name in outputs:
        output = path.with_name(name)
        if not output.exists():
            reasons.append(f"{name} missing")
        elif changed_outputs(outputs, [output]):
            reasons.append(f"{name} changed")
    return PlotsStatus(PlotsState.STALE, tuple(reasons)) if reasons else PlotsStatus(PlotsState.CURRENT)
