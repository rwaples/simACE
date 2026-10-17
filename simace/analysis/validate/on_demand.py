"""On-demand checks: run by ``simace validate`` on finished reps, never by ``simace run`` (ADR 0028).

Every calibrated test registers here, whatever it costs, and so would a
quality check that needs outcomes a rep does not store or costs too much for
every rep. Each takes the rep's :class:`~.calibrated.RepInputs` and returns
one section of the rep's ``validation.yaml``, under ``checks.{name}``.

``validation.yaml`` carries a required ``schema`` block, and
:func:`load_validation` refuses a file without it. Every file written before
ADR 0028 has none: the April files from before ADR 0008, and ADR 0027's first
onset-τ files, whose ``summary.passed`` would otherwise read as a verdict.
"""

from __future__ import annotations

__all__ = ["ON_DEMAND_CHECKS", "SCHEMA", "load_validation", "one_warning_for_refused", "run_on_demand"]

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from simace.core.yaml_io import load_yaml

from .calibrated import CalibratedTest
from .onset_tau import run_onset_tau

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from .calibrated import RepInputs

logger = logging.getLogger(__name__)

#: The ``schema`` block of every ``validation.yaml`` this simace writes and reads.
SCHEMA = {"kind": "on_demand_checks", "version": 1}

ON_DEMAND_CHECKS: dict[str, CalibratedTest] = {
    test.name: test
    for test in (
        CalibratedTest(
            "onset_tau",
            "frailty onset τ between relatives against its realized expectation (ADR 0027)",
            run_onset_tau,
        ),
    )
}


def run_on_demand(inputs: RepInputs, checks: Sequence[str]) -> dict[str, Any]:
    """Run the named on-demand checks on one rep and return its ``validation.yaml`` content."""
    from importlib.metadata import version

    return {
        "schema": dict(SCHEMA),
        "simace_version": version("simace"),
        "seed": inputs.seed,
        "checks": {name: ON_DEMAND_CHECKS[name].per_rep(inputs) for name in checks},
    }


#: The refused files of the innermost :func:`one_warning_for_refused`, or None outside one.
_refused: ContextVar[list[Path] | None] = ContextVar("_refused", default=None)


def load_validation(path: Path) -> dict[str, Any] | None:
    """A rep's ``validation.yaml``, or None when the rep has none or the file lacks this simace's schema.

    A refused file is logged by name, so a reader that skips it says why;
    inside :func:`one_warning_for_refused` it is counted instead.
    """
    if not path.exists():
        return None
    data = load_yaml(path)
    found = data.get("schema") if isinstance(data, dict) else None
    if found == SCHEMA:
        return data
    refused = _refused.get()
    if refused is None:
        logger.warning("Ignoring %s: schema %s is not %s; rerun `simace validate` on the rep", path, found, SCHEMA)
    else:
        refused.append(path)
    return None


@contextmanager
def one_warning_for_refused() -> Iterator[None]:
    """Log one warning for every file :func:`load_validation` refuses inside: their count and the first."""
    refused: list[Path] = []
    token = _refused.set(refused)
    try:
        yield
    finally:
        _refused.reset(token)
        if refused:
            logger.warning(
                "Ignoring %d validation.yaml file(s) without schema %s, first %s; rerun `simace validate` on those reps",
                len(refused),
                SCHEMA,
                refused[0],
            )
