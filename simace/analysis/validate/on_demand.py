"""On-demand validation checks: run by ``simace validate`` on finished reps, never by ``simace run``.

Analyze runs the cheap checks on every rep. A check that costs too much for
that, or that needs outcomes the rep does not store (the phenotyped
population's raw onsets, regenerated from the rep seed), registers here. Each
check takes the rep's recorded pedigree, its seed and its phenotype-stage
arguments, and returns one section of the rep's ``validation.yaml``, which
carries a ``summary`` with ``passed``, ``n_scored`` and ``n_failed``.
"""

from __future__ import annotations

__all__ = ["ON_DEMAND_CHECKS", "OnDemandCheck", "run_on_demand"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .onset_tau import run_onset_tau

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import polars as pl


@dataclass(frozen=True)
class OnDemandCheck:
    """One on-demand check family: its name in ``validation.yaml``, a one-line description, and its runner."""

    name: str
    description: str
    run: Callable[..., dict[str, Any]]


ON_DEMAND_CHECKS: dict[str, OnDemandCheck] = {
    check.name: check
    for check in (
        OnDemandCheck(
            "onset_tau",
            "frailty onset τ between relatives against its realized expectation (ADR 0027)",
            run_onset_tau,
        ),
    )
}


def run_on_demand(
    pedigree: pl.DataFrame, *, seed: int, phenotype: Mapping[str, Any], checks: Sequence[str]
) -> dict[str, Any]:
    """Run the named on-demand checks on one rep and return its ``validation.yaml`` content."""
    from importlib.metadata import version

    sections = {name: ON_DEMAND_CHECKS[name].run(pedigree, seed=seed, phenotype=phenotype) for name in checks}
    return {
        "simace_version": version("simace"),
        "seed": seed,
        "summary": {
            "passed": all(section["summary"]["passed"] for section in sections.values()),
            "n_failed": sum(section["summary"]["n_failed"] for section in sections.values()),
        },
        "checks": sections,
    }
