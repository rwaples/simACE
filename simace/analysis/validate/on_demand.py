"""On-demand checks: run by ``simace validate`` on finished reps, never by ``simace run`` (ADR 0028).

Every calibrated test registers here, whatever it costs, and so would a
quality check that needs outcomes a rep does not store or costs too much for
every rep. Each takes the rep's :class:`~.calibrated.RepInputs` and returns
one section of the rep's ``validation.yaml``, under ``checks.{name}``.

``validation.yaml`` carries a required ``schema`` block, and
:func:`load_validation` refuses a file without it (:mod:`.schema_file`).
"""

from __future__ import annotations

__all__ = ["ON_DEMAND_CHECKS", "SCHEMA", "load_validation", "run_on_demand"]

from typing import TYPE_CHECKING, Any

from .calibrated import FAMILYWISE_ALPHA, CalibratedTest
from .generation0 import CROSS_TRAIT_RC_CELLS, TOTAL_VARIANCE_CELLS, run_cross_trait_rc, run_total_variance
from .inheritance import HOUSEHOLD_DRAW_CELLS, MENDELIAN_RESIDUAL_CELLS, run_household_draw, run_mendelian_residual
from .mate_correlation import CELL_NAMES as MATE_CORRELATION_CELLS
from .mate_correlation import run_mate_correlation
from .onset_tau import CELL_NAMES as ONSET_TAU_CELLS
from .onset_tau import run_onset_tau
from .schema_file import load_schemad
from .verdict import ZCalibration

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from .calibrated import RepInputs

#: The ``schema`` block of every ``validation.yaml`` this simace writes and reads.
SCHEMA = {"kind": "on_demand_checks", "version": 1}

ON_DEMAND_CHECKS: dict[str, CalibratedTest] = {
    test.name: test
    for test in (
        CalibratedTest(
            "onset_tau",
            "frailty onset τ between relatives against its realized expectation (ADR 0027)",
            run_onset_tau,
            ZCalibration(rep_rate=FAMILYWISE_ALPHA),
            min_reps=20,
            cell_names=ONSET_TAU_CELLS,
            own_page=True,
        ),
        CalibratedTest(
            "mendelian_residual",
            "A minus midparent: uncorrelated between FS, MHS and PHS, variance A/2 (ADR 0029)",
            run_mendelian_residual,
            ZCalibration(rep_rate=FAMILYWISE_ALPHA),
            min_reps=20,
            cell_names=MENDELIAN_RESIDUAL_CELLS,
        ),
        CalibratedTest(
            "household_draw",
            "C minus its transmitted part: uncorrelated between PHS (ADR 0029)",
            run_household_draw,
            ZCalibration(rep_rate=FAMILYWISE_ALPHA),
            min_reps=20,
            cell_names=HOUSEHOLD_DRAW_CELLS,
        ),
        CalibratedTest(
            "total_variance",
            "Var(A + C + E) over recorded generation 0 against A + C + E, household-cluster SE (ADR 0030)",
            run_total_variance,
            ZCalibration(rep_rate=FAMILYWISE_ALPHA),
            min_reps=20,
            cell_names=TOTAL_VARIANCE_CELLS,
        ),
        CalibratedTest(
            "cross_trait_rC",
            "r(C1, C2) over recorded generation 0 against rC, household-cluster SE (ADR 0030)",
            run_cross_trait_rc,
            ZCalibration(rep_rate=FAMILYWISE_ALPHA),
            min_reps=20,
            cell_names=CROSS_TRAIT_RC_CELLS,
        ),
        CalibratedTest(
            "mate_correlation",
            "mate correlation per parents' generation against its target, SE by pairing mode (ADR 0031)",
            run_mate_correlation,
            ZCalibration(rep_rate=FAMILYWISE_ALPHA),
            min_reps=20,
            cell_names=MATE_CORRELATION_CELLS,
            own_page=True,
        ),
    )
}


def run_on_demand(inputs: RepInputs, checks: Sequence[str], source: str | None) -> dict[str, Any]:
    """Run the named on-demand checks on one rep and return its ``validation.yaml`` content.

    ``source`` is the ``git describe`` of the checkout the build runs from (None for a wheel install).
    """
    from importlib.metadata import version

    return {
        "schema": dict(SCHEMA),
        "simace_version": version("simace"),
        "source": source,
        "seed": inputs.seed,
        "checks": {name: ON_DEMAND_CHECKS[name].per_rep(inputs) for name in checks},
    }


def load_validation(path: Path) -> dict[str, Any] | None:
    """A rep's ``validation.yaml``, or None when the rep has none or the file lacks this simace's schema."""
    return load_schemad(path, SCHEMA, "validation.yaml")
