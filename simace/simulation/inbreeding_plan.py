"""Which inbreeding engine a simulation runs, decided from the config alone (ADR 0033 §2-§5).

:func:`resolve` maps ``(mode, depth, N, G_sim)`` to an :class:`InbreedingPlan`
with the same answer on every host: the thread count and ``--max-memory`` never
enter it. It imports no numba, because config load and the ``simace run``
parent call it; the engines live in :mod:`simace.simulation.inbreeding`.

The rules:

- ``approx`` with ``depth >= G_sim`` runs exact: its F is exact by construction.
- Small-N handoff: ``approx`` runs exact when ``8 N^2 <= 1 GiB`` and
  ``N^2 G_sim <= 1e9``.
- The exact engine is densep when ``8 N^2 <= 16 GiB`` and its predicted time
  does not exceed walkd2's, else walkd2.
- An exact run is refused when twice the chosen engine's predicted time
  exceeds one hour.
- xexact packs each tail entry into one uint32 when
  ``bitlen(N - 1) + (depth - 1) + bitlen(depth - 1) <= 32``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "COST_MODEL",
    "DEFAULT_DEPTH",
    "INBREEDING_MODES",
    "OFF",
    "InbreedingPlan",
    "InbreedingRefused",
    "plan_for",
    "predicted_seconds",
    "resolve",
]

#: Allowed values of ``pedigree.inbreeding.mode``.
INBREEDING_MODES: tuple[str, ...] = ("off", "approx", "exact")

#: Default of ``pedigree.inbreeding.depth``.
DEFAULT_DEPTH = 6

#: The exact engines' fitted cost models at the fixed reference of 8 threads, as
#: printed in ``notes/issue-76-recommendation/summ/fit_exact2.txt`` (fitted by
#: ``fit_exact2.py`` beside it; the recommendation §8.5). Train pred/meas ratios:
#: densep 0.95-1.42, walkd2 0.56-1.82.
COST_MODEL: dict[str, float] = {
    "densep_t0": 0.145,
    "densep_c": 4.66e-10,  # s per N^2 G
    "walkd2_t0": 0.139,
    "walkd2_a": 3.69e9,  # visits/s = a N^b w^c
    "walkd2_b": -0.163,
    "walkd2_c": -0.023,
    "walkd2_beta": 0.684,  # parents per generation / N
    "walkd2_gamma": 0.792,  # walks per individual per generation
}

_GIB = 2**30
_HANDOFF_DENSE_BYTES = _GIB
_HANDOFF_N2G = 1e9
_DENSEP_MAX_BYTES = 16 * _GIB
_REFUSE_SECONDS = 3600.0
_PACKED_BITS = 32


class InbreedingRefused(ValueError):
    """An exact inbreeding run predicted to take too long (ADR 0033 §4)."""


@dataclass(frozen=True)
class InbreedingPlan:
    """The inbreeding computation one simulation runs; a pure function of ``(mode, depth, N, G_sim)``.

    ``mode`` is the effective mode, after the small-N handoff and the
    ``depth >= G_sim`` rule. ``depth`` is set for approx only, ``packed`` is
    the xexact tail layout and False for every other engine.
    """

    mode: Literal["off", "approx", "exact"]
    engine: Literal["none", "xexact", "densep", "walkd2"]
    depth: int | None = None
    packed: bool = False

    def record(self) -> dict[str, Any]:
        """Return the plan as the ``{mode, engine, depth, packed}`` record run.yaml and the pedigree metadata carry."""
        return {"mode": self.mode, "engine": self.engine, "depth": self.depth, "packed": self.packed}


OFF = InbreedingPlan("off", "none")


def _walkd2_visits(N: int, G: int) -> float:
    c = COST_MODEL
    P = c["walkd2_beta"] * N
    total = 0.0
    cum = 0.0
    for k in range(G):
        cum += P * (1.0 - math.exp(-(2.0**k) / P))
        total += cum
    return c["walkd2_gamma"] * N * total


def predicted_seconds(engine: str, N: int, G: int) -> float:
    """Return the cost model's wall time at 8 threads for an exact engine on ``N`` per generation and ``G`` generations."""
    c = COST_MODEL
    if engine == "densep":
        return c["densep_t0"] + c["densep_c"] * N * N * G
    if engine == "walkd2":
        V = _walkd2_visits(N, G)
        w = V / (c["walkd2_gamma"] * N * G)
        return c["walkd2_t0"] + V / (c["walkd2_a"] * N ** c["walkd2_b"] * w ** c["walkd2_c"])
    raise ValueError(f"no cost model for engine {engine!r}")


def _refused(seconds: float) -> bool:
    return 2.0 * seconds > _REFUSE_SECONDS


def _packed(N: int, depth: int) -> bool:
    # The prototype's field widths (fusedx_pre.py): at least one bit for the id and the generation offset.
    ib = max(1, (N - 1).bit_length())
    gb = max(1, (depth - 1).bit_length())
    return ib + (depth - 1) + gb <= _PACKED_BITS


def _exact(N: int, G_sim: int, hint: str) -> InbreedingPlan:
    t_walk = predicted_seconds("walkd2", N, G_sim)
    if 8 * N * N <= _DENSEP_MAX_BYTES and predicted_seconds("densep", N, G_sim) <= t_walk:
        engine, t = "densep", predicted_seconds("densep", N, G_sim)
    else:
        engine, t = "walkd2", t_walk
    if _refused(t):
        raise InbreedingRefused(
            f"exact inbreeding at N={N}, G_sim={G_sim} is refused: {engine} is predicted to take {t:,.0f} s at 8 "
            f"threads, and twice that exceeds {_REFUSE_SECONDS:,.0f} s (ADR 0033 §4); {hint}"
        )
    return InbreedingPlan("exact", engine)  # type: ignore[arg-type]


def resolve(mode: str, depth: int, N: int, G_sim: int) -> InbreedingPlan:
    """Return the plan for ``mode`` and ``depth`` on ``N`` per generation and ``G_sim`` simulated generations.

    Raises:
        ValueError: for a mode outside :data:`INBREEDING_MODES` or a depth
            that is not an integer >= 1.
        InbreedingRefused: when the run would be exact and its predicted
            time fails the refusal rule.
    """
    if mode not in INBREEDING_MODES:
        raise ValueError(f"inbreeding mode must be one of {list(INBREEDING_MODES)}, got {mode!r}")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError(f"inbreeding depth must be an integer >= 1, got {depth!r}")
    if mode == "off":
        return OFF
    if mode == "exact":
        return _exact(N, G_sim, "use mode: approx")
    if depth >= G_sim:
        return _exact(N, G_sim, f"lower depth below G_sim ({G_sim}; default {DEFAULT_DEPTH})")
    if 8 * N * N <= _HANDOFF_DENSE_BYTES and N * N * G_sim <= _HANDOFF_N2G:
        return _exact(N, G_sim, "use mode: approx")  # unreachable: N^2 G_sim <= 1e9 predicts < 1 s
    return InbreedingPlan("approx", "xexact", depth, _packed(N, depth))


def plan_for(params: Mapping[str, Any]) -> InbreedingPlan:
    """Resolve the plan of a flat scenario or rep parameter dict; absent inbreeding keys mean off (pre-ADR-0033)."""
    G_sim = params.get("G_sim") or params["G_ped"]
    return resolve(
        params.get("inbreeding_mode", "off"), params.get("inbreeding_depth", DEFAULT_DEPTH), params["N"], G_sim
    )
