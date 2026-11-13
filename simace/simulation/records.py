"""What a simulation reports besides its pedigree: one pairing record per mating iteration and the C-transmission clips.

Each record's ``to_dict`` is the row ``mating.yaml`` and the pedigree's
``simace_c_transmission_clips`` metadata store, with the keys in the order
those files have always had.
"""

from __future__ import annotations

__all__ = [
    "AssortativePairing",
    "CClip",
    "GenerationClip",
    "MatingIteration",
    "PairingRecord",
    "RandomPairing",
    "SimulationResult",
    "WrightFisherPairing",
]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from simace.simulation.assortment import METROPOLIS_TOL

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl


@dataclass(frozen=True)
class RandomPairing:
    """Random mating: ``M`` couples, no target to pin."""

    M: int

    def to_dict(self) -> dict[str, Any]:
        """This record as its ``mating.yaml`` row."""
        return {"mode": "random", "M": self.M}


@dataclass(frozen=True)
class WrightFisherPairing:
    """Wright-Fisher mating: each offspring draws its parents independently, so there are no couples to record."""

    def to_dict(self) -> dict[str, Any]:
        """This record as its ``mating.yaml`` row."""
        return {"mode": "wright_fisher"}


@dataclass(frozen=True)
class AssortativePairing:
    """One assortative iteration: the Metropolis effort and the final slot-level errors (correlation minus target).

    ``errors`` is empty when nothing was pinned (``M < 2``, or a constant
    active trait); such a record is not converged.
    """

    mode: Literal["single", "both"]
    M: int
    proposals: int
    max_proposals: int
    errors: Mapping[str, float]
    duplicates_left: int

    @property
    def converged(self) -> bool:
        """Every error is within ``METROPOLIS_TOL``, and there is at least one."""
        return bool(self.errors) and all(abs(e) < METROPOLIS_TOL for e in self.errors.values())

    def to_dict(self) -> dict[str, Any]:
        """This record as its ``mating.yaml`` row."""
        return {
            "mode": self.mode,
            "M": self.M,
            "proposals": self.proposals,
            "max_proposals": self.max_proposals,
            "errors": dict(self.errors),
            "duplicates_left": self.duplicates_left,
            "converged": self.converged,
        }


PairingRecord = RandomPairing | WrightFisherPairing | AssortativePairing


@dataclass(frozen=True)
class MatingIteration:
    """The pairing of mating iteration ``iteration``, whose offspring are recorded generation ``generation``.

    ``generation`` is negative during burn-in.
    """

    iteration: int
    generation: int
    pairing: PairingRecord

    def to_dict(self) -> dict[str, Any]:
        """This record as its ``mating.yaml`` row."""
        return {"iteration": self.iteration, "generation": self.generation, **self.pairing.to_dict()}


@dataclass(frozen=True)
class CClip:
    """A negative eigenvalue of a fresh C draw's covariance, clipped to 0 under C transmission (ADR 0025).

    ``excess`` is the clipped eigenvalue's magnitude, within ``tolerance``,
    and ``direction`` its eigenvector.
    """

    excess: float
    tolerance: float
    direction: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """This record as its ``simace_c_transmission_clips`` metadata entry."""
        return {"excess": self.excess, "tolerance": self.tolerance, "direction": list(self.direction)}


@dataclass(frozen=True)
class GenerationClip:
    """A clip in the reproduction that made recorded generation ``generation`` (negative during burn-in)."""

    generation: int
    clip: CClip

    def to_dict(self) -> dict[str, Any]:
        """This record as its ``simace_c_transmission_clips`` metadata entry."""
        return {"generation": self.generation, **self.clip.to_dict()}


@dataclass(frozen=True)
class SimulationResult:
    """A simulation's recorded pedigree, its pairing record per mating iteration, and its C-transmission clips."""

    pedigree: pl.DataFrame
    pairing: tuple[MatingIteration, ...]
    clips: tuple[GenerationClip, ...]
