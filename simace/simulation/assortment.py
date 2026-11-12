"""Standard-mating assortative-mating plan.

Encapsulates the standard-model-only assortment logic that used to be two
``if mating_model == "standard"`` blocks inside ``run_simulation``: the
per-generation assortment resolution, the within-person cross-trait liability
correlation (``rho_w``), the ``|rho_w| < 1`` guard, the full 4x4 PSD check, and
the per-generation mate-correlation matrix (``R_mf_i``) handed to
``_mating_standard``.

Wright-Fisher has no assortative mating, so ``run_simulation`` only builds an
``AssortmentPlan`` on the standard path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from simace.simulation.params import resolve_per_gen_param

if TYPE_CHECKING:
    from pathlib import Path

#: Assortment's Metropolis swaps stop once every slot-level mate correlation
#: they pin is within this of its target. ``simace validate``'s
#: ``mate_correlation`` test drops a cell whose SE this no longer dwarfs; it
#: lives here so the test reads it without importing the simulator.
METROPOLIS_TOL = 5e-4


def write_mating_log(mating_log: list[dict[str, Any]], path: str | Path) -> None:
    """Publish ``run_simulation``'s ``mating_log`` as a rep's ``mating.yaml``."""
    from simace.core.publish import publish
    from simace.core.yaml_io import dump_yaml

    with publish(path) as (tmp,):
        dump_yaml({"metropolis_tol": METROPOLIS_TOL, "iterations": mating_log}, tmp)


def read_mating_log(path: str | Path) -> list[dict[str, Any]]:
    """The pairing records of a ``mating.yaml``, one per mating iteration (:func:`write_mating_log`)."""
    from simace.core.yaml_io import load_yaml

    return load_yaml(path)["iterations"]


def parent_ce_gen(i: int) -> int:
    """C/E generation of the population being mated at loop iteration ``i``.

    Founders (``i=0``) carry gen-0 C/E and offspring from iteration ``j`` carry
    ``per_gen[j]`` C/E, so the parents at iteration ``i`` carry
    ``max(0, i - 1)``.  Shared by :meth:`AssortmentPlan.build`'s guards and
    :meth:`AssortmentPlan.for_generation` so the validated schedule and the
    runtime schedule cannot drift apart.
    """
    return max(0, i - 1)


def assortment_schedule(
    assort1: float | dict[int, float],
    assort2: float | dict[int, float],
    *,
    G_sim: int,
    assort_matrix: np.ndarray | list[list[float]] | None = None,
    mating_model: str = "standard",
) -> tuple[list[float], list[float]]:
    """The ``assort1`` and ``assort2`` the simulator mates on in each of the ``G_sim`` mating iterations.

    A set ``assort_matrix`` replaces both by its diagonal. The Wright-Fisher
    mating model pairs parents at random whatever assortment it inherits, so
    both are zero there. ``run_simulation`` and the analysis both read the
    schedule from here.
    """
    if mating_model != "standard":
        return [0.0] * G_sim, [0.0] * G_sim
    if assort_matrix is not None:
        matrix = np.asarray(assort_matrix, dtype=np.float64)
        return [float(matrix[0, 0])] * G_sim, [float(matrix[1, 1])] * G_sim

    return (
        resolve_per_gen_param(assort1, G_sim, name="assort1", allow_negative=True),
        resolve_per_gen_param(assort2, G_sim, name="assort2", allow_negative=True),
    )


def _cross_am_matrix(assort1: float, assort2: float, rho_w: float) -> np.ndarray:
    """Build the 2x2 mate-correlation matrix ``R_mf`` for both-trait assortment.

    The off-diagonal cross-AM term is derived from ``rho_w``:
    ``c = rho_w * sqrt(|assort1 * assort2|) * sign(assort1 * assort2)``.
    """
    c = rho_w * np.sqrt(abs(assort1 * assort2)) * np.sign(assort1 * assort2)
    return np.array([[assort1, c], [c, assort2]])


@dataclass(frozen=True)
class AssortmentPlan:
    """Per-generation assortative-mating schedule for the standard mating model.

    Built once per simulation via :meth:`build` (which validates ``rho_w`` and
    the 4x4 PSD constraint up front); :meth:`for_generation` then yields the
    per-iteration assortment inputs for ``_mating_standard``.
    """

    assort1_per_gen: list[float]
    assort2_per_gen: list[float]
    rho_w_per_ce: list[float]
    R_mf_user: np.ndarray | None

    @classmethod
    def build(
        cls,
        *,
        assort1_per_gen: list[float],
        assort2_per_gen: list[float],
        R_mf_user: np.ndarray | None,
        rA: float,
        rC: float,
        rE: float,
        A1: float,
        A2: float,
        C1: float,
        C2: float,
        E1_per_gen: list[float],
        E2_per_gen: list[float],
        G_sim: int,
    ) -> AssortmentPlan:
        """Validate rho_w + PSD over the mating iterations of an :func:`assortment_schedule`.

        Args:
            assort1_per_gen: Trait-1 mate correlation per mating iteration.
            assort2_per_gen: Trait-2 mate correlation per mating iteration.
            R_mf_user: Explicit 2x2 mate-correlation matrix, or ``None`` to
                auto-compute the cross-AM off-diagonal from ``rho_w``.
            rA: Cross-trait additive-genetic correlation.
            rC: Cross-trait shared-environment correlation.
            rE: Cross-trait unique-environment correlation.
            A1: Trait-1 additive-genetic variance (constant across gens).
            A2: Trait-2 additive-genetic variance (constant across gens).
            C1: Trait-1 shared-environment variance (constant across gens).
            C2: Trait-2 shared-environment variance (constant across gens).
            E1_per_gen: Per-generation trait-1 unique-environment variance.
            E2_per_gen: Per-generation trait-2 unique-environment variance.
            G_sim: Total simulated generations.

        Raises:
            ValueError: if both-trait AM is requested with ``|rho_w| >= 1`` at
                any generation, or the full 4x4 ``Sigma_4`` is not PSD.
        """
        # Within-person cross-trait liability correlation per C/E generation
        _rho_w_AC = rA * np.sqrt(A1 * A2) + rC * np.sqrt(C1 * C2)
        rho_w_per_ce = [_rho_w_AC + rE * np.sqrt(E1_per_gen[g] * E2_per_gen[g]) for g in range(G_sim)]

        # Both guards below iterate over *mating iterations* and pair each
        # iteration's assortment with the parental C/E generation the runtime
        # actually uses -- ``rho_w_per_ce[max(0, i - 1)]``, matching
        # :meth:`for_generation`.  Validating the ``g``-indexed pairing instead
        # could reject a safe schedule or accept an invalid one under
        # generation-dependent C/E, and validated the final offspring C/E
        # generation even though that population is never mated.
        for i in range(G_sim):
            rw = rho_w_per_ce[parent_ce_gen(i)]
            if assort1_per_gen[i] != 0 and assort2_per_gen[i] != 0 and abs(rw) >= 1.0 - 1e-10:
                raise ValueError(
                    f"Both-trait assortative mating requires |rho_w| < 1 "
                    f"(got rho_w={rw:.4f} at mating iteration {i}, "
                    f"parental C/E generation {parent_ce_gen(i)}). "
                    f"Traits are perfectly correlated; "
                    f"use single-trait assortment instead."
                )

        # Validate PSD of the full 4x4 Sigma actually used at each iteration.
        for i in range(G_sim):
            rw = rho_w_per_ce[parent_ce_gen(i)]
            a1_i = assort1_per_gen[i]
            a2_i = assort2_per_gen[i]
            if R_mf_user is None and not (a1_i != 0 and a2_i != 0):
                continue  # PSD check is only meaningful when both-trait AM is active
            if R_mf_user is not None:
                R_mf_i = R_mf_user
            else:
                R_mf_i = _cross_am_matrix(a1_i, a2_i, rw)
            R_ff = np.array([[1.0, rw], [rw, 1.0]])
            Sigma_4 = np.block([[R_ff, R_mf_i.T], [R_mf_i, R_ff]])
            eigvals = np.linalg.eigvalsh(Sigma_4)
            if eigvals[0] < -1e-8:
                raise ValueError(
                    f"Full 4x4 mate correlation matrix Sigma_4 is not PSD "
                    f"(min eigenvalue = {eigvals[0]:.6f} at mating iteration {i}, "
                    f"parental C/E generation {parent_ce_gen(i)}). "
                    f"Reduce the magnitude of assort_matrix off-diagonal entries "
                    f"or per-gen assort1/assort2."
                )

        return cls(
            assort1_per_gen=assort1_per_gen,
            assort2_per_gen=assort2_per_gen,
            rho_w_per_ce=rho_w_per_ce,
            R_mf_user=R_mf_user,
        )

    def for_generation(self, i: int) -> tuple[float, float, float, np.ndarray | None]:
        """Return ``(assort1_i, assort2_i, rho_w_i, R_mf_i)`` for loop iteration ``i``.

        ``rho_w`` is indexed by the *parental* C/E generation: founders (``i=0``)
        carry gen-0 C/E, and offspring from iteration ``j`` carry ``per_gen[j]``
        C/E — hence ``parent_ce_gen = max(0, i - 1)``. ``R_mf_i`` is auto-computed
        from ``rho_w_i`` when both-trait AM is active and no explicit matrix was
        supplied, else it is the user matrix (possibly ``None``).
        """
        rho_w_i = self.rho_w_per_ce[parent_ce_gen(i)]

        # Per-iter AM values (constant across iters for scalar assort1/assort2).
        a1_i = self.assort1_per_gen[i]
        a2_i = self.assort2_per_gen[i]

        # Auto-compute R_mf for this generation's rho_w if not user-provided.
        if self.R_mf_user is None and a1_i != 0 and a2_i != 0:
            R_mf_i = _cross_am_matrix(a1_i, a2_i, rho_w_i)
        else:
            R_mf_i = self.R_mf_user

        return a1_i, a2_i, rho_w_i, R_mf_i
