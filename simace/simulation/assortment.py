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

import numpy as np


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
        assort1: float | dict[int, float],
        assort2: float | dict[int, float],
        R_mf_user: np.ndarray | None,
        rA: float,
        rC: float,
        rE: float,
        A1: float,
        A2: float,
        C1_per_gen: list[float],
        C2_per_gen: list[float],
        E1_per_gen: list[float],
        E2_per_gen: list[float],
        G_sim: int,
    ) -> AssortmentPlan:
        """Resolve per-generation assortment and validate rho_w + PSD.

        Args:
            assort1: Trait-1 mate correlation; scalar or per-gen dict.
            assort2: Trait-2 mate correlation; scalar or per-gen dict.
            R_mf_user: Explicit 2x2 mate-correlation matrix, or ``None`` to
                auto-compute the cross-AM off-diagonal from ``rho_w``.
            rA: Cross-trait additive-genetic correlation.
            rC: Cross-trait shared-environment correlation.
            rE: Cross-trait unique-environment correlation.
            A1: Trait-1 additive-genetic variance (constant across gens).
            A2: Trait-2 additive-genetic variance (constant across gens).
            C1_per_gen: Per-generation trait-1 shared-environment variance.
            C2_per_gen: Per-generation trait-2 shared-environment variance.
            E1_per_gen: Per-generation trait-1 unique-environment variance.
            E2_per_gen: Per-generation trait-2 unique-environment variance.
            G_sim: Total simulated generations.

        Raises:
            ValueError: if both-trait AM is requested with ``|rho_w| >= 1`` at
                any generation, or the full 4x4 ``Sigma_4`` is not PSD.
        """
        from simace.simulation.simulate import resolve_per_gen_param

        # assort1/assort2 may be scalar or per-gen dict (raw-iter keyed).
        # AM correlations can be negative so we pass allow_negative=True.
        assort1_per_gen = resolve_per_gen_param(assort1, G_sim, name="assort1", allow_negative=True)
        assort2_per_gen = resolve_per_gen_param(assort2, G_sim, name="assort2", allow_negative=True)

        # Within-person cross-trait liability correlation per C/E generation
        _rho_w_A = rA * np.sqrt(A1 * A2)
        rho_w_per_ce = [
            _rho_w_A + rC * np.sqrt(C1_per_gen[g] * C2_per_gen[g]) + rE * np.sqrt(E1_per_gen[g] * E2_per_gen[g])
            for g in range(G_sim)
        ]

        # Validate |rho_w| < 1 for all C/E generations where both-trait AM is on.
        # With per-gen AM the both-trait check is per-iteration.
        for g, rw in enumerate(rho_w_per_ce):
            if assort1_per_gen[g] != 0 and assort2_per_gen[g] != 0 and abs(rw) >= 1.0 - 1e-10:
                raise ValueError(
                    f"Both-trait assortative mating requires |rho_w| < 1 "
                    f"(got rho_w={rw:.4f} at C/E generation {g}). "
                    f"Traits are perfectly correlated; "
                    f"use single-trait assortment instead."
                )

        # Validate PSD of full 4x4 Sigma for each generation's rho_w + AM.
        for g, rw in enumerate(rho_w_per_ce):
            a1_g = assort1_per_gen[g]
            a2_g = assort2_per_gen[g]
            if R_mf_user is None and not (a1_g != 0 and a2_g != 0):
                continue  # PSD check is only meaningful when both-trait AM is active
            if R_mf_user is not None:
                R_mf_g = R_mf_user
            else:
                R_mf_g = _cross_am_matrix(a1_g, a2_g, rw)
            R_ff = np.array([[1.0, rw], [rw, 1.0]])
            Sigma_4 = np.block([[R_ff, R_mf_g.T], [R_mf_g, R_ff]])
            eigvals = np.linalg.eigvalsh(Sigma_4)
            if eigvals[0] < -1e-8:
                raise ValueError(
                    f"Full 4x4 mate correlation matrix Sigma_4 is not PSD "
                    f"(min eigenvalue = {eigvals[0]:.6f} at C/E generation {g}). "
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
        # rho_w for the current parental population:
        # founders (i=0) have gen-0 C/E; offspring from iter j have per_gen[j] C/E
        parent_ce_gen = max(0, i - 1)
        rho_w_i = self.rho_w_per_ce[parent_ce_gen]

        # Per-iter AM values (constant across iters for scalar assort1/assort2).
        a1_i = self.assort1_per_gen[i]
        a2_i = self.assort2_per_gen[i]

        # Auto-compute R_mf for this generation's rho_w if not user-provided.
        if self.R_mf_user is None and a1_i != 0 and a2_i != 0:
            R_mf_i = _cross_am_matrix(a1_i, a2_i, rho_w_i)
        else:
            R_mf_i = self.R_mf_user

        return a1_i, a2_i, rho_w_i, R_mf_i
