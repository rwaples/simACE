"""Theoretical expected mate liability correlation matrix.

Companion to the generative ``_assortative_pair_partners`` in
``simace.simulation.simulate``: given the same assortative-mating parameters
and ACE variance components passed to the simulator, this module returns the
expected ``R_mf`` matrix that the simulator should produce. Used by
validation and plotting code to compare observed vs. expected mate
correlations. Change one, change the other.
"""

from __future__ import annotations

__all__ = ["expected_mate_corr_matrix"]

import numpy as np


def expected_mate_corr_matrix(
    assort1: float,
    assort2: float,
    rA: float,
    rC: float,
    A1: float,
    C1: float,
    A2: float,
    C2: float,
    assort_matrix: np.ndarray | list | None = None,
    rE: float = 0.0,
    E1: float = 0.0,
    E2: float = 0.0,
) -> np.ndarray:
    """Compute the 2x2 expected mate liability correlation matrix.

    Returns E[corr(F_i, M_j)] for i,j in {1,2} given assortative mating
    parameters and ACE variance components.

    With the 4-variate copula algorithm, assort1 and assort2 are target
    Pearson mate correlations. The cross-mate cross-trait correlation follows
    from the mechanistic path: c = rho_w * sqrt(|r1*r2|) * sign(r1*r2),
    where rho_w is the within-person cross-trait liability correlation.

    When ``assort_matrix`` is provided, it is returned directly (the user
    has specified the full R_mf).

    Args:
        assort1: Target mate Pearson correlation for trait 1.
        assort2: Target mate Pearson correlation for trait 2.
        rA: Genetic correlation between traits.
        rC: Shared-environment correlation between traits.
        A1: Additive genetic variance for trait 1.
        C1: Shared-environment variance for trait 1.
        A2: Additive genetic variance for trait 2.
        C2: Shared-environment variance for trait 2.
        assort_matrix: If provided, returned directly as the full R_mf matrix.
        rE: Unique-environment correlation between traits.
        E1: Unique-environment variance for trait 1.
        E2: Unique-environment variance for trait 2.

    Returns:
        2x2 array of expected mate liability correlations ``E[corr(F_i, M_j)]``.
    """
    if assort_matrix is not None:
        return np.asarray(assort_matrix, dtype=np.float64)

    if assort1 == 0 and assort2 == 0:
        return np.zeros((2, 2))

    # Within-person cross-trait correlation
    rho_w = rA * np.sqrt(A1 * A2) + rC * np.sqrt(C1 * C2) + rE * np.sqrt(E1 * E2)

    if assort1 != 0 and assort2 != 0:
        # Both traits: diagonal = targets, off-diagonal from rho_w mediation
        c = rho_w * np.sqrt(abs(assort1 * assort2)) * np.sign(assort1 * assort2)
        return np.array([[assort1, c], [c, assort2]])
    if assort1 != 0:
        # Single-trait on trait 1: propagate via rho_w
        a = assort1
        return np.array(
            [
                [a, a * rho_w],
                [a * rho_w, a * rho_w**2],
            ]
        )
    # Single-trait on trait 2: propagate via rho_w
    a = assort2
    return np.array(
        [
            [a * rho_w**2, a * rho_w],
            [a * rho_w, a],
        ]
    )
