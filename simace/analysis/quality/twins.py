"""MZ-twin validation checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .check import FamilyResult, QualityCheck

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

    from simace.core.pedigree_arrays import PedigreeArrays
    from simace.simulation.params import SimulationParams


@dataclass(frozen=True)
class TwinMeasurements:
    """What the twin family measures.

    Attributes:
        twin_rate: The share of non-founders who are MZ twins; ``0.0`` with no
            twins at all, ``None`` when twins exist but no non-founder does.
        expected_twin_rate: Its expectation ``2q``; ``None`` exactly when
            ``twin_rate`` is.
    """

    twin_rate: float | None
    expected_twin_rate: float | None


def validate_twins(
    df: pd.DataFrame | pl.DataFrame, sim: SimulationParams, ped: PedigreeArrays
) -> FamilyResult[TwinMeasurements]:
    """Validate MZ twin properties for two-trait simulation.

    Checks bidirectional twin pointers, shared parents, identical A values
    and sex for MZ pairs, and that the observed twin-individual rate among
    non-founders matches ``2q``, ``q = p_mztwin / (1 + p_mztwin)``: each
    generation draws ``Binomial(N, q)`` MZ pairs (``place_twins``), so the
    non-founders' pairs are ``Binomial(n, q)``, scored within 4 SE.

    Under ``mating_model="wright_fisher"`` no MZ twins are produced by
    design (see ADR 0002), so ``expected_rate`` is ``0.0`` and any
    twin present in the pedigree is a violation that fails the rate
    check.  Structural checks still run when twins are present so a
    corrupted WF output is also flagged for malformed twin records.

    Args:
        df: Pedigree DataFrame.
        sim: The rep's parameters.
        ped: The same pedigree as id-addressable arrays.

    Returns:
        The checks and the twin rate. With no twin records only the rate is
        checked; with no non-founders it is not checked.
    """
    checks = []
    p_mztwin = sim.p_mztwin
    is_wf = sim.mating_model == "wright_fisher"
    pair_share = 0.0 if is_wf else float(p_mztwin) / (1 + float(p_mztwin))
    expected_rate = 2 * pair_share

    twin_all = df["twin"].to_numpy()
    id_all = df["id"].to_numpy()
    twin_mask = twin_all != -1
    n_twins = int(twin_mask.sum())

    if n_twins == 0:
        # No twin records, so the structural twin checks have nothing to test.
        # The rate is still tested: zero twins is the WF invariant, and under
        # standard mating it fails unless p_mztwin is small enough that zero
        # observed twins is plausible.
        rate_pass = True if is_wf else p_mztwin < 0.01
        rate_msg = (
            "No twins expected under mating_model=wright_fisher"
            if is_wf
            else f"No twins found, expected rate: {p_mztwin}"
        )
        rate = QualityCheck("twin_rate", rate_pass, rate_msg, observed=0.0, expected=expected_rate)
        return FamilyResult(TwinMeasurements(twin_rate=0.0, expected_twin_rate=expected_rate), (rate,))

    # Get unique twin pairs
    twin_ids = id_all[twin_mask]
    twin_partners = twin_all[twin_mask]
    mask = twin_ids < twin_partners
    t1_arr = twin_ids[mask]
    t2_arr = twin_partners[mask]
    n_pairs = len(t1_arr)

    # Bidirectional check. Tolerates a partner missing from the pedigree the
    # way the reindex it replaces did: absent means the check simply fails.
    partners_present = ped.contains(t2_arr)
    bidirectional = bool(partners_present.all()) and np.all(ped.gather("twin", t2_arr) == t1_arr)
    checks.append(
        QualityCheck(
            "twin_bidirectional",
            bool(bidirectional),
            f"All {n_twins} twin references are bidirectional: {bidirectional}",
        )
    )

    # Same parents
    t1_mother = ped.gather("mother", t1_arr)
    t2_mother = ped.gather("mother", t2_arr)
    t1_father = ped.gather("father", t1_arr)
    t2_father = ped.gather("father", t2_arr)
    same_parents = np.all((t1_mother == t2_mother) & (t1_father == t2_father))
    checks.append(
        QualityCheck("twin_same_parents", bool(same_parents), f"All {n_pairs} twin pairs share parents: {same_parents}")
    )

    # Same A values and same sex - loop over traits for A
    for t in [1, 2]:
        col = f"A{t}"
        v1 = ped.gather(col, t1_arr)
        v2 = ped.gather(col, t2_arr)
        same = np.allclose(v1, v2)
        checks.append(
            QualityCheck(f"twin_same_{col}", bool(same), f"All MZ twin pairs have identical {col} values: {same}")
        )

    # Same sex
    t1_sex = ped.gather("sex", t1_arr)
    t2_sex = ped.gather("sex", t2_arr)
    same_sex = np.all(t1_sex == t2_sex)
    checks.append(QualityCheck("twin_same_sex", bool(same_sex), f"All MZ twin pairs have same sex: {same_sex}"))

    # Twin rate (count only non-founder twin pairs; founders have twins but no parents in pedigree)
    nf_mask = df["mother"].to_numpy() != -1
    if nf_mask.any():
        n_nf = int(nf_mask.sum())
        nf_twin_mask = nf_mask & twin_mask
        nf_twin_ids = id_all[nf_twin_mask]
        nf_twin_partners = twin_all[nf_twin_mask]
        nf_pairs = int(np.sum(nf_twin_ids < nf_twin_partners))
        observed_rate = nf_pairs * 2 / n_nf
        if is_wf:
            # WF documents zero twins (ADR 0002).  Any presence is a
            # violation of the no-MZ-twins invariant.
            rate_ok = nf_pairs == 0
            rate_msg = (
                f"Twin rate under mating_model=wright_fisher: {observed_rate:.4f} "
                f"(expected 0.0); pedigree contains {nf_pairs} unexpected twin pair(s)."
            )
        else:
            # +1 pair keeps the bound from rejecting on a single pair at tiny n.
            pair_tol = 4 * np.sqrt(n_nf * pair_share * (1 - pair_share)) + 1
            rate_ok = abs(nf_pairs - n_nf * pair_share) <= pair_tol
            rate_msg = (
                f"Twin rate in non-founders: {observed_rate:.4f} (expected {expected_rate:.4f} "
                f"from p_mztwin={p_mztwin:.4f}; {nf_pairs} pairs, tol ±{pair_tol:.1f})"
            )
        checks.append(
            QualityCheck("twin_rate", rate_ok, rate_msg, observed=float(observed_rate), expected=expected_rate)
        )
        measurements = TwinMeasurements(twin_rate=float(observed_rate), expected_twin_rate=expected_rate)
    else:
        measurements = TwinMeasurements(twin_rate=None, expected_twin_rate=None)

    return FamilyResult(measurements, tuple(checks))
