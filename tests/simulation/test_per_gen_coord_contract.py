"""Regression test for the per-generation E coordinate contract.

simACE's `resolve_per_gen_param` indexes E1/E2/C1/C2 dicts by *raw
simulation iteration* (0..G_sim-1), while the phenotyped slice of the
output pedigree carries *output-generation* labels (rebased to
0..G_ped-1 by subtracting burn-in). This means a schedule written as
``{0: 0.5, 6: 1.0}`` is interpreted in raw-iter coords with forward-fill
— so under G_sim=20 / G_ped=8, every recorded generation gets E=1.0
(because forward-fill from raw iter 6 onward dominates the 12 burn-in +
8 recorded iterations).

This test locks in the current behaviour as a contract. If simACE is
later refactored so E/C dicts accept output-gen-keyed schedules, the
assertions here will flip (or the test will be migrated to test the
new behaviour).

See `notes/heritability/epimight_h2_temporal.md` §"Generation-coordinate
contract" for the design discussion.
"""

from __future__ import annotations

import pytest

from simace.simulation.simulate import run_simulation


@pytest.mark.regression
def test_per_gen_E_uses_raw_iter_keys():
    """E1 dict `{0: 0.5, 6: 1.0}` produces var(E1)≈1.0 across all recorded gens.

    With G_sim=10, G_ped=5, burnin=5. The recorded generations correspond
    to raw iterations 5..9. The dict {0: 0.5, 6: 1.0} forward-fills:
      * raw iters 0..5: E1=0.5
      * raw iters 6..9: E1=1.0
    So the *output gens* covering raw iters 5..9 will have mixed values:
    gen 0 (raw iter 5) → E=0.5; gens 1..4 (raw iters 6..9) → E=1.0.
    This documents the dual coord system and lets us validate that
    schedules written for the *recorded* range need careful key choices.
    """
    pedigree = run_simulation(
        N=200,
        G_ped=5,
        G_sim=10,
        seed=42,
        # Constant A, varying E via raw-iter keyed dict.
        A1=0.5,
        A2=0.5,
        rA=0.0,
        C1=0.0,
        C2=0.0,
        rC=0.0,
        rE=0.0,
        E1={0: 0.5, 6: 1.0},
        E2={0: 0.5, 6: 1.0},
        mating_lambda=0.5,
        p_mztwin=0.0,
        assort1=0.0,
        assort2=0.0,
    )

    e1_by_gen = pedigree.groupby("generation")["E1"].var()

    # Output gen 0 corresponds to raw iter 5 (still E=0.5)
    # Output gens 1..4 correspond to raw iters 6..9 (E=1.0)
    assert e1_by_gen[0] == pytest.approx(0.5, rel=0.2), (
        f"output gen 0 (raw iter 5) should have var(E1)≈0.5, got {e1_by_gen[0]:.3f}"
    )
    for g in (1, 2, 3, 4):
        assert e1_by_gen[g] == pytest.approx(1.0, rel=0.2), (
            f"output gen {g} should have var(E1)≈1.0 from forward-fill, got {e1_by_gen[g]:.3f}"
        )


@pytest.mark.regression
def test_per_gen_assort1_dict_accepted():
    """assort1 as a per-gen dict should be accepted (not rejected as invalid scalar).

    Tests the resolver accepts dicts for assort1/assort2 and threads per-iter
    values into the mating loop. Smoke-only — does not assert on observed
    spousal correlation, which is noisy at small N.
    """
    pedigree = run_simulation(
        N=200,
        G_ped=4,
        G_sim=8,
        seed=99,
        A1=0.5,
        A2=0.5,
        rA=0.0,
        C1=0.0,
        C2=0.0,
        rC=0.0,
        rE=0.0,
        E1=0.5,
        E2=0.5,
        mating_lambda=0.5,
        p_mztwin=0.0,
        # Raw-iter keyed AM schedule. Forward-filled: iters 0..3 → 0, 4..7 → 0.4.
        assort1={0: 0.0, 4: 0.4},
        assort2={0: 0.0, 4: 0.4},
    )
    assert len(pedigree) == 200 * 4, "expected N * G_ped rows"
    assert set(pedigree["generation"].unique()) == {0, 1, 2, 3}


@pytest.mark.regression
def test_assort_matrix_rejects_per_gen_assort_dict():
    """assort_matrix is incompatible with per-gen assort dicts (v1 restriction)."""
    with pytest.raises(ValueError, match="assort_matrix is incompatible"):
        run_simulation(
            N=100,
            G_ped=3,
            G_sim=4,
            seed=11,
            A1=0.5,
            A2=0.5,
            rA=0.0,
            C1=0.0,
            C2=0.0,
            rC=0.0,
            rE=0.0,
            E1=0.5,
            E2=0.5,
            mating_lambda=0.5,
            p_mztwin=0.0,
            assort1={0: 0.0, 2: 0.3},
            assort2=0.3,
            assort_matrix=[[0.3, 0.1], [0.1, 0.3]],
        )


@pytest.mark.regression
def test_per_gen_E_with_dense_raw_iter_schedule():
    """Dense raw-iter schedule lets us target specific output generations.

    Under G_sim=10, G_ped=5, burnin=5: output gens 0..4 map to raw iters 5..9.
    A dense schedule ``{0: 0.5, 5: 0.5, 6: 0.7, 7: 0.9, 8: 1.0, 9: 1.0}`` produces
    a per-output-gen rising E variance.
    """
    pedigree = run_simulation(
        N=200,
        G_ped=5,
        G_sim=10,
        seed=43,
        A1=0.5,
        A2=0.5,
        rA=0.0,
        C1=0.0,
        C2=0.0,
        rC=0.0,
        rE=0.0,
        E1={0: 0.5, 5: 0.5, 6: 0.7, 7: 0.9, 8: 1.0, 9: 1.0},
        E2={0: 0.5, 5: 0.5, 6: 0.7, 7: 0.9, 8: 1.0, 9: 1.0},
        mating_lambda=0.5,
        p_mztwin=0.0,
        assort1=0.0,
        assort2=0.0,
    )
    e1_by_gen = pedigree.groupby("generation")["E1"].var()
    # Expected: output gen 0 = raw 5 → 0.5; gen 1 = raw 6 → 0.7; gen 2 = raw 7 → 0.9;
    # gens 3, 4 = raw 8, 9 → 1.0.
    expected = {0: 0.5, 1: 0.7, 2: 0.9, 3: 1.0, 4: 1.0}
    for g, target in expected.items():
        assert e1_by_gen[g] == pytest.approx(target, rel=0.25), (
            f"output gen {g}: realized var(E1)={e1_by_gen[g]:.3f} vs target {target}"
        )
