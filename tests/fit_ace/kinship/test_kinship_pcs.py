"""Unit tests for fit_ace.kinship.kinship_pcs.compute_kinship_pcs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fit_ace.kinship.kinship import build_sparse_kinship
from fit_ace.kinship.kinship_pcs import compute_kinship_pcs
from sim_ace.simulation.simulate import run_simulation

_BASE_SIM = {
    "mating_lambda": 0.5,
    "p_mztwin": 0.0,
    "A1": 0.5,
    "C1": 0.0,
    "A2": 0.0,
    "C2": 0.0,
    "rA": 0.0,
    "rC": 0.0,
}


def _small_pedigree(seed: int = 1, N: int = 50, G_ped: int = 3) -> pd.DataFrame:
    return run_simulation(seed=seed, N=N, G_ped=G_ped, **_BASE_SIM)


def _kinship_from_ped(ped: pd.DataFrame, min_kinship: float = 1e-10):
    """Helper: build 2φ and return (matrix, ids) ordered like *ped*."""
    K = build_sparse_kinship(
        ped["id"].to_numpy(),
        ped["mother"].to_numpy(),
        ped["father"].to_numpy(),
        ped["twin"].to_numpy(),
        min_kinship=min_kinship,
    )
    return K, ped["id"].to_numpy()


def test_shape_and_dtype():
    ped = _small_pedigree(seed=1, N=50, G_ped=3)
    K, ids_in = _kinship_from_ped(ped)
    ids, pcs, eigs, trace = compute_kinship_pcs(K, ids_in, n_pcs=5, seed=42)
    assert ids.shape == (len(ped),)
    assert ids.dtype == np.int32
    assert pcs.shape == (len(ped), 5)
    assert pcs.dtype == np.float32
    assert eigs.shape == (5,)
    assert eigs.dtype == np.float64
    assert np.isfinite(trace)
    assert trace > 0


def test_eigenvalues_descending_and_nonneg():
    ped = _small_pedigree(seed=2, N=60, G_ped=3)
    K, ids_in = _kinship_from_ped(ped)
    _, _, eigs, _ = compute_kinship_pcs(K, ids_in, n_pcs=10, seed=0)
    finite = eigs[~np.isnan(eigs)]
    assert np.all(np.diff(finite) <= 1e-8)
    assert np.all(finite >= -1e-8)


def test_separates_connected_components():
    """Two disjoint pedigrees should be separable by a top PC."""
    pedA = _small_pedigree(seed=10, N=40, G_ped=2)
    pedB = _small_pedigree(seed=11, N=40, G_ped=2)

    offset = int(pedA["id"].max()) + 1
    pedB = pedB.copy()
    for col in ("id", "mother", "father", "twin"):
        mask = pedB[col].to_numpy() >= 0
        pedB.loc[mask, col] = pedB.loc[mask, col] + offset
    pedB["household_id"] = pedB["household_id"] + int(pedA["household_id"].max()) + 1

    combined = pd.concat([pedA, pedB], ignore_index=True)
    combined["_comp"] = np.concatenate([np.zeros(len(pedA)), np.ones(len(pedB))]).astype(int)

    K, ids_in = _kinship_from_ped(combined)
    _, pcs, _, _ = compute_kinship_pcs(K, ids_in, n_pcs=5, seed=7)
    best_sep = 0.0
    for j in range(5):
        mA = pcs[combined["_comp"].to_numpy() == 0, j].mean()
        mB = pcs[combined["_comp"].to_numpy() == 1, j].mean()
        s = np.nanstd(pcs[:, j])
        if s > 0:
            best_sep = max(best_sep, abs(mA - mB) / s)
    assert best_sep > 0.5, f"Expected a PC to separate components, got best_sep={best_sep}"


def test_deterministic():
    ped = _small_pedigree(seed=3, N=40, G_ped=3)
    K, ids_in = _kinship_from_ped(ped)
    _, pcs1, eigs1, _ = compute_kinship_pcs(K, ids_in, n_pcs=6, seed=123)
    _, pcs2, eigs2, _ = compute_kinship_pcs(K, ids_in, n_pcs=6, seed=123)
    np.testing.assert_array_equal(pcs1, pcs2)
    np.testing.assert_array_equal(eigs1, eigs2)


def test_small_N_pads_nan():
    """When N is too small for n_pcs, trailing PC columns must be NaN."""
    ped = _small_pedigree(seed=4, N=5, G_ped=2)
    n = len(ped)
    K, ids_in = _kinship_from_ped(ped)
    _, pcs, eigs, trace = compute_kinship_pcs(K, ids_in, n_pcs=30, seed=5)

    assert pcs.shape == (n, 30)
    assert eigs.shape == (30,)

    k_expected = min(30, n - 2)
    assert np.all(np.isfinite(pcs[:, :k_expected]))
    assert np.all(np.isnan(pcs[:, k_expected:]))
    assert np.all(np.isfinite(eigs[:k_expected]))
    assert np.all(np.isnan(eigs[k_expected:]))
    assert np.isfinite(trace)
    assert trace > 0


def test_trace_matches_diagonal():
    """trace equals sum of diag(matrix)."""
    ped = _small_pedigree(seed=6, N=40, G_ped=3)
    K, ids_in = _kinship_from_ped(ped)
    _, _, _, trace = compute_kinship_pcs(K, ids_in, n_pcs=5, seed=0)
    assert trace == pytest.approx(float(K.diagonal().sum()))


def test_threshold_raises_eigenvalue_error_monotonically():
    """Higher min_kinship → sparser matrix → more eigenvalue drift from baseline."""
    ped = _small_pedigree(seed=7, N=80, G_ped=3)
    K_full, ids_in = _kinship_from_ped(ped, min_kinship=1e-10)
    _, _, e_full, _ = compute_kinship_pcs(K_full, ids_in, n_pcs=10, seed=0)

    errs = []
    for thr in (1e-6, 1e-3, 1e-2):
        K_th, _ = _kinship_from_ped(ped, min_kinship=thr)
        _, _, e_th, _ = compute_kinship_pcs(K_th, ids_in, n_pcs=10, seed=0)
        errs.append(float(np.max(np.abs(e_full - e_th))))
    # Error is non-decreasing as threshold grows (allow ties)
    assert errs[0] <= errs[1] + 1e-10
    assert errs[1] <= errs[2] + 1e-10
