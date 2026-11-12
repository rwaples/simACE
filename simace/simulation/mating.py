"""Mating for one generation: slot counts, partner pairing (random or assortative), offspring, and twins."""

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from simace.simulation.assortment import METROPOLIS_TOL, _cross_am_matrix
from simace.simulation.pairing_kernels import (
    _fast_rank,
    _find_duplicate_pairs,
    _kd_match,
    _metropolis_full,
    _metropolis_single,
)
from simace.simulation.pedigree_buffers import _liability

if TYPE_CHECKING:
    from typing import Any


logger = logging.getLogger(__name__)


def draw_mating_counts(rng: np.random.Generator, n: int, mating_lambda: float) -> np.ndarray:
    """Draw zero-truncated Poisson mating counts for *n* individuals.

    Args:
        rng: numpy random generator
        n: number of individuals
        mating_lambda: Poisson lambda for the ZTP distribution

    Returns:
        Array of shape ``(n,)`` with all values >= 1.
    """
    counts = rng.poisson(lam=mating_lambda, size=n)
    zeros = counts == 0
    while zeros.any():
        counts[zeros] = rng.poisson(lam=mating_lambda, size=zeros.sum())
        zeros = counts == 0
    return counts


def balance_mating_slots(
    rng: np.random.Generator, male_counts: np.ndarray, female_counts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Balance total mating slots between males and females.

    Takes ``T = min(sum(male), sum(female))`` and randomly trims the larger
    side so both sum to *T*.

    Returns:
        ``(balanced_male_counts, balanced_female_counts)``
    """
    m_total = int(male_counts.sum())
    f_total = int(female_counts.sum())
    T = min(m_total, f_total)

    def _trim(counts: np.ndarray, total: int, target: int) -> np.ndarray:
        if total == target:
            return counts.copy()
        # Expand to per-slot array, keep a random subset of size target
        idx = np.repeat(np.arange(len(counts)), counts)
        keep = rng.choice(len(idx), size=target, replace=False)
        keep.sort()
        kept_idx = idx[keep]
        return np.bincount(kept_idx, minlength=len(counts)).astype(counts.dtype)

    return _trim(male_counts, m_total, T), _trim(female_counts, f_total, T)


def pair_partners(
    rng: np.random.Generator,
    male_idxs: np.ndarray,
    male_counts: np.ndarray,
    female_idxs: np.ndarray,
    female_counts: np.ndarray,
) -> np.ndarray:
    """Create mating pairs via random bipartite matching.

    Expands each sex into slot arrays using ``np.repeat``, shuffles one side,
    and pairs positionally. Each duplicate ``(mother, father)`` pair swaps its
    father with a uniformly random non-duplicate row's, for at most 5 passes;
    a duplicate left after them is kept, next to its first row.

    Returns:
        ``(M, 2)`` array of ``[mother_idx, father_idx]``.
    """
    male_slots = np.repeat(male_idxs, male_counts)
    female_slots = np.repeat(female_idxs, female_counts)
    rng.shuffle(male_slots)

    matings = np.column_stack([female_slots, male_slots])

    # Deduplicate: swap conflicting entries (rare at low lambda)
    for _attempt in range(5):
        is_dup = _find_duplicate_pairs(matings)
        if not is_dup.any():
            return matings  # mothers in index order, each couple once
        dup_idxs = np.where(is_dup)[0]
        non_dup_idxs = np.where(~is_dup)[0]
        if len(non_dup_idxs) == 0:
            break
        for d in dup_idxs:
            swap_with = rng.choice(non_dup_idxs)
            matings[d, 1], matings[swap_with, 1] = matings[swap_with, 1], matings[d, 1]

    return _family_order(matings)


def _family_order(matings: np.ndarray) -> np.ndarray:
    """Sort ``[mother, father]`` rows by mother then father, so a repeated couple's rows are adjacent."""
    return matings[np.lexsort((matings[:, 1], matings[:, 0]))]


#: Smallest standard deviation a trait may have among one sex's mating slots in
#: two-trait assortment; below it the trait is treated as constant.
_MIN_TRAIT_SD = 1e-12


#: Largest |within-sex trait correlation| two-trait assortment accepts; above it
#: the two traits are treated as collinear and ``R_ff`` / ``R_mm`` as singular.
_MAX_TRAIT_CORR = 1.0 - 1e-6


#: Roundoff tolerance on the conditional covariance's smallest eigenvalue: within
#: it the eigenvalue is clipped to 0, beyond it the requested ``R_mf`` is infeasible.
_CONDITIONAL_EIG_TOL = 1e-8


def _standardized_traits(liab1: np.ndarray, liab2: np.ndarray, sex: str) -> np.ndarray:
    """Z-score both liabilities of one sex's slots; raise if either is constant."""
    columns = []
    for trait, liab in ((1, liab1), (2, liab2)):
        sd = liab.std()
        if not sd > _MIN_TRAIT_SD:
            raise ValueError(
                f"Two-trait assortative mating: {sex} trait-{trait} liability is constant across "
                f"mating slots (SD {sd:.3g}); the within-sex trait correlation is undefined. "
                f"Use single-trait assortment instead."
            )
        columns.append((liab - liab.mean()) / sd)
    return np.column_stack(columns)


def _realised_trait_corr(z: np.ndarray, sex: str) -> np.ndarray:
    """The realised 2x2 trait correlation of one sex's slots; raise if singular."""
    r = np.corrcoef(z, rowvar=False)[0, 1]
    if not abs(r) <= _MAX_TRAIT_CORR:
        raise ValueError(
            f"Two-trait assortative mating: the realised {sex} within-person trait correlation "
            f"is {r:.6f}; the traits are collinear across mating slots, so the conditional mate "
            f"draw is undefined. Use single-trait assortment instead."
        )
    return np.array([[1.0, r], [r, 1.0]])


def _assortative_pair_partners(
    rng: np.random.Generator,
    male_idxs: np.ndarray,
    male_counts: np.ndarray,
    female_idxs: np.ndarray,
    female_counts: np.ndarray,
    pheno: np.ndarray,
    assort1: float,
    assort2: float,
    rho_w: float = 0.0,
    assort_matrix: np.ndarray | None = None,
    pairing_log: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """Create mating pairs with assortative mating.

    Single-trait case (one of assort1/assort2 zero): bivariate Gaussian copula
    on the active trait, then, after dedup, Metropolis father swaps that never
    create a duplicate couple until the trait's slot-level Pearson correlation
    is within ``METROPOLIS_TOL`` of its target (ADR 0032; the copula alone
    falls short, issue #74).

    Both-nonzero case: 4-variate Gaussian copula on the mate correlation
    structure of Border et al. (2022, Science). Each female's ideal mate is
    drawn from the Gaussian conditional given her two standardized
    liabilities, males are matched to those draws with ``_kd_match``, and
    Metropolis greedy swaps remove the remaining sampling error in the four
    Pearson correlations. Mate liabilities are jointly normal, so thresholded
    mate concordance matches ``R_mf`` (issue #34: the previous moment
    matching hit the Pearson targets with heavy joint tails).

    Args:
        rng: numpy random generator
        male_idxs: indices of males in the parental population
        male_counts: balanced mating slot counts per male
        female_idxs: indices of females in the parental population
        female_counts: balanced mating slot counts per female
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2] for parents
        assort1: target mate correlation on trait 1 liability, in [-1, 1]
        assort2: target mate correlation on trait 2 liability, in [-1, 1]
        rho_w: configured within-person cross-trait liability correlation;
            sets the default cross-trait mate correlation when
            ``assort_matrix`` is None
        assort_matrix: optional full 2x2 mate correlation matrix R_mf.
            When provided, overrides the auto-computed off-diagonals.
        pairing_log: receives the pairing record: ``mode`` (``single`` or
            ``both``), ``M``, ``proposals``, ``max_proposals``, the final
            post-dedup slot-level ``errors`` (correlation minus target),
            ``duplicates_left`` and ``converged``.

    Returns:
        ``(M, 2)`` array of ``[mother_idx, father_idx]``, in mother order,
        with any repeated couple's rows adjacent.

    Raises:
        ValueError: in the both-nonzero case with two or more slots, if a
            trait is constant or the two traits are collinear within either
            sex (ADR 0024: use single-trait assortment), or if ``R_mf`` is
            infeasible for the realised within-sex trait correlations.
    """
    # Imported here, not at module level: scipy.spatial costs ~160 ms to import
    # and only assortative mating needs the tree.
    from scipy.spatial import cKDTree  # ty: ignore[unresolved-import]  # real class; scipy stubs omit it

    # 1. Expand slots
    female_slots = np.repeat(female_idxs, female_counts)
    male_slots = np.repeat(male_idxs, male_counts)
    M = len(female_slots)
    mode = "both" if assort1 != 0 and assort2 != 0 else "single"
    if M < 2:
        # Nothing to sort: zero or one possible pairing.
        matings = np.column_stack([female_slots, male_slots])
        if pairing_log is not None:
            pairing_log.append(_pairing_record(mode, M, 0, 0, {}, 0))
        return matings

    # 2. Compute liability per slot
    liab1_f = _liability(pheno, 1, female_slots)
    liab2_f = _liability(pheno, 2, female_slots)
    liab1_m = _liability(pheno, 1, male_slots)
    liab2_m = _liability(pheno, 2, male_slots)

    if mode == "both":
        # --- Both traits nonzero: 4-variate Gaussian copula on R_mf ---
        r1, r2 = assort1, assort2

        # Build full R_mf with cross-trait off-diagonals
        if assort_matrix is not None:
            R_mf = np.asarray(assort_matrix, dtype=np.float64)
        else:
            R_mf = _cross_am_matrix(r1, r2, rho_w)
        c_target = R_mf[0, 1]

        fz = _standardized_traits(liab1_f, liab2_f, "female")
        males_z = _standardized_traits(liab1_m, liab2_m, "male")

        # Draw each female's ideal mate from the Gaussian conditional
        # m | f ~ N(f R_ff^-1 R_mf, R_mm - R_mf' R_ff^-1 R_mf), then match
        # males to those draws. R_ff and R_mm are the realised within-sex
        # correlations, not rho_w: cross-trait AM moves the population's
        # trait correlation away from the configured value, and a mismatch
        # leaves the targets and the males with different distributions,
        # which bends the joint tails. Whitening gives the k-d split
        # independent axes.
        R_ff = _realised_trait_corr(fz, "female")
        R_mm = _realised_trait_corr(males_z, "male")
        B = np.linalg.solve(R_ff, R_mf)
        eigval, eigvec = np.linalg.eigh(R_mm - R_mf.T @ B)
        if eigval[0] < -_CONDITIONAL_EIG_TOL:
            raise ValueError(
                f"Requested mate correlation matrix R_mf={np.round(R_mf, 4).tolist()} is not attainable with the "
                f"realised within-sex trait correlations (female {R_ff[0, 1]:.4f}, male {R_mm[0, 1]:.4f}): "
                f"the conditional covariance R_mm - R_mf' R_ff^-1 R_mf has eigenvalue {eigval[0]:.3g} < 0. "
                f"Reduce the assortment or cross-trait mate correlations."
            )
        noise_factor = eigvec * np.sqrt(np.clip(eigval, 0.0, None))
        target = fz @ B + rng.standard_normal((M, 2)) @ noise_factor.T
        whiten = np.linalg.inv(np.linalg.cholesky(R_mm)).T
        male_perm = _kd_match(target @ whiten, males_z @ whiten)

        # Metropolis swaps remove the sampling error of the draw so the
        # Pearson correlations land on R_mf.
        mz = males_z[male_perm]

        T1 = r1 * M
        T2 = r2 * M
        T12 = c_target * M
        T21 = c_target * M
        S1 = float(fz[:, 0] @ mz[:, 0])
        S2 = float(fz[:, 1] @ mz[:, 1])
        S12 = float(fz[:, 0] @ mz[:, 1])
        S21 = float(fz[:, 1] @ mz[:, 0])

        max_proposals = 8 * M
        metro_seed = int(rng.integers(2**63))

        S1, S2, S12, S21, proposals_done = _metropolis_full(
            fz,
            mz,
            male_perm,
            S1,
            S2,
            S12,
            S21,
            T1,
            T2,
            T12,
            T21,
            M,
            METROPOLIS_TOL,
            max_proposals,
            metro_seed,
        )

        logger.info(
            "Metropolis: %d/%d proposals (%.1f%%), M=%d, err1=%.5f, err2=%.5f, err12=%.5f, err21=%.5f",
            proposals_done,
            max_proposals,
            100.0 * proposals_done / max_proposals,
            M,
            abs(S1 / M - r1),
            abs(S2 / M - r2),
            abs(S12 / M - c_target),
            abs(S21 / M - c_target),
        )
        matings = np.column_stack([female_slots, male_slots[male_perm]])

    else:
        # --- Single-trait: bivariate Gaussian copula ---
        rank1_f = _fast_rank(liab1_f)
        rank2_f = _fast_rank(liab2_f)
        rank1_m = _fast_rank(liab1_m)
        rank2_m = _fast_rank(liab2_m)

        if assort1 < 0:
            rank1_m = 1.0 - rank1_m
        if assort2 < 0:
            rank2_m = 1.0 - rank2_m

        score_f = abs(assort1) * rank1_f + abs(assort2) * rank2_f
        score_m = abs(assort1) * rank1_m + abs(assort2) * rank2_m

        order_f = np.argsort(score_f)
        order_m = np.argsort(score_m)
        male_sorted = male_slots[order_m]

        r_eff = min(np.sqrt(assort1**2 + assort2**2), 1.0)

        cov = [[1.0, r_eff], [r_eff, 1.0]]
        z = rng.multivariate_normal([0.0, 0.0], cov, size=M)

        rank_f = np.argsort(np.argsort(z[:, 0]))
        rank_m = np.argsort(np.argsort(z[:, 1]))

        # Row k's mother is female slot slot_of_row[k]; dedup and the swaps move only fathers.
        slot_of_row = order_f[rank_f]
        matings = np.column_stack([female_slots[slot_of_row], male_sorted[rank_m]])

    n_swaps, duplicates_left = _dedup_matings(matings, liab1_f, liab2_f, cKDTree)

    if mode == "both":
        proposals = proposals_done
        targets = {"r1": (0, 0, r1), "r2": (1, 1, r2), "r12": (0, 1, c_target), "r21": (1, 0, c_target)}
        if n_swaps == 0:
            # No dedup swap: the Metropolis sums are the final slot-level correlations.
            sums = {"r1": S1, "r2": S2, "r12": S12, "r21": S21}
            errors = {name: sums[name] / M - r for name, (_, _, r) in targets.items()}
        else:
            errors = _slot_errors(pheno, matings, targets)
    else:
        # The copula's correlation is random around the target and dedup moves it
        # further (issue #74); swaps that never create a duplicate couple pin it.
        t = 0 if assort1 != 0 else 1
        target_r = assort1 if assort1 != 0 else assort2
        max_proposals = 8 * M
        female_liab = liab1_f if t == 0 else liab2_f
        refined = _refine_single_trait(rng, pheno, matings, slot_of_row, female_liab, t, target_r, max_proposals)
        # A constant active trait has no correlation to pin: no refinement and no errors.
        proposals, errors = (0, {}) if refined is None else (refined[0], {f"r{t + 1}": refined[1] / M - target_r})
        if duplicates_left:  # the swaps never add a duplicate couple, but they can remove one
            duplicates_left = int(_find_duplicate_pairs(matings).sum())

    record = _pairing_record(
        mode, M, proposals, max_proposals, {k: float(v) for k, v in errors.items()}, duplicates_left
    )
    if not errors:
        logger.warning(
            "Assortative mating (single): trait %d's liability is constant across mating slots; nothing was pinned",
            t + 1,
        )
    elif not record["converged"]:
        logger.warning(
            "Assortative mating (%s) missed its target after %d proposals: %s", mode, proposals, record["errors"]
        )
    if pairing_log is not None:
        pairing_log.append(record)
    if mode == "single":
        # Back to female-slot order, i.e. mother order (row k holds slot slot_of_row[k])
        in_slot_order = np.empty_like(matings)
        in_slot_order[slot_of_row] = matings
        matings = in_slot_order
    return _family_order(matings) if record["duplicates_left"] else matings


def _refine_single_trait(
    rng: np.random.Generator,
    pheno: np.ndarray,
    matings: np.ndarray,
    slot_of_row: np.ndarray,
    female_liab: np.ndarray,
    t: int,
    target_r: float,
    max_proposals: int,
) -> tuple[int, float] | None:
    """Swap fathers in ``matings`` (in place) until trait ``t``'s slot-level correlation is within tolerance.

    ``slot_of_row[k]`` is the female slot of row ``k``'s mother, and
    ``female_liab`` trait ``t``'s liability per female slot. Female slots are
    in female order (``np.repeat`` over ascending ids), so inverting
    ``slot_of_row`` groups the rows by mother without a sort.

    Returns:
        The number of proposals made and the final sum S = M · r of the
        slot-level correlation, or None when trait ``t`` is constant on
        either side, which leaves the correlation undefined and the pairing
        as the copula and dedup left it.
    """
    M = len(matings)
    f_liab, m_liab = female_liab[slot_of_row], _liability(pheno, t + 1, matings[:, 1])
    if not (f_liab.std() > _MIN_TRAIT_SD and m_liab.std() > _MIN_TRAIT_SD):
        return None
    fz, mz = _zscore(f_liab), _zscore(m_liab)
    fathers = matings[:, 1].astype(np.int64)
    mothers = matings[:, 0].astype(np.int64)
    # Rows grouped by mother (CSR), for the duplicate-couple check.
    rows = np.empty(M, dtype=np.int64)
    rows[slot_of_row] = np.arange(M, dtype=np.int64)
    rows_ptr = np.zeros(int(mothers.max()) + 2, dtype=np.int64)
    np.cumsum(np.bincount(mothers, minlength=len(rows_ptr) - 1), out=rows_ptr[1:])
    seed = int(rng.integers(2**63))
    S, proposals = _metropolis_single(
        fz, mz, fathers, mothers, rows_ptr, rows, float(fz @ mz), target_r * M, M, METROPOLIS_TOL, max_proposals, seed
    )
    matings[:, 1] = fathers
    return int(proposals), float(S)


def _pairing_record(
    mode: str, M: int, proposals: int, max_proposals: int, errors: dict[str, float], duplicates_left: int
) -> dict[str, Any]:
    """One assortative iteration's pairing record (``run_simulation``'s ``mating_log``).

    ``converged`` needs at least one error, so a record with nothing pinned
    (M < 2, or a constant active trait) does not read as converged.
    """
    return {
        "mode": mode,
        "M": int(M),
        "proposals": int(proposals),
        "max_proposals": int(max_proposals),
        "errors": errors,
        "duplicates_left": int(duplicates_left),
        "converged": bool(errors) and all(abs(e) < METROPOLIS_TOL for e in errors.values()),
    }


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardize with ddof 0, so a mean of products is a Pearson correlation."""
    return (x - x.mean()) / x.std()


def _slot_errors(
    pheno: np.ndarray, matings: np.ndarray, targets: dict[str, tuple[int, int, float]]
) -> dict[str, float]:
    """Slot-level Pearson correlation minus target, per ``name: (mother trait, father trait, target)`` (0-based)."""
    # One row gather per side, then each trait's liability from its columns.
    sides = [pheno[matings[:, side]] for side in (0, 1)]
    z = {(side, t): _zscore(_liability(sides[side], t + 1)) for side in (0, 1) for t in (0, 1)}
    return {name: float(np.mean(z[0, i] * z[1, j]) - r) for name, (i, j, r) in targets.items()}


def _dedup_matings(matings: np.ndarray, liab1_f: np.ndarray, liab2_f: np.ndarray, cKDTree: Any) -> tuple[int, int]:
    """Swap the father of each duplicate couple with a nearby female's (in place), for at most 5 passes.

    Returns:
        The number of father swaps made and the number of duplicate couples left.

    ``liab1_f`` / ``liab2_f`` are the female slots' liabilities in slot order.
    The single-trait copula reorders the rows, so there the partner is not a
    near female. Aligning them is worse: a duplicate's nearest row is then
    often the same woman's other slot, and the swap changes nothing.
    """
    # Deduplicate: swap with female-proximity partner to preserve correlations
    t_dedup = time.perf_counter()
    n_swaps = 0
    clean = False
    for _attempt in range(5):
        is_dup = _find_duplicate_pairs(matings)
        if not is_dup.any():
            clean = True
            break
        dup_idxs = np.where(is_dup)[0]
        non_dup_idxs = np.where(~is_dup)[0]
        if len(non_dup_idxs) == 0:
            break
        n_swaps += len(dup_idxs)

        # KD-tree nearest-neighbor lookup: O(M log M) build + O(dups × log M) query
        nondup_liabs = np.column_stack([liab1_f[non_dup_idxs], liab2_f[non_dup_idxs]])
        tree = cKDTree(nondup_liabs)
        dup_liabs = np.column_stack([liab1_f[dup_idxs], liab2_f[dup_idxs]])
        _, nn_idx = tree.query(dup_liabs, k=1)

        for i, d in enumerate(dup_idxs):
            swap_with = non_dup_idxs[nn_idx[i]]
            matings[d, 1], matings[swap_with, 1] = matings[swap_with, 1], matings[d, 1]
    logger.debug(
        "Dedup: %d father swaps in %.1fs",
        n_swaps,
        time.perf_counter() - t_dedup,
    )
    return n_swaps, 0 if clean else int(_find_duplicate_pairs(matings).sum())


def allocate_offspring(rng: np.random.Generator, n_matings: int, N: int) -> np.ndarray:
    """Distribute *N* offspring across *n_matings* matings via multinomial.

    Returns:
        Array of shape ``(n_matings,)`` summing to exactly *N*.
    """
    probs = np.ones(n_matings) / n_matings
    return rng.multinomial(N, probs)


def place_twins(rng: np.random.Generator, n_matings: int, N: int, p_mztwin: float) -> tuple[np.ndarray, np.ndarray]:
    """Allocate *N* offspring as births, a fraction *p_mztwin* of them MZ twin births, over *n_matings* couples.

    Draws ``K ~ Binomial(N, p / (1 + p))`` MZ pairs (at most ``N // 2``), so
    ``B = N - K`` births with ``E[K / B] = p``. The births are spread over
    the couples by :func:`allocate_offspring`, and ``K`` of them, uniformly
    at random, are twin births. Each couple's children are laid out in birth
    order, so co-twins are adjacent, at any position in the family, and a
    couple can have several MZ pairs.

    Returns:
        ``(offspring_counts, first_twins)``: children per couple, summing to
        *N*, and the sorted offspring positions (into the couples'
        contiguous blocks, in mating order) of each pair's first co-twin;
        the second is the next position.
    """
    K = min(int(rng.binomial(N, p_mztwin / (1 + p_mztwin))), N // 2) if p_mztwin > 0 else 0
    births = allocate_offspring(rng, n_matings, N - K)
    twin_births = np.sort(rng.choice(N - K, size=K, replace=False, shuffle=False))
    couple = np.searchsorted(np.cumsum(births), twin_births, side="right")
    # Each earlier twin birth shifts a birth's first child one position on.
    return births + np.bincount(couple, minlength=n_matings), twin_births + np.arange(K)


def _mating_standard(
    rng: np.random.Generator,
    parental_sex: np.ndarray,
    mating_lambda: float,
    p_mztwin: float,
    pheno: np.ndarray | None = None,
    assort1: float = 0.0,
    assort2: float = 0.0,
    rho_w: float = 0.0,
    assort_matrix: np.ndarray | None = None,
    pairing_log: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate parent-offspring pairings via a modular mating pipeline.

    1. Separate males/females and draw ZTP(mating_lambda) mating counts.
    2. Balance slots so both sexes have equal total.
    3. Pair partners randomly (or assortatively if assort1/assort2 != 0).
    4. Allocate N offspring as births across matings, some of them MZ twin
       births (:func:`place_twins`).
    5. Build output arrays.

    Args:
        rng: numpy random generator
        parental_sex: array of sex values (0=female, 1=male) for parents
        mating_lambda: Poisson lambda for zero-truncated mating count distribution
        p_mztwin: MZ twin rate per birth (see :func:`place_twins`)
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2]; required when
            assort1 or assort2 is nonzero
        assort1: target mate correlation on trait 1 liability
        assort2: target mate correlation on trait 2 liability
        rho_w: within-person cross-trait liability correlation, used to
            derive off-diagonal entries of the mate correlation matrix
        assort_matrix: optional full 2x2 mate correlation matrix R_mf
        pairing_log: receives this iteration's pairing record (``mode``
            ``"random"``, ``"single"`` or ``"both"``; for assortment also
            the final slot-level errors and whether they are within
            ``METROPOLIS_TOL``)

    Returns:
        parent_idxs: (N, 2) array of [mother_idx, father_idx] for each offspring
        twins: (m, 2) array of [twin1_idx, twin2_idx] pairs for MZ twins
        household_ids: (N,) array mapping each offspring to a household index

    Raises:
        ValueError: if assort is nonzero but pheno is None
        ValueError: if parental_sex contains zero males or zero females.
    """
    if (assort1 != 0 or assort2 != 0) and pheno is None:
        raise ValueError("pheno must be provided when assort1 or assort2 is nonzero")
    N = len(parental_sex)
    male_idxs = np.where(parental_sex == 1)[0]
    female_idxs = np.where(parental_sex == 0)[0]
    if len(female_idxs) == 0:
        raise ValueError("Standard mating: parent population has 0 female individuals; cannot sample mothers.")
    if len(male_idxs) == 0:
        raise ValueError("Standard mating: parent population has 0 male individuals; cannot sample fathers.")

    # 1. Draw mating counts per individual
    male_counts = draw_mating_counts(rng, len(male_idxs), mating_lambda)
    female_counts = draw_mating_counts(rng, len(female_idxs), mating_lambda)

    # 2. Balance slots
    male_counts, female_counts = balance_mating_slots(rng, male_counts, female_counts)

    # 3. Pair partners -> (M, 2) of [mother_idx, father_idx], in mother order with
    # repeated couples adjacent. Each pairing function keeps that order itself:
    # a lexsort here cost 8-14% of an assortative mating call (issue #78).
    if assort1 != 0 or assort2 != 0:
        assert pheno is not None  # required whenever assortment is active (see docstring)
        matings = _assortative_pair_partners(
            rng,
            male_idxs,
            male_counts,
            female_idxs,
            female_counts,
            pheno,
            assort1,
            assort2,
            rho_w=rho_w,
            assort_matrix=assort_matrix,
            pairing_log=pairing_log,
        )
    else:
        matings = pair_partners(rng, male_idxs, male_counts, female_idxs, female_counts)
        if pairing_log is not None:
            pairing_log.append({"mode": "random", "M": len(matings)})
    M = len(matings)

    # 4. Allocate offspring as births, some of them MZ twin births
    offspring_counts, first_twins = place_twins(rng, M, N, p_mztwin)
    twins = np.column_stack([first_twins, first_twins + 1])

    # 5. Build output arrays: each couple's children form one block, in birth order
    parent_idxs = np.repeat(matings, offspring_counts, axis=0)

    # Household: all offspring of the same mother share a household
    _, household_ids = np.unique(parent_idxs[:, 0], return_inverse=True)

    return parent_idxs, twins, household_ids


def _mating_wf(
    rng: np.random.Generator,
    parental_sex: np.ndarray,
    N: int,
    generation: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sex-structured idealized Wright-Fisher mating for one generation.

    For each of ``N`` offspring, draw one mother uniformly at random from the
    prior generation's females and one father uniformly at random from the
    males (both with replacement; selfing is impossible because parents are
    constrained to opposite sexes).  Multinomial offspring counts per parent;
    no persistent mating-pair structure; no MZ twins.

    Args:
        rng: numpy random generator.
        parental_sex: array of sex values (0=female, 1=male) for parents.
        N: number of offspring to produce.
        generation: simulation iteration index (used only for error messages
            so an empty-pool failure names the offending generation).

    Returns:
        parent_idxs: ``(N, 2)`` array of ``[mother_idx, father_idx]``.
        twins: ``(0, 2)`` empty int array (WF has no MZ twins).
        household_ids: ``(N,)`` contiguous ``[0, n_mothers)`` mapping each
            offspring to a maternal household, matching the standard path's
            grouping convention.

    Raises:
        ValueError: if ``parental_sex`` contains zero males or zero females.
    """
    male_idxs = np.where(parental_sex == 1)[0]
    female_idxs = np.where(parental_sex == 0)[0]
    if len(female_idxs) == 0:
        raise ValueError(f"WF mating: parent generation {generation} has 0 female individuals; cannot sample mothers.")
    if len(male_idxs) == 0:
        raise ValueError(f"WF mating: parent generation {generation} has 0 male individuals; cannot sample fathers.")

    mother_idx = rng.choice(female_idxs, size=N, replace=True)
    father_idx = rng.choice(male_idxs, size=N, replace=True)
    parent_idxs = np.column_stack([mother_idx, father_idx])

    twins = np.empty((0, 2), dtype=int)
    _, household_ids = np.unique(mother_idx, return_inverse=True)

    return parent_idxs, twins, household_ids
