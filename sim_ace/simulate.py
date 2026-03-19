"""
ACE Pedigree Simulation

Simulates multi-generational pedigrees with:
- A: Additive genetic component
- C: Common/shared environment component
- E: Unique environment component

Supports single-trait and two-trait (bivariate) modes with configurable
cross-trait correlations for genetic (rA) and common environment (rC) components.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import yaml

from sim_ace.utils import save_parquet

logger = logging.getLogger(__name__)


def generate_correlated_components(
    rng: np.random.Generator, n: int, sd1: float, sd2: float, correlation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two correlated normal variables via multivariate normal.

    Args:
        rng: numpy random generator
        n: number of samples
        sd1: standard deviation for component 1
        sd2: standard deviation for component 2
        correlation: correlation between components

    Returns:
        (comp1, comp2): tuple of arrays, each shape (n,)

    Raises:
        ValueError: if sd1 or sd2 is negative, or correlation is outside [-1, 1]
    """
    if sd1 < 0 or sd2 < 0:
        raise ValueError(f"Standard deviations must be non-negative, got sd1={sd1}, sd2={sd2}")
    if not (-1 <= correlation <= 1):
        raise ValueError(f"Correlation must be in [-1, 1], got {correlation}")

    cov = [
        [sd1**2, correlation * sd1 * sd2],
        [correlation * sd1 * sd2, sd2**2],
    ]
    samples = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
    return samples[:, 0], samples[:, 1]


def generate_mendelian_noise(
    rng: np.random.Generator, n: int, sd_A1: float, sd_A2: float, rA: float
) -> tuple[np.ndarray, np.ndarray]:
    """Generate correlated Mendelian sampling noise for two traits.

    Under the infinitesimal model, the Mendelian sampling variance is
    0.5 * Var(A) for each trait, so sd_noise = sd_A / sqrt(2)
    (Bulmer, 1971, Am. Nat., 105, 201-211).

    Args:
        rng: numpy random generator
        n: number of offspring
        sd_A1: standard deviation of A1 (sqrt of A1 variance)
        sd_A2: standard deviation of A2 (sqrt of A2 variance)
        rA: genetic correlation between traits

    Returns:
        (noise1, noise2): tuple of arrays, each shape (n,)
    """
    sd_noise1 = sd_A1 / np.sqrt(2)
    sd_noise2 = sd_A2 / np.sqrt(2)
    return generate_correlated_components(rng, n, sd_noise1, sd_noise2, rA)


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
    and pairs positionally. Duplicate ``(mother, father)`` pairs are resolved
    by swapping conflicting entries.

    Returns:
        ``(M, 2)`` array of ``[mother_idx, father_idx]``.
    """
    male_slots = np.repeat(male_idxs, male_counts)
    female_slots = np.repeat(female_idxs, female_counts)
    rng.shuffle(male_slots)

    matings = np.column_stack([female_slots, male_slots])

    # Deduplicate: swap conflicting entries (rare at low lambda)
    for _attempt in range(5):
        seen = set()
        dups = []
        for i in range(len(matings)):
            key = (matings[i, 0], matings[i, 1])
            if key in seen:
                dups.append(i)
            else:
                seen.add(key)
        if not dups:
            break
        # Swap father between duplicate and a random non-duplicate
        non_dups = np.setdiff1d(np.arange(len(matings)), dups)
        if len(non_dups) == 0:
            break
        for d in dups:
            swap_with = rng.choice(non_dups)
            matings[d, 1], matings[swap_with, 1] = matings[swap_with, 1], matings[d, 1]

    return matings


def allocate_offspring(rng: np.random.Generator, n_matings: int, N: int) -> np.ndarray:
    """Distribute *N* offspring across *n_matings* matings via multinomial.

    Returns:
        Array of shape ``(n_matings,)`` summing to exactly *N*.
    """
    probs = np.ones(n_matings) / n_matings
    return rng.multinomial(N, probs)


def assign_twins(rng: np.random.Generator, offspring_counts: np.ndarray, p_mztwin: float) -> np.ndarray:
    """Decide which matings produce an MZ twin pair.

    Only matings with >= 2 offspring are eligible. At most one twin pair
    per mating.

    Returns:
        Boolean mask of shape ``(M,)`` — True where a twin pair occurs.
    """
    mask = np.zeros(len(offspring_counts), dtype=bool)
    eligible = offspring_counts >= 2
    if eligible.any():
        rolls = rng.uniform(size=eligible.sum()) < p_mztwin
        mask[eligible] = rolls
    return mask


def mating(
    rng: np.random.Generator, parental_sex: np.ndarray, mating_lambda: float, p_mztwin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate parent-offspring pairings via a modular mating pipeline.

    1. Separate males/females and draw ZTP(mating_lambda) mating counts.
    2. Balance slots so both sexes have equal total.
    3. Pair partners randomly.
    4. Allocate N offspring across matings via multinomial.
    5. Assign MZ twins to eligible matings.
    6. Build output arrays.

    Args:
        rng: numpy random generator
        parental_sex: array of sex values (0=female, 1=male) for parents
        mating_lambda: Poisson lambda for zero-truncated mating count distribution
        p_mztwin: probability of a mating producing MZ twins (if >= 2 offspring)

    Returns:
        parent_idxs: (N, 2) array of [mother_idx, father_idx] for each offspring
        twins: (m, 2) array of [twin1_idx, twin2_idx] pairs for MZ twins
        household_ids: (N,) array mapping each offspring to a household index
    """
    N = len(parental_sex)
    male_idxs = np.where(parental_sex == 1)[0]
    female_idxs = np.where(parental_sex == 0)[0]

    # 1. Draw mating counts per individual
    male_counts = draw_mating_counts(rng, len(male_idxs), mating_lambda)
    female_counts = draw_mating_counts(rng, len(female_idxs), mating_lambda)

    # 2. Balance slots
    male_counts, female_counts = balance_mating_slots(rng, male_counts, female_counts)

    # 3. Pair partners -> (M, 2) of [mother_idx, father_idx]
    matings = pair_partners(rng, male_idxs, male_counts, female_idxs, female_counts)
    M = len(matings)

    # 4. Allocate offspring
    offspring_counts = allocate_offspring(rng, M, N)

    # 5. Assign twins
    twin_mask = assign_twins(rng, offspring_counts, p_mztwin)

    # 6. Build output arrays
    parent_idxs = np.repeat(matings, offspring_counts, axis=0)

    # Household: all offspring of the same mother share a household
    _, household_ids = np.unique(parent_idxs[:, 0], return_inverse=True)

    # Twin pairs: first two offspring of each twin-flagged mating
    starts = np.zeros(M + 1, dtype=int)
    np.cumsum(offspring_counts, out=starts[1:])
    twin_indices = np.where(twin_mask)[0]
    if len(twin_indices) > 0:
        t1 = starts[twin_indices]
        t2 = t1 + 1
        twins = np.column_stack([t1, t2])
    else:
        twins = np.array([], dtype=int).reshape(0, 2)

    return parent_idxs, twins, household_ids


def reproduce(
    rng: np.random.Generator,
    pheno: np.ndarray,
    parents: np.ndarray,
    twins: np.ndarray,
    household_ids: np.ndarray,
    sd_A1: float,
    sd_E1: float,
    sd_C1: float,
    sd_A2: float,
    sd_E2: float,
    sd_C2: float,
    rA: float,
    rC: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate offspring phenotypes from parents for two correlated traits.

    Additive genetic values are inherited as midparent + Mendelian noise.
    Common environment (C) is drawn freshly per household — it is NOT
    inherited from parents but represents the offspring's own shared rearing
    environment (siblings share C; parents and children do not). Unique
    environment (E) is drawn independently per individual.

    Args:
        rng: numpy random generator
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2] for parents
        parents: (n, 2) array of [mother_idx, father_idx]
        twins: array of MZ twin index pairs
        household_ids: (n,) array mapping each offspring to a household
        sd_A1: standard deviation of A for trait 1
        sd_E1: standard deviation of E for trait 1
        sd_C1: standard deviation of C for trait 1
        sd_A2: standard deviation of A for trait 2
        sd_E2: standard deviation of E for trait 2
        sd_C2: standard deviation of C for trait 2
        rA: genetic correlation between traits
        rC: common environment correlation between traits

    Returns:
        offspring: (n, 6) array of [A1, C1, E1, A2, C2, E2]
        sex_offspring: (n,) array of sex values (0=female, 1=male)
    """
    n = len(parents)

    # Sex assignment
    sex_offspring = rng.binomial(size=n, n=1, p=0.5)

    # Additive genetic: midparent + correlated Mendelian noise
    mp1 = pheno[parents, 0].mean(axis=1)  # A1 midparent
    mp2 = pheno[parents, 3].mean(axis=1)  # A2 midparent

    noise1, noise2 = generate_mendelian_noise(rng, n, sd_A1, sd_A2, rA)
    a1_offspring = mp1 + noise1
    a2_offspring = mp2 + noise2

    # Common environment: freshly drawn per household each generation.
    # C is NOT inherited from parents -- it reflects the offspring's own
    # shared rearing environment. Siblings share C; parents and children do not.
    # This is the standard ACE model assumption (no autoregressive C transmission).
    unique_hh, hh_indices = np.unique(household_ids, return_inverse=True)
    n_hh = len(unique_hh)
    hh_c1, hh_c2 = generate_correlated_components(rng, n_hh, sd_C1, sd_C2, rC)
    c1_offspring = hh_c1[hh_indices]
    c2_offspring = hh_c2[hh_indices]

    # Unique environment: independent draws for each trait
    e1_offspring = rng.normal(size=n, loc=0, scale=sd_E1)
    e2_offspring = rng.normal(size=n, loc=0, scale=sd_E2)

    # MZ twins share A values and sex for both traits
    if len(twins) > 0:
        a1_offspring[twins[:, 1]] = a1_offspring[twins[:, 0]]
        a2_offspring[twins[:, 1]] = a2_offspring[twins[:, 0]]
        sex_offspring[twins[:, 1]] = sex_offspring[twins[:, 0]]

    offspring = np.stack(
        [
            a1_offspring,
            c1_offspring,
            e1_offspring,
            a2_offspring,
            c2_offspring,
            e2_offspring,
        ],
        axis=-1,
    )

    return offspring, sex_offspring


def add_to_pedigree(
    pheno: np.ndarray,
    sex: np.ndarray,
    parents: np.ndarray,
    twins: np.ndarray,
    household_ids: np.ndarray,
    generation: int,
    pedigree: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add a generation to the pedigree DataFrame.

    Args:
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2]
        sex: (n,) array of sex values
        parents: (n, 2) array of [mother_idx, father_idx]
        twins: array of MZ twin index pairs
        household_ids: (n,) array mapping each offspring to a household index
        generation: generation number (0 for founders)
        pedigree: existing pedigree DataFrame or None for first generation

    Returns:
        Updated pedigree DataFrame with the new generation appended.
    """
    n = len(pheno)
    twin_col = np.full(n, -1, dtype=int)
    if len(twins) > 0:
        twin_col[twins[:, 0]] = twins[:, 1]
        twin_col[twins[:, 1]] = twins[:, 0]

    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "sex": sex,
            "mother": parents[:, 0],
            "father": parents[:, 1],
            "twin": twin_col,
            "generation": generation,
            "household_id": household_ids,
            "A1": pheno[:, 0],
            "C1": pheno[:, 1],
            "E1": pheno[:, 2],
            "liability1": pheno[:, 0] + pheno[:, 1] + pheno[:, 2],
            "A2": pheno[:, 3],
            "C2": pheno[:, 4],
            "E2": pheno[:, 5],
            "liability2": pheno[:, 3] + pheno[:, 4] + pheno[:, 5],
        }
    )

    if pedigree is not None:
        offset_id = len(pedigree)
        offset_parent = offset_id - n
        offset_household = pedigree["household_id"].iloc[-1] + 1
        df["id"] = df["id"] + offset_id
        df["mother"] = df["mother"] + offset_parent
        df["father"] = df["father"] + offset_parent
        df.loc[df["twin"] != -1, "twin"] = df.loc[df["twin"] != -1, "twin"] + offset_id
        df["household_id"] = df["household_id"] + offset_household
        pedigree = pd.concat([pedigree, df], copy=False).reset_index(drop=True)
    else:
        # First generation: no known parents
        df["mother"] = -1
        df["father"] = -1
        pedigree = df

    return pedigree


def run_simulation(
    seed: int,
    N: int,
    G_ped: int,
    mating_lambda: float,
    p_mztwin: float,
    A1: float,
    C1: float,
    A2: float,
    C2: float,
    rA: float,
    rC: float,
    G_sim: int | None = None,
) -> pd.DataFrame:
    """Run the full ACE simulation for two correlated traits.

    Total phenotypic variance is fixed to 1.0 for each trait. Only A and C are
    free parameters; E is the residual: E = 1 - A - C. This means all variance
    components are proportions of total variance (i.e., h2 = A, c2 = C, e2 = E).

    Args:
        seed: Random seed
        N: Population size per generation (positive integer)
        G_ped: Number of generations to record in pedigree (integer >= 1)
        mating_lambda: Poisson lambda for zero-truncated mating count distribution (> 0)
        p_mztwin: Probability of a mating producing MZ twins, in [0, 1)
        A1, C1: Trait 1 variance components, each in [0, 1] with A1 + C1 <= 1
        A2, C2: Trait 2 variance components, each in [0, 1] with A2 + C2 <= 1
        rA: Genetic correlation between traits, in [-1, 1]
        rC: Common environment correlation between traits, in [-1, 1]
        G_sim: Total generations to simulate (default: G_ped). First G_sim - G_ped
               generations are burn-in and discarded from output.

    Returns:
        pedigree DataFrame

    Raises:
        ValueError: if any parameter is outside its valid range
    """
    if G_sim is None:
        G_sim = G_ped

    # --- Input validation ---
    for name, val in [("A1", A1), ("C1", C1), ("A2", A2), ("C2", C2)]:
        if not (0 <= val <= 1):
            raise ValueError(f"{name} must be between 0 and 1, got {val}")

    if 1.0 - A1 - C1 < -1e-10:
        raise ValueError(f"A1 + C1 must be <= 1.0 (got A1={A1}, C1={C1}, E1={1.0 - A1 - C1:.4f})")
    if 1.0 - A2 - C2 < -1e-10:
        raise ValueError(f"A2 + C2 must be <= 1.0 (got A2={A2}, C2={C2}, E2={1.0 - A2 - C2:.4f})")

    if not (int(N) == N and N > 0):
        raise ValueError(f"N must be a positive integer, got {N}")
    if not (G_ped == int(G_ped) and G_ped >= 1):
        raise ValueError(f"G_ped must be an integer >= 1, got {G_ped}")
    if not (mating_lambda > 0):
        raise ValueError(f"mating_lambda must be > 0, got {mating_lambda}")
    if not (0 <= p_mztwin < 1):
        raise ValueError(f"p_mztwin must be in [0, 1), got {p_mztwin}")
    if not (-1 <= rA <= 1):
        raise ValueError(f"rA must be in [-1, 1], got {rA}")
    if not (-1 <= rC <= 1):
        raise ValueError(f"rC must be in [-1, 1], got {rC}")

    if G_sim < G_ped:
        raise ValueError(f"G_sim ({G_sim}) must be >= G_ped ({G_ped})")

    logger.info("Starting simulation: N=%d, G_ped=%d, seed=%d", N, G_ped, seed)
    t0 = time.perf_counter()

    rng = np.random.default_rng(seed)

    # E is residual variance (total variance fixed to 1.0)
    E1 = 1.0 - A1 - C1
    E2 = 1.0 - A2 - C2

    sd_A1, sd_C1, sd_E1 = np.sqrt(A1), np.sqrt(C1), np.sqrt(E1)
    sd_A2, sd_C2, sd_E2 = np.sqrt(A2), np.sqrt(C2), np.sqrt(E2)

    # Initialize founders with correlated components
    sex = rng.binomial(size=N, n=1, p=0.5)

    # A components: correlated via rA
    a1, a2 = generate_correlated_components(rng, N, sd_A1, sd_A2, rA)

    # C components: correlated via rC
    c1, c2 = generate_correlated_components(rng, N, sd_C1, sd_C2, rC)

    # E components: independent (no correlation)
    e1 = rng.normal(size=N, loc=0, scale=sd_E1)
    e2 = rng.normal(size=N, loc=0, scale=sd_E2)

    pheno = np.stack([a1, c1, e1, a2, c2, e2], axis=-1)

    # Simulate generations
    burnin = G_sim - G_ped
    pedigree = None
    for i in range(G_sim):
        parents, twins, household_ids = mating(rng, sex, mating_lambda, p_mztwin)
        pheno, sex = reproduce(
            rng,
            pheno,
            parents,
            twins,
            household_ids,
            sd_A1,
            sd_E1,
            sd_C1,
            sd_A2,
            sd_E2,
            sd_C2,
            rA,
            rC,
        )
        if i >= burnin:
            pedigree = add_to_pedigree(
                pheno,
                sex,
                parents,
                twins,
                household_ids,
                generation=i - burnin,
                pedigree=pedigree,
            )
        # Per-generation data shape checkpoints
        fam_sizes = np.bincount(household_ids)
        logger.info(
            "Generation %d: %d twins, mean family size %.2f",
            i,
            len(twins) * 2,
            fam_sizes.mean(),
        )

    elapsed = time.perf_counter() - t0
    assert pedigree is not None
    logger.info(
        "Simulation complete in %.1fs: pedigree has %d individuals",
        elapsed,
        len(pedigree),
    )

    return pedigree


def cli() -> None:
    """Command-line interface for running ACE simulations."""
    from sim_ace.cli_base import add_logging_args, init_logging

    parser = argparse.ArgumentParser(description="Run ACE pedigree simulation")
    add_logging_args(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--N", type=int, default=1000, help="Founder population size")
    parser.add_argument("--G-ped", type=int, default=3, help="Number of pedigree generations")
    parser.add_argument("--G-sim", type=int, default=None, help="Number of burn-in generations (default: G_ped)")
    parser.add_argument("--mating-lambda", type=float, default=0.5, help="ZTP mating count lambda")
    parser.add_argument("--p-mztwin", type=float, default=0.02, help="Probability of MZ twinning")
    parser.add_argument("--A1", type=float, default=0.5, help="Additive genetic variance for trait 1")
    parser.add_argument("--C1", type=float, default=0.2, help="Shared environment variance for trait 1")
    parser.add_argument("--A2", type=float, default=0.5, help="Additive genetic variance for trait 2")
    parser.add_argument("--C2", type=float, default=0.2, help="Shared environment variance for trait 2")
    parser.add_argument("--rA", type=float, default=0.5, help="Cross-trait genetic correlation")
    parser.add_argument("--rC", type=float, default=0.3, help="Cross-trait shared environment correlation")
    parser.add_argument("--output-pedigree", required=True, help="Output pedigree parquet path")
    parser.add_argument("--output-params", required=True, help="Output params YAML path")
    parser.add_argument("--rep", type=int, default=1, help="Replicate number")
    args = parser.parse_args()

    init_logging(args)

    pedigree = run_simulation(
        seed=args.seed,
        N=args.N,
        G_ped=args.G_ped,
        mating_lambda=args.mating_lambda,
        p_mztwin=args.p_mztwin,
        A1=args.A1,
        C1=args.C1,
        A2=args.A2,
        C2=args.C2,
        rA=args.rA,
        rC=args.rC,
        G_sim=args.G_sim,
    )

    save_parquet(pedigree, args.output_pedigree)

    params_dict = {
        "seed": args.seed,
        "rep": args.rep,
        "A1": args.A1,
        "C1": args.C1,
        "E1": 1.0 - args.A1 - args.C1,
        "A2": args.A2,
        "C2": args.C2,
        "E2": 1.0 - args.A2 - args.C2,
        "rA": args.rA,
        "rC": args.rC,
        "N": args.N,
        "G_ped": args.G_ped,
        "G_sim": args.G_sim or args.G_ped,
        "mating_lambda": args.mating_lambda,
        "p_mztwin": args.p_mztwin,
    }
    with open(args.output_params, "w", encoding="utf-8") as f:
        yaml.dump(params_dict, f, default_flow_style=False)
