"""
ACE Pedigree Simulation

Simulates multi-generational pedigrees with:
- A: Additive genetic component
- C: Common/shared environment component
- E: Unique environment component

Supports single-trait and two-trait (bivariate) modes with configurable
cross-trait correlations for genetic (rA) and common environment (rC) components.
"""

import numpy as np
import pandas as pd
import yaml


def generate_correlated_components(rng, n, sd1, sd2, correlation):
    """Generate two correlated normal variables via multivariate normal.

    Args:
        rng: numpy random generator
        n: number of samples
        sd1: standard deviation for component 1
        sd2: standard deviation for component 2
        correlation: correlation between components

    Returns:
        (comp1, comp2): tuple of arrays, each shape (n,)
    """
    cov = [
        [sd1**2, correlation * sd1 * sd2],
        [correlation * sd1 * sd2, sd2**2],
    ]
    samples = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
    return samples[:, 0], samples[:, 1]


def generate_mendelian_noise(rng, n, sd_A1, sd_A2, rA):
    """Generate correlated Mendelian sampling noise for two traits.

    The Mendelian noise has variance = 0.5 * Var(A) for each trait,
    so sd_noise = sd_A / sqrt(2).

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


def mating(rng, parental_sex, fam_size, p_nonsocial_father, p_mztwin):
    """Generate parent-offspring pairings.

    Args:
        rng: numpy random generator
        parental_sex: array of sex values for parents
        fam_size: mean family size (Poisson lambda)
        p_nonsocial_father: proportion of non-social fathers
        p_mztwin: target empirical rate of MZ twins (fraction of individuals who are twins)

    Returns:
        parent_idxs: (n, 2) array of [mother_idx, father_idx] for each offspring
        twins: array of [twin1_idx, twin2_idx] pairs for MZ twins
    """
    n = len(parental_sex)

    # Convert target twin rate to per-birth probability
    # If each birth has prob p of being twins, expected twin rate = 2p/(1+p)
    # Solving for p given target rate t: p = t / (2 - t)
    p_twin_birth = p_mztwin / (2 - p_mztwin)
    nmale = parental_sex.sum()
    nfemale = n - nmale
    male_idxs = np.where(parental_sex)[0]
    female_idxs = np.where(parental_sex == 0)[0]

    rng.shuffle(female_idxs)
    rng.shuffle(male_idxs)

    # Generate family sizes until we have enough offspring
    while True:
        family_sizes = rng.poisson(lam=fam_size, size=nfemale)
        if family_sizes.sum() >= n:
            break

    parent_idxs = np.zeros(shape=(n, 2), dtype=int)
    remaining_offspring = n
    twins = []

    for i in range(nfemale):
        mother = female_idxs[i]
        social_father = male_idxs[i]
        target_size = min(family_sizes[i], remaining_offspring)
        offspring_created = 0

        while offspring_created < target_size and remaining_offspring > 0:
            # Determine biological father
            if rng.uniform() > p_nonsocial_father:
                bio_father = social_father
            else:
                bio_father = rng.choice(male_idxs)

            # Check if twins: need room for 2 in both family and population
            slots_in_family = target_size - offspring_created
            is_twin = (
                rng.uniform() <= p_twin_birth
                and slots_in_family >= 2
                and remaining_offspring >= 2
            )

            if is_twin:
                # MZ twins
                idx1 = remaining_offspring - 1
                idx2 = remaining_offspring - 2
                parent_idxs[idx1] = [mother, bio_father]
                parent_idxs[idx2] = [mother, bio_father]
                twins.append([idx1, idx2])
                remaining_offspring -= 2
                offspring_created += 2
            else:
                # Singleton
                parent_idxs[remaining_offspring - 1] = [mother, bio_father]
                remaining_offspring -= 1
                offspring_created += 1

        if remaining_offspring == 0:
            break

    return parent_idxs, (
        np.array(twins, dtype=int) if twins else np.array([], dtype=int).reshape(0, 2)
    )


def reproduce(rng, pheno, parents, twins, sd_A1, sd_E1, sd_A2, sd_E2, rA):
    """Simulate offspring phenotypes from parents for two correlated traits.

    Args:
        rng: numpy random generator
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2] for parents
        parents: (n, 2) array of [mother_idx, father_idx]
        twins: array of MZ twin index pairs
        sd_A1: standard deviation of A for trait 1
        sd_E1: standard deviation of E for trait 1
        sd_A2: standard deviation of A for trait 2
        sd_E2: standard deviation of E for trait 2
        rA: genetic correlation between traits

    Returns:
        offspring: (n, 6) array of [A1, C1, E1, A2, C2, E2]
        sex_offspring: (n,) array of sex values (0=female, 1=male)
    """
    n = len(parents)

    # Sex assignment
    sex_offspring = rng.binomial(size=n, n=1, p=0.5)

    # Additive genetic: midparent + correlated Mendelian noise
    mp1 = pheno[:, 0][parents].mean(1)  # A1 midparent
    mp2 = pheno[:, 3][parents].mean(1)  # A2 midparent

    noise1, noise2 = generate_mendelian_noise(rng, n, sd_A1, sd_A2, rA)
    a1_offspring = mp1 + noise1
    a2_offspring = mp2 + noise2

    # Common environment: inherited from mother (both traits)
    c1_offspring = pheno[:, 1][parents[:, 0]]
    c2_offspring = pheno[:, 4][parents[:, 0]]

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


def add_to_pedigree(pheno, sex, parents, twins, generation, pedigree=None):
    """Add a generation to the pedigree DataFrame.

    Args:
        pheno: (n, 6) array of [A1, C1, E1, A2, C2, E2]
        sex: (n,) array of sex values
        parents: (n, 2) array of [mother_idx, father_idx]
        twins: array of MZ twin index pairs
        generation: generation number (0 for founders)
        pedigree: existing pedigree DataFrame or None for first generation

    Returns:
        Updated pedigree DataFrame
    """
    df = pd.DataFrame(pheno, columns=["A1", "C1", "E1", "A2", "C2", "E2"])
    df["liability1"] = df["A1"] + df["C1"] + df["E1"]
    df["liability2"] = df["A2"] + df["C2"] + df["E2"]

    df["sex"] = sex
    df[["mother", "father"]] = parents
    df["twin"] = -1
    df["generation"] = generation

    if len(twins) > 0:
        df.loc[twins[:, 0], "twin"] = twins[:, 1]
        df.loc[twins[:, 1], "twin"] = twins[:, 0]

    df["id"] = df.index.values
    df = df[
        [
            "id",
            "sex",
            "mother",
            "father",
            "twin",
            "generation",
            "A1",
            "C1",
            "E1",
            "liability1",
            "A2",
            "C2",
            "E2",
            "liability2",
        ]
    ]

    if pedigree is not None:
        n = len(df)
        offset_id = pedigree["id"].max() + 1
        offset_parent = offset_id - n
        df["id"] = df["id"] + offset_id
        df["mother"] = df["mother"] + offset_parent
        df["father"] = df["father"] + offset_parent
        df.loc[df["twin"] != -1, "twin"] = df.loc[df["twin"] != -1, "twin"] + offset_id
        pedigree = pd.concat([pedigree, df]).reset_index(drop=True)
    else:
        # First generation: no known parents
        df["mother"] = -1
        df["father"] = -1
        pedigree = df

    return pedigree


def run_simulation(
    seed,
    N,
    G_ped,
    fam_size,
    p_mztwin,
    p_nonsocial_father,
    A1,
    C1,
    A2,
    C2,
    rA,
    rC,
    G_sim=None,
):
    """Run the full ACE simulation for two correlated traits.

    Args:
        seed: Random seed
        N: Population size per generation
        G_ped: Number of generations to record in pedigree
        fam_size: Mean family size
        p_mztwin: Proportion of MZ twins
        p_nonsocial_father: Proportion of non-social fathers
        A1, C1: Trait 1 variance components
        A2, C2: Trait 2 variance components
        rA: Genetic correlation between traits
        rC: Common environment correlation between traits
        G_sim: Total generations to simulate (default: G_ped). First G_sim - G_ped
               generations are burn-in and discarded from output.

    Returns:
        pedigree DataFrame
    """
    if G_sim is None:
        G_sim = G_ped
    if G_sim < G_ped:
        raise ValueError(f"G_sim ({G_sim}) must be >= G_ped ({G_ped})")

    rng = np.random.default_rng(seed)

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
        parents, twins = mating(rng, sex, fam_size, p_nonsocial_father, p_mztwin)
        pheno, sex = reproduce(
            rng, pheno, parents, twins, sd_A1, sd_E1, sd_A2, sd_E2, rA
        )
        if i >= burnin:
            pedigree = add_to_pedigree(
                pheno, sex, parents, twins, generation=i - burnin, pedigree=pedigree
            )

    return pedigree


if __name__ == "__main__":
    # Get parameters from Snakemake
    params = snakemake.params
    output_pedigree = snakemake.output.pedigree
    output_params = snakemake.output.params

    # Run simulation
    pedigree = run_simulation(
        seed=params.seed,
        N=params.N,
        G_ped=params.G_ped,
        fam_size=params.fam_size,
        p_mztwin=params.p_mztwin,
        p_nonsocial_father=params.p_nonsocial_father,
        A1=params.A1,
        C1=params.C1,
        A2=params.A2,
        C2=params.C2,
        rA=params.rA,
        rC=params.rC,
        G_sim=params.G_sim,
    )

    # Save pedigree
    pedigree.to_parquet(output_pedigree, index=False)

    # Save parameters used for this simulation
    params_dict = {
        "seed": params.seed,
        "rep": params.rep,
        "A1": params.A1,
        "C1": params.C1,
        "E1": 1.0 - params.A1 - params.C1,
        "A2": params.A2,
        "C2": params.C2,
        "E2": 1.0 - params.A2 - params.C2,
        "rA": params.rA,
        "rC": params.rC,
        "N": params.N,
        "G_ped": params.G_ped,
        "G_sim": params.G_sim,
        "fam_size": params.fam_size,
        "p_mztwin": params.p_mztwin,
        "p_nonsocial_father": params.p_nonsocial_father,
    }
    with open(output_params, "w") as f:
        yaml.dump(params_dict, f, default_flow_style=False)
