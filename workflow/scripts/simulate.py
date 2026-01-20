"""
ACE Pedigree Simulation

Simulates multi-generational pedigrees with:
- A: Additive genetic component
- C: Common/shared environment component
- E: Unique environment component
"""

import numpy as np
import pandas as pd
import yaml


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
            is_twin = (rng.uniform() <= p_twin_birth and
                       slots_in_family >= 2 and
                       remaining_offspring >= 2)

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

    return parent_idxs, np.array(twins, dtype=int) if twins else np.array([], dtype=int).reshape(0, 2)


def reproduce(rng, pheno, parents, twins, sd_A, sd_E):
    """Simulate offspring phenotypes from parents.

    Args:
        rng: numpy random generator
        pheno: (n, 3) array of [A, C, E] values for parents
        parents: (n, 2) array of [mother_idx, father_idx]
        twins: array of MZ twin index pairs
        sd_A: standard deviation of additive genetic component
        sd_E: standard deviation of unique environment component

    Returns:
        offspring: (n, 3) array of [A, C, E] values
        sex_offspring: (n,) array of sex values (0=female, 1=male)
    """
    n = len(pheno)

    # Additive genetic: midparent value + noise
    mp = pheno[:, 0][parents].mean(1)
    a_offspring = rng.normal(size=n, loc=mp, scale=sd_A / np.sqrt(2))

    # Common environment: inherited from mother
    c_offspring = pheno[:, 1][parents[:, 0]]

    # Unique environment: random per individual
    e_offspring = rng.normal(size=n, loc=0, scale=sd_E)

    # Sex
    sex_offspring = rng.binomial(size=n, n=1, p=0.5)

    # MZ twins share additive genetic factor and sex
    if len(twins) > 0:
        a_offspring[twins[:, 1]] = a_offspring[twins[:, 0]]
        sex_offspring[twins[:, 1]] = sex_offspring[twins[:, 0]]

    offspring = np.stack([a_offspring, c_offspring, e_offspring], axis=-1)
    return offspring, sex_offspring


def add_to_pedigree(pheno, sex, parents, twins, pedigree=None):
    """Add a generation to the pedigree DataFrame.

    Args:
        pheno: (n, 3) array of [A, C, E] values
        sex: (n,) array of sex values
        parents: (n, 2) array of [mother_idx, father_idx]
        twins: array of MZ twin index pairs
        pedigree: existing pedigree DataFrame or None for first generation

    Returns:
        Updated pedigree DataFrame
    """
    df = pd.DataFrame(pheno, columns=['A', 'C', 'E'])
    df['sex'] = sex
    df[['mother', 'father']] = parents
    df['twin'] = -1

    if len(twins) > 0:
        df.loc[twins[:, 0], 'twin'] = twins[:, 1]
        df.loc[twins[:, 1], 'twin'] = twins[:, 0]

    df['id'] = df.index.values
    df = df[['id', 'sex', 'mother', 'father', 'twin', 'A', 'C', 'E']]

    if pedigree is not None:
        n = len(df)
        offset_id = pedigree['id'].max() + 1
        offset_parent = offset_id - n
        df['id'] = df['id'] + offset_id
        df['mother'] = df['mother'] + offset_parent
        df['father'] = df['father'] + offset_parent
        df.loc[df['twin'] != -1, 'twin'] = df.loc[df['twin'] != -1, 'twin'] + offset_id
        pedigree = pd.concat([pedigree, df]).reset_index(drop=True)
    else:
        # First generation: no known parents
        df['mother'] = -1
        df['father'] = -1
        pedigree = df

    return pedigree


def run_simulation(seed, A, C, N, ngen, fam_size, p_mztwin, p_nonsocial_father):
    """Run the full ACE simulation.

    Args:
        seed: Random seed
        A: Additive genetic variance component
        C: Common environment variance component
        N: Population size per generation
        ngen: Number of generations
        fam_size: Mean family size
        p_mztwin: Proportion of MZ twins
        p_nonsocial_father: Proportion of non-social fathers

    Returns:
        pedigree DataFrame
    """
    rng = np.random.default_rng(seed)

    E = 1.0 - A - C
    sd_A = np.sqrt(A)
    sd_C = np.sqrt(C)
    sd_E = np.sqrt(E)

    # Initialize first generation
    sex = rng.binomial(size=N, n=1, p=0.5)
    a_pheno = rng.normal(size=N, loc=0, scale=sd_A)
    c_household = rng.normal(size=N, loc=0, scale=sd_C)
    e = rng.normal(size=N, loc=0, scale=sd_E)
    pheno = np.stack([a_pheno, c_household, e], axis=-1)

    # Simulate generations
    pedigree = None
    for i in range(ngen):
        parents, twins = mating(
            rng, sex, fam_size, p_nonsocial_father, p_mztwin
        )
        pheno, sex = reproduce(rng, pheno, parents, twins, sd_A, sd_E)
        pedigree = add_to_pedigree(pheno, sex, parents, twins, pedigree)

    return pedigree


if __name__ == "__main__":
    # Get parameters from Snakemake
    params = snakemake.params
    output_pedigree = snakemake.output.pedigree
    output_params = snakemake.output.params

    # Run simulation
    pedigree = run_simulation(
        seed=params.seed,
        A=params.A,
        C=params.C,
        N=params.N,
        ngen=params.ngen,
        fam_size=params.fam_size,
        p_mztwin=params.p_mztwin,
        p_nonsocial_father=params.p_nonsocial_father,
    )

    # Save pedigree
    pedigree.to_parquet(output_pedigree, index=False)

    # Save parameters used for this simulation
    params_dict = {
        'seed': params.seed,
        'A': params.A,
        'C': params.C,
        'E': 1.0 - params.A - params.C,
        'N': params.N,
        'ngen': params.ngen,
        'fam_size': params.fam_size,
        'p_mztwin': params.p_mztwin,
        'p_nonsocial_father': params.p_nonsocial_father,
    }
    with open(output_params, 'w') as f:
        yaml.dump(params_dict, f, default_flow_style=False)
