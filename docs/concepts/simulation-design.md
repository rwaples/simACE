# Simulation Design

## Multi-generational pedigree

The simulation generates `G_sim` total generations:

- `G_sim - G_ped` are **burn-in** generations (simulated but not recorded)
- `G_ped` generations are recorded in the pedigree
- The last `G_pheno` of `G_ped` are phenotyped

Each generation contains `N` individuals. With default settings
($N = 100{,}000$, $G_{ped} = 6$), the recorded pedigree contains
approximately $600{,}000$ individuals.

## Mating and reproduction

In each generation, couples are formed from the potential parent pool
according to the following rules:

- An individual may participate in multiple couples.
- Males and females are paired randomly by default, or assortatively
  on liability via `assort1` and `assort2`.
- Offspring are distributed across matings by a multinomial draw.
- Population size is held constant; some couples produce no offspring.
- MZ twins are assigned to matings with two or more offspring.

At default settings (`mating_lambda = 0.5`), approximately 77% of
individuals have a single partner and 23% have two or more, producing
a natural mix of full sibs, maternal half-sibs, and paternal half-sibs.

## Pedigree relationship types

`PedigreeGraph` extracts 23 relationship categories using sparse matrix algebra.
Each type is parameterised by `(up, down, n_ancestors)`:

- **up**: meioses from individual A up to common ancestor(s)
- **down**: meioses down from ancestor(s) to individual B
- **n_ancestors**: 1 (half/lineal) or 2 (full, mated-pair)
- **kinship**: $n_{\text{ancestors}} \times (1/2)^{(\text{up} + \text{down} + 1)}$

| Code | Label | Up | Down | Ancestors | Kinship | Degree |
|------|-------|---:|-----:|----------:|--------:|-------:|
| MZ | MZ twin | -- | -- | -- | 1/2 | 0 |
| FS | Full sib | 1 | 1 | 2 | 1/4 | 1 |
| MHS | Maternal half sib | 1 | 1 | 1 | 1/8 | 2 |
| PHS | Paternal half sib | 1 | 1 | 1 | 1/8 | 2 |
| GP | Grandparent | 2 | 0 | 1 | 1/8 | 2 |
| Av | Avuncular | 1 | 2 | 2 | 1/8 | 2 |
| 1C | 1st cousin | 2 | 2 | 2 | 1/16 | 3 |

The `max_degree` parameter controls extraction depth (default 2).
Degree 3–5 types require deeper matrix products and are computed only
when requested.

The full registry of 23 types is available as `REL_REGISTRY` and `PAIR_KINSHIP`
from `simace.core.pedigree_graph`.

## Pipeline stages

The simulation is conceptually split into four stages, plus downstream analysis:

1. **Simulate** -- generate multi-generational pedigree with ACE liability components
2. **Phenotype** -- map liability to age-of-onset via time-to-event models
3. **Censor** -- apply age-window and competing-risk mortality censoring
4. **Sample** -- optionally subsample and apply ascertainment bias

Followed by: validation, summary statistics, model fitting, and plotting.
