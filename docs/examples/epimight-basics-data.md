# EPIMIGHT basics: simulated data description

Two simulated multi-generational pedigrees for testing EPIMIGHT on liability
heritability, genetic correlation, prevalence, and age of onset. Both carry the
same two binary traits with an age of onset and about 100,000 phenotyped
individuals, with one replicate each. They differ in how much realism the
pedigree and observation process carry.

| | `eb_simple` | `eb_realistic` |
|---|---|---|
| Phenotyped individuals | 4 generations x 25,000 | 10 generations x 10,000 |
| Recorded pedigree depth | 5 generations | 10 generations |
| MZ twins | none | 2% of eligible couples |
| Assortative mating (spousal r) | none | 0.30 on trait 1, 0.15 on trait 2 |
| Death (competing risk) | none | Weibull, scale 164, shape 2.73 |
| Observation window | ages 0 to 80 for every generation | shrinks for recent cohorts (see below) |
| Prevalence by sex | equal | male = 2 x female, same mean |
| Random dropout | none | 10% of individuals removed |

## Trait model

Each trait follows a liability threshold model. Liability is the sum of an
additive genetic component A and a unique environment component E, with the
shared environment component C set to zero. The variance of A is the liability
heritability. Case status is liability above the threshold for lifetime
prevalence K. Age of onset among cases follows a logistic cumulative incidence
curve with midpoint `cip_x0` (the median onset age) and slope 0.15.

The two traits are the same in both scenarios. Trait 1 is heritable, rare, and
late onset. Trait 2 is less heritable, common, and early onset.

| Parameter | Trait 1 | Trait 2 |
|---|---|---|
| Heritability h2 | 0.7 | 0.3 |
| Lifetime prevalence K | 0.02 | 0.15 |
| Median onset age | 50 | 20 |
| Genetic correlation rA between traits | 0.5 | |

Environmental correlation between the traits is zero, so the liability
correlation between traits is rA x sqrt(0.7 x 0.3) = 0.229.

## Censoring in the realistic scenario

The realistic scenario observes each birth cohort only up to an age that
shrinks for later generations, mimicking registers where young cohorts have
not yet reached the ages of onset. Generation 9 is the youngest.

| Generation | Observed ages |
|---|---|
| 0, 1, 2 | 0 to 80 |
| 3 | 0 to 70 |
| 4 | 0 to 60 |
| 5 | 0 to 50 |
| 6 | 0 to 40 |
| 7 | 0 to 30 |
| 8 | 0 to 20 |
| 9 | 0 to 12 |

Death from the competing-risk mortality curve also ends observation. An
individual is a case only if onset happens before both the end of the
observation window and death. The observed case fraction is therefore below
the lifetime prevalence K, most strongly in the youngest generations and for
the late-onset trait 1.

Realized values in the two datasets:

| | `eb_simple` | `eb_realistic` |
|---|---|---|
| Phenotyped rows | 100,000 | 90,000 |
| Observed case fraction, trait 1 | 0.020 | 0.011 |
| Observed case fraction, trait 2 | 0.150 | 0.125 |
| Median observed onset, trait 1 / 2 | 49.3 / 19.9 | 47.4 / 18.6 |
| Trait 1 case fraction, generation 0 / 9 | | 0.018 / 0.000 |
| Trait 2 case fraction, generation 0 / 9 | | 0.147 / 0.038 |

In the realistic scenario the late-onset trait 1 has essentially no observed
cases in the two youngest generations.

## Files

Each scenario writes to `results/epimight_basics/<scenario>/rep1/`.

`pedigree.parquet` has one row per individual in the recorded pedigree:
`id`, `sex` (0 female, 1 male), `mother`, `father` (-1 for founders), `twin`
(id of the MZ co-twin, -1 if none), `generation`, `household_id`, and the true
liability components `A1`, `E1`, `liability1`, `A2`, `E2`, `liability2`.

`trait.parquet` has one row per phenotyped individual, joined on `id`:

| Column | Meaning |
|---|---|
| `t1`, `t2` | Latent age of onset. 1e6 means never a case. |
| `death_age` | Age at death from the mortality curve |
| `t_observed1`, `t_observed2` | Age at onset, or age at end of follow-up if not a case |
| `affected1`, `affected2` | Observed case status |
| `age_censored1`, `age_censored2` | True when onset fell outside the observation window |
| `death_censored1`, `death_censored2` | True when death came before onset |

`params.yaml` in the same directory records every parameter of the scenario.
