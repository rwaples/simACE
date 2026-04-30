# simACE documentation

simACE simulates age-of-onset phenotypes for related individuals. It uses realistic pedigrees 
and family structures to generate multi-generational family relationships and to simulate heritable traits.

## What is simACE?

simACE generates synthetic registry-scale datasets with millions of individuals
across multiple generations.  It produces liabilities, time-to-event phenotypes, and allows censoring and ascertainment. It uses the ACE 
(Additive genetic, Common environment, unique Environment) liability model. 
It is designed for evaluating and benchmarking statistical methods that
estimate heritability and familial correlations from population health registries. 

## Key features

- Multi-generational pedigree simulation with realistic mating patterns, half-siblings, and MZ twins
- ACE trait liability model for two heritable traits at a time
- Multiple phenotype models: Weibull frailty, cure-frailty, ADuLT LTM, ADuLT Cox, and simple liability threshold
- Age-window and competing-risk mortality censoring
- Case ascertainment and pedigree dropout
- Statistical validation of simulated data
- Built-in diagnostic plots
- Snakemake pipeline for reproducible, parallelised execution
- It is fast - less than 3 minutes for 1M individuals x 3 generations x 3 reps using 4 cores and <16 GB RAM

## Quick links

- [Installation](getting-started/installation.md) — conda environment setup
- [Quick Start](getting-started/quickstart.md) — running an initial simulation
- [Configuration](user-guide/configuration.md) — parameter reference
- [API Reference](api/index.md) — Python API documentation for `simace`
