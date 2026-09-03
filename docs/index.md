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
- Five phenotype model families: proportional-hazards frailty (six baseline hazards), cure-frailty, ADuLT (LTM and Cox), first-passage time, and simple liability threshold
- Age-window and competing-risk mortality censoring
- Unified ascertainment stage: random dropout + case-weighted sampling (per ADR 0001)
- Statistical validation of simulated data
- Built-in diagnostic plots
- Snakemake pipeline for reproducible, parallelised execution
- Scales to N = 1,000,000 per generation: the stock `baseline1M` scenario peaks at about 8 GB RSS per replicate (`benchmarks/base/baseline1M/`)

## Quick links

- [Installation](getting-started/installation.md) — pixi environment setup
- [Quick Start](getting-started/quickstart.md) — running an initial simulation
- [Configuration](user-guide/configuration.md) — parameter reference
- [Examples](examples/minimal-ace.md) — story-first walkthroughs of simulation mechanisms
- [API Reference](api/index.md) — Python API documentation for `simace`
