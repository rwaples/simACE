# simACE documentation

simACE simulates age-of-onset phenotypes for related individuals. It builds
multi-generational pedigrees and simulates heritable traits on them.

## What is simACE?

simACE generates synthetic registry-scale datasets with millions of individuals
across multiple generations. It produces liabilities and time-to-event
phenotypes, then applies censoring and ascertainment. Liability follows the ACE
model (additive genetic, common environment, unique environment). It is designed
for evaluating and benchmarking statistical methods that estimate heritability
and familial correlations from population health registries.

## Key features

- Multi-generational pedigree simulation with multi-partner mating, half-siblings, and MZ twins
- ACE trait liability model for two heritable traits at a time
- Five phenotype model families: proportional-hazards frailty (six baseline hazards), cure-frailty, ADuLT (LTM and Cox), first-passage time, and simple liability threshold
- Age-window and competing-risk mortality censoring
- Unified ascertainment stage: random dropout + case-weighted sampling (per ADR 0001)
- Statistical validation of simulated data
- Built-in diagnostic plots
- Snakemake pipeline for reproducible, parallelised execution
- Scales to N = 1,000,000 per generation: the stock `baseline1M` scenario peaks at about 8 GB RSS per replicate (`benchmarks/base/baseline1M/`)

## Quick links

- [Installation](getting-started/installation.md): pixi environment setup
- [Quick start](getting-started/quickstart.md): running an initial simulation
- [Configuration](user-guide/configuration.md): parameter reference
- [Examples](examples/minimal-ace.md): worked walkthroughs of simulation mechanisms
- [API reference](api/index.md): Python API documentation for `simace`
