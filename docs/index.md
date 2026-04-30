# simACE documentation

simACE simulates registry-scale age-of-onset phenotypes with realistic
multi-generational pedigrees and the ACE (Additive genetic, Common environment,
unique Environment) liability model.

## What is simACE?

simACE generates synthetic registry-scale datasets with millions of individuals
across multiple generations, producing two correlated time-to-event traits governed
by additive genetic (A), shared environment (C), and unique environment (E) variance
components. It is designed for evaluating and benchmarking statistical methods that
estimate heritability and familial correlations from population health registries.

## Key features

- Multi-generational pedigree simulation with realistic mating patterns, half-siblings, and MZ twins
- Multiple phenotype models: Weibull frailty, cure-frailty, ADuLT LTM, ADuLT Cox, and simple liability threshold
- Age-window and competing-risk mortality censoring
- Subsampling with case ascertainment bias and pedigree dropout
- Automated structural and statistical validation of simulated data
- Publication-quality diagnostic plots compiled into PDF atlases
- Snakemake pipeline for reproducible, parallelised execution

## Quick links

- [Installation](getting-started/installation.md) -- set up the conda environment
- [Quick Start](getting-started/quickstart.md) -- run your first simulation
- [Configuration](user-guide/configuration.md) -- parameter reference
- [API Reference](api/index.md) -- Python API docs for `simace`

## Two packages

| Package | Purpose | Install |
|---|---|---|
| `simace` | Simulation, phenotyping, censoring, sampling, analysis, plotting | `pip install -e .` |
| `fitACE` | Model fitting (EPIMIGHT, PA-FGRS, sparseREML, iter_reml, Stan, PCGC) | `pip install -e fitACE/` |

This documentation covers `simace`. Model fitting lives in the sister
[`fitACE`](https://github.com/rwaples/fitACE) repo, checked out under
`./fitACE/`. See [Project Structure](concepts/project-structure.md) for the
full repo map.
