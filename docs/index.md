# simACE documentation

!!! warning "Under construction"
    This documentation is a work in progress. Some pages may be incomplete.

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
- [API Reference](api/index.md) -- Python API docs for `sim_ace`

## Two packages

| Package | Purpose | Install |
|---|---|---|
| `sim_ace` | Simulation, phenotyping, censoring, sampling, analysis, plotting | `pip install -e .` |
| `fit_ace` | Model fitting (EPIMIGHT, PA-FGRS, Stan) | `pip install -e fit_ace/` |

This documentation covers `sim_ace`. See the fit_ace directory for model fitting documentation.
