# simACE documentation

simACE simulates registry-scale age-of-onset phenotypes with realistic
multi-generational pedigrees and the ACE (Additive genetic, Common environment,
unique Environment) liability model.

## What is simACE?

<!-- TODO: 2-3 sentences: purpose, scale (millions of individuals), two correlated traits -->

## Key features

<!-- TODO: Bullet list: pedigree simulation, multiple phenotype models, censoring,
     subsampling, validation, publication-quality plots -->

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
