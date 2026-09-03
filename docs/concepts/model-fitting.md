# Model fitting

simACE simulates data. Fitting variance-component models to that data is the job of the sister repo [fitACE](https://github.com/rwaples/fitACE), a private monorepo checked out at `./fitACE/`. This page says which estimates each repo produces. Usage instructions are in the fitACE README.

## What simACE estimates itself

The stats stage in `simace.analysis.stats` runs three estimators on every replicate. They exist to validate the simulation, not to compete with fitACE.

- **Tetrachoric correlations** by relationship type and generation, from binary affection status.
- **Falconer's $h^2$**, from the tetrachoric correlations of MZ and full-sibling pairs.
- **Parent-offspring regression** of offspring liability on midparent liability.

The maths is in [Methods, validation via statistical analysis](methods.md#validation-via-statistical-analysis).

## What fitACE estimates

fitACE reads the outcomes-only trait files and the analysis pedigree that the ascertainment stage writes. See [Pipeline schema](pipeline-schema.md) for the file contract. Its method packages are:

- **PCGC**.
- **Iterative and sparse REML**, including the `ace_iter_reml` C++ binary.
- **TetraHer**, through the `tetraher_simace` LDAK fork.
- **PA-FGRS**, family genetic risk scores.
- **Stan** models.
- **Frailty**, a pairwise Weibull maximum-likelihood estimator of liability correlation from censored onset times. The likelihood is in [Methods, pairwise Weibull survival correlation](methods.md#pairwise-weibull-survival-correlation-estimation).
- **EPIMIGHT**, an R package for heritability from family data, integrated through the separate `fitACE_epimight` repo.

## What the simulation gives a fitter

Every simulated individual has known $A$, $C$, and $E$ values, so every fitted variance component can be compared to its true value. Three parts of the pipeline shape what a fitter sees:

- The **phenotype models** map liability to affected status and age at onset. See [Methods, phenotype models](methods.md#phenotype-models).
- The **censoring** stage hides events outside each generation's age window and events after death. See [Methods, censoring](methods.md#censoring).
- The **ascertainment** stage drops individuals at random, then draws a case-weighted sample. See [Methods, ascertainment](methods.md#ascertainment) and [ADR 0001](../adr/0001-unified-ascertainment-stage.md).
