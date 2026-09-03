# Project Structure

## Repository layout

```
simACE/
├── Snakefile                            # Root entry point (no -s flag needed)
├── pixi.toml, pixi.lock                 # Authoritative environment pins (ADR 0016)
├── config/
│   ├── _default.yaml                    # Default simulation parameters
│   └── {folder}.yaml                    # Per-folder scenario definitions
│
├── simace/                              # Simulation package (pip install -e .)
│   ├── __init__.py                       # Package init
│   ├── config.py                         # Config resolution and parameter coercion
│   ├── core/                             # Shared infrastructure
│   │   ├── _numba_utils.py               # Shared Numba-compiled utilities
│   │   ├── cli_base.py                   # Shared CLI boilerplate (add_logging_args, init_logging)
│   │   ├── compute_hazard_terms.py       # Baseline hazard computation for parametric survival models
│   │   ├── numerics.py                   # safe_corrcoef, safe_linregress, numba-accelerated helpers
│   │   ├── parquet.py                    # Parquet reader/writer with dtype narrowing and the null contract
│   │   ├── parquet_to_tsv.py             # `simace-parquet-to-tsv` CLI entry point
│   │   ├── pedigree_arrays.py            # Pedigree columns as numpy arrays, addressable by id
│   │   ├── pedigree_filter.py            # Filter a pedigree to observed IDs plus their ancestors
│   │   ├── relationships.py              # Relationship-pair and sex vocabulary (re-exports PAIR_KINSHIP from pedigree_graph)
│   │   ├── schema.py                     # PEDIGREE schema and hydrated in-memory trait schemas
│   │   ├── snakemake_adapter.py          # Signature bridge between Snakemake script wrappers and domain functions
│   │   ├── stage.py                      # @stage decorator: input/output schema assertions on stage functions
│   │   ├── trait_schema.py               # Outcomes-only trait file schemas (RAW_TRAIT, CENSORED_TRAIT) and hydrate_trait
│   │   └── yaml_io.py                    # load_yaml, dump_yaml helpers
│   ├── simulation/
│   │   ├── simulate.py                   # ACE pedigree simulation (mating, reproduce, run_simulation)
│   │   ├── params.py                     # Validated parameter record for run_simulation
│   │   ├── assortment.py                 # Standard-mating assortative-mating plan
│   │   ├── mate_correlation.py           # Expected mate liability correlation matrix
│   │   ├── am_equilibrium.py             # Assortative-mating additive-variance equilibrium
│   │   └── emit_params.py                # Echo scenario parameters to a YAML sidecar
│   ├── phenotype/
│   │   ├── runner.py                     # run_phenotype dispatcher and CLI (re-exported from __init__.py)
│   │   ├── hazards.py                    # Baseline-hazard registry (Weibull, exponential, Gompertz, ...)
│   │   ├── blended_post.py               # Post-hoc blended-diagnosis transform
│   │   ├── models/                       # PhenotypeModel subclasses (one file per family)
│   │   │   ├── _base.py                  # PhenotypeModel ABC
│   │   │   ├── _prevalence.py            # Prevalence resolution shared by threshold-based models
│   │   │   ├── adult.py                  # ADuLT
│   │   │   ├── cure_frailty.py           # Mixture cure-frailty
│   │   │   ├── first_passage.py          # First-passage time
│   │   │   ├── frailty.py                # Proportional-hazards frailty
│   │   │   └── simple_ltm.py             # Simple liability threshold
│   │   └── _prototypes/                  # Experimental models not wired into the dispatcher
│   ├── censoring/
│   │   └── censor.py                     # Age-window and competing-risk death censoring
│   ├── ascertainment/
│   │   ├── __init__.py                   # Stage docstring and re-exports
│   │   └── runner.py                     # run_ascertainment: dropout, case-weighted N_sample, pedigree closure, CLI (per ADR 0001)
│   ├── analysis/
│   │   ├── analyze.py                    # Combined Analyze stage: produce the curated report.yaml
│   │   ├── report.py                     # Assemble the curated report from validation + stats outputs
│   │   ├── report_schema.py              # Schema constants and contract checks for the report
│   │   ├── gather.py                     # Gather per-replicate report summaries into report_summary.tsv
│   │   ├── stats/                        # Per-concern stats package
│   │   │   ├── runner.py                 # Stats orchestrator
│   │   │   ├── correlations.py           # Pairwise correlations, parent-offspring regressions, h² estimators
│   │   │   ├── tetrachoric.py            # Tetrachoric correlation primitives
│   │   │   ├── pedigree.py               # Family size and parent-presence summaries
│   │   │   ├── incidence.py              # Prevalence, mortality, cumulative incidence, joint affection
│   │   │   ├── censoring.py              # Censoring window, confusion matrix, cascade, person-years
│   │   │   ├── effective_size.py         # Per-rep effective population size summary
│   │   │   └── sampling.py               # Downsampling for scatter/histogram plot inputs
│   │   └── validate/                     # Validation package (one module per check family)
│   │       ├── runner.py                 # Validation orchestrator and CLI
│   │       ├── structural.py, statistical.py
│   │       ├── heritability.py, twins.py, half_sibs.py, consanguinity.py
│   │       ├── assortative_mating.py, am_equilibrium.py, am_relatedness.py
│   │       ├── population.py, effective_size.py
│   │       └── _common.py                # Shared validation helpers
│   └── plotting/
│       ├── plot_utils.py                 # Shared plotting helpers (finalize_plot, violin, heatmap)
│       ├── plot_style.py                 # Color palette and shared style tokens
│       ├── plot_phenotype.py             # Phenotype plot orchestrator + CLI
│       ├── plot_distributions.py         # Mortality, age-at-onset, cumulative incidence
│       ├── plot_liability.py             # Joint liability, violin, affection plots
│       ├── plot_correlations.py          # Tetrachoric + parent-offspring correlations
│       ├── plot_heritability.py          # Heritability plots (by generation, sex, etc.)
│       ├── plot_pedigree_counts.py       # Pedigree relationship pair counts diagram
│       ├── plot_effective_size.py        # Effective-size atlas plots
│       ├── plot_am_equilibrium.py        # Assortative-mating equilibrium plot
│       ├── plot_validation.py            # Validation summary plots
│       ├── compare_scenarios.py          # Cross-scenario comparison plots for the Examples pages
│       ├── atlas_manifest.py             # Atlas registry: ordered plots and section breaks
│       ├── render_atlas.py               # Format-dispatching seam for atlas rendering
│       ├── plot_atlas_html.py            # Single-page HTML atlas (the default artifact)
│       ├── plot_atlas.py                 # Multi-page PDF atlas (on demand)
│       ├── stats_report.py               # Adapter from the curated report to the flat plotting view
│       ├── plot_pipeline.py              # Pipeline DAG diagram
│       └── plot_table1.py                # Epidemiological Table 1
│
├── fitACE/                              # Model-fitting monorepo checkout (gitignored, see Repo Map)
│
├── workflow/
│   ├── common.py                         # Shared helpers (get_param, get_folder, etc.)
│   ├── envs/                             # Conda env specs for named external tools
│   ├── scripts/simace/                   # Thin script wrappers called by the rules
│   └── rules/simace/                     # Modular Snakemake rule files
│       ├── targets.smk                   # Target rules: all, scenario, per-stage sentinels
│       ├── simulate.smk                  # Pedigree simulation
│       ├── phenotype.smk                 # Phenotype + censor rules
│       ├── ascertainment.smk             # Unified dropout + case-weighted N_sample (per ADR 0001)
│       ├── validate.smk, stats.smk       # Validation and statistics
│       ├── analyze.smk                   # Curated report.yaml
│       ├── effective_size.smk            # Effective population size
│       ├── examples.smk                  # Example-page targets (minimal-ace, with-c, ...)
│       ├── tskit_preprocess.smk          # tskit founder preprocessing for gene-drop
│       ├── tstrait_phenotype.smk         # tstrait-based phenotype models
│       ├── genotype_drop.smk             # Gene-drop pipeline (tskit-based recombination)
│       └── utils.smk                     # Shared Snakemake utilities
├── scripts/                             # Standalone helper scripts (regen_rulegraph.sh, bench_*.py, sweep generators, etc.)
├── tools/                               # Maintenance tooling (release.py, family typecheck)
├── tests/                               # Mirrors simace/ sub-package structure
├── docs/                                # MkDocs sources, ADRs, plans
├── external/                            # Reference implementations and the pedigree-graph / pedsum checkouts (gitignored)
├── results/{folder}/{scenario}/         # Per-scenario simulation outputs
├── logs/{folder}/{scenario}/            # Log files
└── benchmarks/{folder}/{scenario}/      # Runtime and memory benchmarks
```

## Repo map

Five repos, all under `rwaples/` on GitHub. simACE is the umbrella working
directory; fitACE and its nested fitACE_epimight are checkouts inside it
(gitignored from simACE — no submodules). Since ADR 0017, fitACE is a
monorepo: the method packages, the `ace_iter_reml` C++ source, and the
`tetraher_simace` LDAK fork live in subdirectories of `./fitACE/` rather than
in separate repos.

| Repo | Visibility | Local path | Role |
|---|---|---|---|
| [`simACE`](https://github.com/rwaples/simACE) | public | `.` (this repo) | Simulation pipeline: simulate → phenotype → censor → ascertainment → validate → stats → plot |
| [`fitACE`](https://github.com/rwaples/fitACE) | private | `./fitACE/` | Model-fitting monorepo: core + Snakemake orchestrator + method packages in `fitACE_<x>/` subdirs (PCGC, iter/sparse REML + the `ace_iter_reml` C++ source under `fitACE_iter_reml/`, TetraHer + the `tetraher_simace` LDAK fork, PA-FGRS, Stan, frailty). Consumes simACE outputs. |
| [`fitACE_epimight`](https://github.com/rwaples/fitACE_epimight) | private | `./fitACE/fitACE_epimight/` | EPIMIGHT integration: long-form input emitter, R driver, Snakemake rules, atlas/bias plotting. Its own repo, tracking the BioPsyk/epimight R upstream; included by `fitACE/Snakefile`. |
| [`pedigree-graph`](https://github.com/rwaples/pedigree-graph) | public | `./external/pedigree-graph/` | Sparse-matrix pedigree relationship extraction and kinship computation. |
| [`pedsum`](https://github.com/rwaples/pedsum) | public | `./external/pedsum/` | Pedigree summary CLI: structure, relatedness, inbreeding, Ne estimators. Built on `pedigree-graph`. |

Each nested repo has its own `origin` wired to the matching GitHub repo.
Build artifacts (`build-fp*/`, `ldak6.2.simace`, Stan binaries) are
gitignored — rebuild from source.
