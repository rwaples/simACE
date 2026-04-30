# Project Structure

## Repository layout

```
simACE/
├── Snakefile                            # Root entry point (no -s flag needed)
├── config/
│   ├── _default.yaml                    # Default simulation parameters
│   └── {folder}.yaml                    # Per-folder scenario definitions
│
├── simace/                              # Simulation package (pip install -e .)
│   ├── __init__.py                       # Package init
│   ├── config.py                         # Config loading and parameter coercion
│   ├── core/                             # Shared infrastructure
│   │   ├── cli_base.py                   # Shared CLI boilerplate (add_logging_args, init_logging)
│   │   ├── compute_hazard_terms.py       # Baseline hazard functions (Weibull, Gompertz, etc.)
│   │   ├── numerics.py                   # safe_corrcoef, safe_linregress, scipy.special wrappers
│   │   ├── parquet.py                    # save_parquet and parquet I/O helpers
│   │   ├── parquet_to_tsv.py             # `simace-parquet-to-tsv` CLI entry point
│   │   ├── pedigree_graph.py             # Sparse-matrix pedigree relationship extraction
│   │   ├── relationships.py              # Relationship registry (REL_REGISTRY, PAIR_KINSHIP, PAIR_TYPES)
│   │   ├── schema.py                     # Pipeline data-schema contracts (phenotype → censor → sample handoff)
│   │   └── yaml_io.py                    # load_yaml, dump_yaml helpers
│   ├── simulation/
│   │   ├── simulate.py                   # Pedigree simulation (mating, reproduce, run_simulation)
│   │   └── mate_correlation.py           # Assortative-mating helpers
│   ├── phenotyping/
│   │   ├── phenotype.py                  # PhenotypeModel ABC + frailty / cure-frailty / ADuLT / first-passage models
│   │   ├── threshold.py                  # Liability-threshold binary phenotype
│   │   └── hazards.py                    # Baseline-hazard registry (Weibull, exponential, Gompertz, ...)
│   ├── censoring/
│   │   └── censor.py                     # Age-window and competing-risk death censoring
│   ├── sampling/
│   │   ├── dropout.py                    # Pedigree dropout (random individual removal)
│   │   └── sample.py                     # Subsampling with case ascertainment bias
│   ├── analysis/
│   │   ├── stats/                        # Per-concern stats package (split from old stats.py)
│   │   │   ├── runner.py                 # Orchestrator (computes phenotype_stats.yaml)
│   │   │   ├── correlations.py           # Pairwise correlations and parent-offspring regressions
│   │   │   ├── tetrachoric.py            # Tetrachoric correlations + Falconer h²
│   │   │   ├── pedigree.py               # Pair counts and family-structure stats
│   │   │   ├── incidence.py              # Cumulative incidence curves
│   │   │   ├── censoring.py              # Censoring confusion / cascade
│   │   │   └── sampling.py               # Sample-summary statistics
│   │   ├── validate.py                   # Structural + statistical validation
│   │   └── gather.py                     # Gather validation results into validation_summary.tsv
│   └── plotting/
│       ├── plot_utils.py                 # Shared plotting helpers (finalize_plot, violin, heatmap)
│       ├── plot_style.py                 # Color palette and shared style tokens
│       ├── plot_phenotype.py             # Phenotype plot orchestrator + CLI
│       ├── plot_distributions.py         # Mortality, age-at-onset, cumulative incidence
│       ├── plot_liability.py             # Joint liability, violin, affection plots
│       ├── plot_correlations.py          # Tetrachoric + parent-offspring correlations
│       ├── plot_heritability.py          # Heritability plots (by generation, sex, etc.)
│       ├── plot_pedigree_counts.py       # Pedigree relationship pair counts diagram
│       ├── plot_validation.py            # Validation summary plots
│       ├── compare_scenarios.py          # Cross-scenario comparison plots
│       ├── plot_atlas.py                 # Multi-page PDF atlas with figure captions
│       ├── atlas_manifest.py             # Atlas registry + dispatch
│       ├── plot_pipeline.py              # Pipeline DAG diagram
│       └── plot_table1.py                # Epidemiological Table 1
│
├── fitACE/                              # Sister repo with model fitting (gitignored, see Repo Map)
│
├── workflow/
│   ├── common.py                         # Shared helpers (get_param, get_folder, etc.)
│   └── rules/simace/                     # Modular Snakemake rule files
│       ├── targets.smk                   # Target rules: all, scenario, per-stage sentinels
│       ├── simulate.smk, dropout.smk     # Pedigree simulation and dropout
│       ├── phenotype.smk, sample.smk     # Phenotyping and sampling
│       ├── validate.smk, stats.smk       # Validation and statistics
│       ├── examples.smk                  # Example-page targets (minimal-ace, with-c, ...)
│       ├── tskit_preprocess.smk          # tskit founder preprocessing for gene-drop
│       ├── tstrait_phenotype.smk         # tstrait-based phenotype models
│       ├── genotype_drop.smk             # Gene-drop pipeline (tskit-based recombination)
│       └── utils.smk                     # Shared Snakemake utilities
├── scripts/                             # Standalone helper scripts (regen_rulegraph.sh, run_epimight.py, bench_*.py, etc.)
├── tests/                               # Mirrors simace/ sub-package structure
├── external/                            # Reference implementations (gitignored)
├── results/{folder}/{scenario}/         # Per-scenario simulation outputs
├── logs/{folder}/{scenario}/            # Log files
└── benchmarks/{folder}/{scenario}/      # Runtime and memory benchmarks
```

## Repo map

simACE is the umbrella working directory; model fitting lives in nested
checkouts of sister repos (gitignored from simACE — no submodules):

| Repo | Visibility | Local path | Role |
|---|---|---|---|
| [`simACE`](https://github.com/rwaples/simACE) | public | `.` (this repo) | Simulation pipeline: simulate → phenotype → censor → sample → validate → stats → plot |
| [`fitACE`](https://github.com/rwaples/fitACE) | private | `./fitACE/` | Model fitting (EPIMIGHT, PA-FGRS, sparseREML, iter_reml, Stan, PCGC). Consumes simACE outputs. |
| [`ace_iter_reml`](https://github.com/rwaples/ace_iter_reml) | private | `./fitACE/fitace/ace_iter_reml/` | C++ PCG-AI-REML binary. |
| [`tetraher_simace`](https://github.com/rwaples/tetraher_simace) | private | `./external/tetraher_simace/` | Fork of LDAK 6.2 (grouping + warm-start + OMP opt-in). |

Each nested repo has its own `origin` wired to the matching GitHub repo.
Build artifacts (`build-fp*/`, `ldak6.2.simace`, Stan binaries) are
gitignored — rebuild from source.
