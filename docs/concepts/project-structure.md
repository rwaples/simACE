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
│   ├── __init__.py                       # setup_logging(), _snakemake_tag()
│   ├── core/                             # Shared infrastructure
│   │   ├── utils.py                      # save_parquet, safe_corrcoef, yaml_loader, PAIR_TYPES, etc.
│   │   ├── cli_base.py                   # Shared CLI boilerplate (add_logging_args, init_logging)
│   │   ├── pedigree_graph.py             # Sparse-matrix pedigree relationship extraction
│   │   ├── compute_hazard_terms.py       # Baseline hazard functions (Weibull, Gompertz, etc.)
│   │   ├── weibull_mle.py                # Gauss-Hermite MLE for Weibull frailty correlation
│   │   └── _numba_utils.py               # Shared Numba kernels
│   ├── simulation/
│   │   └── simulate.py                   # Pedigree simulation (mating, reproduce, run_simulation)
│   ├── phenotyping/
│   │   ├── phenotype.py                  # Frailty, cure-frailty, ADuLT phenotype models
│   │   └── threshold.py                  # Liability-threshold binary phenotype
│   ├── censoring/
│   │   └── censor.py                     # Age-window and competing-risk death censoring
│   ├── sampling/
│   │   ├── dropout.py                    # Pedigree dropout (random individual removal)
│   │   └── sample.py                     # Subsampling with case ascertainment bias
│   ├── analysis/
│   │   ├── stats.py                      # Tetrachoric correlations, heritability, pair counts
│   │   ├── simple_ltm_stats.py           # Simple LTM phenotype statistics
│   │   ├── validate.py                   # Structural + statistical validation
│   │   ├── ltm_falconer.py               # Falconer h² from tetrachoric correlations
│   │   ├── survival_stats.py             # Pairwise frailty correlation wrappers
│   │   └── gather.py                     # Gather validation results into TSV
│   └── plotting/
│       ├── plot_utils.py                 # Shared plotting helpers (finalize_plot, violin, heatmap)
│       ├── plot_phenotype.py             # Phenotype plot orchestrator + CLI
│       ├── plot_distributions.py         # Mortality, age-at-onset, cumulative incidence
│       ├── plot_liability.py             # Joint liability, violin, affection plots
│       ├── plot_correlations.py          # Tetrachoric + parent-offspring correlations
│       ├── plot_pedigree_counts.py       # Pedigree relationship pair counts diagram
│       ├── plot_simple_ltm.py            # Simple LTM phenotype plots
│       ├── plot_validation.py            # Validation summary plots
│       ├── plot_atlas.py                 # Multi-page PDF atlas with figure captions
│       ├── plot_pipeline.py              # Pipeline DAG diagram
│       └── plot_table1.py                # Epidemiological Table 1
│
├── fitACE/                              # Sister repo with model fitting (gitignored, see Repo Map)
│
├── workflow/
│   ├── common.py                         # Shared helpers (get_param, get_folder, etc.)
│   ├── rules/                            # Modular Snakemake rule files
│   │   ├── targets.smk                   # Target rules: all, scenario, per-stage sentinels
│   │   ├── simulate.smk, dropout.smk     # Pedigree simulation and dropout
│   │   ├── phenotype.smk, sample.smk     # Phenotyping and sampling
│   │   ├── validate.smk, stats.smk       # Validation and statistics
│   │   ├── epimight.smk, pafgrs.smk      # fitACE pipeline rules
│   │   └── epimight_bias.smk             # EPIMIGHT bias analysis
│   └── scripts/                          # Snakemake script wrappers
├── scripts/                             # Standalone helper scripts (e.g., regen_rulegraph.sh)
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
