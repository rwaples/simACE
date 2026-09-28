# ADR 0020: A standalone `simace` CLI replaces Snakemake

Date: 2026-09-25

Status: accepted

## Context

simACE ran its pipeline through Snakemake: rule files under `workflow/rules/`,
thin script wrappers under `workflow/scripts/`, and a signature-introspection
adapter (`simace/core/snakemake_adapter.py`) between them. The stage modules
already had flag-driven CLIs with explicit paths, and config resolution
already lived in `simace.config` with no Snakemake dependency. What Snakemake
still owned was the results-path convention, the per-rep seed offset, the
order of stages, and incremental rebuilds.

That split cost more than it gave. Every stage had two entry points (the
wrapper and the CLI) that could drift. The rule files duplicated each stage's
parameter list a third time. Incrementality was by file timestamp, so a
config change did not invalidate anything, and `temp()` deleted the plotting
sample a later standalone plot run needed. fitACE reads simACE's outputs
through its own Snakemake and did not depend on simACE's rules.

## Decision

- `simace <command>` is the one entry point. Stage subcommands (`simulate`,
  `phenotype`, `censor`, `ascertain`, `analyze`, `plot`, `atlas`, and the
  debug and utility commands) take explicit path flags and one flag per
  domain keyword argument. They never read YAML config, scenario names, or
  the results layout. The ten `simace-*` console scripts are retired.
- `simace run <scenario>` is the only config reader. It resolves the
  scenario through `simace.config`, applies `seed + rep - 1`, and imposes
  `results/{folder}/{scenario}/rep{rep}/` through one `Layout` object
  (`simace/cli/layout.py`). `simace gather <folder>` builds the folder
  summary and validation atlas.
- **Resume granularity is the replicate.** `run` writes a rep's `run.yaml`
  manifest after every stage of the rep exits 0. It records the scenario,
  rep, and seed, the values of the config keys the rep's stages and
  `params.yaml` read (`REP_PARAM_KEYS`), and the stage list. A rep is
  complete only when that manifest matches all of these and every output
  the rep declares (`REP_OUTPUTS`) exists, so a manifest copied from another
  rep or a deleted artifact never passes for finished work. A rerun skips a
  complete rep, recomputes a rep with no manifest or a missing output from
  scratch, and refuses a rep whose manifest differs unless `--force`. There
  is no per-stage resume, no stage windows, and no intermediate deletion.
  Plots and the atlas are rebuilt on every run.
  The manifest records the simace version, but a version change never makes
  a rep stale: comparing versions would invalidate every rep on any commit.
  `simace ls` flags reps built by another version instead.
- One `simace run` per scenario at a time. `run` holds an exclusive `flock`
  on `results/{folder}/{scenario}/.run.lock`. Stage children inherit the lock
  descriptor, so a killed orchestrator cannot start a second run while its
  child is still writing. The kernel releases the lock when the last child
  exits. Parallelism within a scenario is `--jobs`, not concurrent invocations.
- Every stage runs as its own `python -m simace <stage>` subprocess, and
  its outputs are published atomically (a unique `<path>.<random>.tmp`
  per writer, then `os.replace`).
  `run` records each child's wall time and `wait4` peak RSS in the rep's
  `timing.tsv`, which replaces Snakemake's benchmark TSVs for `gather` and
  `tools/benchmark`.
- Structured parameters cross the subprocess boundary losslessly. Maps keyed
  by generation use JSON with integer-key coercion; the phenotype parameter
  dicts and the atlas metadata use YAML flow mappings, which keep nested
  integer keys (per-generation and sex-specific prevalence). A test runs
  every configured scenario's parameters through each stage's argv and
  requires the domain function to receive exactly the config values.
- Gene-drop scenarios (`use_gene_drop`, `drop_from`) are not run by
  `simace run`; their scripts moved to `scripts/gene_drop/`. The example
  comparison scripts moved to `scripts/examples/`.

### Amendment 2026-09-28: what "complete" covers

- The run manifest also records each output's size and nanosecond mtime.
  Every output is published by renaming a fresh temporary, so a stage rerun
  by hand (the documented debugging path) rewrites the file and the replicate
  becomes incomplete: `run` recomputes it, `gather` skips it, `ls` names the
  file. A manifest without the block is stale and refused until `--force`.

## Consequences

- The fitACE compatibility contract is unchanged: `pedigree.parquet`,
  `trait.parquet`, `report.yaml`, and `params.yaml` keep their paths under
  `results/{folder}/{scenario}/rep{rep}/`, and `params.yaml` keeps every key.
  `params.yaml` gains `G_pheno`, which fitACE's dev-grid summaries had been
  defaulting to 3.
- `trait.raw.parquet` and `plotting_sample.parquet` are now durable, so a
  standalone `simace plot` works on any finished scenario.
- Every stage runs with OpenMP/BLAS pinned to one thread, as Snakemake's
  `threads: 1` rules and `--cores 1` runs had it. Numba, polars, and
  pedigree-graph's Rust pool get every core with `--jobs 1` and one thread
  each with `--jobs N > 1`. pedigree-graph defaults to one thread, so `run`
  sets `PEDIGREE_GRAPH_THREADS` to the CPUs the process may use unless the
  caller already set it. Measured at
  `baseline100K`, a single OpenMP/BLAS thread is as fast as four or five for
  simulate and analyze and about 6% faster for plot.
- Lost relative to Snakemake: cluster submission, per-job memory estimates,
  and per-stage selective rebuilds. `simace show` prints each stage's peak
  RSS and median wall time from the complete reps so `--jobs` and
  `--max-memory` can be sized by hand.
- `--max-memory SIZE` caps each stage process's resident memory. `run`
  polls the child's `VmRSS` in `/proc` every 0.1 s and kills it once it is
  over. Kernel limits were measured and rejected: at `small_test`, a stage's
  peak virtual size (`RLIMIT_AS`) was 15 to 25 times its peak RSS and its
  data segment (`RLIMIT_DATA`) 2 to 4 times, so either limit would kill
  stages that fit. The cap is per stage, not per run, and a spike shorter
  than the poll interval can pass it.

## Amendment (2026-09-28): folder targets, config-aware gather, source ref

Decided after reviewing how the CLI would be used day to day.

- `simace run` takes any number of targets. A target is a scenario name,
  or else a folder name standing for every runnable scenario whose
  `folder` it is. All reps of all targets share one `--jobs` pool; every
  scenario's lock is taken before anything starts; plots are drawn after
  the pool drains, for each scenario whose reps are all complete. This
  takes back the cross-scenario scheduling the original decision gave up.
  `--rep` accepts ranges and needs exactly one scenario. `--no-plots`
  skips the plot pass.
- `simace gather` reads the config when the directory exists and skips
  reps that are stale or incomplete under it, as it already skipped reps
  without a manifest. "Only `run` reads config" holds for stage
  subcommands, not for the folder-level tools.
- `run.yaml` records `source`, the `git describe --tags --always --dirty`
  of the checkout that built the rep (null for a wheel install). It never
  makes a rep stale; `simace ls` shows it when it differs from the running
  checkout, since in an editable install the version alone does not move
  between `pixi install`s.
- Every command finds the repository root by walking up from the current
  directory to `config/_default.yaml`; paths stay relative when the current
  directory is the root.

## Verification

Recorded in `plans/standalone-simace-cli-v2.md` §6 at implementation time.

- Parity: `simace run small_test` against a fresh
  `snakemake --cores 1 results/test/small_test/scenario.done` gave equal
  `pedigree.full.parquet`, `pedigree.parquet`, `trait.full.parquet`,
  `trait.parquet`, `params.yaml`, `report.yaml`, and `plot_payload.yaml` for
  all three reps, and the `test` folder's `report_summary.tsv` matched on
  every non-timing column.
- Performance: `tools.benchmark`, three measured runs, Snakemake
  `--cores 1` against `simace run --jobs 1`. Median whole-pipeline wall time
  fell 9.5% for `small_test` and 4.5% for `baseline100K`, no stage got slower,
  and whole-run peak RSS fell about 1%. Per-stage peak RSS measured exactly
  with GNU time matched the Snakemake-era stage code.
