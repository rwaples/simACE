# ADR 0020: A standalone `simace` CLI replaces Snakemake

Date: 2026-09-25

Status: accepted. [ADR 0021](0021-two-canonical-replicate-parquets.md)
replaced the replicate Parquet output contract and the per-rep stage list
described here, implemented 2026-09-29: a rep is `simulate`, `cohort`,
`analyze`.

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
- `run --format pdf` and `gather --format pdf` build the PDF atlas as a
  further stage (`atlas-pdf`, its own log and timing row) beside the HTML
  atlas, which is always built (ADR 0010).
- `gather` summarizes the configured folder: every rep of every runnable
  scenario in `config/{folder}.yaml`, naming each rep it leaves out and its
  state, with a per-scenario `n of m reps` line. `--all` gathers by disk
  instead, for archived folders the config no longer lists.
- A scenario's plots have their own state. `plots/plots.yaml` fingerprints
  the `run.yaml` of every rep the plot pass was built from and the atlas
  files it wrote. `ls`, `show`, and the run summary report the plots as
  current, stale with the reason, or absent. `--no-plots` and a partial
  `--rep` run still exit 0 when their reps succeed; the reported plot state
  is what tells a reader the atlas is not of the current reps.

## Consequences

- The fitACE compatibility contract is unchanged: `pedigree.parquet`,
  `trait.parquet`, `report.yaml`, and `params.yaml` keep their paths under
  `results/{folder}/{scenario}/rep{rep}/`, and `params.yaml` keeps every key.
  `params.yaml` gains `G_pheno`, which fitACE's dev-grid summaries had been
  defaulting to 3.
- `trait.raw.parquet` and `plotting_sample.parquet` are now durable, so a
  standalone `simace plot` works on any finished scenario.
- Every stage runs with OpenMP/BLAS pinned to one thread, as Snakemake's
  `threads: 1` rules and `--cores 1` runs had it. Numba and polars get
  every core with `--jobs 1` and one thread each with `--jobs N > 1`.
  pedigree-graph defaults to one thread, so `run` sets
  `PEDIGREE_GRAPH_THREADS` to the CPUs the process may use divided by
  `--jobs` (at least 1) unless the caller already set it. Measured on 12
  cores, splitting the cores this way made analyze 29% to 56% faster than
  one thread per rep at `--jobs 2` and `--jobs 3` (baseline100K, bench1M),
  with byte-identical outputs and analyze peak RSS up 1% to 7%. Measured at
  `baseline100K`, a single OpenMP/BLAS thread is as fast as four or five for
  simulate and analyze and about 6% faster for plot.
- Lost relative to Snakemake: cluster submission, per-job memory estimates,
  and per-stage selective rebuilds. `simace show` prints each stage's peak
  RSS and median wall time from the complete reps so `--jobs` and
  `--max-memory` can be sized by hand.
- `--max-memory SIZE` caps each stage's resident memory. `run` polls the
  `VmRSS` of the child and every descendant in `/proc` every 0.1 s, sums
  them, and kills the whole process tree once the sum is over. The sum
  covers `simace plot`'s worker processes; it counts shared pages once
  per process, so it overstates a tree's footprint by its shared
  libraries. The same samples set `max_rss_mb` in `timing.tsv` when they
  exceed the stage process's own `wait4` peak. Kernel limits were measured
  and rejected: at `small_test`, a stage's peak virtual size (`RLIMIT_AS`)
  was 15 to 25 times its peak RSS and its data segment (`RLIMIT_DATA`) 2
  to 4 times, so either limit would kill stages that fit. The cap is per
  stage, not per run, and a spike shorter than the poll interval can pass
  it.

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

## Amendment (2026-09-30): `--until` and partial reps

Consumers that read only `pedigree.parquet` and `cohort.parquet` (the
fitACE_epimight CI testbed) paid for `analyze`, the costliest stage on small
reps: on `citb_K05_x50_c80` it takes 16.9 s of a 25.9 s rep, and its outputs
went unread (simACE #28).

- `simace run --until STAGE` stops each rep after `STAGE`. Resume
  granularity becomes the stage prefix: `run.yaml` is written once every
  stage up to `STAGE` exits 0, and its `stages` list records exactly those
  stages. Each prefix is still all or nothing.
- A rep is checked against the stages its `run.yaml` records: their config
  keys (`rep_param_keys`; the `params.yaml` keys count only once `analyze`,
  which reads that file, has run) and their outputs. A rep complete through
  fewer stages than a run asks for is `partial`. The run resumes it at the
  first missing stage, keeping the earlier outputs, rewriting `params.yaml`
  from the current config, and adding to `timing.tsv`. A change to a key
  only a missing stage reads (`max_degree`) does not refuse a partial rep.
- Plots, `simace gather`, and `simace show`'s timing count only reps
  complete through every stage. `--until` before `analyze` skips the plot
  pass, and `simace ls` lists partial reps with the stage they reached.

## Amendment (2026-09-30): exact memory meters through cgroup v2

`timing.tsv`'s `max_rss_mb` was the larger of `wait4`'s `ru_maxrss` and the
summed `VmRSS` of the stage's tree sampled every 0.1 s. Which of the two it
held changed from run to run, the sum counted a shared page once per
process, and a spike shorter than a poll went unrecorded.

- Each stage runs in its own child cgroup of a delegated cgroup v2 root
  (`simace/cli/cgroups.py`). The root is one transient systemd scope per
  `simace run`, started with `Delegate=yes`, or the directory
  `SIMACE_CGROUP_ROOT` names. A shell shim moves the stage into its cgroup
  before `exec`, because memory is charged to the cgroup a process is in
  when it allocates and does not move with the process later.
- `timing.tsv` gains `tree_peak_mb`, the cgroup's `memory.peak`: the exact
  high-water mark of the stage and every descendant, a shared page counted
  once, page cache and kernel memory included. `max_rss_mb` is `ru_maxrss`
  alone. Without a delegated cgroup `tree_peak_mb` is empty, so no column
  holds two meters.
- `--max-memory` becomes `memory.max` on the stage's cgroup, with
  `memory.swap.max` at 0 and `memory.oom.group` at 1, so the kernel kills
  the stage and its workers together. The Consequences above rejected
  kernel limits after measuring `RLIMIT_AS` and `RLIMIT_DATA`, which bound
  address space per process, most of it never resident. `memory.max` bounds
  the pages actually charged to the whole tree. Page cache counts toward it
  and is reclaimed before an OOM kill, so a stage near the cap slows under
  reclaim instead of dying at once. Without a delegated cgroup the `/proc`
  poll still enforces the cap.
- Opening the scope took under 0.1 s here, and removing a stage's cgroup
  (kill whatever is left, wait until it is empty, `rmdir`) takes
  milliseconds.

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
