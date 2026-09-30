# Benchmark pipeline performance

The pipeline benchmark records repeated `simace run --force` runs with enough provenance to
compare two results. Each invocation writes a new directory. It never appends to
an earlier result.

## Run a benchmark

Run the small profile to check the harness:

```bash
pixi run python -m tools.benchmark run --profile smoke
```

Run the release profile to measure `baseline10K`, `baseline100K`, and
`baseline1M`:

```bash
pixi run python -m tools.benchmark run --profile release
```

Both profiles run one replicate at a time (`--jobs 1`) and three measured repetitions by default. Warm
mode first runs an unmeasured pass with a new run-local Numba cache. To measure
cold starts, give every measured execution an empty cache:

```bash
pixi run python -m tools.benchmark run --profile smoke --cache-mode cold
```

To benchmark a custom set, name its folder and scenarios:

```bash
pixi run python -m tools.benchmark run \
  --folder bench_scale \
  --scenarios bench100K bench1M bench10M
```

The command prints the new `bench-logs/<run-id>/` path when it finishes. Use
`--out` only with a path that does not exist. The command rejects existing paths
and has no resume or overwrite mode.

## Read a result

Print the stored medians and observed ranges:

```bash
pixi run python -m tools.benchmark summarize bench-logs/<run-id>
```

Each run directory contains two schema-versioned documents:

- `manifest.json` records the commit and dirty state, input hashes, host and tool
  versions, the `--jobs` value and thread budgets, the command, cache policy, and scenario order.
- `results.json` records every execution, its raw-artifact paths, and the
  per-scenario and per-stage summaries.

The `runs/` directory keeps the `simace run` log, GNU time output, process samples,
and copies of each replicate's `timing.tsv` for each execution. `cache/` contains
the run-local Numba cache.

## Compare two runs

Compare a candidate with a baseline:

```bash
pixi run python -m tools.benchmark compare \
  bench-logs/<baseline-run-id> \
  bench-logs/<candidate-run-id>
```

The command compares matching scenario and stage medians. It returns exit status
1 when wall time or peak memory regresses by more than 5%. Change the gates
with `--time-threshold-percent` and `--memory-threshold-percent`.

The command returns exit status 2 when critical provenance differs. This
includes the CPU, lock file, scenario inputs, cache policy, `--jobs` and thread
budgets, command, and sampling interval. Use `--allow-incompatible` only when
you intend to compare unlike environments. The command prints every mismatch.

## Interpret memory metrics

The benchmark records these memory measurements for each execution:

- `cgroup_peak_kb` is `memory.peak` of a fresh delegated cgroup that holds
  the whole `simace run` process and every stage cgroup it creates, which
  adopt it through `SIMACE_CGROUP_ROOT`. It is the exact high-water mark of
  the whole run, with a shared page counted once. It includes page cache and
  kernel memory, and a page is charged to the cgroup that first touched it.
  It is null when no delegated cgroup is available; see
  [Stage timing](output-structure.md#stage-timing).
- `gnu_time_max_rss_kb` is GNU time's maximum RSS, the largest single
  process's high-water mark. It does not sum processes that are resident at
  the same time.
- `max_individual_rss_kb` is the largest process observed by the 250 ms
  sampler.
- `peak_summed_rss_kb` is the largest concurrent sum of resident memory
  across the benchmark process group, sampled every 250 ms. It counts a
  shared page once per process, and short spikes can fall between samples.

Each stage row also carries `max_rss_mb` and `tree_peak_mb` from its
`timing.tsv`.

Each summary's `peak_rss_kb`, which the memory gate reads, names its source
in `memory_meter`. It is `cgroup` when every measured execution in the
summary has a cgroup figure: `cgroup_peak_kb` for the whole run, and each
stage's largest `tree_peak_mb` for a stage row. Otherwise it is `sampled`:
`peak_summed_rss_kb` for the whole run, and the stage's summed sampled
resident memory for a stage row. The two meters measure different things, so
`compare` does not compare memory across them. It prints a `NOT COMPARED`
line for each such row and gates only wall time there. `tools/bench_plot.py`
refuses a run whose summaries mix meters.

The sampler identifies the launched process group through Linux `/proc`. It
does not select processes by name, so unrelated host work
does not enter the measurement. It attributes each process to the `python -m simace <stage>`
process it descends from. It still records the `processes.jsonl` time series
and CPU frequency when a cgroup is available.

The runtime and memory pages in the validation atlas are separate. They read
only the `simulate` row of each replicate's `timing.tsv`; they do not measure the `cohort`,
`analyze`, or plotting stages. Statistical validation
and fitACE estimator bias or RMSE studies do not measure computational
performance.

## Run the manual workflow

The `Pipeline benchmark` GitHub Actions workflow has only a manual dispatch. It
runs the smoke profile and uploads the full run directory even when the command
fails. It does not gate pull requests or compare timing on shared runners.

## Reproduce from a fresh clone

Install the committed lock before running either profile:

```bash
git clone https://github.com/rwaples/simACE.git
cd simACE
pixi install --locked
pixi run python -m tools.benchmark run --profile smoke
```
