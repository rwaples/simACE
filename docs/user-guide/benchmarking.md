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
1 when wall time or sampled peak RSS regresses by more than 5%. Change the gates
with `--time-threshold-percent` and `--memory-threshold-percent`.

The command returns exit status 2 when critical provenance differs. This
includes the CPU, lock file, scenario inputs, cache policy, `--jobs` and thread
budgets, command, and sampling interval. Use `--allow-incompatible` only when
you intend to compare unlike environments. The command prints every mismatch.

## Interpret memory metrics

The benchmark reports three different memory measurements:

- GNU time maximum RSS is a process high-water value. It does not sum processes
  that are resident at the same time.
- Maximum individual RSS is the largest process observed by the 250 ms sampler.
- Peak summed RSS is the largest concurrent sum across the benchmark process
  group. Use this value to size the machine for the whole pipeline run.

The sampler identifies the launched process group through Linux `/proc`. It
does not select processes by name, so unrelated host work
does not enter the measurement. It attributes each process to the `python -m simace <stage>`
process it descends from. Short spikes can fall between samples. Each stage's `timing.tsv` peak comes from `wait4` and cannot miss one.

The runtime and memory pages in the validation atlas are separate. They read
only the `simulate` row of each replicate's `timing.tsv`; they do not measure the phenotype,
censoring, ascertainment, analysis, or plotting stages. Statistical validation
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
