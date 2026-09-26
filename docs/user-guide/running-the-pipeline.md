# Running the pipeline

Run every command from the repository root. `simace run` reads the
scenario files in `config/` and writes under `results/` and `logs/`.

## Run one scenario

```bash
pixi run simace run baseline10K
```

`simace run` computes every replicate of the scenario, then draws the
scenario's plots and its HTML atlas. Each replicate goes through the stages
in this order: simulate, phenotype, censor, ascertain, analyze. Each stage
reads the files the previous one wrote.

| Flag | Effect |
|---|---|
| `--rep 2 3` | Compute only these replicates |
| `--jobs 3` | Compute three replicates at once. Each stage process is limited to one thread |
| `--dry-run` | Print the exact stage commands and write nothing |
| `--force` | Recompute the requested replicates even when they are complete |
| `--fail-fast` | Stop starting new replicates after the first failure |
| `--max-memory 8G` | Kill any stage process whose resident memory goes over 8 GiB, which fails its replicate. The cap is per stage, so `--jobs 3` can use up to three times it |
| `--format pdf` | Write `plots/atlas.pdf` instead of `plots/atlas.html` |
| `--results DIR`, `--logs DIR`, `--config-dir DIR` | Use other roots than `results/`, `logs/`, `config/` |

To run every scenario in a folder, loop over the names:

```bash
for s in baseline10K baseline100K; do pixi run simace run "$s"; done
```

## Preview the run

To see the commands without running them, add `--dry-run`:

```bash
pixi run simace run baseline10K --dry-run
```

Before a run that takes more than a few minutes, preview it.

## Summarize a folder

```bash
pixi run simace gather base
```

`simace gather` reads every `report.yaml` under `results/base/`, writes
`results/base/report_summary.tsv`, and draws the validation plots and atlas in
`results/base/plots/`. Pass `--format pdf` for the PDF atlas.

## Rerun and resume

`simace run` writes a replicate's `run.yaml` only after every stage of the
replicate succeeds. It records the scenario, replicate number, seed, and
parameters the replicate was computed from. A replicate is complete when its
`run.yaml` matches the current config and every output file of the
replicate exists.

When you rerun a scenario, each requested replicate is handled as a whole:

| State | What `simace run` does |
|---|---|
| Complete | Skips the replicate |
| No `run.yaml`, for example after an interruption | Recomputes the replicate from the first stage |
| `run.yaml` matches, but an output file is missing (`incomplete`) | Recomputes the replicate from the first stage and names the missing files |
| `run.yaml` differs from the current config, or belongs to another replicate (`stale`) | Refuses and names the differing keys. `--force` recomputes |

Plots and the atlas are rebuilt on every run. After changing a plotting
module, rerun the scenario: complete replicates are skipped and only the
plots are redrawn.

A code change does not make a replicate stale. `run.yaml` records the simace
version that built the replicate, and `simace ls` shows it when it differs
from the version you are running. After a fix that changes results, rerun the
affected scenarios with `--force`.

To see the state of every replicate, run `simace ls`:

```bash
pixi run simace ls base
```

A replicate built by another version shows as `complete (built by simace
2026.9.1)`. In an editable checkout the running version is the one from the
last `pixi install`, so commits since then do not show up.

Only one `simace run` of a scenario can run at a time. The run holds a lock on
`results/{folder}/{scenario}/.run.lock`, and a second run of the same scenario
exits with the first run's pid. To compute several replicates at once, pass
`--jobs` to one run instead of starting several. Runs of different scenarios
do not block each other, and `--dry-run` does not take the lock.

To see a scenario's resolved parameters, per-replicate seeds, and paths, run
`simace show baseline10K`.

## Run one stage by hand

Every stage is also a subcommand that takes explicit input and output paths
and one flag per parameter. It never reads `config/`. To see a stage's flags,
pass `--help`:

```bash
pixi run simace simulate --help
```

The commands `simace run --dry-run` prints are valid stage invocations, so
you can copy one and rerun a single stage while debugging. Each stage's log is
in `logs/{folder}/{scenario}/rep{N}/{stage}.log`, and its wall time and peak
memory are in `results/{folder}/{scenario}/rep{N}/timing.tsv`.

`simace validate`, `simace stats`, and `simace effective-size` are not part of
`simace run`. Run `effective-size` by hand on a replicate's
`pedigree.parquet`, `trait.parquet`, and `params.yaml`.

## Convert parquet to TSV

To read a parquet file in R or a spreadsheet, convert it with
`simace parquet-to-tsv`. It writes a `.tsv.gz` file next to each parquet file.

```bash
pixi run simace parquet-to-tsv results/base/baseline10K/rep1/pedigree.parquet
pixi run simace parquet-to-tsv results/base/baseline10K/rep1/*.parquet
```

For an uncompressed `.tsv`, pass `--no-gzip`. To write eight decimal places
instead of the default four, pass `-p 8`.

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'simace'` | Run the command through `pixi run` from the repository root |
| `FileNotFoundError: config/_default.yaml` | Run the command from the repository root, or pass `--config-dir` |
| `refused: run.yaml differs in ...` | The scenario changed after that replicate ran. Rerun with `--force` |
| `simace run: another simace run (pid ...) is running ...` | Wait for that run to finish. The lock is released when its process exits, even if it is killed |
| `simace run: scenario ... uses gene drop` | Gene-drop scenarios run through `scripts/gene_drop/`, not `simace run` |
| A stage failed | Read the log the error line names, in `logs/{folder}/{scenario}/rep{N}/` |
| A large-N simulation is killed or hangs | Lower `--jobs` so fewer replicates share memory, or pass `--max-memory` so an oversized stage fails with a clear error instead of pushing the machine into swap |
| `FAILED (exit -9, killed for going over --max-memory)` | The stage needed more than the cap. Raise `--max-memory` or lower `N` |
