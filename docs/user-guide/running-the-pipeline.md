# Running the pipeline

`simace run` reads the scenario files in `config/` and writes under
`results/` and `logs/`. Run it from anywhere inside the repository: the
commands find the directory that holds `config/_default.yaml` by walking up
from the current directory. `--config-dir`, `--results` and `--logs` point
them elsewhere.

## Run scenarios

```bash
pixi run simace run baseline10K            # one scenario
pixi run simace run base                   # every scenario in config/base.yaml
pixi run simace run base dev --jobs 6      # two folders, six replicates at a time
```

Each target is a scenario name, or else a folder name, which stands for
every scenario whose `folder` it is (gene-drop scenarios are left out). A
scenario named more than once runs once. `simace run` computes every
replicate of every target through one pool of `--jobs` workers, then draws
each scenario's plots and HTML atlas. Each replicate goes through the stages
in this order: simulate, phenotype, censor, ascertain, analyze. Each stage
reads the files the previous one wrote.

| Flag | Effect |
|---|---|
| `--rep 2 5-8` | Compute only these replicates, given as numbers or inclusive ranges. Needs exactly one scenario |
| `--jobs 3` | Compute three replicates at once, across every target. Each stage process is limited to one thread |
| `--dry-run` | Print the exact stage commands and write nothing |
| `--force` | Recompute the requested replicates even when they are complete |
| `--fail-fast` | Stop starting new replicates after the first failure |
| `--no-plots` | Skip the plots and atlas |
| `--max-memory 8G` | Kill any stage process whose resident memory goes over 8 GiB, which fails its replicate. The cap is per stage, so `--jobs 3` can use up to three times it |
| `--format pdf` | Also write `plots/atlas.pdf`; `plots/atlas.html` is always built |
| `--results DIR`, `--logs DIR`, `--config-dir DIR` | Use other roots than `results/`, `logs/`, `config/` |

## Preview the run

To see the commands without running them, add `--dry-run`:

```bash
pixi run simace run baseline10K --dry-run
```

Before a run that takes more than a few minutes, preview it, and check what
its stages needed last time:

```bash
pixi run simace show baseline10K
```

`simace show` prints the scenario's resolved parameters, per-replicate seeds
and paths, and a `timing` block with, per stage over the complete
replicates, the median wall time, the peak resident memory, and which
replicate peaked. Size `--jobs` and `--max-memory` from it.

## Summarize a folder

```bash
pixi run simace gather base
```

`simace gather` reads every `report.yaml` of a complete replicate under
`results/base/`, writes `results/base/report_summary.tsv`, and draws the
validation plots and atlas in `results/base/plots/`. Pass `--format pdf` to
write the PDF atlas beside the HTML one.

A replicate is left out, with a line on stderr saying why, when it has no
`run.yaml` (it failed or was interrupted) or when the config directory still
lists its scenario and the replicate is stale or incomplete under the current
config. Replicates of scenarios the config has since dropped are kept.

## Rerun and resume

`simace run` writes a replicate's `run.yaml` only after every stage of the
replicate succeeds. It records the scenario, replicate number, seed, and
parameters the replicate was computed from, and the size and modification
time of every output file. A replicate is complete when its `run.yaml`
matches the current config and every output file of the replicate exists
as recorded.

When you rerun a scenario, each requested replicate is handled as a whole:

| State | What `simace run` does |
|---|---|
| Complete | Skips the replicate |
| No `run.yaml`, for example after an interruption | Recomputes the replicate from the first stage |
| `run.yaml` matches, but an output file is missing or was rewritten after `run.yaml` (`incomplete`) | Recomputes the replicate from the first stage and names the files |
| `run.yaml` differs from the current config, or belongs to another replicate (`stale`) | Refuses and names each differing key with its recorded and current value (`N: 999 -> 300`). `--force` recomputes |

After the replicates, `simace run` prints one summary line per scenario with
the counts (`summary: 2 skipped, 3 computed, 1 failed`), then one line per
failed replicate naming the stage and its log, and one per refused replicate
naming the changed keys. With several scenarios a `[total]` line follows.

Plots and the atlas are rebuilt on every run for each scenario whose
replicates are all complete. After changing a plotting module, rerun the
scenario: complete replicates are skipped and only the plots are redrawn.
`--no-plots` skips them.

A code change does not make a replicate stale. `run.yaml` records the simace
version and, in a git checkout, the commit (`git describe --tags --always
--dirty`) that built the replicate, and `simace ls` shows either when it
differs from what you are running. After a fix that changes results, rerun
the affected scenarios with `--force`.

To see the state of every replicate, run `simace ls`:

```bash
pixi run simace ls base
```

Each scenario prints one line with the count of replicates in each state.
Complete replicates are counted only; every other replicate is listed with
the reason, grouped by state:

```
base/baseline10K  5 reps: 3 complete, 1 stale (rep2: N: 999 -> 10000), 1 absent (rep5)
base/baseline100K  3 reps: 3 complete (reps 1-3: built at v2026.9-3-gabc123)
```

In an editable checkout the version is the one from the last
`pixi install`, so commits since then show up in the `built at` note, not
the version.

Only one `simace run` of a scenario can run at a time. The run holds a lock on
`results/{folder}/{scenario}/.run.lock`; a run that names several scenarios
takes every lock before starting, and a second run of a locked scenario
exits naming the lock and the first run's pid. To compute several replicates
at once, pass `--jobs` to one run instead of starting several. Runs of
different scenarios do not block each other, and `--dry-run` does not take
the lock. If the run process is killed, the lock stays held until its active
stage processes exit, so the pid in the message may be gone while a stage it
started still holds the lock.

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

A stage rerun this way rewrites its output, which no longer matches the size
and time recorded in the replicate's `run.yaml`. `simace ls` then shows the
replicate as incomplete (`rep2: trait.parquet changed`), `simace gather`
leaves it out, and the next `simace run` recomputes it from the first stage.
To keep a debugging output out of the results, point the stage's output
flags at another directory.

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
| `ModuleNotFoundError: No module named 'simace'` | Run the command through `pixi run` |
| `FileNotFoundError: config/_default.yaml` | Run the command from inside the repository, or pass `--config-dir` |
| `refused: run.yaml differs in ...` | The scenario changed after that replicate ran. Rerun with `--force` |
| `simace run: ... is locked (...) by simace run pid ..., or by a stage it started` | Wait for that run to finish. The lock is released when its last process exits, even if the run was killed |
| `simace run: unknown target ...` | Neither a scenario nor a folder in `config/`; the message lists both |
| `simace run: scenario ... uses gene drop` | Gene-drop scenarios run through `scripts/gene_drop/`, not `simace run` |
| A stage failed | Read the log the summary names, in `logs/{folder}/{scenario}/rep{N}/` |
| A large-N simulation is killed or hangs | Check `simace show` for the peak memory of past replicates, lower `--jobs` so fewer replicates share memory, or pass `--max-memory` so an oversized stage fails with a clear error instead of pushing the machine into swap |
| `FAILED (exit -9, killed for going over --max-memory)` | The stage needed more than the cap. Raise `--max-memory` or lower `N` |
