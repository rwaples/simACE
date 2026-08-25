# simACE pipeline benchmark — fresh GitHub clone, 2026-08-20

Fully isolated re-baseline: nothing from the working tree, the conda env, or the
existing `.pixi/` was reused.

## What was benchmarked

| | |
|---|---|
| Source | `git clone https://github.com/rwaples/simACE.git`, checked out at tag **`v2026.08`** (`cee3bec`), tree clean |
| Location | `/data/Documents/simACE-bench-2026-08-20` (sibling of the working tree, same filesystem; **deleted 2026-08-25**, see below) |
| Environment | `pixi install --locked` → **9.0 s**, own `.pixi/envs/default` (3.2 GB) |
| Interpreter | `.../simACE-bench-2026-08-20/.pixi/envs/default/bin/python3.13` (3.13.15) |
| simace | `2026.8`, editable from the clone |
| pedigree-graph | `0.7.0`, PyPI wheel in site-packages (not editable) |
| Stack | polars 1.43.2 · pandas 3.0.5 · numpy 2.4.6 · scipy 1.18.0 · numba 0.66.0 · pyarrow 25.0.0 · snakemake 9.25.1 |
| Machine | i7-9750H (6c/12t), 30 GB RAM, `/data` NVMe |

**Ref choice.** `origin/master` (`1c1006b`) is an *ancestor* of the release tag —
it predates the polars migration entirely and has no `pixi.toml`/`pixi.lock`, so
it is not installable by the documented path. `v2026.08` == `origin/dev`.

## Method

- Target per scenario: `results/base/<sc>/stats.done` + `plots/atlas.html` = **21 jobs**
  (3 reps × emit_params/simulate/phenotype/censor_weibull/ascertainment/analyze,
  then plot_phenotype + assemble_atlas). Same DAG shape at all three sizes.
- **Not** `scenario.done`: that rule depends on the folder-wide
  `results/base/report_summary.tsv`, which gathers `report.yaml` across *every*
  scenario in the `base` folder — in a fresh clone it schedules 89 jobs including
  `baseline2M` and `baseline10M` simulations.
- 3 sweeps × 3 scenarios, `--cores 4`, `-F` (full re-execution each sweep).
- Sweep 1 is genuinely cold: 0 numba `.nbi/.nbc` files in the fresh clone.
- Per-rule benchmark TSVs snapshotted per sweep before the next overwrites them
  → n=9 samples per per-replicate rule, n=3 for the two per-scenario rules.
- Every snakemake exit status checked; a failure aborts rather than recording a
  spuriously fast number. **9/9 runs OK.**
- Fastest-core frequency sampled throughout: median 3.84–4.20 GHz, max 4.40 GHz.
  **No thermal/power clamp** — the artifact that corrupted the earlier polars A/B.

## Whole-scenario wall clock (`--cores 4`)

| Scenario | Cold (sweep 1) | Warm median | Warm range | Cold penalty | Peak RSS |
|---|---|---|---|---|---|
| baseline10K | 44.8 s | **42.3 s** | 41.7–42.8 | +2.5 s | 0.49 GB |
| baseline100K | 55.2 s | **52.7 s** | 51.0–54.4 | +2.5 s | 1.40 GB |
| baseline1M | 213.0 s | **189.4 s** | 186.7–191.9 | +23.6 s | 9.47 GB |

Output on disk: 33 MB / 179 MB / 1.7 GB.

## Per-rule, warm (sweeps 2–3), median seconds and peak RSS MB

| Rule | 10K | 100K | 1M |
|---|---|---|---|
| emit_params | 0.24 / 26 | 0.23 / 26 | 0.21 / 28 |
| simulate | 0.89 / 135 | 1.15 / 304 | 4.46 / 774 |
| phenotype | 1.42 / 257 | 1.42 / 319 | 2.24 / 1001 |
| censor_weibull | 0.33 / 8 | 0.45 / 8 | 2.85 / 1055 |
| ascertainment | 0.41 / 9 | 1.23 / 350 | 10.88 / 2429 |
| analyze | 2.29 / 357 | 4.81 / 1196 | 39.44 / 2095* |
| plot_phenotype | 23.93 / 487 | 24.68 / 538 | 26.03 / 538 |
| assemble_atlas | 1.14 / 110 | 1.16 / 107 | 1.16 / 109 |

\* see the RSS caveat below — the true `analyze` peak at 1M is ~9 GB, not 2.1 GB.

## Findings

### 1. `plot_phenotype` is size-independent and dominates small scenarios

24–26 s regardless of N. It is **57 % of the 10K scenario** and 47 % of 100K,
falling to 12 % at 1M. Single job, not per-replicate. This is the largest
available win for the 10K/100K turnaround, and it is pure fixed cost — matplotlib
import plus ~35 figures — not data volume.

### 2. Snakemake's benchmark TSVs understate peak memory by ~3×

`benchmarks/.../analyze.tsv` reports 2.1–3.1 GB at 1M. Direct process-tree
sampling at 0.4 s intervals shows `analyze` peaking at **8.2–9.0 GB** per
replicate, confirmed independently by `/usr/bin/time -v` (9.47 GB whole-tree
high-water). Snakemake's psutil sampler misses the transient spike.

This matters because those TSVs are the natural input for sizing SLURM `mem_mb`.
Cross-checked three ways: peak sum-of-tree 9.63 GB, peak single-process 8.69 GB,
kernel high-water 9.47 GB — all consistent with one process, `analyze`, spiking.
Identified via the wrapper filename `.snakemake/scripts/tmp*.analyze.py`.

### 3. The declared memory budget is nonetheless well calibrated

`analyze` declares `_scale_mem(..., "G_ped")` = `N × G_ped × 2/1000` = **12 GB**
at 1M against a 9.0 GB observed peak — 25 % headroom. The linear model holds.
Extrapolating: 2M ≈ 18 GB actual (24 GB declared), 10M ≈ 90 GB actual
(120 GB declared) — which is why `baseline10M` has no completed reports on this
30 GB machine.

### 4. Cold-start cost is small except at 1M

+2.5 s at 10K and 100K (numba JIT), but +23.6 s at 1M — larger than JIT alone
explains; the cold `analyze` rep also recorded its highest RSS (3.1 GB sampled)
and slowest time (50.5 s vs 39.4 s warm), consistent with allocator/page-cache
pressure on first touch rather than compilation.

### 5. Repo hygiene: `master` is stale on GitHub

`local master == remote master == 1c1006b`, an ancestor of `v2026.08`. The
release was tagged on `dev` and `master` was never fast-forwarded, departing from
the usual merge-dev-to-master-then-tag convention. Anyone cloning the default
branch gets pre-polars code with no pixi manifest.

## Raw data and reproduction

The benchmark clone (`/data/Documents/simACE-bench-2026-08-20`, 5.0 GB) was
**deleted on 2026-08-25**. Its 5 GB was a clean `v2026.08` checkout, a `.pixi`
env, and generated `results/` — all reproducible. The raw measurements and the
driver scripts were preserved to:

    plans/bench-fresh-clone-2026-08-20-data/   # local only, gitignored

That directory is **not tracked** (`plans/*` is gitignored) — the raw
measurements live on the machine that produced them, not in the repo. The
numbers in this document are the distributed record.

- `wallclock.tsv` — whole-scenario wall/RSS/CPU/clock, 9 runs
- `tsv/sweep{1,2,3}/<scenario>/rep{1,2,3}/*.tsv` — per-rule snapshots
- `runs/*.{log,time,freq}` — snakemake logs, `time -v`, clock samples
- `memprobe-*.{txt,log,time}` — process-tree memory attribution

The driver scripts are **tracked**, generalized out of this one-off run:

- `tools/bench_pipeline.sh` — the sweep driver
- `tools/bench_aggregate.py` — the per-rule median/range tables
- `tools/bench_memprobe.sh` — per-rule peak-memory attribution

To repeat this benchmark on a clean tree:

    git clone https://github.com/rwaples/simACE.git <dir> && cd <dir>
    git checkout v2026.08 && pixi install --locked
    SWEEPS=3 CORES=4 bash tools/bench_pipeline.sh
    python tools/bench_aggregate.py bench-logs

To re-read the numbers in this document from the preserved data:

    python tools/bench_aggregate.py plans/bench-fresh-clone-2026-08-20-data
