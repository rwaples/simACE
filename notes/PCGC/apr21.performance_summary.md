# PCGC performance summary — 2026-04-21

Session scope: Round 2 PCGC scale test at n=300k, resource-heuristic
calibration from Track 1 data, and a rare-K C/E identifiability
investigation that surfaced while validating the no-AM σ²_A
cross-check.  This note is the wall/RSS/scaling roll-up; bias
methodology and the full K-sweep interpretation live in
`notes/pcgc_phase1_bias.md`.

## Headline

1. **Mem heuristic calibrated** — fitACE's PCGC rule now uses a
   dedicated `_pcgc_mem` helper fit to Track 1 data (80 MB / 1k
   individuals empirical, 60 MB / 1k in the heuristic for ~1.7×
   headroom).  Previous heuristic was ~17× too small at n=100k and
   would have OOM'd on any scheduler enforcing `mem_mb`.  Commit
   `e1c045f` (fitACE).
2. **Round 2 scale test at n=300k passed** — wall 151 s (AM) / 122 s
   (noam), RSS 21.5 / 14.4 GB.  RSS scaling exponent ≈ 1.0 (linear,
   as Track 1 predicted); wall exponent ≈ 1.44 (mildly super-linear,
   Python wrapper + kinship build dominate at production scale).  1M
   extrapolation updated: ~9–10 min wall, ~70 GB RSS — well inside
   8h / 500GB ceiling.
3. **Bias finding logged** (see `pcgc_phase1_bias.md` follow-up
   section): at K=0.05 no-AM, V(C) absorbs Ve across all n
   (10k → 300k).  Pattern is K-driven, not n-driven; clean recovery
   at K ≥ 0.30.  Replication across 5 seeds at n=300k confirms the
   bias is reproducible (SD across reps ≈ single-rep SE, within
   ~40%).

## Resource heuristic calibration (fitACE `_pcgc_mem`, `_pcgc_runtime`)

Before: `mem_mb = max(1024, _scale_mem(G_ped))` which resolves to the
4 GB floor for any scenario below ~330k founders.  Commit history
called this "iter_reml scaling halved" but only runtime was halved;
mem was inherited unchanged.

After: `_pcgc_mem = _scale_mem(G_ped, mb_per_1k=60, floor=2000)`.
Coefficient 60 chosen for ~1.7× headroom over measured RSS.  Runtime
uses `min_per_1M=5, floor=10`.

Cross-check against Track 1 + Round 2 benchmarks:

| scenario | N | measured RSS | new mem_mb | headroom |
|---|---:|---:|---:|---:|
| iter_reml_10k       |   3,400 |   0.73 GB |  2,000 MB (floor) | 2.7× |
| iter_reml_50k       |  17,000 |   3.78 GB |  6,120 MB         | 1.6× |
| iter_reml_100k      |  34,000 |   7.12 GB | 12,240 MB         | 1.7× |
| iter_reml_300k      | 100,000 |  21.46 GB | 36,000 MB         | 1.7× |
| iter_reml_300k_noam | 100,000 |  14.43 GB | 36,000 MB         | 2.5× |
| iter_reml_1M (extr) | 333,000 |  ~70 GB   | ~120,000 MB       | ~1.7× |

1.7× is the design target and it's holding.  The 2.5× at 300k_noam
vs 1.7× at 300k (AM) shows AM adds ~50% RSS at the same n — worth
knowing, and captured in the heuristic's margin.

## Round 2 scale test (n=300k)

Fixtures added this round (`config/iter_reml_bench.yaml` sim-side;
PCGC overlay in `fitACE/config/iter_reml_bench.yaml`):

- `iter_reml_300k` — N=100000, assort1=0.3, K=0.05, replicates=5
- `iter_reml_300k_noam` — N=100000, assort1=0.0, K=0.05, replicates=5

### Per-fixture rep1 wall + RSS

| scenario | n_phen | n_pairs_A | n_pairs_C | wall_snake | wall_internal | max_rss |
|---|---:|---:|---:|---:|---:|---:|
| iter_reml_300k      | 300,000 | 104,701,641 | 353,632 | 150.98 s | 8.66 s | 21,459 MB |
| iter_reml_300k_noam | 300,000 | 104,685,122 | 355,251 | 122.47 s | 6.64 s | 14,428 MB |

Jackknife: `pair_blocks` fallback in both (single giant component).
Both converged in 1 iteration (closed-form moment estimator).  Sim
was cheap: 0.6–1.0 s for pedigree build at N=100000, 0.2 s for
phenotype — the user's earlier note that the pipeline has handled
pedigrees larger than 100k is confirmed, these were not new territory
for simACE.

### Scale table (full)

| scenario    | n_phen  | wall_snake | wall_internal | max_rss  | n_pairs_A |
|---|---:|---:|---:|---:|---:|
| iter_reml_10k       |   10,200 |   4.35 s  | 0.14 s |  0.73 GB |   3.4 M |
| iter_reml_50k       |   51,000 |  17.89 s  | 0.84 s |  3.78 GB |  17.5 M |
| iter_reml_100k      |  102,000 |  35.73 s  | 1.78 s |  7.12 GB |  35.0 M |
| iter_reml_300k      |  300,000 | 150.98 s  | 8.66 s | 21.46 GB | 104.7 M |
| iter_reml_300k_noam |  300,000 | 122.47 s  | 6.64 s | 14.43 GB | 104.7 M |

### Scaling laws, updated

**RSS:** linear.  100k → 300k ratio = 3.01× for 2.94× n → exponent 1.01.
Track 1's `RSS ≈ 80 · n^0.99 MB` is confirmed.

**Wall:** mildly super-linear.  100k → 300k ratio = 4.22× (AM) / 3.43×
(noam) for 2.94× n → exponent ≈ 1.44 / 1.30.  Track 1's
`wall ≈ 9.8e-4 · n^0.91` was fit through 10k/50k/100k and
under-predicts at 300k by ~1.6×.  The Python wrapper + kinship build
scale slightly worse than the cpp kernel itself (internal wall
exponent is ~1.44 as well between 100k and 300k).

**n=1M extrapolation (updated):**

| metric | old (Track 1 fit) | new (with 300k) |
|---|---:|---:|
| wall | 4.6 min | ~9–10 min |
| RSS  | 70 GB   | ~70–75 GB (unchanged) |
| mem_mb budget | — | ~120 GB (1.7× headroom) |

Both comfortably inside the 8h / 500GB project ceiling.

## Rare-K C/E identifiability (summary; details in bias note)

While running the no-AM σ²_A cross-check (Track 1 open question #3),
the `iter_reml_100k_noam` PCGC fit surfaced V(A)=0.465, V(C)=0.431,
Ve=0.103 against truth A=0.4, C=0.2, Ve=0.4 — V(C) had absorbed Ve.
A K-sweep on the same fixture (changing only τ = Φ⁻¹(1−K)) showed:

| K | V(A) | V(C) | Ve |
|---:|---:|---:|---:|
| 0.05 | 0.465 | **0.431** | **0.103** |
| 0.15 | 0.428 | 0.259 | 0.313 |
| 0.30 | 0.417 | 0.205 | 0.378 |
| 0.50 | 0.407 | 0.206 | 0.387 |

Bias is monotone in K.  At K=0.50 the partition is essentially
perfect (≤ 0.02 in all components) — **the sim truth, kinship build,
and 2×2 moment solver all work.**  The bias is purely an
ascertainment-driven artifact at rare K.  See the bias note
follow-up section for the full size sweep and revised
recommendations.

### Replication at n=300k (5 reps each, K=0.05)

Confirms bias is seed-stable, not a fluke of rep1:

iter_reml_300k (AM=0.3):

| component | mean | SD(reps) | mean SE_within | bias vs truth |
|---|---:|---:|---:|---:|
| V(A) | 0.765 | 0.016 | 0.020 | **+0.365** (AM-inflated) |
| V(C) | 0.320 | 0.025 | 0.021 | +0.120 |
| Ve   | **−0.084** | 0.020 | 0.028 | −0.484 |

Ve is negative in all 5 reps (−0.059 to −0.106) — AM-inflation
sentinel is deterministic.

iter_reml_300k_noam (AM=0.0):

| component | mean | SD(reps) | mean SE_within | bias vs truth |
|---|---:|---:|---:|---:|
| V(A) | 0.490 | 0.026 | 0.018 | +0.090 |
| V(C) | 0.387 | 0.020 | 0.020 | **+0.187** |
| Ve   | 0.123 | 0.025 | 0.025 | **−0.277** |

### SE calibration check (`SD(reps) / SE_within`)

| scenario | V(A) | V(C) | Ve |
|---|---:|---:|---:|
| 300k AM    | 0.80 | 1.19 | 0.71 |
| 300k_noam  | 1.44 | 1.00 | 1.00 |

Jackknife SEs honest within ~±40%.  Only V(A) on the no-AM fixture
has SD noticeably larger than SE (1.44×), suggesting the jackknife
slightly under-estimates variability there.

## Artifacts produced this session

**Commits:**
- fitACE `e1c045f` — empirical resource directives from Track 1 data
- simACE `0c7edd5` — 300k fixtures + bias note follow-up
- fitACE `03974cd` — enable PCGC fits on no-AM + 300k scenarios

**Fit outputs** (under `results/iter_reml_bench/`):
- `iter_reml_{10k,50k,100k}_noam/rep1/pcgc/` (no-AM cross-check)
- `iter_reml_300k/rep{1..5}/pcgc/` and `iter_reml_300k_noam/rep{1..5}/pcgc/`
  (Round 2 scale + replication)

**Benchmarks** (under `benchmarks/iter_reml_bench/`):
- Matching `pcgc.tsv` Snakemake OS-level rows for all of the above.
- `benchmarks/pcgc/pcgc_scale.tsv` (Track 1 roll-up) not yet extended
  with the 300k row; run `workflow/scripts/collect_pcgc_scale.py`
  with the updated scenario list to refresh.

## Open questions → Round 3 (n=1M)

1. Direct confirmation of linear RSS scaling at n=1M (expected ~72 GB).
2. Direct confirmation of the revised wall-scaling exponent at n=1M
   (predicted ~9–10 min; a systematic deviation above 15 min would
   suggest a hidden quadratic term somewhere).
3. Jackknife stability with a single giant component at 1M.  The
   `pair_blocks` fallback has held at every n so far but dependence
   between adjacent pair blocks at biobank scale is presumed tolerable
   and not yet measured.
4. (Separate track) Decide how to report C/Ve at rare K in production.
   Options: (a) suppress partition and report h² only, (b) require
   K ≥ 0.15 fits for partition claims, (c) derive or empirically
   correct the rare-K C/Ve bias.  The first two are cheap, the
   third is research-scope.
