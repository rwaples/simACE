# PA-FGRS bivariate scorer — performance roadmap

Working notes from profiling the bivariate PA-FGRS scorer
(`fitACE_pafgrs/workflow/scripts/pafgrs_bivariate_score.py` +
`fitace_pafgrs/pafgrs_bivariate.py`). All numbers measured 2026-07-02..04 on
the `cure_rA50_200k` / `cure_rA50_400k` scenarios (12-core box, numba 0.65.1).

## TL;DR

- `score_loop` dominates (76% at ndeg3) and is **linear in N** — the n_rel
  distribution is invariant to founder count, so cost scales with #probands.
- Per-proband cost is set by `ndegree` (~n^2.6 between the O(n²) covmat build
  and O(n³) Pearson–Aitken conditioning), *not* by N.
- The scoring kernel is numba-parallel over probands but caps at ~4.4× on 12
  cores (≈50% efficiency at 8) — memory-bound on per-iteration covmat allocs.
- `cip_h2` (~6.2s, degree-independent) is mostly **duplicated pair
  extraction** — a cheap ~10× win on that phase.

## Measured baseline (200k, real snakemake run, pruned 4-combo sweep)

| phase | ndeg2 (s) | ndeg3 (s) | notes |
|---|---|---|---|
| load | 0.10 | 0.11 | parquet read |
| kinship_build | 1.53 | 4.57 | fitACE core; linear in N |
| prep_relatives | 0.41 | 1.22 | extract relatives + dense kin cache |
| cip_h2 | 6.13 | 6.24 | **degree-independent**; 2× Falconer extraction |
| rA_estimate | 0.64 | 0.64 | estimate_rg (own FS extraction) |
| score_loop | 9.27 | 43.06 | 4 variants, serial loop; each internally parallel |
| write | 0.45 | 0.45 | parquet |
| **total** | ~18.6 | ~56.3 | peak RSS 1.67 / 2.51 GB |

## Scaling: N and ndegree are orthogonal knobs

Doubling N (200k → 400k) at fixed degree:

| | ndeg2 200k | ndeg2 400k | × | ndeg3 200k | ndeg3 400k | × |
|---|---|---|---|---|---|---|
| mean n_rel | 12.6 | 12.6 | 1.00 | 27.0 | 27.0 | 1.00 |
| score/variant (warm) | 0.63s | 1.29s | 2.07 | 3.72s | 8.10s | 2.18 |

`score_loop` is ~linear in N. The mean relative count is **identical** across
sizes (family structure is set by mating params, not N), so per-proband cost is
a function of `ndegree` only: ndeg2→3 raises it ~5.9× for a 2.1× rise in mean
n_rel. Cost is spread, not tail-locked (top 10% of probands ≈ 45% of Σn_rel³;
max n_rel only 60/107) — so there's no hub to trim; wins are per-proband.

**2M extrapolation (10× the 200k, warm):** score_loop ≈ 25s (ndeg2) / ~150s
(ndeg3); cip_h2 ≈ 62s (pre-fix). Compute is ~2–5 min. **Real risk is memory:**
peak RSS ~2.5 GB at 200k/ndeg3 → ~25 GB at 2M; if that hits swap, linearity
breaks. Validate RSS before a full-scale run.

## Parallelism (measured)

Only `_nb_score_batch_bivariate_prepped` (`prange` over probands) and the
prep-stage `_extract_rel_kinship` are parallel. The 4 variants, kinship_build,
cip_h2, rA_estimate, DataFrame assembly, and write are all serial.

Thread scan (ndeg3 200k, per variant, separate processes):

| threads | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| time (s) | 15.39 | 8.74 | 5.38 | 3.73 | 3.51 |
| speedup | 1.0× | 1.76× | 2.86× | 4.12× | 4.38× |
| efficiency | 100% | 88% | 72% | 52% | 37% |

Memory-bound: each proband iteration heap-allocates a fresh `covmat`
(`np.empty((sz,sz))`, sz up to 2+2·136≈274 → ~0.6 MB) plus small arrays inside
the parallel region. Sweet spot ~4–8 threads.

## Backlog (ranked by ROI / effort)

### 1. Dedup + degree-restrict Falconer pair extraction — **DONE (2026-07-03)**
`cip_h2` ran `from_subsample(ped,phe).extract_pairs(md=3)` once **per trait**
(pure duplication) when only FS (degree-1) is consumed. Now extract FS pairs
**once** at `max_degree=1` and inject into both `compute_ltm_falconer` calls.
- fitACE core `falconer.py`: optional `pairs=` arg (None → old behavior).
- Both scorers (`pafgrs_bivariate_score.py`, `pafgrs_score.py`) extract once.
- **Result (200k ndeg2 rerun): cip_h2 6.13s → 0.43s (~14×)**; scores.parquet
  byte-identical (30 cols × 199,998 rows); 13 falconer + 116 sister tests pass.
  Scales linearly → ~57s saved at 2M.
- Committed (fitACE `b9b5cf7`, fitACE_pafgrs `669a4b5`).

### 2. Fix the Snakemake threads / numba mismatch — **DONE (2026-07-03)**
Rules declared `threads: 1` while numba grabbed all cores → oversubscription
when Snakemake packs many jobs. Fixed by mirroring the PCGC idiom:
- New `pafgrs_threads` config knob (default 8; tunable — 4 packs better for
  throughput-heavy runs given the ~72%/52% efficiency at 4/8 threads).
- Both scorers set every thread pool (numba/OMP/MKL/BLAS/NUMEXPR/VECLIB) to
  `snakemake.threads` before any heavy import (numba locks at first dispatch).
- The four scoring rules take `threads:` from the config knob, so the scheduler
  reservation and the in-process cap match.
- Verified: dry-run resolves `threads: 8`; scores byte-identical.
- Committed (fitACE `0f7c21d`, fitACE_pafgrs `409c18a`).

Note single-scenario latency is slightly higher (fewer threads than grabbing
all 12); the win is aggregate throughput under multi-job packing.

### 3. Score-kernel parallel efficiency — **INVESTIGATED (2026-07-03)**
Hypothesis was that per-iteration `np.empty((sz,sz))` allocation inside `prange`
capped efficiency at ~50%. **Disproven by diagnostic.** Steps taken:
- Threading layer: default already resolves to `tbb` (scalable allocator) — no
  free win; explicit tbb/omp/workqueue all within noise.
- Removed one of ~4 big per-proband matrices + a full copy (the dead `scm` →
  `cov = scm.copy()` fusion, committed `dc8e827`, byte-identical). Controlled
  thread-scan before/after: efficiency curve **unchanged** (T4 71.5%→71.9%,
  T8 51.5%→48.4%). ~5% serial speedup only, washes out by 8 threads.
- Conclusion: the ceiling is **load imbalance** (per-proband cost ∝ n_rel³, and
  static `prange` chunking gives unbalanced chunks) and/or memory bandwidth —
  **not** allocation. The per-thread-preallocation rewrite would add real
  silent-bias risk to the statistical core for no efficiency gain — **not worth
  it.**

Real lever if ever revisited: load-balance the `prange` (cost-sort probands so
static chunks are balanced, or `numba.set_parallel_chunksize` for finer/dynamic
scheduling). Separate, harder investigation; modest payoff given #2 already
recovers throughput by packing jobs at higher per-job efficiency.

### 4. Dedup estimate_rg (rA) + symmetrize rho_cross — **DONE (2026-07-03)**
`estimate_rg` did a 3rd FS extraction and computed a *directional* `rho_cross`
(sib_a·t1 × sib_b·t2), so reusing shared pairs would have perturbed rA.
Fixed both: `rho_cross` is now symmetrised (each unordered FS pair contributes
both directions → order-invariant + 2× the observations), and `estimate_rg`
takes an optional `fs_pairs=`; the scorer passes the shared FS pairs.
- Validated (cure_rA50_200k true 0.5 / cure_rA00_200k true 0.0): dedup ==
  self-extraction (1e-9), order-invariant under a/b flip (1e-12), rA within
  noise of truth (0.481→0.475 / 0.004→−0.021). Order-invariance unit test added.
- Rule re-run: rA_source=true columns byte-identical, rA_source=estimated
  est/var/cov changed (nrel unchanged, as expected).
- rA_estimate phase 0.64s → 0.002s (3rd extraction eliminated).

### 5. score_loop hot-loop memory traffic — **PARTIALLY DONE (2026-07-04)**
The rank-1 downdate double-loop is O(n_valid³) per proband; cost is spread (not
tail-locked), so it's genuine per-proband work. Three memory-traffic cuts landed
(fitACE_pafgrs `d1005c5`), all preserving the exact score path:
- **Upper-triangle downdate**: the covariance rank-1 update now writes only the
  upper triangle (`for b in range(a, j)` vs `range(j)`) — subsequent conditioning
  steps read only `cov[a, j]` (a<j), the diagonal, and the final `cov[0,1]`. Halves
  the inner-loop *writes* of the O(n³) hot path (exact arithmetic, bit-identical).
- **Reuse marginal survival** `sf_u_marg` instead of recomputing `_nb_norm_sf`
  twice per relative (new `_nb_trunc_norm_mixture_with_marginal_sf` entry point).
- **float32 relative-kinship cache** (`kin_flat`): halves that array's bandwidth.
  Safe *and* bit-identical here because pedigree kinships are dyadic rationals
  (k/2ⁿ), exactly representable in float32 → promote back to identical float64.

**Measured (warm, same-session old-vs-new, median of 2×6 reps, 200k):**

| scen | ndeg | old s/variant | new s/variant | speedup |
|---|---|---|---|---|
| rA50 | 2 | 0.571 | 0.510 | 1.12× |
| rA50 | 3 | 3.407 | 3.013 | 1.13× |
| rA00 | 2 | 0.631 | 0.542 | 1.16× |
| rA00 | 3 | 3.686 | 3.188 | 1.16× |

~12–16% warm per-variant, consistent across degrees. **Scores byte-identical**
(20/20 est/var/cov arrays, max Δ = 0, verified old-vs-new checkout). Peak RSS at
ndeg3 also fell ~20–43 MB (mostly the CSC-kinship change below).

Still open (higher effort, not done): **(a) cap least-informative relatives** —
PA already sorts by informativeness, so truncating the tail gives *cubic* savings
on the worst probands (accuracy-vs-speed study needed); (b) exploit covmat block
structure.

### 6. Avoid rebuilding the covmat skeleton per variant — **INVESTIGATED, WON'T DO (2026-07-03)**
Premise turned out to be inaccurate: the kinship skeleton is **already** built
once in `prepare_bivariate_scoring` (`rel_flat_kin`, `rel_kin_flat`) and shared
across every `score_bivariate_variant` call. What runs per variant is the
h2-scaled covmat fill (O(sz²)) + conditioning (O(sz³)) — both inherently
per-variant (each variant conditions on its own h2-scaled matrix).
- Build-vs-condition split (fitted from warm ndeg2/ndeg3 timings + real sz
  moments): O(sz²) work 46–63%, O(sz³) 37–54%. But most of that O(sz²) is
  per-variant (covmat value fills, cm→cov reorder, conditioning's inner
  O(sz²) loops), not hoistable.
- The only variant-invariant residual (kinship reads/index arithmetic +
  phenotype gather + sort order) is small, and capturing it needs fusing the 4
  variants into one kernel pass — a #3-class rewrite of the statistical core for
  a modest gain. Declined for the same risk reason as #3.
- Possible future safe-ish within-kernel dedup (independent of fusion): build
  `cov` directly from `covmat` via a composed valid+reorder index map, skipping
  the intermediate `cm` (one fewer O(sz²) materialization per variant,
  byte-identical verifiable). Deferred — #3-class kernel edit.

### 7. (context) kinship_build + 2M memory wall — **memory trimmed (2026-07-04)**
`kinship_build` (fitACE core, shared) scales linearly in N; out of scope for
compute here. fitACE core `4e40ba7` now builds the pair-kinship matrix directly
as CSC with int32 pair indices + float32 dyadic coefficients (no intermediate CSR
conversion, duplicate paths capped in place). Build wall-time is within run-to-run
noise; the win is **peak RSS** (ndeg3 200k: −19 MB rA50 / −43 MB rA00) and lower
allocation churn — relevant to the 2M memory wall. Track the RSS wall (see 2M
extrapolation) before any full-scale 2M run.

## Reproduce

Profiling scripts (scratchpad, not committed):
`profile_bivariate.py` (n_rel dist + warm timings, takes a rep dir),
`thread_scan.py` (parallel efficiency), `cip_breakdown.py` (cip_h2 breakdown).
Scaling data: `results/pafgrs/scaling_bivariate.tsv`.
