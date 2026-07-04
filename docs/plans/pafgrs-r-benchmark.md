# PA-FGRS: Python port vs reference R — univariate speed benchmark

Head-to-head of the Python univariate PA-FGRS scorer (`fitace_pafgrs`) against
the reference R implementation (`BioPsyk/PAFGRS`, Krebs/Gådin/Schork; vendored at
`external/PAFGRS`, R 4.5.3 env `pafgrs_r`). Measured 2026-07-04 on the 12-core
box. **Univariate only** — the R package has no bivariate PA-FGRS (`pa_fgrs`,
`pa_fgrs_adt`, `pa_fgrs_cont`, batch `FGRS_wrapper`; no multivariate entry point).

## Setup

- Data: `cure_rA50_200k` / `cure_rA50_400k` (trait 1, h²=0.5, empirical CIP).
  "100k" = a 100k-proband subset of the 200k pedigree — throughput-identical to a
  native 100k, since the relative-count distribution is set by mating params, not
  founder count (same N-invariance the score_loop roadmap relies on).
- **Identical inputs**: both implementations are fed the *same* sparse kinship K
  (phenotyped×phenotyped, x ≥ 0.5^(ndeg+1), pedigree ids) and the same pheno
  (id, aff, thr, w). Kinship construction is excluded from both timers.
- R timer wraps only the `FGRS_wrapper(method="PAFGRS")` call (excludes I/O +
  package load). `FGRS_wrapper` is an interpreted `sapply` over probands calling
  `pa_fgrs` per proband — **inherently serial, single-threaded**.
- Python timer wraps `prepare_univariate_scoring` + `score_univariate_variant`
  ("from-K": relative extraction + covmat + PA solve — the analog of what R's
  call does). Warm (JIT pre-compiled). Pipeline default is `pafgrs_threads=8`.

## Correctness (same computation)

R and Python agree to sort-tie noise — the comparison is valid:

| set | n | corr(postM, est) | mean \|Δmean\| | max \|Δmean\| | n_rel match |
|---|---|---|---|---|---|
| ndeg2 100k | 100,000 | 0.999973 | 6.0e-4 | 4.6e-2 | 100% |
| ndeg3 100k | 100,000 | 0.999976 | 7.2e-4 | 3.6e-2 | 100% |

Max diff is PA sort-order ties on a few probands (documented Level-2 agreement);
mean diff ~1e-3. Relative counts match exactly.

## Throughput (probands/s, warm)

| degree | **R** (1-thread) | Python 1-thread (kernel) | Python 8-thread (kernel) | Python 8-thread (from-K) |
|---|---|---|---|---|
| ndeg2 | 157 | 238,000 | 1,077,000 | 591,000 |
| ndeg3 | 130 | 76,800 | 330,000 | 172,000 |

Throughput is N-stable (200k and 400k rates match within noise), so wall-clock =
N ÷ rate at any size. Python parallel efficiency ≈ 4.5×/4.3× on 8 threads
(ndeg2/ndeg3), consistent with the memory-bound score kernel.

## Wall-clock at 100k and 400k (from-K, warm)

| degree | scale | **R** (1-thread) | **Python** (8-thread) | speedup | R basis |
|---|---|---|---|---|---|
| ndeg2 | 100k | 636 s (10.6 min) | 0.17 s | **~3,800×** | measured |
| ndeg2 | 400k | 2,542 s (42.4 min) | 0.68 s | **~3,800×** | projected 4× |
| ndeg3 | 100k | 771 s (12.9 min) | 0.58 s | **~1,300×** | measured |
| ndeg3 | 400k | 3,086 s (51.4 min) | 2.33 s | **~1,300×** | projected 4× |

R at 100k is the **measured anchor** (ndeg2 635.5 s @ 157 prob/s; ndeg3 771.4 s @
130 prob/s). 400k is projected as 4× the 100k time — justified: R's `FGRS_wrapper`
is marginal-dominated (fit across N=100/1000/4000: ~0.5 s fixed + 6.6 ms/proband
ndeg2, 8.9 ms/proband ndeg3) with N-invariant per-proband cost.

- **Pure PA-solve kernel** (excluding Python's prep): speedup rises to ~6,900×
  (ndeg2) / ~2,540× (ndeg3) at 8 threads.
- **Single-thread Python vs single-thread R** (pure language/algorithm, no
  parallelism): ndeg2 ~1,500×, ndeg3 ~590×.

## Takeaway

The numba port is **3–4 orders of magnitude** faster than the reference R at the
same task and same numeric result. R's per-proband interpreted `sapply` makes
full-cohort scoring a ~10–50 min job even at 100k–400k; the Python scorer does it
in sub-second-to-seconds. This is why the pipeline can afford to score full
cohorts across many (h², CIP-source) variants where R would be prohibitive.

## Reproduce

Scratchpad (not committed): `export_for_r.py` (identical-input exporter),
`r_bench_run.R` (R `FGRS_wrapper` timer), `py_fullscale.py` (Python throughput),
`r_100k.sh` (100k anchor runs). TSV: `results/pafgrs/r_vs_python_benchmark.tsv`.
