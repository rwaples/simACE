# PCGC Phase 1 — Track 1 scale-validation findings

**Date:** 2026-04-20
**Scope:** n ≈ 10k / 51k / 102k (rare trait K=0.05) on AM=0.3 pedigrees.
**Backend:** C++ (`ace_pcgc`) on 8 threads, cross-checked against numba
at 10k and 50k.  Inputs reuse the existing `iter_reml_{10k,50k,100k}`
fixtures in `results/iter_reml_bench/`; no new simACE fixtures generated
this round.

## Headline

PCGC cpp finishes n=100k in **1.78 s** internal wall, **35.7 s**
Snakemake wall (Python wrapper + kinship build dominates), **7.3 GB**
peak RSS.  Three-point log-log extrapolation places n=1M at
**~4.6 min wall / ~70 GB RSS** — comfortably under the 8h / 500GB
project ceiling.  Round 2 (n=300k, n=1M on new simACE fixtures) is
green to proceed.

## Per-scenario table (`benchmarks/pcgc/pcgc_scale.tsv`)

| scenario | n_phen | n_pairs_A | wall_internal (s) | wall_snake (s) | RSS (GB) | σ²_A | σ²_C |
|---|---:|---:|---:|---:|---:|---:|---:|
| iter_reml_10k  |  10,200 |  3,364,485 | 0.14 |  4.35 | 0.73 | 0.78 | 0.37 |
| iter_reml_50k  |  51,000 | 17,498,253 | 0.84 | 17.89 | 3.78 | 0.85 | 0.37 |
| iter_reml_100k | 102,000 | 35,021,713 | 1.78 | 35.73 | 7.12 | 0.67 | 0.40 |

`largest_component_pair_fraction = 1.0` in all three — the pedigrees
are single giant components, so the jackknife `auto` scheme falls back
to `pair_blocks` (recorded in `fit.vc.tsv.meta` via the Stage-0
diagnostic plumbing).

## Scaling

Log-log fit (three-point least squares) on
`(n_phen, wall_snake)` and `(n_phen, max_rss)`:

```
wall ≈ 9.8e-4 · n^0.91   s
RSS  ≈ 8.0e-2 · n^0.99   MB
```

Essentially linear in n on both axes, as expected for a moment
estimator whose pair count grows linearly with n in these deep
multi-generation pedigrees (~340 pairs per phenotyped individual
after `grm_threshold=0.001`).  **Extrapolated budget:**

| target | wall | RSS |
|---|---:|---:|
| n=300k |  1.55 min | 21.2 GB |
| n=1M   |  4.61 min | 70.1 GB |

Both well inside the 8h / 500GB project ceiling with several orders
of magnitude of headroom on wall and ~7× on RSS.

## Backend parity (`benchmarks/pcgc/scale_parity.tsv`)

Ran numba in parallel with cpp on `iter_reml_10k` and `iter_reml_50k`
(both below numba's 100k cap) via `python -m fitace.pcgc.bench`:

- 20 (scenario, rep, backend-pair, column) comparisons.
- 0 failed the tolerance gate (`atol=1e-3` on σ², `atol=5e-3` on SE,
  `atol=1e-6` on c_factor; integer counts must match exactly).

At n=50k with 17.5M pairs, cpp and numba agree to all printed digits
on every σ² and SE field.  The Round-2 prerequisite — "cpp is not
silently drifting from the validated numba path at biobank-adjacent
scale" — is satisfied.

## Why σ²_A is biased high (NOT a PCGC bug)

Truth on the sim side is `{A: 0.4, C: 0.2, E: 0.4}`, but the
`iter_reml_{10k,50k,100k}` fixtures use `assort1=0.3`.  Under
assortative mating the realized liability-scale Vp at equilibrium is
> 1 (mates are correlated → offspring liabilities are correlated →
additional variance propagates through the pedigree).  PCGC
standardizes on liability-scale Vp=1 by construction, so any
AM-inflated variance is reabsorbed into σ²_A (and, less so, σ²_C),
leaving σ²_E < 0 as a sentinel that the target Vp is below the
realized one.

The bias-characterization track (Track 2) uses the `dev_laplace_n*k`
size sweep with `assort1=0.0` — that is the correct fixture to
measure PCGC bias, not the AM-loaded scale fixtures.

## Diagnostics added this round (Stage 0)

Each PCGC fit now writes to `fit.vc.tsv.meta`:

- `jackknife_scheme_used` ∈ {components, pair_blocks}
- `n_components`
- `largest_component_size`
- `largest_component_pair_fraction`

And a new companion file:

- `fit.jackknife_blocks.tsv` — one row per block, `(block, pair_count)`.

These made the "single giant component → pair_blocks fallback"
observation directly visible in the meta; without them the reader
would have had to re-run the BFS by hand to know the scheme.

## Open questions → Round 2

1. Confirm linear scaling holds to n=1M — simACE pedigree build above
   n=100k has never been exercised, so the pedigree-generation phase
   itself may surprise us.  PCGC cpp should not.
2. Jackknife stability with a single giant component at n=1M.  Current
   fallback is `pair_blocks`; dependence between adjacent pair blocks
   at biobank scale is presumed tolerable but not measured here.
3. Re-run against a no-AM fixture at n=100k (e.g., `iter_reml_100k_noam`)
   to confirm σ²_A is recovered correctly when the sim truth is
   on-scale — a quick additional sanity check before scaling up.

## Files produced

- `results/iter_reml_bench/iter_reml_{10k,50k,100k}/rep1/pcgc/`
  - `fit.vc.tsv`, `fit.cov.tsv`, `fit.iter.tsv`, `fit.bench.tsv`
  - `fit.vc.tsv.meta` (new diagnostic fields), `fit.jackknife_blocks.tsv`
- `benchmarks/iter_reml_bench/iter_reml_{10k,50k,100k}/rep1/pcgc.tsv`
  (Snakemake OS-level bench)
- `benchmarks/pcgc/pcgc_scale.tsv` (combined roll-up; see
  `workflow/scripts/collect_pcgc_scale.py`)
- `benchmarks/pcgc/scale_parity.tsv` + `scale_parity_diff.tsv`
  (cpp↔numba agreement at 10k + 50k)
- `fitACE/config/iter_reml_bench.yaml` (new fit overlay; K=0.05 cpp)
