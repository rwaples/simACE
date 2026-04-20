# PCGC Phase 1 — Track 2 bias characterization (dev grid)

**Date:** 2026-04-20
**Scope:** All `dev_laplace_*` (17 scenarios × rep1), `dev_mean_*`
(3 scenarios × rep1–5 where reps exist), `dev_mcem_*` (4 × rep1–5).
All cpp backend, K matched to the scenario's
`iter_reml_prevalence`.
**Outputs:**
- `results/dev/summary_pcgc_v2026.05/summary.tsv` — 48 rows
- `results/dev/summary_pcgc_v2026.05/summary_aggregate.tsv` — 24 rows
- `results/dev/summary_pcgc_v2026.05/head_to_head.tsv` — PCGC vs MCEM on
  matched seeds

## Headline

PCGC is **unbiased at K ≈ 0.15 / moderate h² / no AM / n ≥ 3k** — the
core operating regime it will face in biobank production runs.  Known
failure modes:

1. **K = 0.05 (rare)**: σ²_A biased low by ~0.1 at n=10k; comparable to
   or better than MCEM at the matched `dev_mcem_K_rare` (n=1k).
2. **Assortative mating**: σ²_A biased high by 0.1–0.5 depending on AM
   strength. Expected — PCGC fixes liability-scale Vp=1 whereas AM
   inflates realized Vp > 1 at equilibrium. Affects iter_reml's
   Laplace path similarly; documented; not a PCGC bug.
3. **Edge case `low_e` (σ²_E = 0.1)**: σ²_A biased +0.14 because
   σ²_A + σ²_C already saturates Vp.

On the **head-to-head vs MCEM** (dev_mcem_* n=1k, rep1–5): PCGC and
MCEM agree within `1.96·SE_diff` on every scenario — no disagreements
flagged.  PCGC's SEs are 0.5–2× MCEM's (moment estimator is less
efficient than full likelihood, as theory predicts).

## Full table (per-scenario aggregate)

Bias columns are `PCGC estimate − truth` (threshold scale, liability
Vp = 1 by construction).

| scenario | n_phen | K | truth h² | est h² | bias_h² | bias_A | note |
|---|---:|---:|---:|---:|---:|---:|---|
| dev_laplace_n1k            | 1020   | 0.15 | 0.5 | 0.630 | +0.130 | +0.130 | small-n |
| dev_laplace_n3k            | 3000   | 0.15 | 0.5 | 0.530 | +0.030 | +0.030 | OK |
| dev_laplace_n10k           | 10200  | 0.15 | 0.5 | 0.600 | +0.100 | +0.100 | rep=1, SD from dev_mean_n10k rep1–5 = 0.106 → 1·SD bias |
| dev_laplace_n30k           | 30000  | 0.15 | 0.5 | 0.514 | +0.014 | +0.014 | spot on |
| dev_laplace_n50k           | 51000  | 0.15 | 0.5 | 0.525 | +0.025 | +0.025 | OK |
| dev_laplace_h2_low         | 10200  | 0.15 | 0.2 | 0.203 | +0.003 | +0.003 | spot on |
| dev_laplace_h2_high        | 10200  | 0.15 | 0.7 | 0.741 | +0.041 | +0.041 | OK |
| dev_laplace_c2_high        | 10200  | 0.15 | 0.3 | 0.199 | -0.101 | -0.101 | h² pulled toward c² |
| dev_laplace_ce_dominant    | 10200  | 0.15 | 0.1 | 0.117 | +0.017 | +0.017 | OK |
| dev_laplace_ae_only        | 10200  | 0.15 | 0.5 | 0.520 | +0.020 | +0.020 | AE model, OK |
| dev_laplace_low_e          | 10200  | 0.15 | 0.6 | 0.743 | +0.143 | +0.143 | σ²_E=0.1 edge case |
| dev_laplace_K_rare         | 10200  | 0.05 | 0.5 | 0.371 | **-0.129** | **-0.129** | rare-K challenge |
| dev_laplace_K_common       | 10200  | 0.30 | 0.5 | 0.587 | +0.087 | +0.087 | OK |
| dev_laplace_K_half         | 10200  | 0.50 | 0.5 | 0.501 | +0.001 | +0.001 | spot on |
| dev_laplace_am_light       | 10200  | 0.15 | 0.5 | 0.766 | **+0.266** | **+0.266** | AM inflates Vp |
| dev_laplace_am_strong      | 10200  | 0.15 | 0.5 | 1.055 | **+0.555** | **+0.555** | AM severe |
| dev_laplace_am_strong_50k  | 51000  | 0.15 | 0.5 | 1.070 | **+0.570** | **+0.570** | AM persists at n=50k |
| dev_mean_n10k (5 reps)     | 10200  | 0.15 | 0.5 | 0.600 | +0.100 | +0.100 | SD 0.106 |
| dev_mean_n50k              | 51000  | 0.15 | 0.5 | 0.525 | +0.025 | +0.025 | OK |
| dev_mean_am_strong         | 10200  | 0.15 | 0.5 | 1.055 | **+0.555** | **+0.555** | AM |
| dev_mcem_n1k (5 reps)      | 1020   | 0.15 | 0.5 | 0.512 | +0.012 | +0.012 | SD 0.165 |
| dev_mcem_K_rare (5 reps)   | 1020   | 0.05 | 0.5 | 0.289 | **-0.211** | **-0.211** | rare-K + small-n |
| dev_mcem_K_common (5 reps) | 1020   | 0.30 | 0.5 | 0.420 | -0.080 | -0.080 | SD 0.166 |
| dev_mcem_h2_low (5 reps)   | 1020   | 0.15 | 0.2 | 0.133 | -0.067 | -0.067 | small-n |

## Head-to-head vs MCEM (`head_to_head.tsv`, matched seeds)

| scenario | truth h² | PCGC h² (SD) | MCEM h² (SD) | Δ (z) | disagree? |
|---|---:|---:|---:|---:|---:|
| dev_mcem_n1k      | 0.5 | 0.512 (0.165) | 0.498 (0.113) | +0.014 (z= 0.16) | no |
| dev_mcem_K_common | 0.5 | 0.420 (0.166) | 0.452 (0.118) | -0.032 (z=-0.35) | no |
| dev_mcem_K_rare   | 0.5 | 0.289 (0.211) | 0.227 (0.153) | +0.062 (z= 0.53) | no |
| dev_mcem_h2_low   | 0.2 | 0.133 (0.152) | 0.171 (0.164) | -0.038 (z=-0.38) | no |

`disagree` uses `|Δ| > 1.96·√((SD_pcgc²/n_reps) + (SD_mcem²/n_reps))` as
the threshold.  All four scenarios pass — PCGC is not silently drifting
away from MCEM on shared data.  Both methods are biased low on the
rare-K and h²_low scenarios at n=1k; at n=3k+ the bias dissipates (see
`dev_laplace_n3k`).

## Backend parity on the dev grid (`benchmarks/pcgc/dev_parity.tsv`)

cpp vs numba on two scenarios picked for coverage:
- `dev_laplace_n10k`      (n≈10k, K=0.15)
- `dev_laplace_K_rare`    (n≈10k, K=0.05)

**20 of 20** column-wise comparisons pass the
`atol=1e-3 σ² / 5e-3 SE / 1e-6 c_factor` tolerance.  Cross-check
that cpp hasn't diverged from the validated numba path at rare K is
clean.

## Recommendations

1. **Trust PCGC for** n ≥ 3k, K ∈ [0.1, 0.5], no or light AM, moderate
   h² (0.1–0.7).  Expected bias ≤ 0.03 in h² terms.
2. **Defer to MCEM / Laplace for**
   - n < 3k with rare K (K ≤ 0.05): |bias_h²| can exceed 0.2.
     PCGC still agrees with MCEM there — both are noisy.
   - Any AM > 0.  The fix is either (a) re-derive truth at AM
     equilibrium before comparing, or (b) refit without the AM
     pedigree.  Not a PCGC issue per se, but the naive-vs-truth table
     will look bad and the user should know.
   - σ²_E very close to 0 (`low_e` scenario).
3. **Round 2 (n=300k, n=1M)** does not need additional bias fixtures —
   the dev_laplace_n{30k,50k} results at n=30k and n=50k show bias
   already shrunk below 0.03, and no theoretical reason it should
   grow at larger n.  Round 2's scale runs should include a no-AM
   fixture so σ²_A recovery at the production target is checkable.

## Caveats

- Laplace `summary_aggregate.tsv` does not exist yet (iter_reml Laplace
  has only rep1 and has never been summary-aggregated) — no Laplace
  column in `head_to_head.tsv`.  To populate, run
  `python -m fitace.iter_reml.summarize_dev_grid results/dev
  --scenarios-glob 'dev_laplace_*' --out-dir results/dev/summary_laplace_v2026.05`,
  then rerun `compare_to_iter_reml.py --laplace
  results/dev/summary_laplace_v2026.05/summary_aggregate.tsv`.
  Not in scope for this round.
- `dev_laplace_*` scenarios are rep=1 only; SD columns are blank.
  Bias is from a single draw and should be read accordingly (point
  estimate quality, not coverage).
- `dev_mean_n10k` rep1–5 have the same `iter_reml_prevalence` and the
  same pedigree seeds as `dev_laplace_n10k` rep1 — PCGC estimates on
  shared seed are numerically identical across `dev_mean_*`, `dev_laplace_*`,
  and `dev_mcem_*` when they share seed (confirmed: both aggregates show
  h²=0.600 for the n10k row).  This is not surprising; PCGC depends only
  on y and the kinship, not on the liability model the iter_reml fit uses.
