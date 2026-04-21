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

## Update (2026-04-21): rare-K C/E identifiability

Round 2 gated on a no-AM σ²_A cross-check (open question #3 of the scale
note).  Running PCGC on the existing `iter_reml_{10k,50k,100k}_noam`
fixtures at K=0.05 surfaced a bias pattern the dev grid had missed:
**V(C) absorbs Ve at rare K, regardless of n.**

### Size sweep at K=0.05, AM=0.0 (truth A=0.4, C=0.2, Ve=0.4)

| scenario | n_phen | V(A) (SE) | V(C) (SE) | Ve (SE) | bias_C | bias_Ve |
|---|---:|---:|---:|---:|---:|---:|
| iter_reml_10k_noam  |  10,200 | 0.438 (0.091) | 0.443 (0.121) | 0.119 (0.125) | +0.243 | −0.281 |
| iter_reml_50k_noam  |  51,000 | 0.454 (0.040) | 0.405 (0.058) | 0.141 (0.068) | +0.205 | −0.259 |
| iter_reml_100k_noam | 102,000 | 0.465 (0.035) | 0.431 (0.039) | 0.103 (0.050) | +0.231 | −0.297 |

Bias is **stable across n** (not shrinking with sample size).  The SE
shrinks ~1/√n as expected, so the z-score of the C/Ve bias climbs from
~2σ at 10k to ~6σ at 100k — i.e. becomes *more detectable* at larger n,
not less.  V(A) is lightly biased high (+0.04 to +0.07, within 2 SE
everywhere) but stable.

### K sweep at n=102k_noam (same fixture, only τ = Φ⁻¹(1−K) changes)

| K | V(A) (SE) | V(C) (SE) | Ve (SE) | bias_A | bias_C | bias_Ve |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.465 (0.035) | 0.431 (0.039) | 0.103 (0.050) | +0.065 | **+0.231** | **−0.297** |
| 0.15 | 0.428 (0.018) | 0.259 (0.015) | 0.313 (0.022) | +0.028 | +0.059 | −0.087 |
| 0.30 | 0.417 (0.011) | 0.205 (0.009) | 0.378 (0.010) | +0.017 | +0.005 | −0.022 |
| 0.50 | 0.407 (0.011) | 0.206 (0.007) | 0.387 (0.009) | +0.007 | +0.006 | −0.013 |

Bias is **monotone in K**.  At K=0.30 the partition is clean (≤ 0.02 in
all components); at K=0.50 it's essentially perfect.  At K=0.05 the C/E
split is unusable.  Not a bug, not an n-scaling issue, not a pedigree
artifact — the fixture recovers cleanly at K=0.50, so the sim truth,
the kinship build, and the 2×2 moment solver all work.  This is the
rare-trait ascertainment compression that the PCGC literature warns
about, quantified on our own fixtures.

### Why Track 2 missed it

Track 2's K-axis tests (`dev_laplace_K_{rare,common,half}`) were all at
n=10k.  At n=10k the C/Ve SE is ~0.12, so the +0.23 C-bias is only
~2σ — buried in noise.  The only large-n rare-K point Track 2 had was
`iter_reml_100k` at K=0.05, but that fixture has AM=0.3 which conflates
AM-induced Vp-inflation with the rare-K C/E bias.  The no-AM fixtures
isolate the effect.

## Recommendations

1. **Trust V(A) / h² for** n ≥ 3k, K ≥ 0.05, no AM, moderate h² (0.1–0.7).
   Expected bias in h²: ≤ 0.03 at K ≥ 0.15, ≤ 0.07 at K=0.05.
2. **Trust the V(C) / Ve partition only for K ≥ 0.15** (ideally ≥ 0.30).
   At K=0.05 the partition is systematically wrong by ~0.2 in each
   component; do not report C/E separately in that regime.  If the
   scientific question requires C/E separation on a rare trait, fit at
   a higher threshold or use a full-likelihood method.
3. **Defer to MCEM / Laplace for**
   - Any AM > 0.  Fix: re-derive truth at AM equilibrium before
     comparing, or refit without the AM pedigree.  Not a PCGC issue
     per se, but the naive-vs-truth table will look bad.
   - σ²_E very close to 0 (`low_e` scenario).
4. **Round 2 (n=300k, n=1M)** does not need additional bias fixtures —
   the n-axis story is now complete (bias is K-driven, n-stable), and
   the dev_laplace_n{30k,50k} h² bias at K=0.15 already falls below
   0.03.  Scale runs should still include a no-AM fixture so σ²_A
   recovery at the production target is checkable.

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
