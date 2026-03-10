# Plan: Frailty-Model Heritability Estimation for EPIMIGHT Pipeline

## Context

EPIMIGHT estimates h² using Falconer's formula on binary affected/unaffected outcomes, which systematically underestimates liability-scale h² because the frailty model's stochastic hazard process adds ~π²/6 Gumbel noise. The fix: estimate liability correlation directly from survival times using the shared-frailty MLE already implemented in `sim_ace/survival_corr.py`, then convert to h² via known kinship coefficients. This avoids the binary-outcome bottleneck.

## Key Existing Code to Reuse

- **`sim_ace/survival_corr.py`** — `pairwise_weibull_corr_se()` (same-trait pair MLE), `cross_trait_weibull_corr_se()` (cross-trait MLE), `compute_pair_corr()` (wrapper)
- **`epimight/create_parquet.py`** — `KIND_TO_PAIRS` mapping, `BASE_YEAR` constant
- **`sim_ace/pedigree_graph.py`** — `extract_relationship_pairs()` returns all 10 pair types
- **`workflow/scripts/compute_phenotype_stats.py`** — pattern for passing frailty params from Snakemake config

## Files to Create/Modify

### 1. CREATE `epimight/frailty_h2.py` — Core computation

Main function: `compute_frailty_h2_for_epimight(phenotype_path, output_dir, beta1, hazard_model1, hazard_params1, beta2, hazard_model2, hazard_params2, kinds, seed, n_quad, max_pairs)`

Logic:
- Load phenotype.parquet, extract pairs via `extract_relationship_pairs()`
- For each kind: combine pair types via `KIND_TO_PAIRS`, concatenate indices
- Filter MZ twins OUT of FS (they have r=1.0 and would bias FS upward)
- Call `pairwise_weibull_corr_se()` for each kind × trait
- Convert: `h² = r / kinship`, `se_h² = se_r / kinship`, `l95/u95 = h² ± 1.96·se_h²`
- Per-generation stratification: filter pairs by younger member's generation
- Cross-trait GC per kind: pair (trait1 of person A, trait2 of person B) using `cross_trait_weibull_corr_se()`
- Output: `frailty_h2.json`, `tsv/frailty_h2_d1.tsv`, `tsv/frailty_h2_d2.tsv`, `tsv/frailty_gc.tsv`

Kind-to-kinship mapping:
```
PO=0.5, FS=0.5, HS=0.25, mHS=0.25, pHS=0.25, Av=0.25, 1G=0.25, 1C=0.125
```

TSV format (one row per kind × born_at_year):
```
born_at_year  kind  r  r_se  h2  h2_se  h2_l95  h2_u95  n_pairs
```

Handle non-Weibull hazard models gracefully (skip with NaN, log warning).

### 2. CREATE `workflow/scripts/epimight_frailty_h2.py` — Snakemake wrapper

Thin wrapper following `compute_phenotype_stats.py` pattern. Passes frailty params from `snakemake.params` to `compute_frailty_h2_for_epimight()`.

### 3. MODIFY `workflow/rules/epimight.smk` — Add rule + update atlas deps

Add `epimight_frailty_h2` rule:
- Input: `phenotype.parquet`
- Output: `frailty_h2.json`, `tsv/frailty_h2_d{1,2}.tsv`, `tsv/frailty_gc.tsv`
- Params: beta1/2, hazard_model1/2, hazard_params1/2, epimight_kinds, seed
- Resources: mem_mb=8000, runtime=120, threads=4 (Numba prange)

Update `epimight_atlas` inputs to include `frailty_h2.json`.

### 4. MODIFY `epimight/plot_epimight.py` — Overlay frailty estimates

Changes:
- Add `load_frailty_h2_tsv()` loader
- Modify `plot_h2_bar()`: accept optional frailty data, overlay diamond markers with error bars at each birth cohort's frailty h², plus a dashed line for the pooled frailty h²
- Modify `plot_gc_bar()`: same overlay approach for frailty GC
- Modify `plot_summary_table()`: add "h² d1 (frailty)" and "h² d2 (frailty)" columns
- Add `plot_frailty_vs_falconer()`: grouped bar chart (blue=Falconer, orange=Frailty) per kind, one figure per disorder
- Update `_PLOT_BASENAMES`, `EPIMIGHT_CAPTIONS`, and `assemble_epimight_atlas()`

### 5. MODIFY `sim_ace/survival_corr.py` — Proper left-censoring likelihood

Add `left_censored_i` and `left_censored_j` optional boolean arrays to `pairwise_weibull_corr_se()` and the Numba kernels.

Current likelihood contributions:
- Event (δ=1): `h(t|L) · S(t|L)` → `log g = const + β·L - H_base · exp(β·L)`
- Right-censored (δ=0): `S(t|L)` → `log g = -H_base · exp(β·L)`

Add for left-censored (event before observation window):
- Left-censored: `F(t|L) = 1 - S(t|L)` → `log g = log(1 - exp(-H_base · exp(β·L)))`
- Use `log1p(-exp(-x))` for numerical stability (or `log(-expm1(-x))` when x is small)

The `frailty_h2.py` code passes `age_censored` from phenotype.parquet as the left-censoring indicator. `pairwise_weibull_corr_se` defaults to `left_censored=None` (all False) for backward compatibility.

No changes to `sim_ace/utils.py` — PAIR_TYPES is not extended. The new code iterates over KIND_TO_PAIRS and calls `pairwise_weibull_corr_se` directly.

## Design Details

### MZ Twin Handling

MZ twins share 100% of genetic variance (kinship = 1.0) vs full siblings (kinship = 0.5). Since `KIND_TO_PAIRS["FS"]` includes both "Full sib" and "MZ twin", pooling them would bias the FS correlation estimate upward. The frailty code filters MZ twins out of the FS pool by excluding pairs where either member has `twin >= 0` in `phenotype.parquet`.

### Per-Generation Stratification

For each `born_at_year` (= `BASE_YEAR + generation`):
- Same-generation pairs (FS, HS, mHS, pHS, 1C): both members in target generation
- Cross-generation pairs (PO, Av, 1G): younger member in target generation

This matches EPIMIGHT's convention where cohort membership is determined by the index individual.

### Cross-Trait Genetic Correlation

For each relationship kind, estimate cross-trait liability correlation by pairing trait 1 of person A with trait 2 of person B (where A and B are relatives). The existing `cross_trait_weibull_corr_se()` accepts separate beta/scale/rho per "trait side", so we pass trait 1 params for person A and trait 2 params for person B. The genetic correlation is then:

```
ρ_g = r_cross / (kinship × √(h²_d1 × h²_d2))
```

where `h²_d1` and `h²_d2` come from the same-trait frailty estimates.

### Non-Weibull Hazard Models

Some scenarios use lognormal, gamma, loglogistic, or Gompertz baselines. The frailty MLE in `survival_corr.py` currently only supports Weibull. For non-Weibull scenarios, the frailty rule outputs NaN values and logs a warning. The atlas handles NaN gracefully (empty panels).

## Verification

1. Dry run: `snakemake -n results/long/long_term/rep1/epimight/frailty_h2.json`
2. Run: `snakemake --cores 4 results/long/long_term/rep1/epimight/frailty_h2.json`
3. Check frailty h² is closer to true h² (0.50) than Falconer h² (~0.10-0.13)
4. Rebuild atlas: `snakemake --cores 4 -f results/long/long_term/rep1/epimight/plots/atlas.pdf`
5. Verify overlays render correctly on bar charts and summary table
