# ACE Simulation Validation

*Last updated: 2026-02-16*

This document describes every validation step in the ACE simulation pipeline.
The pipeline validates pedigree structure, variance-component statistics, twin
and half-sibling relationships, heritability estimates, and two downstream
phenotype models (Weibull time-to-event and liability threshold).

---

## Pipeline data flow

```
pedigree.parquet + params.yaml
        │
        ▼
   validate.py            ──►  validation.yaml   (per-rep structural checks)
        │
        ▼
  gather_validation.py    ──►  validation_summary.tsv  (all scenarios × reps)
        │
        ▼
  plot_validation.py      ──►  results/plots/*.png     (cross-scenario summary)

phenotype.weibull.parquet
        │
        ▼
  compute_phenotype_stats.py  ──►  phenotype_stats.yaml + phenotype_samples.parquet
        │
        ▼
  plot_phenotype.py           ──►  results/{scenario}/plots/*.png

phenotype.liability_threshold.parquet
        │
        ▼
  compute_threshold_stats.py  ──►  threshold_stats.yaml + threshold_samples.parquet
        │
        ▼
  plot_threshold.py           ──►  results/{scenario}/plots/*.png
```

---

## 1. Pedigree validation (`validate.py`)

`validate.py` reads `pedigree.parquet` and `params.yaml` for a single
scenario replicate and produces `validation.yaml`. Every check returns
`{passed, details}` plus observed/expected values.

### 1.1 Structural checks (`validate_structural`)

| Check | Rule | Tolerance |
|---|---|---|
| **id_integrity** | IDs are 0 .. N×G_ped − 1, contiguous | Exact match |
| **parent_references** | Every `mother`/`father` value is a valid ID or −1 (founder); no self-parenting | Exact |
| **sex_parent_consistency** | All mothers are female (sex = 0), all fathers are male (sex = 1) | Exact |
| **sex_distribution** | Proportion male ∈ [0.45, 0.55] | ±0.05 from 0.5 |

### 1.2 Twin checks (`validate_twins`)

| Check | Rule | Tolerance |
|---|---|---|
| **twin_bidirectional** | If A→B twin link exists, B→A must also exist | Exact |
| **twin_same_parents** | Both members of every MZ pair share the same mother and father | Exact |
| **twin_same_A1 / twin_same_A2** | MZ twins have identical additive genetic values per trait (`np.allclose`) | Floating-point allclose |
| **twin_same_sex** | MZ twins have the same sex | Exact |
| **twin_rate** | Observed MZ twin rate among non-founders ≈ `p_mztwin` | max(4 × SE, 0.005), where SE = √(p(1−p)/n) |

### 1.3 Half-sibling checks (`validate_half_sibs`)

Sibling pairs are counted via vectorized self-merge on `mother`/`father`
columns. Twins are excluded.

| Check | Rule | Formula | Tolerance |
|---|---|---|---|
| **half_sib_pair_proportion** | Fraction of maternal sibling pairs that are half-sibs ≈ expected | Expected = 1 − (1 − p_nonsocial)² | max(4 × SE, 0.02) |
| **offspring_with_half_sib** | Fraction of offspring (who have siblings) that have ≥ 1 maternal half-sib ≈ expected | Expected = 1 − (1 − p_nonsocial) × exp(−fam_size × p_nonsocial) | max(4 × SE, 0.02) |

### 1.4 Statistical checks (`validate_statistical`)

Computed on **founders** (generation 0) unless noted otherwise.

| Check | Rule | Tolerance |
|---|---|---|
| **variance_A1/C1/E1/A2/C2/E2** | Var(component) in founders ≈ parametric value | abs(obs − expected) < 0.1 |
| **total_variance_trait1/trait2** | Sum of A+C+E variances ≈ 1.0 | abs(total − 1.0) < 0.15 |
| **cross_trait_rA** | cor(A1, A2) in founders ≈ `rA` param | abs(obs − expected) < 0.15 |
| **cross_trait_rC** | cor(C1, C2) in founders ≈ `rC` param | abs(obs − expected) < 0.15 |
| **cross_trait_rE** | cor(E1, E2) in founders ≈ 0 | abs(obs) < 0.1 |
| **c1_inheritance / c2_inheritance** | Within each family (same mother), all non-founder siblings share the same C value | Proportion shared > 0.99 |
| **e1_independence** | E1 correlation between siblings ≈ 0 (sampled from up to 500 families, 1000 pairs) | abs(r) < 0.1 |

### 1.5 Heritability checks (`validate_heritability`)

#### MZ twin correlations
| Check | Rule | Tolerance |
|---|---|---|
| **mz_twin_A1/A2_correlation** | cor(A) between MZ co-twins ≈ 1.0 (shared genotype) | r > 0.99 |
| **mz_twin_liability1/liability2_correlation** | cor(P) between MZ co-twins (informational, no pass/fail) | — |

#### DZ (full-sib) correlations

Full-sib pairs are extracted via merge (same mother AND same father, excluding
twins). Subsampled to 5000 pairs max.

| Check | Rule | Tolerance |
|---|---|---|
| **dz_sibling_A1/A2_correlation** | cor(A) between DZ siblings ≈ 0.5 | max(4 × SE, 0.05), where SE = (1 − r²)/√(n − 1) |
| **dz_sibling_liability1/liability2_correlation** | cor(P) between DZ siblings (informational) | — |

#### Falconer's heritability estimate

| Check | Formula | Tolerance |
|---|---|---|
| **falconer_estimate_trait1/trait2** | h² = 2(r_MZ − r_DZ) using liability correlations | max(4 × SE_falconer, 0.05), where SE = 2√(SE_MZ² + SE_DZ²) |

#### Parent–offspring regression

Midparent–offspring regressions on both the A component and full liability.
Requires > 100 valid offspring with both parents present.

| Metric | Interpretation |
|---|---|
| **parent_offspring_A1/A2_regression** | Slope ≈ 1.0 for additive genetic values |
| **parent_offspring_liability1/liability2_regression** | Slope ≈ h² for liability |

### 1.6 Population checks (`validate_population`)

| Check | Rule | Tolerance |
|---|---|---|
| **generation_sizes** | Every generation has exactly N individuals | Exact |
| **generation_count** | Number of generations = G_ped | Exact |
| **family_size** | Mean offspring per mother ≈ `fam_size` | abs(obs − expected) < fam_size × 0.5 |

### 1.7 Per-generation statistics (`compute_per_generation_stats`)

Informational only (no pass/fail). For each generation and each trait,
records:

- Liability mean, variance, SD
- Mean and variance of A, C, E components

### 1.8 Family size distribution (`compute_family_size_distribution`)

Informational. Reports mean, median, std, and parent count for offspring per
mother and per father.

### 1.9 Summary

The validation summary counts all checks with a `passed` field:

```yaml
summary:
  passed: true/false
  checks_passed: <int>
  checks_failed: <int>
  checks_total: <int>
```

---

## 2. Weibull phenotype statistics (`compute_phenotype_stats.py`)

Reads `phenotype.weibull.parquet` (one rep) and produces `phenotype_stats.yaml`.
All statistics are computed per trait (trait 1 and trait 2).

### 2.1 Prevalence

Simple affected-fraction: `mean(affected1)`, `mean(affected2)`.

### 2.2 Mortality

Decadal mortality rates from 0 to `censor_age` (e.g., 0–9, 10–19, ...).
For each decade: `died_in_decade / alive_at_start`.

### 2.3 Regression (liability vs age at onset)

Linear regression of `liability` → `t_observed` among affected individuals.
Reports slope, intercept, R², and n.

### 2.4 Cumulative incidence

200-point curves from age 0 to `censor_age`:
- **Observed**: fraction affected with `t_observed ≤ age`
- **True**: fraction with raw `t ≤ age` (ignoring censoring)
- **half_target_age**: age at which 50% of lifetime observed cases are diagnosed

### 2.5 Censoring windows

Per-generation breakdown showing:
- True vs observed incidence curves
- Percent affected, left-censored, right-censored, death-censored

Requires `gen_censoring` parameter: a dict mapping generation number to
`[lo, hi]` age windows (e.g. `{0: [40, 80], 1: [0, 80], 3: [0, 45]}`).

### 2.6 Relationship pair extraction (`extract_relationship_pairs`)

Identifies aligned row-index arrays for seven relationship types, shared by
both liability and tetrachoric correlation computations:

| Relationship | Detection method |
|---|---|
| MZ twin | `twin != -1`, deduplicated (id < twin) |
| Full sib | Same mother AND same father, non-twin non-founders |
| Maternal half sib | Same mother, different father |
| Paternal half sib | Same father, different mother |
| Mother–offspring | Non-founders mapped to mother row |
| Father–offspring | Non-founders mapped to father row |
| 1st cousin | Share a grandparent via different parents (capped at 100K grandparents) |

### 2.7 Liability correlations

Pearson correlation of `liability` values for each pair type (minimum 10 pairs).

### 2.8 Tetrachoric correlations

MLE-based tetrachoric correlation on binary `affected` status for each pair
type. Uses Owen's T function for the bivariate normal CDF. Reports r, SE
(from observed Fisher information), and n_pairs.

### 2.9 Downsampled parquet

A random 100K-row sample saved as `phenotype_samples.parquet` for plotting.

---

## 3. Liability threshold statistics (`compute_threshold_stats.py`)

Reads `phenotype.liability_threshold.parquet` and produces
`threshold_stats.yaml`. Shares `extract_relationship_pairs`,
`compute_liability_correlations`, and `tetrachoric_corr_se` from
`compute_phenotype_stats.py`.

### 3.1 Prevalence by generation

Per-trait, per-generation prevalence plus overall prevalence.

### 3.2 Joint affection

2 × 2 contingency table for trait 1 × trait 2 affection status:
`{both, trait1_only, trait2_only, neither}` as counts and proportions.

### 3.3 Liability by status

Mean and SD of liability for affected vs unaffected individuals, per trait.

### 3.4 Liability correlations

Same as Weibull model (Pearson on continuous liability), seven pair types.

### 3.5 Tetrachoric correlations

Same MLE approach as Weibull model, seven pair types, per trait.

### 3.6 Downsampled parquet

Random 100K-row sample saved as `threshold_samples.parquet` for plotting.

---

## 4. Aggregation (`gather_validation.py`)

Reads every `results/{scenario}/rep{N}/validation.yaml` and writes a single
`results/validation_summary.tsv` with one row per scenario × rep. Columns
include:

- Scenario metadata: `scenario`, `rep`, `N`, `G_ped`, `G_sim`, `seed`
- Variance parameters: `A1`, `C1`, `E1`, `A2`, `C2`, `E2`, `rA`, `rC`
- Population parameters: `p_mztwin`, `p_nonsocial_father`, `fam_size`
- Summary: `checks_failed`
- Observed variances: `variance_A1` ... `variance_E2`
- Observed cross-trait correlations: `observed_rA`, `observed_rC`, `observed_rE`
- Twin correlations: `mz_twin_A1_corr`, `mz_twin_liability1_corr`, etc.
- Sibling correlations: `dz_sibling_A1_corr`, `dz_sibling_liability1_corr`, etc.
- Half-sib stats: `half_sib_prop_expected/observed`, `offspring_with_half_sib_expected/observed`
- Heritability: `falconer_h2_trait1/trait2`
- Regressions: `parent_offspring_A1_slope/r2`, `parent_offspring_liability1_slope/r2`, etc.
- Family size: `mother_mean_offspring`, `father_mean_offspring`

---

## 5. Summary plots (`plot_validation.py`)

Reads `validation_summary.tsv` and produces nine PNG files in `results/plots/`:

| Plot | What it shows |
|---|---|
| `variance_components.png` | Observed vs expected A/C/E variances (2 traits × 3 components) |
| `twin_rate.png` | Observed vs expected MZ twin rate |
| `correlations_A.png` | MZ twin A1 corr, DZ sib A1 corr, half-sib A1 corr, midparent–offspring A1 R² |
| `correlations_phenotype.png` | MZ/DZ/half-sib liability correlations and midparent–offspring liability slope |
| `heritability_estimates.png` | Falconer h² and midparent–offspring slopes for both traits |
| `half_sib_proportions.png` | Observed vs expected half-sib pair proportion and offspring-with-half-sib fraction |
| `cross_trait_correlations.png` | Observed vs expected rA, rC, rE |
| `family_size.png` | Mean offspring per mother and per father vs parametric `fam_size` |
| `summary_bias.png` | Bias (observed − expected) for A1, C1, E1 variances, twin rate, DZ A1 corr, half-sib A1 corr |

All plots use stripplots with orange expected-value markers across scenarios.

---

## 6. Phenotype-model plots

### Weibull plots (`plot_phenotype.py` → `results/{scenario}/plots/`)

| Plot | Content |
|---|---|
| `death_age_distribution.png` | Death age histogram |
| `phenotype_traits.png` | Per-trait distributions of age at onset |
| `liability_regression.png` | Liability vs observed age at onset scatterplot |
| `liability_joint.png` | Liability1 vs liability2 joint scatter |
| `liability_violin.png` | Liability distributions for affected vs unaffected |
| `cumulative_incidence.png` | True vs observed cumulative incidence curves |
| `censoring_windows.png` | Per-generation censoring window incidence |
| `joint_affection.png` | 2 × 2 heatmap of trait 1 × trait 2 affection |
| `tetrachoric_relationship.png` | Tetrachoric correlations by relationship type |

### Threshold plots (`plot_threshold.py` → `results/{scenario}/plots/`)

| Plot | Content |
|---|---|
| `prevalence_by_generation.png` | Per-generation prevalence for each trait |
| `liability_threshold_violin.png` | Liability distributions for affected vs unaffected |
| `threshold_tetrachoric.png` | Tetrachoric correlations by relationship type |
| `threshold_joint_affection.png` | 2 × 2 heatmap of trait 1 × trait 2 affection |
| `threshold_liability_joint.png` | Liability1 vs liability2 joint scatter |

---

## 7. Tolerance reference

Most tolerance formulas follow the pattern **max(k × SE, floor)** where k = 4
(≈ 99.99% under normality) and floor is a minimum tolerance to prevent
false failures at very large sample sizes.

| Check | SE formula | k | Floor |
|---|---|---|---|
| twin_rate | √(p(1−p)/n) | 4 | 0.005 |
| half_sib_pair_proportion | √(p(1−p)/n_pairs) | 4 | 0.02 |
| offspring_with_half_sib | √(p(1−p)/n_sibs) | 4 | 0.02 |
| DZ sibling A correlation | (1−r²)/√(n−1) | 4 | 0.05 |
| Falconer h² | 2√(SE_MZ² + SE_DZ²) | 4 | 0.05 |
| Variance components | — (fixed) | — | 0.1 |
| Total variance | — (fixed) | — | 0.15 |
| Cross-trait correlations | — (fixed) | — | 0.15 |
| E correlation (sibs) | — (fixed) | — | 0.1 |
| Sex distribution | — (fixed) | — | 0.05 |
| Family size | — | — | fam_size × 0.5 |
