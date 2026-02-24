# ACE Simulation Framework: Diagnostic Plots

The pipeline produces diagnostic and summary plots organised into five categories. All per-scenario plots are generated from pre-computed per-replicate statistics and downsampled data (capped at 100,000 points) to avoid loading full phenotype files at plot time.

---

## 1. Distribution Plots

Source: `sim_ace/plot_distributions.py`. Generated for Weibull frailty phenotypes.

### 1.1 Mortality rate by decade (`death_age_distribution.png`)

Two-panel figure (14 $\times$ 5 in).

**Left panel — Mortality Rate by Decade.**
- **Bars**: Per-decade mortality rate (deaths in decade / alive at start of decade), averaged across replicates. Semi-transparent blue fill with black edges.
- **x-axis**: Age decade labels (e.g., "0-10", "10-20", ...).
- **y-axis**: Mortality rate (proportion).

**Right panel — Cumulative Mortality by Decade.**
- **Bars**: Cumulative mortality $F(t) = 1 - \prod_{d \leq t}(1 - r_d)$, same style as left.
- **Text annotations**: Above each bar, survival probability $S = 1 - F(t)$ formatted as "S=0.XX".

### 1.2 Age-at-onset and death-age histograms (`trait_phenotype.png`)

A $2 \times 2$ grid (14 $\times$ 10 in), rows = traits 1 and 2.

**Left column — Affected individuals.**
- **Histogram**: Density histogram (50 bins) of observed time-to-event (`t_observed`) for individuals with $\delta = 1$. Red fill (`C3`), black edges.
- **Title**: "Trait $k$: Age at Onset (affected)".

**Right column — Death-censored unaffected.**
- **Histogram**: Density histogram of observed time for individuals with $\delta = 0$ and death-censored = True. Blue fill (`C0`), black edges.
- **Title**: "Trait $k$: Age at Death (death-censored, unaffected)".

### 1.3 Liability vs. age-at-onset (`trait_regression.png`)

Side-by-side joint plots (16 $\times$ 7 in), one per trait. Each joint plot has:

- **Central scatter**: Liability ($x$) vs. observed age-at-onset ($y$) for affected individuals. Translucent blue points ($\alpha = 0.05$, size 3, rasterised).
- **Regression line**: Red (`C3`) line from pre-computed slope and intercept, averaged across replicates.
- **$R^2$ annotation**: Upper-left corner, "R$^2$ = X.XXXX".
- **Marginal histograms**: Top (liability) and right (age-at-onset), 50 bins, no edges.
- **Panel title**: "Trait $k$".

### 1.4 Cumulative incidence curves (`cumulative_incidence.png`)

Two-panel figure (14 $\times$ 5 in), shared $y$-axis, one panel per trait.

- **Blue solid line** ("Observed"): Mean cumulative incidence from censored data across replicates, with shaded min-max band when $>1$ replicate.
- **Grey solid line** ("True"): Mean cumulative incidence from uncensored event times, with shaded band.
- **Grey dashed crosshairs**: Horizontal line at half the observed lifetime prevalence; vertical line at the age when 50% of lifetime cases have occurred. A red dot marks the intersection.
- **Arrow annotation**: "50% of cases by age $X$" pointing to the crosshair intersection.
- **Text box** (upper-left, white background): "Affected: $X.X$%", "True prev: $X.X$%", "Censored: $X.X$%".
- **Legend**: Lower-right, showing "True" (grey) and "Observed" (blue) line styles.

### 1.5 Censoring windows by generation (`censoring_windows.png`)

Grid of panels: rows = traits ($\times 2$), columns = generations ($\times N$). Panel size $5N \times 8$ in.

- **Grey solid line** ("True"): True cumulative incidence within this generation, with grey fill to baseline.
- **Blue solid line** ("Observed"): Observed cumulative incidence, with blue fill. Min-max bands across replicates when $>1$ rep.
- **Text box** (upper-left, white background): "Affected: $X.X$%", "Left-cens: $X.X$%", "Right-cens: $X.X$%", "Death-cens: $X.X$%".
- **Column titles**: "Gen $g$" with observation window annotation "[lo, hi]" when available.
- **Row labels**: "Trait $k$ / Cumulative Incidence".
- **Legend** (top-right panel): Grey = True, Blue = Observed.

---

## 2. Liability Plots

Source: `sim_ace/plot_liability.py`. Generated for Weibull frailty phenotypes.

### 2.1 Cross-trait liability joint plots (`liability_joint.png`)

$2 \times 2$ grid of joint plots (14 $\times$ 12 in). Panels: Liability, $A$ (Additive genetic), $C$ (Common environment), $E$ (Unique environment).

- **Central scatter**: Trait 1 ($x$) vs. Trait 2 ($y$) for the given component. Blue points ($\alpha = 0.05$, size 3, rasterised).
- **Marginal histograms**: Top and right, 50 bins, semi-transparent.
- **Pearson $r$ annotation**: Upper-left of scatter, "r = X.XXXX".
- **Axis labels**: "$\text{Component}$ (Trait 1)" and "$\text{Component}$ (Trait 2)".

### 2.2 Liability joint plots coloured by affection (`liability_joint_affected.png`)

Same $2 \times 2$ layout as 2.1, but with affection-status colouring based on trait 1:

- **Blue points** ($\alpha = 0.03$): Unaffected individuals.
- **Red points** ($\alpha = 0.15$, `C3`): Affected individuals (trait 1).
- **Marginal histograms**: Stacked blue (unaffected) and red (affected).
- **Figure legend** (upper-right): Circle markers labelled "Unaffected" (blue) and "Affected (T1)" (red).
- **Pearson $r$ annotation**: Same as 2.1.

### 2.3 Liability violin plots by affection status (`liability_violin.png`)

Single figure (8 $\times$ 6 in) with split violins.

- **Split violins**: One per trait ($x$-axis). Left half = unaffected, right half = affected (coloured by seaborn hue).
- **Diamond markers** ($\blacklozenge$, black): Mean liability for each group, positioned slightly left (unaffected) or right (affected) of centre.
- **Text annotations**: "$\mu = X.XX$" next to each diamond.
- **Prevalence annotations**: Below each trait, "Prevalence: $X.X$%" in italic.
- **Legend**: Seaborn-generated hue legend for Affected True/False.

### 2.4 Liability violin plots by generation (`liability_violin_by_generation.png`)

Grid: rows = traits ($\times 2$), columns = recorded generations. Size $4N \times 8$ in.

- **Split violins**: Affected vs. unaffected within each generation, trimmed at data range (`cut=0`).
- **Diamond markers and $\mu$ annotations**: Same as 2.3, per panel.
- **x-axis label per panel**: "prev: $X.X$%" showing observed generation-specific prevalence.
- **Column titles**: "Gen $g$ (oldest)" for the first column, "Gen $g$ (youngest)" for the last.
- **Legend**: Upper-right panel only.

### 2.5 Joint affection heatmap (`joint_affection.png`)

Single figure (7 $\times$ 6 in).

- **$2 \times 2$ heatmap** (Blues colourmap): Rows = Trait 2 (Affected/Unaffected), Columns = Trait 1 (Affected/Unaffected).
- **Cell annotations**: Two-line labels showing proportion ("0.XX") and count ("(n=XXXXX)").
- **Colour bar**: Right side, labelled "Proportion".
- **Title**: "Joint Affected Status (Weibull) [$\text{scenario}$]" with subtitle showing:
  - `r_tet = X.XXX`: tetrachoric correlation on censored binary affected status.
  - `r_weibull = X.XXX`: cross-trait liability correlation estimated from uncensored Weibull survival data (oracle reference).
  - `(stratified: X.XXX)`: generation-stratified Weibull estimate — computes per-generation cross-trait correlations and combines via inverse-variance weighting ($\hat{r} = \sum w_g r_g / \sum w_g$ where $w_g = 1/\text{SE}_g^2$). This reduces bias from heterogeneous censoring across generations.
  - `(naive: X.XXX)`: unweighted pooled censored Weibull estimate for comparison (biased when censoring varies by generation). All values averaged across replicates.

---

## 3. Correlation Plots

Source: `sim_ace/plot_correlations.py`. Generated for Weibull frailty phenotypes.

### 3.1 Tetrachoric correlations by relationship type (`tetrachoric_sibling.png`)

Two-panel figure (16 $\times$ 6 in), one per trait. $y$-axis range: $[-0.1, 1.1]$.

- **Coloured violins**: Distribution of tetrachoric correlations (computed from censored binary affected status) across replicates, one per relationship type (MZ twin, Full sib, Mother-offspring, Father-offspring, Maternal half sib, Paternal half sib, 1st cousin). These represent the correlations estimable from observed data after censoring.
- **Black dots**: Individual per-replicate tetrachoric correlation estimates, jittered horizontally ($\pm 0.08$).
- **Black dashed horizontal segments**: Mean Pearson liability correlation for each pair type (averaged across replicates). This is the ground-truth correlation computed directly on the continuous latent liability values, serving as the theoretical reference.
- **Green dash-dot horizontal segments** (when available): Mean uncensored Weibull pairwise survival-time correlation. This shows what the correlation would be without censoring distortion.
- **Pair count annotations**: "N=$X$" above each violin (mean pairs per replicate).
- **Legend** (upper-right): "Liability r" (black dashed), "Weibull r (uncensored)" (green dash-dot, when present).
- **x-axis labels**: Relationship type names, rotated 15$^\circ$.
- **Interpretation**: The gap between the violins and the black dashed lines reflects attenuation due to censoring and the dichotomization inherent in the tetrachoric estimate. The green dash-dot lines show the intermediate effect of censoring alone (before dichotomization).

### 3.2 Tetrachoric correlations by generation (`tetrachoric_by_generation.png`)

Grid: rows = traits ($\times 2$), columns = generations. Size $5N \times 10$ in.

- **Violins, dots, dashed segments, pair counts**: Same encoding as 3.1 (violins = observed tetrachoric correlations, black dashed = true liability correlations, dots = per-replicate estimates), but computed within each generation separately.
- **x-axis labels**: Relationship type names, rotated 30$^\circ$, smaller font.
- **Column titles**: "Gen $g$".
- **Legend** (upper-right of top-right panel): "Liability r" (black dashed).

### 3.3 Parent-offspring liability regressions (`parent_offspring_liability.png`)

Grid: rows = traits ($\times 2$), columns = last 3 non-founder generations. Size $5N \times 8$ in.

- **Scatter**: Midparent liability ($x$) vs. offspring liability ($y$). Blue points ($\alpha = 0.15$, size 3, rasterised).
- **Red regression line** (`C3`): Least-squares fit through the scatter, linewidth 2.
- **Text box** (upper-left, white background): "r = X.XXX" (Pearson correlation, averaged across replicates) and "n = XXXXX" (mean pair count).
- **Column titles**: "Gen $g$".
- **Row labels**: "Trait $k$ / Offspring Liability" ($y$-axis), "Midparent Liability" ($x$-axis, bottom row only).

### 3.4 Narrow-sense heritability by generation (`heritability.by_generation.png`)

$1 \times 2$ figure (10 $\times$ 5 in), one panel per trait.

- **Blue dots**: Per-replicate $h^2 = \text{Var}(A) / (\text{Var}(A) + \text{Var}(C) + \text{Var}(E))$ for each generation, computed from per-generation variance components in `validation.yaml`.
- **Blue line**: Mean $h^2$ across replicates, connecting generation means.
- **Orange dashed line**: Configured heritability ($A_k$ parameter), the expected $h^2$.
- **$y$-axis**: $[0, 1]$, labelled $h^2 = \text{Var}(A) / \text{Var}(L)$.
- **$x$-axis**: Generation number.
- **Interpretation**: Stable $h^2$ across generations confirms that the ACE variance decomposition is maintained through the simulation. Founders (generation 1) have variance components set exactly; subsequent generations should converge to equivalent values under random mating.

Source: `sim_ace/plot_correlations.py`. Data from `validation.yaml` → `per_generation`.

### 3.5 Broad-sense heritability by generation (`broad_heritability.by_generation.png`)

$1 \times 2$ figure (10 $\times$ 5 in), one panel per trait.

- **Blue dots**: Per-replicate $H^2 = (\text{Var}(A) + \text{Var}(C)) / (\text{Var}(A) + \text{Var}(C) + \text{Var}(E))$ for each generation.
- **Blue line**: Mean $H^2$ across replicates.
- **Orange dashed line**: Parametric value ($A_k + C_k$).
- **$y$-axis**: $[0, 1]$, labelled $H^2 = (\text{Var}(A)+\text{Var}(C)) / \text{Var}(L)$.
- **$x$-axis**: Generation number.
- **Interpretation**: Comparing $H^2$ with the narrow-sense $h^2$ (3.4) isolates the contribution of shared environment to familial resemblance.

Source: `sim_ace/plot_correlations.py`. Data from `validation.yaml` → `per_generation`.

---

## 4. Threshold-Model Plots

Source: `sim_ace/plot_threshold.py`. Generated for liability-threshold binary phenotypes.

### 4.1 Prevalence by generation (`prevalence_by_generation.png`)

Bar chart comparing observed vs. configured prevalence per generation and trait. Configured values shown as reference markers.

### 4.2 Liability violin plots (`liability_violin.png`)

Split violins showing liability for affected vs. unaffected under the threshold model. Same encoding as Weibull violins (2.3): diamond mean markers, $\mu$ annotations, prevalence text.

### 4.3 Liability violin plots by generation (`liability_violin_by_generation.png`)

Per-generation split violins with configured prevalence annotated. Same encoding as Weibull per-generation violins (2.4).

### 4.4 Tetrachoric correlations (`tetrachoric.png`)

Violin plots of tetrachoric correlations by relationship type for threshold affected status indicators. Same encoding as 3.1: coloured violins show observed tetrachoric correlations from binary affected status, black dots are per-replicate estimates, black dashed lines are the ground-truth Pearson liability correlations, and pair counts are annotated above each violin.

### 4.5 Joint affection heatmap (`joint_affection.png`)

$2 \times 2$ heatmap of joint affection proportions and counts with tetrachoric correlation annotated. Same encoding as 2.5.

### 4.6 Cross-trait liability joint plot (`liability_joint.png`)

Scatter of trait 1 vs. trait 2 liability coloured by threshold affection status.

---

## 5. Validation Summary Plots

Source: `sim_ace/plot_validation.py`. Generated from `validation_summary.tsv` across all scenarios in a folder. Seaborn "whitegrid" theme with "Set2" palette.

All strip plots share a common encoding:
- **Blue dots** (`C0`, $\alpha = 0.7$, jitter 0.15): Per-replicate observed values, one column per scenario on the $x$-axis.
- **Orange dash markers** (`C1`, `_` marker, size 200, linewidth 3): Expected value from configuration or theory, one per scenario.
- **$x$-axis labels**: Rotated 45$^\circ$ when $>4$ scenarios.

### 5.1 Variance components (`variance_components.png`)

$2 \times 3$ grid (width scales with scenario count, height 10 in). Rows = traits 1 and 2. Columns = $A$, $C$, $E$.

- **Blue dots**: Observed founder-generation variance for each component.
- **Orange dashes**: Configured variance parameter.
- **$y$-axis**: "Variance Proportion".

### 5.2 Twin rate (`twin_rate.png`)

Single panel (width scales, height 5 in).

- **Blue dots**: Observed MZ twin individual rate per replicate.
- **Orange dashes**: Configured `p_mztwin`.
- **$y$-axis**: "Twin Rate".
- **Title**: "MZ Twin Rate: Observed vs Expected".

### 5.3 A-component correlations (`correlations_A.png`)

$2 \times 2$ grid.

- **Panel 1**: MZ twin $A_1$ correlation. Expected = 1.0.
- **Panel 2**: DZ (full-sibling) $A_1$ correlation. Expected = 0.5.
- **Panel 3**: Half-sibling $A_1$ correlation. Expected = 0.25.
- **Panel 4**: Midparent-offspring $A_1$ $R^2$. Expected = 0.5.

Each panel: blue dots + orange dashes + orange dashed horizontal reference line at the expected value.

### 5.4 Phenotype (liability) correlations (`correlations_phenotype.png`)

$2 \times 2$ grid. Expected values are computed per-scenario from configured variance components:

- **Panel 1**: MZ twin liability$_1$ correlation. Expected = $A_1 + C_1$.
- **Panel 2**: DZ sibling liability$_1$ correlation. Expected = $0.5 A_1 + C_1$.
- **Panel 3**: Half-sibling liability$_1$ correlation. Expected = $0.25 A_1$.
- **Panel 4**: Midparent-offspring liability$_1$ slope. Expected = $A_1$.

Blue dots + orange dashes (scenario-specific expected, computed from config).

### 5.5 Heritability estimates (`heritability_estimates.png`)

$2 \times 2$ grid. Rows = traits 1 and 2. Columns = estimation method.

- **Left**: Falconer's $\hat{h}^2$ vs. configured $A_k$.
- **Right**: Midparent-offspring liability slope vs. configured $A_k$.

Blue dots + orange dashes.

### 5.6 Half-sibling proportions (`half_sib_proportions.png`)

$1 \times 2$ figure.

- **Left panel**: Observed maternal half-sibling pair proportion. Orange dashes at expected $1 - (1 - p_{\text{nonsocial}})^2$.
- **Right panel**: Proportion of offspring with at least one half-sibling (no expected reference).

### 5.7 Cross-trait correlations (`cross_trait_correlations.png`)

$1 \times 3$ figure.

- **Panel 1**: Observed $r_A$ vs. configured `rA`. Blue dots + orange dashes.
- **Panel 2**: Observed $r_C$ vs. configured `rC`. Blue dots + orange dashes.
- **Panel 3**: Observed $r_E$. No configured expected; orange dashed horizontal line at 0 (theoretical independence).

### 5.8 Family size (`family_size.png`)

Single panel (width scales, height 5 in).

- **Blue dots** (left-offset): Mean offspring per mother (among mothers with $\geq 1$ child).
- **Red dots** (`C3`, right-offset): Mean offspring per father (among fathers with $\geq 1$ child).
- **Orange dashes** (centred): Configured Poisson $\lambda$ (`fam_size`).
- **Legend**: "Mother" (blue circle), "Father" (red circle), "Parametric Poisson family size" (orange dash).

### 5.9 Summary bias (`summary_bias.png`)

$2 \times 3$ grid.

- **Strip plots**: Observed $-$ expected for six metrics: $A_1$ bias, $C_1$ bias, $E_1$ bias, twin rate bias, DZ $A_1$ correlation bias (vs. 0.5), half-sibling $A_1$ correlation bias (vs. 0.25).
- **Red dashed horizontal line**: Reference at 0 (no bias).
- **$y$-axis**: Bias (observed $-$ expected).

### 5.10 Runtime (`runtime.png`)

Single log-log scatter (8 $\times$ 6 in).

- **Coloured dots**: One per replicate, coloured by scenario (Set2 palette). $x$ = population size $N$, $y$ = simulation wall-clock seconds.
- **Axes**: Log-scale with scalar (non-scientific) tick formatting.
- **Legend**: Scenario names.

### 5.11 Memory usage (`memory.png`)

Single log-log scatter (8 $\times$ 6 in).

- **Coloured dots**: Same encoding as 5.10. $y$ = peak resident set size (MB).
- **Legend**: Scenario names.
