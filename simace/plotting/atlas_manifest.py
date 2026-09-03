"""Atlas manifest: ordered registry of plots and section breaks.

Each entry is either a :class:`PlotEntry` (a figure to include in the
atlas) or a :class:`SectionBreak` (a divider page). The assembler walks
the manifest linearly and derives ``Figure N`` numbers from the running
:class:`PlotEntry` index — inserting or reordering plots is a single
edit here, with no figure-number cascade.

To add a phenotype plot: write the rendering function in the appropriate
``simace/plotting/plot_*.py`` module so it produces ``<basename>.png``,
then add a :class:`PlotEntry` to :data:`PHENOTYPE_ATLAS` at the
right position. The Snakemake workflow imports
:func:`phenotype_basenames` to declare expected output paths, so adding
the entry automatically wires the new plot into the build.

The :data:`MODEL_SECTION` sentinel is resolved at render time by
:func:`build_phenotype_atlas`: it is replaced with a model-aware
:class:`SectionBreak` carrying equations from
``simace.plotting.plot_atlas.get_model_family`` /
``get_model_equation`` against the scenario params.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlotEntry:
    """One plot in an atlas: filename basename and caption text.

    Attributes:
        basename: Plot filename without extension. Must match what the
            corresponding ``simace.plotting.plot_*`` rendering function writes.
        title: Short caption title (no ``"Figure N:"`` prefix; the
            assembler prepends one at render time).
        body: Multi-line caption body.
    """

    basename: str
    title: str
    body: str


@dataclass(frozen=True)
class SectionBreak:
    """A section divider page between plots.

    Attributes:
        title: Bold heading on the divider page.
        subtitle: Sub-heading below the title.
        equations: Optional LaTeX strings rendered below the subtitle.
    """

    title: str
    subtitle: str
    equations: tuple[str, ...] = ()


AtlasItem = PlotEntry | SectionBreak

# Sentinel for the model-aware phenotype section break. Resolved at
# render time by ``build_phenotype_atlas`` against scenario params; the
# placeholder values below are never displayed.
MODEL_SECTION = SectionBreak(title="<MODEL>", subtitle="<MODEL>")


# ---------------------------------------------------------------------------
# Phenotype atlas
# ---------------------------------------------------------------------------

PHENOTYPE_ATLAS: tuple[AtlasItem, ...] = (
    PlotEntry(
        basename="pedigree_counts.ped",
        title="Pedigree relationship pair counts (G_ped).",
        body=(
            "Schematic multi-generational pedigree diagram of the 10 relationship "
            "categories in the recorded pedigree, across all G_ped "
            "generations. Mean pair counts (averaged across replicates) are superimposed. "
            "Node shapes follow standard pedigree conventions: squares = male, circles = "
            "female."
        ),
    ),
    PlotEntry(
        basename="pedigree_counts",
        title="Pedigree relationship pair counts (G_pheno).",
        body=(
            "Same diagram as the G_ped pair-count figure, but restricted to the phenotyped "
            "population (last G_pheno generations), after ascertainment."
        ),
    ),
    PlotEntry(
        basename="family_structure",
        title="Family structure.",
        body=(
            "Three-panel figure showing offspring and mate count distributions, averaged "
            "across replicates. Left: number of offspring per mating. Center: number of "
            "offspring per person (including childless individuals at 0). Right: fraction of "
            "parents with 1 vs. 2+ mates, grouped by sex."
        ),
    ),
    PlotEntry(
        basename="mate_correlation",
        title="Assortative mating",
        body=(
            "2×2 heatmap of Pearson correlations between mated pairs’ liabilities (female "
            "traits on rows, male traits on columns). Each unique (mother, father) pair "
            "counted once, pooled across all non-founder generations. Bold white text shows "
            "observed r; smaller gray text shows the configured expected correlation on diagonal "
            "cells (off-diagonal not predicted for both-trait assortment). Diverging RdBu "
            "colormap centered at 0, range [−1, 1]."
        ),
    ),
    PlotEntry(
        basename="cross_trait",
        title="Cross-trait liability joint plots.",
        body=(
            "2×2 grid of joint plots for Liability, A (additive genetic), C (common "
            "environment), and E (unique environment). Central scatter of Trait 1 (x) vs. "
            "Trait 2 (y) with Pearson r annotation and marginal histograms."
        ),
    ),
    PlotEntry(
        basename="heritability.by_generation",
        title="Additive genetic and common environment by generation.",
        body=(
            "1×2 figure, one panel per trait. Realized per-generation variance proportions "
            "Var(A) / Var(L) and Var(C) / Var(L) are computed from per-replicate variance "
            "components. Blue points/line show additive genetic A (narrow-sense h²); orange "
            "points/line show common environment C. Dashed lines mark the configured A and C "
            "parameters. A and C are shown separately rather than summed."
        ),
    ),
    PlotEntry(
        basename="am_equilibrium",
        title="Assortative-mating additive-variance equilibrium.",
        body=(
            "1×2 figure, one panel per trait. Blue points/line show the realized per-"
            "generation additive genetic variance Var(A) across replicates. Under phenotypic "
            "assortative mating Var(A) inflates across generations (the Bulmer effect) toward "
            "an equilibrium. The orange dashed line is the infinitesimal-model AM recursion "
            "V_{t+1} = ½·V_t·(1 + r·h²_t) + ½·A, anchored at the founder variance A. The grey "
            "dotted line marks its closed-form equilibrium a². This a² equals the assortative-"
            "mating-only equilibrium additive variance of Herzig et al. (2026, Theoretical "
            "Population Biology 170:26–35, doi:10.1016/j.tpb.2026.06.003): the simACE midparent "
            "model and their explicit gametic-disequilibrium model share the same AM "
            "equilibrium. Traits without assortative mating show the flat configured A as "
            "the expected line. Per-generation A, C, or E suppresses the expected lines."
        ),
    ),
    PlotEntry(
        basename="ge_covariance.by_generation",
        title="Genetic–non-genetic covariance by generation.",
        body=(
            "1×2 figure, one panel per trait. Points show per-replicate realized "
            "2 Cov(A, C+E) / Var(L) by generation; the line shows the replicate mean. "
            "The shaded band is an approximate null guide from finite-N sampling variation. "
            "In current simACE, C and E are drawn independently of A each generation, so the "
            "recorded-pedigree expectation is approximately zero. Without VCT or "
            "autoregressive C, a nonzero value contradicts that independence."
        ),
    ),
    PlotEntry(
        basename="snp_like_h2.by_generation",
        title="SNP-like h² target by generation.",
        body=(
            "1×2 figure, one panel per trait. The SNP-like h² target is not a fitted "
            "GWAS/GRM estimate. It is the population plug-in estimand implied by realized "
            "A and non-genetic C+E covariance: (Var(A)+Cov(A,C+E))²/(Var(A)Var(L)). "
            "Under current simACE assumptions it should match Var(A)/Var(L); under VCT it "
            "can diverge."
        ),
    ),
    PlotEntry(
        basename="observed_h2",
        title="Observed-scale heritability from binary affected status.",
        body=(
            "This diagnostic is provisional and is not a final heritability estimate. "
            "2×2 grid: rows = traits, columns = scale. Left column: observed-scale h² "
            "estimators computed directly from Pearson correlations on binary affected "
            "indicators: Falconer 2(r_MZ − r_FS), Sibs 2·r_FS, PO (midparent-offspring "
            "regression on binary affected status), Half-sibs 4·mean(r_MHS, r_PHS), and Cousins "
            "8·r_1C. Blue per-rep dots at each estimator; grey dotted line marks the "
            "Dempster–Lerner expected value A·z(K)²/(K·(1−K)) at the mean observed prevalence"
            " K. Right column: same per-rep estimates back-transformed to the liability scale"
            " via h²_liab = h²_obs · K(1−K)/z(K)². The D–L correction assumes a threshold-"
            "normal (LTM) mapping from liability to affected status; it is biased under non-"
            "threshold phenotype models such as pure frailty. The worked example in "
            "docs/examples/observed-vs-liability-h2.md shows the bias."
        ),
    ),
    MODEL_SECTION,
    PlotEntry(
        basename="liability_violin.phenotype",
        title="Liability violin plots by affected status.",
        body=(
            "Split violin plots, one per trait. Left half = unaffected, right half = "
            "affected. Diamond markers show mean liability for each group with μ annotations."
            " Observed prevalence annotated below each trait."
        ),
    ),
    PlotEntry(
        basename="liability_components.by_generation",
        title="Liability components by generation.",
        body=(
            "2×3 grid: rows = traits, columns = variance components (A, C, E). Each panel "
            "shows the mean component value among affected (red), unaffected (grey), and "
            "overall (black) individuals per generation. Selection on A is visible as "
            "separation between affected and unaffected lines: affected individuals have "
            "higher mean A (positive selection). C and E show no systematic selection. "
            "Conditional on total liability, they are independent of whether the "
            "liability crosses the threshold. Generation-specific prevalence annotated on x-axis."
        ),
    ),
    SectionBreak(
        title="Age of Onset & Censoring",
        subtitle="Age-at-onset, mortality, cumulative incidence, and censoring analysis",
    ),
    PlotEntry(
        basename="mortality",
        title="mortality",
        body=(
            "Four-panel figure. Left column shows per-decade mortality rate (deaths in "
            "decade / alive at start of decade) and cumulative mortality, averaged across "
            "replicates; cumulative survival probability is annotated above each bar. Right "
            "column shows age-at-death histograms for death-censored unaffected individuals "
            "for traits 1 and 2."
        ),
    ),
    PlotEntry(
        basename="age_at_onset_death",
        title="Age-at-onset.",
        body=(
            "Two-panel figure, one panel per trait. Each panel shows the density histogram "
            "of observed age-at-onset for affected individuals (δ = 1). Death-age "
            "distributions are shown in the mortality figure."
        ),
    ),
    PlotEntry(
        basename="censoring",
        title="Censoring windows by generation.",
        body=(
            "Grid of panels: rows = traits, columns = generations. Grey line = true "
            "cumulative incidence, blue line = observed cumulative incidence. Text box shows "
            "affected %, left-censored %, right-censored %, and death-censored % per "
            "generation. Column titles show observation window [lo, hi]."
        ),
    ),
    PlotEntry(
        basename="cumulative_incidence.phenotype",
        title="Cumulative incidence.",
        body=(
            "Two-panel figure, one per trait. Blue solid line = observed cumulative incidence"
            " from censored data (with min-max band across replicates). Grey solid line = "
            "true cumulative incidence from uncensored event times. Grey dashed crosshairs "
            "mark the ages at which 25% (Q1), 50%, and 75% (Q3) of lifetime cases have "
            "occurred. Text box shows affected %, true prevalence %, and censored %."
        ),
    ),
    PlotEntry(
        basename="cumulative_incidence_aj.phenotype",
        title="Aalen-Johansen cumulative incidence (competing risks).",
        body=(
            "Two-panel figure, one per trait. Solid colored line = Aalen-Johansen trait-event "
            "cumulative incidence treating death as a competing event. Dotted grey line = "
            "true cumulative incidence from uncensored event times. Light overlay = empirical "
            "observed cumulative incidence (#affected/n). The Aalen-Johansen trait curve "
            "carries a min-max band across replicates. The annotation reports the terminal "
            "Aalen-Johansen trait F(∞), Aalen-Johansen death F(∞), and overall survival "
            "S(∞). The three sum to 1. Delayed entry applies each generation's "
            "observation window from gen_censoring."
        ),
    ),
    PlotEntry(
        basename="cumulative_incidence.by_sex",
        title="Cumulative incidence by sex.",
        body=(
            "Two-panel figure, one per trait. Green line = female (sex=0), blue line = male "
            "(sex=1) observed cumulative incidence. Legend shows sample size and observed "
            "prevalence per sex. Statistics computed over the full phenotyped population "
            "(before ascertainment)."
        ),
    ),
    PlotEntry(
        basename="censoring_confusion",
        title="Censoring confusion matrix.",
        body=(
            "Per-trait 2×2 confusion matrix comparing true affected status (event time < "
            "censor_age, from raw simulated times) vs. observed affected status (after "
            "generation-window and death censoring). Rows = true status, columns = observed "
            "status. Unshaded cells use annotations to show proportion and count. Title shows sensitivity "
            "(true positive rate) and specificity (true negative rate). Only phenotyped "
            "generations (those with non-degenerate observation windows) are included. "
            "Statistics computed over the full phenotyped population (before ascertainment)."
        ),
    ),
    PlotEntry(
        basename="censoring_cascade",
        title="Censoring cascade.",
        body=(
            "Per-trait stacked bar chart decomposing true cases (event time < censor_age) by "
            "generation into four mutually exclusive fates: observed (green), death-censored "
            "(red), right-censored (purple), and left-censored (orange). Total bar height "
            "equals true case count per generation. Sensitivity (observed / true) is "
            "annotated per generation; subplot titles show overall sensitivity. Only "
            "generations with non-degenerate observation windows are shown. Statistics "
            "computed over the full phenotyped population (before ascertainment)."
        ),
    ),
    SectionBreak(
        title="Within-Trait Correlations",
        subtitle="Familial tetrachoric correlations",
    ),
    PlotEntry(
        basename="parent_offspring_liability.by_generation",
        title="Parent-offspring liability regressions.",
        body=(
            "Grid: rows = traits, columns = last 3 non-founder generations. Scatter of "
            "midparent liability (x) vs. offspring liability (y) colored by offspring sex "
            "(green = daughters, blue = sons). Observed pooled regression line (solid orange,"
            " with 95% CI band), sex-specific regression lines, and expected slope from "
            "configured A (dashed). Text box shows pooled h² and sex-specific h²♀/h²♂ (slope "
            "± SE), Pearson r, pair count n, and p-value, averaged across replicates."
        ),
    ),
    PlotEntry(
        basename="heritability.by_sex.by_generation",
        title="PO-regression heritability by offspring sex.",
        body=(
            "1×2 figure, one panel per trait. Heritability h² estimated from midparent-"
            "offspring regression slope, stratified by offspring sex. Green dots = per-"
            "replicate daughter h², blue dots = per-replicate son h². Orange dashed line "
            "marks the parametric heritability (A parameter)."
        ),
    ),
    PlotEntry(
        basename="liability_vs_aoo",
        title="Liability vs. age-at-onset.",
        body=(
            "Side-by-side joint plots, one per trait. Central scatter of liability (x) vs. "
            "observed age-at-onset (y) for affected individuals, with regression line and 95%"
            " CI band. Annotations show slope ± SE, Pearson r, n, and p-value, averaged "
            "across replicates. Marginal histograms on top and right."
        ),
    ),
    PlotEntry(
        basename="tetrachoric.phenotype",
        title="Tetrachoric correlations by relationship type.",
        body=(
            "Two-panel figure, one per trait. Coloured violins show the distribution of "
            "tetrachoric correlations (computed from censored binary affected status) across "
            "replicates for each relationship type. Black dots = individual per-replicate "
            "tetrachoric estimates. Black dashed lines = mean Pearson liability correlation "
            "(the true correlation of the continuous latent liability). Green dash-dot "
            "lines = mean uncensored frailty pairwise survival-time correlation (present when"
            " available). Red dotted lines = parametric expected correlation from the "
            "configured ACE variance components (e.g. E[r] = 0.5·A + C for full sibs, 0.5·A "
            "for parent–offspring). N = mean pairs per replicate."
        ),
    ),
    PlotEntry(
        basename="tetrachoric.phenotype.by_sex",
        title="Tetrachoric correlations by sex.",
        body=(
            "2×2 grid: rows = traits, columns = sex (female, male). Same encoding as the "
            "relationship-type tetrachoric correlation plot: colored violins show observed tetrachoric correlations for same-sex pairs "
            "only (FF or MM). Black dashed = liability correlation, red dotted = parametric "
            "E[r]. Opposite-sex pairs are excluded."
        ),
    ),
    SectionBreak(
        title="Cross-Trait Correlations",
        subtitle="Association between the two traits: joint liability, joint affection, and cross-trait tetrachoric correlation",
    ),
    PlotEntry(
        basename="cross_trait.phenotype",
        title="Cross-trait liability joint plots colored by affected status (trait 1).",
        body=(
            "Same 2×2 layout as the cross-trait liability joint plots, but with affected-status "
            "coloring based on trait 1. Blue points = unaffected, orange points = affected (trait 1). Marginal "
            "histograms stacked by affected status."
        ),
    ),
    PlotEntry(
        basename="cross_trait.phenotype.t2",
        title="Cross-trait liability joint plots colored by affected status (trait 2).",
        body=(
            "Same 2×2 layout as the cross-trait liability joint plots, but with affected-status "
            "coloring based on trait 2. Blue points = unaffected, orange points = affected (trait 2). Marginal "
            "histograms stacked by affected status."
        ),
    ),
    PlotEntry(
        basename="joint_affected.phenotype",
        title="Joint affected status heatmap.",
        body=(
            "2×2 heatmap of joint affected status across both traits. Cell annotations show "
            "proportion and count. Title shows cross-trait correlation estimates: 'r_tet' = "
            "tetrachoric correlation on censored binary affected status; 'r_frailty' = "
            "frailty-estimated liability correlation from uncensored survival data; "
            "'stratified' = generation-stratified estimate that computes per-generation "
            "correlations and combines via inverse-variance weighting; 'naive' = unweighted "
            "pooled censored estimate. Statistics computed over the full phenotyped population (before ascertainment)."
        ),
    ),
    PlotEntry(
        basename="cross_trait_tetrachoric",
        title="Cross-trait tetrachoric correlations.",
        body=(
            "Two-panel figure measuring cross-trait association via tetrachoric correlation "
            "between affected1 and affected2. Left panel: same-person cross-trait r by "
            "generation (blue dots per rep, line = mean), with overall r (black dashed) and "
            "uncensored frailty estimate (green dash-dot) reference lines when available. Right panel: "
            "cross-person cross-trait r by relationship type (colored violins + black dots)."
        ),
    ),
    SectionBreak(
        title="Additional per-generation and sex-specific figures.",
        subtitle="Supplemental stratified views moved out of the main narrative flow",
    ),
    PlotEntry(
        basename="liability_violin.phenotype.by_generation",
        title="Liability violin plots by generation.",
        body=(
            "Grid: rows = traits, columns = recorded generations. Split violins for affected "
            "vs. unaffected within each generation. Diamond markers and μ annotations show "
            "per-group means. x-axis labels show observed generation-specific prevalence."
        ),
    ),
    PlotEntry(
        basename="liability_violin.phenotype.by_sex.by_generation",
        title="Liability violin plots by sex and generation.",
        body=(
            "Grid: rows = traits, columns = generations. Each cell has side-by-side split "
            "violins for female (left) and male (right), each showing unaffected vs. affected"
            " distribution. Sex-specific observed prevalence annotated."
        ),
    ),
    PlotEntry(
        basename="cumulative_incidence.by_sex.by_generation",
        title="Cumulative incidence by sex and generation.",
        body=(
            "Grid: rows = traits, columns = generations. Each panel shows cumulative "
            "incidence curves for female (green) and male (blue) separately. Legend shows "
            "per-sex sample size and observed prevalence within each generation. Statistics "
            "computed over the full phenotyped population (before ascertainment)."
        ),
    ),
    PlotEntry(
        basename="tetrachoric.phenotype.by_generation",
        title="Tetrachoric correlations by generation.",
        body=(
            "Grid: rows = traits, columns = generations. Same encoding as the relationship-type "
            "tetrachoric correlation plot (violins = observed tetrachoric correlations, black dashed = true liability correlations,"
            " red dotted = parametric E[r], dots = per-replicate estimates), computed within "
            "each generation separately."
        ),
    ),
    PlotEntry(
        basename="cumulative_incidence_aj.by_sex",
        title="Aalen-Johansen cumulative incidence by sex.",
        body=(
            "Two-panel figure, one per trait. Green line = female (sex=0), blue line = male "
            "(sex=1) Aalen-Johansen trait cumulative incidence. Legend shows sample size and observed "
            "prevalence per sex. Statistics computed over the full phenotyped population "
            "(before ascertainment)."
        ),
    ),
    PlotEntry(
        basename="cumulative_incidence_aj.by_sex.by_generation",
        title="Aalen-Johansen cumulative incidence by sex and generation.",
        body=(
            "Grid: rows = traits, columns = generations. Each panel shows the Aalen-Johansen trait cumulative incidence for "
            "female (green) and male (blue) separately. Legend shows per-sex sample size and "
            "prevalence within each generation. Delayed entry applies each "
            "generation's observation window."
        ),
    ),
)

# ---------------------------------------------------------------------------
# Validation atlas
# ---------------------------------------------------------------------------

VALIDATION_ATLAS: tuple[AtlasItem, ...] = (
    PlotEntry(
        basename="family_size",
        title="Family size.",
        body=(
            "Mean offspring per mother (blue, left-offset) and per father (orange, right-"
            "offset) among parents with ≥1 child. Orange dashes mark expected ~2.0 (N / "
            "n_mothers for balanced sex ratio)."
        ),
    ),
    PlotEntry(
        basename="twin_rate",
        title="MZ twin rate.",
        body=("Observed MZ twin individual rate per replicate (blue dots) vs. configured p_mztwin (orange dashes)."),
    ),
    PlotEntry(
        basename="half_sib_proportions",
        title="Half-sibling proportions.",
        body=(
            "Left panel: Observed maternal half-sibling pair proportion (driven by "
            "mating_lambda). Right panel: Proportion of offspring with at least one half-"
            "sibling."
        ),
    ),
    PlotEntry(
        basename="consanguineous_matings",
        title="Consanguineous matings.",
        body=(
            "Left panel: Number of half-sibling matings per replicate (random mating produces"
            " a small number by chance). Right panel: Missing grandparent-grandchild links "
            "caused by shared grandparents. The reconciliation check confirms that "
            "consanguineous matings account for every missing link."
        ),
    ),
    PlotEntry(
        basename="variance_components",
        title="Variance components.",
        body=(
            "2×3 grid, rows = traits 1 and 2, columns = A, C, E. Blue dots show observed "
            "generation-0 variance for each component per replicate. Orange dashes mark"
            " configured variance parameters."
        ),
    ),
    PlotEntry(
        basename="correlations_A",
        title="A-component correlations.",
        body=(
            "2×2 grid. Panel 1: MZ twin A₁ correlation (expected = 1.0). Panel 2: Full-"
            "sibling (FS) A₁ correlation (expected = 0.5). Panel 3: Half-sibling A₁ correlation "
            "(expected = 0.25). Panel 4: Midparent-offspring A₁ R² (expected = 0.5). Each "
            "panel shows blue dots with the orange random-mating reference line. For scenarios "
            "with assortative mating, a purple marker adds the AM-corrected expectation. It is "
            "computed from the observed genetic and liability mate correlations: FS and PO = "
            "(1+μ_A)/2, half-sib = (1+2μ_A+μ_A·r_ho)/4. Under AM the additive genetic "
            "correlations inflate toward these values. MZ is AM-invariant (no purple marker)."
        ),
    ),
    PlotEntry(
        basename="correlations_phenotype",
        title="Liability correlations.",
        body=(
            "2×2 grid. Expected values computed per-scenario from configured variance "
            "components. Panel 1: MZ twin liability₁ correlation (expected = A₁ + C₁). Panel "
            "2: Full-sibling liability₁ correlation (expected = 0.5A₁ + C₁). Panel 3: Half-"
            "sibling liability₁ correlation (expected = 0.25A₁). Panel 4: Midparent-offspring"
            " liability₁ slope (expected = A₁)."
        ),
    ),
    PlotEntry(
        basename="heritability_estimates",
        title="Heritability estimates.",
        body=(
            "2×2 grid, rows = traits 1 and 2. Left: Falconer's h² vs. configured A. Right: "
            "Midparent-offspring liability slope vs. configured A."
        ),
    ),
    PlotEntry(
        basename="cross_trait_correlations",
        title="Cross-trait correlations.",
        body=(
            "1×3 figure. Panel 1: Observed r_A vs. configured rA. Panel 2: Observed r_C vs. "
            "configured rC. Panel 3: Observed r_E with reference at 0 (theoretical "
            "independence)."
        ),
    ),
    PlotEntry(
        basename="summary_bias",
        title="Summary bias.",
        body=(
            "2×3 grid of strip plots showing observed − expected for six metrics: A₁ bias, C₁"
            " bias, E₁ bias, twin rate bias, Full-sibling A₁ correlation bias (vs. 0.5), half-sibling "
            "A₁ correlation bias (vs. 0.25). Red dashed reference line at 0 (no bias)."
        ),
    ),
    PlotEntry(
        basename="runtime",
        title="Simulation runtime.",
        body=("Log-log scatter of population size N (x) vs. simulation wall-clock seconds (y), colored by scenario."),
    ),
    PlotEntry(
        basename="memory",
        title="Memory usage.",
        body=("Log-log scatter of population size N (x) vs. peak resident set size in MB (y), colored by scenario."),
    ),
)

# ---------------------------------------------------------------------------
# Effective-size atlas
# ---------------------------------------------------------------------------

EFFECTIVE_SIZE_ATLAS: tuple[AtlasItem, ...] = (
    PlotEntry(
        basename="effective_size.estimators",
        title="Ne estimator summary.",
        body=(
            "Per-rep aggregate Ne for all eight estimators on a log y-axis. Blue dots show "
            "per-replicate values. Orange dashed lines show closed-form expectations from "
            "theoretical_expectations() where defined: Ne_V for Ne_V, Ne_iΔF, and Ne_H; N for "
            "Ne_sr; Ne_V/2 for Ne_LTC. The regression estimators Ne_I, Ne_C, and Ne_CT get an "
            "expected line only when N·G² is large enough for the Jensen bias to be small. Per-estimator mean "
            "± SD across reps is annotated above each column. Open markers below the axis "
            "mark reps that returned None (commonly Ne_LTC at G_ped=6). Subtitle reports the scenario header (N, λ, G_ped, "
            "expected Ne_V)."
        ),
    ),
    PlotEntry(
        basename="effective_size.by_generation",
        title="Ne by generation / transition.",
        body=(
            "2×3 grid: per-generation Ne_g for the six estimators that expose a vector. Five "
            "panels (Ne_I, Ne_C, Ne_sr, Ne_iΔF, Ne_CT) index generation 0..G_ped-1 from "
            "ne_per_gen. The Ne_V panel uses ne_per_transition. Its x-axis is labelled "
            "'transition (g→g+1)' with half-integer ticks, so the offset from generation "
            "indexing is visible. One colored line per replicate. Orange "
            "dashed horizontal lines show the closed-form expectation when defined. The plot "
            "skips NaN entries, such as early-generation Ne_I before drift accumulates."
        ),
    ),
    PlotEntry(
        basename="effective_size.drift",
        title="Underlying drift signals (mean F, θ, self-kinship).",
        body=(
            "1×3 grid showing the per-generation drift quantities that the slope-based Ne "
            "estimators invert. Left: mean inbreeding F per generation (input to Ne_I). "
            "Center: mean kinship θ per generation (input to Ne_C). Right: mean "
            "founder self-kinship per generation (input to Ne_CT). One line per "
            "replicate. The estimators assume linear growth in these quantities. A "
            "non-monotone or near-flat trace means the estimator has no slope to fit."
        ),
    ),
    PlotEntry(
        basename="effective_size.family_size_variance",
        title="Family-size variance and covariance (ZTP diagnostic).",
        body=(
            "1×2 diagnostic for Ne_V. Left: per-transition offspring-count "
            "variance for the four parent-sex × offspring-sex quadrants (v_mm, v_mf, v_fm, "
            "v_ff), with the ZTP(λ) closed-form 1 + Var[m]/E[m]² overlaid as an orange dashed "
            "reference. Right: per-transition between-mate covariance (cov_m, cov_f) with the "
            "ZTP closed-form Var[m]/E[m]² overlaid. At λ=0.5 the references sit at ≈1.180 and "
            "≈0.180 respectively. The four v lines converge to the reference when the "
            "mating model produces the ZTP family-size distribution."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def phenotype_basenames() -> list[str]:
    """Ordered phenotype plot basenames (excluding section breaks).

    Used by ``workflow.common`` to declare the Snakemake outputs.
    """
    return [e.basename for e in PHENOTYPE_ATLAS if isinstance(e, PlotEntry)]


def validation_basenames() -> list[str]:
    """Ordered validation plot basenames (excluding section breaks)."""
    return [e.basename for e in VALIDATION_ATLAS if isinstance(e, PlotEntry)]


def effective_size_basenames() -> list[str]:
    """Ordered effective-size plot basenames (excluding section breaks)."""
    return [e.basename for e in EFFECTIVE_SIZE_ATLAS if isinstance(e, PlotEntry)]


def build_phenotype_atlas(params: dict[str, Any] | None) -> list[AtlasItem]:
    """Return ``PHENOTYPE_ATLAS`` with ``MODEL_SECTION`` resolved.

    When ``params`` is ``None`` the model section is omitted (no scenario
    context to derive the title from). Otherwise the sentinel is replaced
    with a :class:`SectionBreak` whose title / subtitle / equations come
    from ``get_model_family`` and ``get_model_equation`` in
    ``simace.plotting.plot_atlas``.
    """
    from simace.plotting.plot_atlas import get_model_equation, get_model_family

    out: list[AtlasItem] = []
    for item in PHENOTYPE_ATLAS:
        if item is MODEL_SECTION:
            if params is None:
                continue
            name, desc = get_model_family(params)
            equations = tuple(get_model_equation(params))
            out.append(SectionBreak(title=f"{name} Phenotype", subtitle=desc, equations=equations))
        else:
            out.append(item)
    return out
