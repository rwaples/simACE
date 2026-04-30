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
            "Schematic multi-generational pedigree diagram showing the 10 relationship "
            "categories extracted from the full simulated pedigree spanning all G_ped "
            "generations. Mean pair counts (averaged across replicates) are superimposed. "
            "Node shapes follow standard pedigree conventions: squares = male, circles = "
            "female."
        ),
    ),
    PlotEntry(
        basename="pedigree_counts",
        title="Pedigree relationship pair counts (G_pheno).",
        body=(
            "Same diagram as Figure 1, but restricted to the phenotyped population (last "
            "G_pheno generations), after any N_sample subsampling."
        ),
    ),
    PlotEntry(
        basename="family_structure",
        title="Family structure.",
        body=(
            "Three-panel figure showing offspring and partner count distributions, averaged "
            "across replicates. Left: number of offspring per couple. Centre: number of "
            "offspring per person (including childless individuals at 0). Right: fraction of "
            "parents with 1 vs. 2+ partners, grouped by sex."
        ),
    ),
    PlotEntry(
        basename="mate_correlation",
        title="Mate liability correlation.",
        body=(
            "2×2 heatmap of Pearson correlations between mated pairs’ liabilities (female "
            "traits on rows, male traits on columns). Each unique (mother, father) pair "
            "counted once, pooled across all non-founder generations. Bold white text shows "
            "observed r; smaller gray text shows expected target correlation on diagonal "
            "cells (off-diagonal not predicted for both-trait assortment). Diverging RdBu "
            "colormap centred at 0, range [−1, 1]."
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
        basename="parent_offspring_liability.by_generation",
        title="Parent-offspring liability regressions.",
        body=(
            "Grid: rows = traits, columns = last 3 non-founder generations. Scatter of "
            "midparent liability (x) vs. offspring liability (y) coloured by offspring sex "
            "(green = daughters, blue = sons). Observed pooled regression line (solid orange,"
            " with 95% CI band), sex-specific regression lines, and expected slope from "
            "configured A (dashed). Text box shows pooled h² and sex-specific h²♀/h²♂ (slope "
            "± SE), Pearson r, pair count n, and p-value, averaged across replicates."
        ),
    ),
    PlotEntry(
        basename="heritability.by_generation",
        title="Narrow-sense liability-scale heritability by generation.",
        body=(
            "1×2 figure, one panel per trait. Narrow-sense heritability h² = Var(A) / (Var(A)"
            " + Var(C) + Var(E)) is computed from the per-generation variance components for "
            "each replicate. Blue dots show per-replicate h² estimates. Orange dashed line "
            "marks the parametric heritability (A parameter)."
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
        basename="additive_shared.by_generation",
        title="Additive genetic and shared environment by generation.",
        body=(
            "1×2 figure, one panel per trait. Combined proportion (Var(A) + Var(C)) / (Var(A)"
            " + Var(C) + Var(E)) is computed from the per-generation variance components for "
            "each replicate. Blue dots show per-replicate estimates. Orange dashed line marks"
            " the parametric value (A + C)."
        ),
    ),
    PlotEntry(
        basename="observed_h2",
        title="Observed-scale heritability from binary affected status.",
        body=(
            "2×2 grid: rows = traits, columns = scale. Left column: observed-scale h² "
            "estimators computed directly from Pearson correlations on binary affected "
            "indicators — Falconer 2(r_MZ − r_FS), Sibs 2·r_FS, PO (midparent-offspring "
            "regression on binary affected status), Half-sibs 4·mean(r_MHS, r_PHS), Cousins "
            "8·r_1C. Blue per-rep dots at each estimator; grey dotted line marks the "
            "Dempster–Lerner expected value A·z(K)²/(K·(1−K)) at the mean observed prevalence"
            " K. Right column: same per-rep estimates back-transformed to the liability scale"
            " via h²_liab = h²_obs · K(1−K)/z(K)². The D–L correction assumes a threshold-"
            "normal (LTM) mapping from liability to affected status; it is biased under non-"
            "threshold phenotype models such as pure frailty (see docs/examples/observed-vs-"
            "liability-h2.md)."
        ),
    ),
    MODEL_SECTION,
    PlotEntry(
        basename="liability_violin.phenotype",
        title="Liability violin plots by affected status (survival model).",
        body=(
            "Split violin plots, one per trait. Left half = unaffected, right half = "
            "affected. Diamond markers show mean liability for each group with μ annotations."
            " Prevalence annotated below each trait."
        ),
    ),
    PlotEntry(
        basename="liability_violin.phenotype.by_generation",
        title="Liability violin plots by generation (survival model).",
        body=(
            "Grid: rows = traits, columns = recorded generations. Split violins for affected "
            "vs. unaffected within each generation. Diamond markers and μ annotations show "
            "per-group means. x-axis labels show observed generation-specific prevalence."
        ),
    ),
    PlotEntry(
        basename="liability_violin.phenotype.by_sex.by_generation",
        title="Liability violin plots by sex and generation (survival model).",
        body=(
            "Grid: rows = traits, columns = generations. Each cell has side-by-side split "
            "violins for female (left) and male (right), each showing unaffected vs. affected"
            " distribution. Sex-specific prevalence annotated."
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
            "higher mean A (positive selection). C and E show no systematic selection since "
            "they are independent of liability threshold crossing conditional on total "
            "liability. Generation-specific prevalence annotated on x-axis."
        ),
    ),
    SectionBreak(
        title="Age of Onset & Censoring",
        subtitle="Age-at-onset, mortality, cumulative incidence, and censoring analysis",
    ),
    PlotEntry(
        basename="age_at_onset_death",
        title="Age-at-onset and death-age histograms.",
        body=(
            "A 2×2 grid, rows = traits 1 and 2. Left column shows density histograms of "
            "observed age-at-onset for affected individuals (δ = 1). Right column shows age-"
            "at-death histograms for death-censored unaffected individuals."
        ),
    ),
    PlotEntry(
        basename="mortality",
        title="Mortality rate by decade.",
        body=(
            "Two-panel figure. Left panel shows per-decade mortality rate (deaths in decade /"
            " alive at start of decade), averaged across replicates. Right panel shows "
            "cumulative mortality, with cumulative survival probability annotated above each "
            "bar."
        ),
    ),
    PlotEntry(
        basename="cumulative_incidence.by_sex",
        title="Cumulative incidence by sex.",
        body=(
            "Two-panel figure, one per trait. Green line = female (sex=0), blue line = male "
            "(sex=1) observed cumulative incidence. Legend shows sample size and prevalence "
            "per sex. Statistics computed on full (non-subsampled) data."
        ),
    ),
    PlotEntry(
        basename="cumulative_incidence.by_sex.by_generation",
        title="Cumulative incidence by sex and generation.",
        body=(
            "Grid: rows = traits, columns = generations. Each panel shows cumulative "
            "incidence curves for female (green) and male (blue) separately. Legend shows "
            "per-sex sample size and prevalence within each generation. Statistics computed "
            "on full (non-subsampled) data."
        ),
    ),
    PlotEntry(
        basename="cumulative_incidence.phenotype",
        title="Cumulative incidence curves.",
        body=(
            "Two-panel figure, one per trait. Blue solid line = observed cumulative incidence"
            " from censored data (with min-max band across replicates). Grey solid line = "
            "true cumulative incidence from uncensored event times. Grey dashed crosshairs "
            "mark the ages at which 25% (Q1), 50%, and 75% (Q3) of lifetime cases have "
            "occurred. Text box shows affected %, true prevalence %, and censored %."
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
        basename="censoring_confusion",
        title="Censoring confusion matrix.",
        body=(
            "Per-trait 2×2 confusion matrix comparing true affected status (event time < "
            "censor_age, from raw simulated times) vs. observed affected status (after "
            "generation-window and death censoring). Rows = true status, columns = observed "
            "status. Cell annotations show proportion and count. Title shows sensitivity "
            "(true positive rate) and specificity (true negative rate). Only phenotyped "
            "generations (those with non-degenerate observation windows) are included. "
            "Statistics computed on full (non-subsampled) data."
        ),
    ),
    PlotEntry(
        basename="censoring_cascade",
        title="Censoring cascade.",
        body=(
            "Per-trait stacked bar chart decomposing true cases (event time < censor_age) by "
            "generation into four mutually exclusive fates: observed (green), death-censored "
            "(red), right-censored (purple), and left-truncated (orange). Total bar height "
            "equals true case count per generation. Sensitivity (observed / true) is "
            "annotated per generation; subplot titles show overall sensitivity. Only "
            "generations with non-degenerate observation windows are shown. Statistics "
            "computed on full (non-subsampled) data."
        ),
    ),
    SectionBreak(
        title="Within-Trait Correlations",
        subtitle="Familial tetrachoric correlations",
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
        title="Tetrachoric correlations by relationship type (survival model).",
        body=(
            "Two-panel figure, one per trait. Coloured violins show the distribution of "
            "tetrachoric correlations (computed from censored binary affected status) across "
            "replicates for each relationship type. Black dots = individual per-replicate "
            "tetrachoric estimates. Black dashed lines = mean Pearson liability correlation "
            "(ground-truth correlation on the continuous latent liability). Green dash-dot "
            "lines = mean uncensored frailty pairwise survival-time correlation (present when"
            " available). Red dotted lines = parametric expected correlation from the "
            "configured ACE variance components (e.g. E[r] = 0.5·A + C for full sibs, 0.5·A "
            "for parent–offspring). N = mean pairs per replicate."
        ),
    ),
    PlotEntry(
        basename="tetrachoric.phenotype.by_sex",
        title="Tetrachoric correlations by sex (survival model).",
        body=(
            "2×2 grid: rows = traits, columns = sex (female, male). Same encoding as Figure "
            "23: coloured violins show observed tetrachoric correlations for same-sex pairs "
            "only (FF or MM). Black dashed = liability correlation, red dotted = parametric "
            "E[r]. Opposite-sex pairs are excluded."
        ),
    ),
    SectionBreak(
        title="Cross-Trait Correlations",
        subtitle="Cross-trait correlation by generation and relationship type",
    ),
    PlotEntry(
        basename="tetrachoric.phenotype.by_generation",
        title="Tetrachoric correlations by generation (survival model).",
        body=(
            "Grid: rows = traits, columns = generations. Same encoding as Figure 23 (violins "
            "= observed tetrachoric correlations, black dashed = true liability correlations,"
            " red dotted = parametric E[r], dots = per-replicate estimates), computed within "
            "each generation separately."
        ),
    ),
    PlotEntry(
        basename="cross_trait.phenotype",
        title="Cross-trait liability joint plots coloured by affected status (trait 1).",
        body=(
            "Same 2×2 layout as Figure 5, but with affected-status colouring based on trait "
            "1. Blue points = unaffected, orange points = affected (trait 1). Marginal "
            "histograms stacked by affected status."
        ),
    ),
    PlotEntry(
        basename="cross_trait.phenotype.t2",
        title="Cross-trait liability joint plots coloured by affected status (trait 2).",
        body=(
            "Same 2×2 layout as Figure 5, but with affected-status colouring based on trait "
            "2. Blue points = unaffected, orange points = affected (trait 2). Marginal "
            "histograms stacked by affected status."
        ),
    ),
    PlotEntry(
        basename="joint_affected.phenotype",
        title="Joint affected status heatmap (survival model).",
        body=(
            "2×2 heatmap of joint affected status across both traits. Cell annotations show "
            "proportion and count. Title shows cross-trait correlation estimates: 'r_tet' = "
            "tetrachoric correlation on censored binary affected status; 'r_frailty' = "
            "frailty-estimated liability correlation from uncensored survival data (oracle); "
            "'stratified' = generation-stratified estimate that computes per-generation "
            "correlations and combines via inverse-variance weighting; 'naive' = unweighted "
            "pooled censored estimate. Statistics computed on full (non-subsampled) data."
        ),
    ),
    PlotEntry(
        basename="cross_trait_tetrachoric",
        title="Cross-trait tetrachoric correlations.",
        body=(
            "Two-panel figure measuring cross-trait association via tetrachoric correlation "
            "between affected1 and affected2. Left panel: same-person cross-trait r by "
            "generation (blue dots per rep, line = mean), with overall r (black dashed) and "
            "frailty oracle (green dash-dot) reference lines when available. Right panel: "
            "cross-person cross-trait r by relationship type (coloured violins + black dots)."
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
            "caused by shared grandparents. Reconciliation verifies that all missing links "
            "are explained by consanguineous matings."
        ),
    ),
    PlotEntry(
        basename="variance_components",
        title="Variance components.",
        body=(
            "2×3 grid, rows = traits 1 and 2, columns = A, C, E. Blue dots show observed "
            "founder-generation variance for each component per replicate. Orange dashes mark"
            " configured variance parameters."
        ),
    ),
    PlotEntry(
        basename="correlations_A",
        title="A-component correlations.",
        body=(
            "2×2 grid. Panel 1: MZ twin A₁ correlation (expected = 1.0). Panel 2: DZ (full-"
            "sibling) A₁ correlation (expected = 0.5). Panel 3: Half-sibling A₁ correlation "
            "(expected = 0.25). Panel 4: Midparent-offspring A₁ R² (expected = 0.5). Each "
            "panel shows blue dots with orange dashed reference line."
        ),
    ),
    PlotEntry(
        basename="correlations_phenotype",
        title="Phenotype (liability) correlations.",
        body=(
            "2×2 grid. Expected values computed per-scenario from configured variance "
            "components. Panel 1: MZ twin liability₁ correlation (expected = A₁ + C₁). Panel "
            "2: DZ sibling liability₁ correlation (expected = 0.5A₁ + C₁). Panel 3: Half-"
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
            " bias, E₁ bias, twin rate bias, DZ A₁ correlation bias (vs. 0.5), half-sibling "
            "A₁ correlation bias (vs. 0.25). Red dashed reference line at 0 (no bias)."
        ),
    ),
    PlotEntry(
        basename="runtime",
        title="Simulation runtime.",
        body=("Log-log scatter of population size N (x) vs. simulation wall-clock seconds (y), coloured by scenario."),
    ),
    PlotEntry(
        basename="memory",
        title="Memory usage.",
        body=("Log-log scatter of population size N (x) vs. peak resident set size in MB (y), coloured by scenario."),
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
