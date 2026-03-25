"""Assemble individual plots into a multi-page PDF atlas with figure captions."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-family lookup
# ---------------------------------------------------------------------------

# Maps phenotype model name → (short display name, one-line description)
MODEL_FAMILY: dict[str, tuple[str, str]] = {
    "weibull": (
        "Weibull Frailty",
        "Proportional hazards with Weibull baseline; frailty exp(\u03b2\u00b7L) scales hazard",
    ),
    "exponential": (
        "Exponential Frailty",
        "Proportional hazards with exponential baseline; frailty exp(\u03b2\u00b7L) scales hazard",
    ),
    "gompertz": (
        "Gompertz Frailty",
        "Proportional hazards with Gompertz baseline; frailty exp(\u03b2\u00b7L) scales hazard",
    ),
    "lognormal": (
        "Log-Normal Frailty",
        "Proportional hazards with log-normal baseline; frailty exp(\u03b2\u00b7L) scales hazard",
    ),
    "loglogistic": (
        "Log-Logistic Frailty",
        "Proportional hazards with log-logistic baseline; frailty exp(\u03b2\u00b7L) scales hazard",
    ),
    "gamma": (
        "Gamma Frailty",
        "Proportional hazards with gamma baseline; frailty exp(\u03b2\u00b7L) scales hazard",
    ),
    "cure_frailty": (
        "Cure Frailty",
        "Mixture cure model: liability threshold for case status, frailty for age-at-onset",
    ),
    "adult_ltm": (
        "ADuLT LTM",
        "Liability threshold for case status, deterministic probit CIP for age-at-onset",
    ),
    "adult_cox": (
        "ADuLT Cox",
        "Ranking for case status, stochastic Weibull CIP for age-at-onset",
    ),
}

# Common frailty model equation (line 1 for all 6 baseline hazard models)
_FRAILTY_LINE = (
    r"$h(t \mid L) = h_0(t) \cdot e^{\beta L \,+\, \beta_{\mathrm{sex}} \cdot \mathrm{sex}},"
    r" \qquad L = A + C + E$"
)

# Model-specific baseline hazard h₀(t) (line 2)
_BASELINE_LINE: dict[str, str] = {
    "weibull": r"$h_0(t) = \dfrac{\rho}{s}\left(\dfrac{t}{s}\right)^{\!\rho-1}$",
    "exponential": r"$h_0(t) = \lambda$",
    "gompertz": r"$h_0(t) = b \, e^{\gamma t}$",
    "lognormal": (
        r"$h_0(t) = \dfrac{\phi(w)}{\sigma\, t\, (1-\Phi(w))},"
        r" \quad w = \dfrac{\ln t - \mu}{\sigma}$"
    ),
    "loglogistic": r"$h_0(t) = \dfrac{(k/\alpha)(t/\alpha)^{k-1}}{1 + (t/\alpha)^k}$",
    "gamma": r"$h_0(t) = f_\Gamma(t;\,k,\theta) \,/\, S_\Gamma(t;\,k,\theta)$",
}

_FRAILTY_MODELS_SET = set(_BASELINE_LINE)


def _equation_lines_for_model(model: str, label: str = "") -> list[str]:
    """Return mathtext equation line(s) for a single phenotype model."""
    prefix = (r"\mathrm{" + label + r"\!:}\ ") if label else ""

    if model in _BASELINE_LINE:
        line1 = r"$" + prefix + _FRAILTY_LINE.strip("$") + r"$" if prefix else _FRAILTY_LINE
        return [line1, _BASELINE_LINE[model]]

    if model == "cure_frailty":
        return [
            r"$"
            + prefix
            + r"\mathrm{case\!:}\ L > \Phi^{-1}(1-K), \qquad"
            + r" t_{\mathrm{case}} \sim h_0(t) \cdot"
            + r" e^{\beta L \,+\, \beta_{\mathrm{sex}} \cdot \mathrm{sex}}$",
        ]
    if model == "adult_ltm":
        return [
            r"$" + prefix + r"\mathrm{CIP}(t) = \frac{K}{1 + e^{-k(t - x_0)}}$",
            r"$\mathrm{case\!:}\ L > \Phi^{-1}(1-K), \qquad"
            + r" t = x_0 + \frac{1}{k}\ln\!\frac{\Phi(-L)}{K - \Phi(-L)}$",
        ]
    if model == "adult_cox":
        return [
            r"$" + prefix + r"t_{\mathrm{raw}} = \sqrt{-\ln U \,/\, e^{L}}," + r" \quad U \sim \mathrm{Uniform}(0,1]$",
            r"$\mathrm{case\!:}\ \mathrm{CIP}_{\mathrm{rank}} < K, \qquad"
            + r" t = x_0 + \frac{1}{k}\ln\!\frac{\mathrm{CIP}}{K - \mathrm{CIP}}$",
        ]
    return []


def get_model_equation(params: dict) -> list[str]:
    """Return mathtext equation lines for the scenario's phenotype model(s)."""
    m1 = str(params.get("phenotype_model1", "weibull"))
    m2 = str(params.get("phenotype_model2", "weibull"))

    if m1 == m2:
        return _equation_lines_for_model(m1)

    lines: list[str] = []
    lines.extend(_equation_lines_for_model(m1, label="Trait 1"))
    lines.extend(_equation_lines_for_model(m2, label="Trait 2"))
    return lines


def get_model_family(params: dict) -> tuple[str, str]:
    """Return (display_name, description) for the scenario's phenotype model(s).

    When both traits use the same family, return that family.
    When they differ, return a combined description.
    """
    m1 = str(params.get("phenotype_model1", "weibull"))
    m2 = str(params.get("phenotype_model2", "weibull"))

    name1, desc1 = MODEL_FAMILY.get(m1, (m1.title(), m1))
    name2, desc2 = MODEL_FAMILY.get(m2, (m2.title(), m2))

    if m1 == m2:
        return name1, desc1

    return (
        f"{name1} / {name2}",
        f"Trait 1: {desc1}; Trait 2: {desc2}",
    )


# ---------------------------------------------------------------------------
# Caption text for each plot, keyed by basename (filename without extension).
# Content drawn from plots.md.
# ---------------------------------------------------------------------------

# Captions ordered to match _PHENOTYPE_BASENAMES in workflow/common.py:
# liability structure -> affected status -> survival & censoring ->
# within-trait correlations -> cross-trait correlations.
PHENOTYPE_CAPTIONS: dict[str, str] = {
    # -- Pedigree structure --
    "pedigree_counts.ped": (
        "Figure 1: Pedigree relationship pair counts (G_ped).\n\n"
        "Schematic multi-generational pedigree diagram showing the 10 relationship "
        "categories extracted from the full simulated pedigree spanning all G_ped "
        "generations. Mean pair counts (averaged across replicates) are superimposed. "
        "Node shapes follow standard pedigree conventions: squares = male, circles = female."
    ),
    "pedigree_counts": (
        "Figure 2: Pedigree relationship pair counts (G_pheno).\n\n"
        "Same diagram as Figure 1, but restricted to the phenotyped population "
        "(last G_pheno generations), after any N_sample subsampling."
    ),
    # -- Family structure --
    "family_structure": (
        "Figure 3: Family structure.\n\n"
        "Three-panel figure showing offspring and mate count distributions, "
        "averaged across replicates. Left: number of offspring per couple. "
        "Centre: number of offspring per person "
        "(including childless individuals at 0). Right: fraction of parents "
        "with 1 vs. 2+ mating partners, grouped by sex."
    ),
    # -- Mate correlation --
    "mate_correlation": (
        "Figure 4: Mate liability correlation.\n\n"
        "2\u00d72 heatmap of Pearson correlations between mated pairs\u2019 liabilities "
        "(female traits on rows, male traits on columns). Each unique "
        "(mother, father) pair counted once, pooled across all non-founder "
        "generations. Bold white text shows observed r; smaller gray text "
        "shows expected target correlation on diagonal cells "
        "(off-diagonal not predicted for both-trait assortment). "
        "Diverging RdBu colormap centred at 0, range [\u22121, 1]."
    ),
    # -- Liability structure --
    "cross_trait": (
        "Figure 5: Cross-trait liability joint plots.\n\n"
        "2\u00d72 grid of joint plots for Liability, A (additive genetic), C (common environment), "
        "and E (unique environment). Central scatter of Trait 1 (x) vs. Trait 2 (y) with "
        "Pearson r annotation and marginal histograms."
    ),
    # -- Liability-scale heritability --
    "parent_offspring_liability.by_generation": (
        "Figure 6: Parent-offspring liability regressions.\n\n"
        "Grid: rows = traits, columns = last 3 non-founder generations. Scatter of "
        "midparent liability (x) vs. offspring liability (y) with regression line. "
        "Text box shows Pearson r, R\u00b2, and pair count n, all averaged across "
        "replicates."
    ),
    "heritability.by_generation": (
        "Figure 7: Narrow-sense liability-scale heritability by generation.\n\n"
        "1\u00d72 figure, one panel per trait. Narrow-sense heritability "
        "h\u00b2 = Var(A) / (Var(A) + Var(C) + Var(E)) is computed from the "
        "per-generation variance components for each replicate. Blue dots show "
        "per-replicate h\u00b2 estimates. "
        "Orange dashed line marks the parametric heritability (A parameter)."
    ),
    "additive_shared.by_generation": (
        "Figure 8: Additive genetic and shared environment by generation.\n\n"
        "1\u00d72 figure, one panel per trait. Combined proportion "
        "(Var(A) + Var(C)) / (Var(A) + Var(C) + Var(E)) is computed from "
        "the per-generation variance components for each replicate. Blue dots show "
        "per-replicate estimates. "
        "Orange dashed line marks the parametric value (A + C)."
    ),
    # -- Liability by affected status --
    "liability_violin.phenotype": (
        "Figure 9: Liability violin plots by affected status (survival model).\n\n"
        "Split violin plots, one per trait. Left half = unaffected, right half = affected. "
        "Diamond markers show mean liability for each group with \u03bc annotations. "
        "Prevalence annotated below each trait."
    ),
    "liability_violin.phenotype.by_generation": (
        "Figure 10: Liability violin plots by generation (survival model).\n\n"
        "Grid: rows = traits, columns = recorded generations. Split violins for affected vs. "
        "unaffected within each generation. Diamond markers and \u03bc annotations show per-group "
        "means. x-axis labels show observed generation-specific prevalence."
    ),
    # -- Survival phenotype & censoring --
    "age_at_onset_death": (
        "Figure 11: Age-at-onset and death-age histograms.\n\n"
        "A 2\u00d72 grid, rows = traits 1 and 2. Left column shows density histograms "
        "of observed age-at-onset for affected individuals (\u03b4 = 1). Right column shows "
        "age-at-death histograms for death-censored unaffected individuals."
    ),
    "mortality": (
        "Figure 12: Mortality rate by decade.\n\n"
        "Two-panel figure. Left panel shows per-decade mortality rate "
        "(deaths in decade / alive at start of decade), averaged across replicates. "
        "Right panel shows cumulative mortality, "
        "with cumulative survival probability annotated above each bar."
    ),
    "cumulative_incidence.by_sex": (
        "Figure 13: Cumulative incidence by sex.\n\n"
        "Two-panel figure, one per trait. Green line = female (sex=0), blue line = male "
        "(sex=1) observed cumulative incidence. Legend shows sample size and prevalence "
        "per sex. Statistics computed on full (non-subsampled) data."
    ),
    "cumulative_incidence.by_sex.by_generation": (
        "Figure 14: Cumulative incidence by sex and generation.\n\n"
        "Grid: rows = traits, columns = generations. Each panel shows cumulative incidence "
        "curves for female (green) and male (blue) separately. Legend shows per-sex sample "
        "size and prevalence within each generation. Statistics computed on full "
        "(non-subsampled) data."
    ),
    "cumulative_incidence.phenotype": (
        "Figure 15: Cumulative incidence curves.\n\n"
        "Two-panel figure, one per trait. Blue solid line = observed cumulative incidence "
        "from censored data (with min-max band across replicates). Grey solid line = true "
        "cumulative incidence from uncensored event times. Grey dashed crosshairs mark the "
        "ages at which 25% (Q1), 50%, and 75% (Q3) of lifetime cases have occurred. "
        "Text box shows affected %, true prevalence %, and censored %."
    ),
    "censoring": (
        "Figure 16: Censoring windows by generation.\n\n"
        "Grid of panels: rows = traits, columns = generations. Grey line = true cumulative "
        "incidence, blue line = observed cumulative incidence. Text box shows affected %, "
        "left-censored %, right-censored %, and death-censored % per generation. Column "
        "titles show observation window [lo, hi]."
    ),
    "censoring_confusion": (
        "Figure 17: Censoring confusion matrix.\n\n"
        "Per-trait 2\u00d72 confusion matrix comparing true affected status "
        "(event time < censor_age, from raw simulated times) vs. observed affected "
        "status (after generation-window and death censoring). "
        "Rows = true status, columns = observed status. Cell annotations show "
        "proportion and count. Title shows sensitivity (true positive rate) and "
        "specificity (true negative rate). Only phenotyped generations "
        "(those with non-degenerate observation windows) are included. "
        "Statistics computed on full (non-subsampled) data."
    ),
    "censoring_cascade": (
        "Figure 18: Censoring cascade.\n\n"
        "Per-trait stacked bar chart decomposing true cases (event time < censor_age) "
        "by generation into four mutually exclusive fates: observed (green), "
        "death-censored (red), right-censored (purple), and left-truncated (orange). "
        "Total bar height equals true case count per generation. Sensitivity "
        "(observed / true) is annotated per generation; subplot titles show overall "
        "sensitivity. Only generations with non-degenerate observation windows are shown. "
        "Statistics computed on full (non-subsampled) data."
    ),
    "liability_vs_aoo": (
        "Figure 19: Liability vs. age-at-onset.\n\n"
        "Side-by-side joint plots, one per trait. Central scatter of liability (x) vs. "
        "observed age-at-onset (y) for affected individuals, with regression line. "
        "Annotations show Pearson r and R\u00b2 averaged across replicates. Marginal "
        "histograms on top and right."
    ),
    # -- Within-trait correlations --
    "joint_affected.phenotype": (
        "Figure 20: Joint affected status heatmap (survival model).\n\n"
        "2\u00d72 heatmap of joint affected status across both traits. Cell annotations "
        "show proportion and count. Title shows cross-trait correlation estimates: "
        "'r_tet' = tetrachoric correlation on censored binary affected status; "
        "'r_frailty' = frailty-estimated liability correlation from uncensored survival "
        "data (oracle); 'stratified' = generation-stratified estimate that "
        "computes per-generation correlations and combines via inverse-variance "
        "weighting; 'naive' = unweighted pooled censored estimate. "
        "Statistics computed on full (non-subsampled) data."
    ),
    "tetrachoric.phenotype": (
        "Figure 21: Tetrachoric correlations by relationship type (survival model).\n\n"
        "Two-panel figure, one per trait. Coloured violins show the distribution of "
        "tetrachoric correlations (computed from censored binary affected status) across "
        "replicates for each relationship type. "
        "Black dots = individual per-replicate tetrachoric estimates. "
        "Black dashed lines = mean Pearson liability correlation (ground-truth "
        "correlation on the continuous latent liability). "
        "Green dash-dot lines = mean uncensored frailty pairwise survival-time "
        "correlation (present when available). "
        "Red dotted lines = parametric expected correlation from the configured ACE "
        "variance components (e.g. E[r] = 0.5\u00b7A + C for full sibs, "
        "0.5\u00b7A for parent\u2013offspring). N = mean pairs per replicate."
    ),
    "tetrachoric.phenotype.by_generation": (
        "Figure 22: Tetrachoric correlations by generation (survival model).\n\n"
        "Grid: rows = traits, columns = generations. Same encoding as Figure 21 "
        "(violins = observed tetrachoric correlations, black dashed = true liability "
        "correlations, red dotted = parametric E[r], dots = per-replicate estimates), "
        "computed within each generation separately."
    ),
    # -- Cross-trait correlations --
    "cross_trait.phenotype": (
        "Figure 23: Cross-trait liability joint plots coloured by affected status (trait 1).\n\n"
        "Same 2\u00d72 layout as Figure 5, but with affected-status colouring based on trait 1. "
        "Blue points = unaffected, orange points = affected (trait 1). Marginal histograms stacked "
        "by affected status."
    ),
    "cross_trait.phenotype.t2": (
        "Figure 24: Cross-trait liability joint plots coloured by affected status (trait 2).\n\n"
        "Same 2\u00d72 layout as Figure 5, but with affected-status colouring based on trait 2. "
        "Blue points = unaffected, orange points = affected (trait 2). Marginal histograms stacked "
        "by affected status."
    ),
    "cross_trait_frailty.by_generation": (
        "Figure 25: Cross-trait frailty correlation by generation.\n\n"
        "Per-generation cross-trait liability correlation estimated from censored "
        "survival data. Blue dots = per-replicate per-generation estimates "
        "with 95% CI error bars; blue line = mean across replicates. "
        "Green dash-dot line = uncensored oracle (ground truth from raw event times). "
        "Orange dashed line = inverse-variance weighted mean across generations "
        "(stratified estimate). Dark orange dotted line = naive pooled estimate."
    ),
    "cross_trait_tetrachoric": (
        "Figure 26: Cross-trait tetrachoric correlations.\n\n"
        "Two-panel figure measuring cross-trait association via tetrachoric "
        "correlation between affected1 and affected2. "
        "Left panel: same-person cross-trait r by generation (blue dots per rep, "
        "line = mean), with overall r (black dashed) and frailty oracle (green "
        "dash-dot) reference lines when available. "
        "Right panel: cross-person cross-trait r by relationship type "
        "(coloured violins + black dots)."
    ),
}

# Captions ordered to match _SIMPLE_LTM_BASENAMES in workflow/common.py:
# prevalence -> liability -> correlations.
SIMPLE_LTM_CAPTIONS: dict[str, str] = {
    "prevalence_by_generation": (
        "Figure 26: Prevalence by generation (threshold model).\n\n"
        "Bar chart comparing observed vs. configured prevalence per generation and trait. "
        "Configured values shown as reference markers."
    ),
    "cross_trait.simple_ltm": (
        "Figure 27: Cross-trait liability joint plot (threshold model).\n\n"
        "Scatter of trait 1 vs. trait 2 liability coloured by threshold affected status."
    ),
    "liability_violin.simple_ltm": (
        "Figure 28: Liability violin plots by affected status (threshold model).\n\n"
        "Split violins showing liability for affected vs. unaffected under the threshold "
        "model. Diamond mean markers with \u03bc annotations and prevalence text."
    ),
    "liability_violin.simple_ltm.by_generation": (
        "Figure 29: Liability violin plots by generation (threshold model).\n\n"
        "Per-generation split violins with configured prevalence annotated. Same encoding "
        "as Figure 9 but for the liability-threshold phenotype."
    ),
    "joint_affected.simple_ltm": (
        "Figure 30: Joint affected status heatmap (threshold model).\n\n"
        "2\u00d72 heatmap of joint affected status proportions and counts with tetrachoric "
        "correlation annotated. Statistics computed on full (non-subsampled) data."
    ),
    "tetrachoric.simple_ltm": (
        "Figure 31: Tetrachoric correlations by relationship type (threshold model).\n\n"
        "Violin plots of tetrachoric correlations for threshold affected status indicators. "
        "Same encoding as Figure 21: coloured violins show observed tetrachoric correlations, "
        "black dots are per-replicate estimates, black dashed "
        "lines are the ground-truth Pearson liability correlations, "
        "red dotted lines are the parametric E[r] from configured ACE components, "
        "and pair counts are annotated above each violin."
    ),
    "cross_trait_tetrachoric.simple_ltm": (
        "Figure 32: Cross-trait tetrachoric correlations (threshold model).\n\n"
        "Same two-panel layout as the survival cross-trait tetrachoric figure. "
        "Left panel: same-person cross-trait r by generation. "
        "Right panel: cross-person cross-trait r by relationship type. "
        "Uses the liability-threshold phenotype affected indicators."
    ),
}

# Captions ordered to match _VALIDATION_BASENAMES in workflow/common.py:
# pedigree structure -> variance & heritability -> cross-trait -> summary -> benchmarks.
VALIDATION_CAPTIONS: dict[str, str] = {
    "family_size": (
        "Figure 1: Family size.\n\n"
        "Mean offspring per mother (blue, left-offset) and per father (orange, right-offset) "
        "among parents with \u22651 child. Orange dashes mark expected ~2.0 "
        "(N / n_mothers for balanced sex ratio)."
    ),
    "twin_rate": (
        "Figure 2: MZ twin rate.\n\n"
        "Observed MZ twin individual rate per replicate (blue dots) vs. configured "
        "p_mztwin (orange dashes)."
    ),
    "half_sib_proportions": (
        "Figure 3: Half-sibling proportions.\n\n"
        "Left panel: Observed maternal half-sibling pair proportion (driven by "
        "mating_lambda). Right panel: Proportion of offspring with "
        "at least one half-sibling."
    ),
    "consanguineous_matings": (
        "Figure 4: Consanguineous matings.\n\n"
        "Left panel: Number of half-sibling matings per replicate (random "
        "mating produces a small number by chance). Right panel: Missing "
        "grandparent-grandchild links caused by shared grandparents. "
        "Reconciliation verifies that all missing links are explained by "
        "consanguineous matings."
    ),
    "variance_components": (
        "Figure 5: Variance components.\n\n"
        "2\u00d73 grid, rows = traits 1 and 2, columns = A, C, E. Blue dots show "
        "observed founder-generation variance for each component per replicate. Orange "
        "dashes mark configured variance parameters."
    ),
    "correlations_A": (
        "Figure 6: A-component correlations.\n\n"
        "2\u00d72 grid. Panel 1: MZ twin A\u2081 correlation (expected = 1.0). "
        "Panel 2: DZ (full-sibling) A\u2081 correlation (expected = 0.5). "
        "Panel 3: Half-sibling A\u2081 correlation (expected = 0.25). "
        "Panel 4: Midparent-offspring A\u2081 R\u00b2 (expected = 0.5). "
        "Each panel shows blue dots with orange dashed reference line."
    ),
    "correlations_phenotype": (
        "Figure 7: Phenotype (liability) correlations.\n\n"
        "2\u00d72 grid. Expected values computed per-scenario from configured variance "
        "components. Panel 1: MZ twin liability\u2081 correlation (expected = A\u2081 + C\u2081). "
        "Panel 2: DZ sibling liability\u2081 correlation (expected = 0.5A\u2081 + C\u2081). "
        "Panel 3: Half-sibling liability\u2081 correlation (expected = 0.25A\u2081). "
        "Panel 4: Midparent-offspring liability\u2081 slope (expected = A\u2081)."
    ),
    "heritability_estimates": (
        "Figure 8: Heritability estimates.\n\n"
        "2\u00d72 grid, rows = traits 1 and 2. Left: Falconer's h\u00b2 vs. configured A. "
        "Right: Midparent-offspring liability slope vs. configured A."
    ),
    "cross_trait_correlations": (
        "Figure 9: Cross-trait correlations.\n\n"
        "1\u00d73 figure. Panel 1: Observed r_A vs. configured rA. Panel 2: Observed "
        "r_C vs. configured rC. Panel 3: Observed r_E with reference at 0 "
        "(theoretical independence)."
    ),
    "summary_bias": (
        "Figure 10: Summary bias.\n\n"
        "2\u00d73 grid of strip plots showing observed \u2212 expected for six metrics: "
        "A\u2081 bias, C\u2081 bias, E\u2081 bias, twin rate bias, DZ A\u2081 correlation "
        "bias (vs. 0.5), half-sibling A\u2081 correlation bias (vs. 0.25). Red dashed "
        "reference line at 0 (no bias)."
    ),
    "runtime": (
        "Figure 11: Simulation runtime.\n\n"
        "Log-log scatter of population size N (x) vs. simulation wall-clock seconds (y), "
        "coloured by scenario."
    ),
    "memory": (
        "Figure 12: Memory usage.\n\n"
        "Log-log scatter of population size N (x) vs. peak resident set size in MB (y), "
        "coloured by scenario."
    ),
}


def _render_params_page(
    pdf: PdfPages,
    scenario: str,
    params: dict,
) -> None:
    """Render a title page with pipeline DAG diagram and parameters."""
    from sim_ace.plot_pipeline import render_pipeline_figure

    fig = render_pipeline_figure(params, scenario=scenario)
    pdf.savefig(fig)
    plt.close(fig)


def _render_section_page(
    pdf: PdfPages,
    title: str,
    subtitle: str = "",
    equations: list[str] | None = None,
) -> None:
    """Render a section divider page with centred title and optional equations."""
    fig = plt.figure(figsize=(11, 8.5))

    if equations:
        # Shift layout to accommodate equation lines
        title_y = 0.62
        fig.text(
            0.5,
            title_y,
            title,
            fontsize=28,
            fontweight="bold",
            fontfamily="serif",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )
        eq_y = 0.49
        for eq_line in equations:
            fig.text(
                0.5,
                eq_y,
                eq_line,
                fontsize=18,
                fontfamily="serif",
                ha="center",
                va="center",
                transform=fig.transFigure,
            )
            eq_y -= 0.09
        if subtitle:
            fig.text(
                0.5,
                eq_y - 0.02,
                subtitle,
                fontsize=13,
                fontfamily="serif",
                color="0.4",
                ha="center",
                va="center",
                transform=fig.transFigure,
            )
    else:
        fig.text(
            0.5,
            0.55,
            title,
            fontsize=28,
            fontweight="bold",
            fontfamily="serif",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )
        if subtitle:
            fig.text(
                0.5,
                0.45,
                subtitle,
                fontsize=16,
                fontfamily="serif",
                color="0.4",
                ha="center",
                va="center",
                transform=fig.transFigure,
            )
    pdf.savefig(fig)
    plt.close(fig)


def _render_table1_page(
    pdf: PdfPages,
    all_stats: list[dict],
    scenario: str,
    params: dict,
) -> None:
    """Render a Table 1 epidemiological summary page."""
    from sim_ace.plot_table1 import render_table1_figure

    fig = render_table1_figure(all_stats, params, scenario=scenario)
    pdf.savefig(fig)
    plt.close(fig)


def assemble_atlas(
    plot_paths: list[Path],
    captions: dict[str, str],
    output_path: Path,
    scenario_params: dict | None = None,
    section_breaks: dict[int, tuple] | None = None,
    stats_data: list[dict] | None = None,
) -> None:
    """Combine saved plot images into a multi-page PDF with captions below each plot.

    Each page contains the plot image in the upper portion and the figure
    caption text below it on the same page.

    Args:
        plot_paths: Ordered list of plot image files (png or pdf).
        captions: Map from plot basename (without extension) to caption text.
        output_path: Path for the combined PDF.
        scenario_params: If provided, a dict with keys 'scenario' and parameter
            names.  A title page with all parameters is rendered first.
        section_breaks: Map from plot index to (title, subtitle) or
            (title, subtitle, equations) tuples.
            A section divider page is inserted before the plot at each index.
        stats_data: If provided, a list of phenotype_stats dicts (one per rep).
            A Table 1 page is rendered after the title page.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atlas_dir = output_path.parent.resolve()
    if section_breaks is None:
        section_breaks = {}

    with PdfPages(str(output_path)) as pdf:
        # Optional title page with scenario parameters
        if scenario_params is not None:
            scenario_name = scenario_params.get("scenario", "unknown")
            _render_params_page(pdf, scenario_name, scenario_params)

            # Table 1 page (requires both params and stats)
            if stats_data:
                _render_table1_page(pdf, stats_data, scenario_name, scenario_params)

        for idx, path in enumerate(plot_paths):
            # Insert section divider page if configured for this index
            if idx in section_breaks:
                brk = section_breaks[idx]
                sec_title, sec_subtitle = brk[0], brk[1]
                sec_equations = brk[2] if len(brk) > 2 else None
                _render_section_page(pdf, sec_title, sec_subtitle, equations=sec_equations)
            path = Path(path)
            if not path.exists():
                logger.warning("Atlas: skipping missing plot %s", path)
                continue

            img = Image.open(str(path))
            basename = path.stem
            caption = captions.get(basename, "")

            # Relative path from the atlas PDF to the source plot
            try:
                rel = path.resolve().relative_to(atlas_dir)
            except ValueError:
                rel = path.name

            # Split caption into title and body
            title, body = "", ""
            if caption:
                lines = caption.split("\n", 1)
                title = lines[0]
                body = lines[1].lstrip("\n") if len(lines) > 1 else ""

            # Use landscape letter page; reserve bottom for caption
            page_w, page_h = 11, 8.5
            # Scale caption space by text length so long captions don't overflow
            if not caption:
                caption_frac = 0.0
            elif len(caption) < 300:
                caption_frac = 0.18
            elif len(caption) < 500:
                caption_frac = 0.25
            else:
                caption_frac = 0.32
            img_frac = 1.0 - caption_frac

            fig = plt.figure(figsize=(page_w, page_h))

            # Image axes in the upper portion
            ax = fig.add_axes([0.005, caption_frac + 0.005, 0.99, img_frac - 0.01])
            ax.imshow(img)
            ax.axis("off")

            # Caption text in the lower portion
            if title:
                fig.text(
                    0.04,
                    caption_frac - 0.02,
                    title,
                    fontsize=14,
                    fontweight="bold",
                    fontfamily="serif",
                    verticalalignment="top",
                    transform=fig.transFigure,
                )
            if body:
                body += f"  [{rel}]"
                fig.text(
                    0.04,
                    caption_frac - 0.05,
                    body,
                    fontsize=12,
                    fontfamily="serif",
                    verticalalignment="top",
                    wrap=True,
                    transform=fig.transFigure,
                )

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    logger.info("Atlas saved to %s (%d plots)", output_path, len(plot_paths))
