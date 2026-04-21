"""Assemble individual plots into a multi-page PDF atlas with figure captions."""

from __future__ import annotations

__all__ = [
    "PHENOTYPE_CAPTIONS",
    "SIMPLE_LTM_CAPTIONS",
    "VALIDATION_CAPTIONS",
    "assemble_atlas",
    "get_model_equation",
    "get_model_family",
]

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-family lookup
# ---------------------------------------------------------------------------

# Display names for distributions and methods
_DISTRIBUTION_DISPLAY: dict[str, str] = {
    "weibull": "Weibull",
    "exponential": "Exponential",
    "gompertz": "Gompertz",
    "lognormal": "Log-Normal",
    "loglogistic": "Log-Logistic",
    "gamma": "Gamma",
}

_METHOD_DISPLAY: dict[str, str] = {
    "ltm": "LTM",
    "cox": "Cox",
}

# Model family descriptions (templates)
_FAMILY_DESC: dict[str, str] = {
    "frailty": "Proportional hazards with {dist} baseline; frailty exp(\u03b2\u00b7L) scales hazard",
    "cure_frailty": "Mixture cure model ({dist} baseline): liability threshold for case status, frailty for age-at-onset",
    "adult": {
        "ltm": "Liability threshold for case status, deterministic probit CIP for age-at-onset",
        "cox": "Ranking for case status, stochastic Weibull CIP for age-at-onset",
    },
    "first_passage": (
        "Inverse Gaussian FPT: liability scales initial distance y\u2080 to boundary; drift \u03bc controls progression"
    ),
}


def _model_display_name(model: str, pp: dict) -> tuple[str, str]:
    """Return (short_name, description) for a phenotype model + params."""
    if model == "frailty":
        dist = pp.get("distribution", "unknown")
        dist_name = _DISTRIBUTION_DISPLAY.get(dist, dist.title())
        return (
            f"{dist_name} Frailty",
            _FAMILY_DESC["frailty"].format(dist=dist_name),
        )
    if model == "cure_frailty":
        dist = pp.get("distribution", "unknown")
        dist_name = _DISTRIBUTION_DISPLAY.get(dist, dist.title())
        return (
            f"Cure Frailty ({dist_name})",
            _FAMILY_DESC["cure_frailty"].format(dist=dist_name),
        )
    if model == "adult":
        method = pp.get("method", "unknown")
        method_name = _METHOD_DISPLAY.get(method, method.upper())
        return (
            f"ADuLT {method_name}",
            _FAMILY_DESC["adult"].get(method, f"ADuLT {method_name} model"),
        )
    if model == "first_passage":
        return ("First-Passage Time", _FAMILY_DESC["first_passage"])
    return (model.title(), model)


# Common frailty model equation (line 1 for all 6 baseline hazard distributions)
_FRAILTY_LINE = (
    r"$h(t \mid L) = h_0(t) \cdot e^{\beta L \,+\, \beta_{\mathrm{sex}} \cdot \mathrm{sex}},"
    r" \qquad L = A + C + E$"
)

# Distribution-specific baseline hazard h₀(t) (line 2)
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


def _equation_lines_for_model(model: str, pp: dict, label: str = "") -> list[str]:
    """Return mathtext equation line(s) for a single phenotype model."""
    prefix = (r"\mathrm{" + label + r"\!:}\ ") if label else ""

    if model in ("frailty", "cure_frailty"):
        dist = pp.get("distribution", "")
        if model == "frailty" and dist in _BASELINE_LINE:
            line1 = r"$" + prefix + _FRAILTY_LINE.strip("$") + r"$" if prefix else _FRAILTY_LINE
            return [line1, _BASELINE_LINE[dist]]
        if model == "cure_frailty":
            return [
                r"$"
                + prefix
                + r"\mathrm{case\!:}\ L > \Phi^{-1}(1-K), \qquad"
                + r" t_{\mathrm{case}} \sim h_0(t) \cdot"
                + r" e^{\beta L \,+\, \beta_{\mathrm{sex}} \cdot \mathrm{sex}}$",
            ]

    if model == "adult":
        method = pp.get("method", "")
        if method == "ltm":
            return [
                r"$" + prefix + r"\mathrm{CIP}(t) = \frac{K}{1 + e^{-k(t - x_0)}}$",
                r"$\mathrm{case\!:}\ L > \Phi^{-1}(1-K), \qquad"
                + r" t = x_0 + \frac{1}{k}\ln\!\frac{\Phi(-L)}{K - \Phi(-L)}$",
            ]
        if method == "cox":
            return [
                r"$"
                + prefix
                + r"t_{\mathrm{raw}} = \sqrt{-\ln U \,/\, e^{L}},"
                + r" \quad U \sim \mathrm{Uniform}(0,1]$",
                r"$\mathrm{case\!:}\ \mathrm{CIP}_{\mathrm{rank}} < K, \qquad"
                + r" t = x_0 + \frac{1}{k}\ln\!\frac{\mathrm{CIP}}{K - \mathrm{CIP}}$",
            ]

    if model == "first_passage":
        return [
            r"$"
            + prefix
            + r"y_0^{(i)} = \sqrt{\lambda}\,"
            + r"e^{-\beta L_i - \beta_{\mathrm{sex}} \cdot \mathrm{sex}_i},"
            + r"\quad Y(t) = y_0^{(i)} + \mu\,t + W(t),"
            + r"\quad T_i = \inf\{t : Y(t) \leq 0\}$",
        ]
    return []


def get_model_equation(params: dict) -> list[str]:
    """Return mathtext equation lines for the scenario's phenotype model(s)."""
    m1 = str(params.get("phenotype_model1", "frailty"))
    m2 = str(params.get("phenotype_model2", "frailty"))
    pp1 = params.get("phenotype_params1", {})
    pp2 = params.get("phenotype_params2", {})

    if m1 == m2 and pp1.get("distribution") == pp2.get("distribution") and pp1.get("method") == pp2.get("method"):
        return _equation_lines_for_model(m1, pp1)

    lines: list[str] = []
    lines.extend(_equation_lines_for_model(m1, pp1, label="Trait 1"))
    lines.extend(_equation_lines_for_model(m2, pp2, label="Trait 2"))
    return lines


def get_model_family(params: dict) -> tuple[str, str]:
    """Return (display_name, description) for the scenario's phenotype model(s).

    When both traits use the same model family and sub-type, return that family.
    When they differ, return a combined description.
    """
    m1 = str(params.get("phenotype_model1", "frailty"))
    m2 = str(params.get("phenotype_model2", "frailty"))
    pp1 = params.get("phenotype_params1", {})
    pp2 = params.get("phenotype_params2", {})

    name1, desc1 = _model_display_name(m1, pp1)
    name2, desc2 = _model_display_name(m2, pp2)

    if m1 == m2 and pp1.get("distribution") == pp2.get("distribution") and pp1.get("method") == pp2.get("method"):
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
        "Three-panel figure showing offspring and partner count distributions, "
        "averaged across replicates. Left: number of offspring per couple. "
        "Centre: number of offspring per person "
        "(including childless individuals at 0). Right: fraction of parents "
        "with 1 vs. 2+ partners, grouped by sex."
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
        "midparent liability (x) vs. offspring liability (y) coloured by offspring sex "
        "(green = daughters, blue = sons). Observed pooled regression line (solid orange, "
        "with 95% CI band), sex-specific regression lines, and expected slope from "
        "configured A (dashed). Text box shows pooled h\u00b2 and sex-specific h\u00b2\u2640/h\u00b2\u2642 "
        "(slope \u00b1 SE), Pearson r, pair count n, and p-value, averaged across replicates."
    ),
    "heritability.by_generation": (
        "Figure 7: Narrow-sense liability-scale heritability by generation.\n\n"
        "1\u00d72 figure, one panel per trait. Narrow-sense heritability "
        "h\u00b2 = Var(A) / (Var(A) + Var(C) + Var(E)) is computed from the "
        "per-generation variance components for each replicate. Blue dots show "
        "per-replicate h\u00b2 estimates. "
        "Orange dashed line marks the parametric heritability (A parameter)."
    ),
    "heritability.by_sex.by_generation": (
        "Figure 8: PO-regression heritability by offspring sex.\n\n"
        "1\u00d72 figure, one panel per trait. Heritability h\u00b2 estimated from "
        "midparent-offspring regression slope, stratified by offspring sex. "
        "Green dots = per-replicate daughter h\u00b2, blue dots = per-replicate son h\u00b2. "
        "Orange dashed line marks the parametric heritability (A parameter)."
    ),
    "additive_shared.by_generation": (
        "Figure 9: Additive genetic and shared environment by generation.\n\n"
        "1\u00d72 figure, one panel per trait. Combined proportion "
        "(Var(A) + Var(C)) / (Var(A) + Var(C) + Var(E)) is computed from "
        "the per-generation variance components for each replicate. Blue dots show "
        "per-replicate estimates. "
        "Orange dashed line marks the parametric value (A + C)."
    ),
    # -- Liability by affected status --
    "observed_h2": (
        "Figure 10: Observed-scale heritability from binary affected status.\n\n"
        "2\u00d72 grid: rows = traits, columns = scale. Left column: observed-scale "
        "h\u00b2 estimators computed directly from Pearson correlations on binary "
        "affected indicators \u2014 Falconer 2(r_MZ \u2212 r_FS), Sibs 2\u00b7r_FS, "
        "PO (midparent-offspring regression on binary affected status), "
        "Half-sibs 4\u00b7mean(r_MHS, r_PHS), Cousins 8\u00b7r_1C. Blue per-rep "
        "dots at each estimator; grey dotted line marks the Dempster\u2013Lerner expected "
        "value A\u00b7z(K)\u00b2/(K\u00b7(1\u2212K)) at the mean observed prevalence K. "
        "Right column: same per-rep estimates back-transformed to the liability scale "
        "via h\u00b2_liab = h\u00b2_obs \u00b7 K(1\u2212K)/z(K)\u00b2. The D\u2013L correction "
        "assumes a threshold-normal (LTM) mapping from liability to affected status; "
        "it is biased under non-threshold phenotype models such as pure frailty "
        "(see docs/examples/observed-vs-liability-h2.md)."
    ),
    # -- Liability by affected status --
    "liability_violin.phenotype": (
        "Figure 11: Liability violin plots by affected status (survival model).\n\n"
        "Split violin plots, one per trait. Left half = unaffected, right half = affected. "
        "Diamond markers show mean liability for each group with \u03bc annotations. "
        "Prevalence annotated below each trait."
    ),
    "liability_violin.phenotype.by_generation": (
        "Figure 12: Liability violin plots by generation (survival model).\n\n"
        "Grid: rows = traits, columns = recorded generations. Split violins for affected vs. "
        "unaffected within each generation. Diamond markers and \u03bc annotations show per-group "
        "means. x-axis labels show observed generation-specific prevalence."
    ),
    "liability_violin.phenotype.by_sex.by_generation": (
        "Figure 13: Liability violin plots by sex and generation (survival model).\n\n"
        "Grid: rows = traits, columns = generations. Each cell has side-by-side "
        "split violins for female (left) and male (right), each showing "
        "unaffected vs. affected distribution. Sex-specific prevalence annotated."
    ),
    "liability_components.by_generation": (
        "Figure 14: Liability components by generation.\n\n"
        "2\u00d73 grid: rows = traits, columns = variance components (A, C, E). "
        "Each panel shows the mean component value among affected (red), unaffected "
        "(grey), and overall (black) individuals per generation. "
        "Selection on A is visible as separation between affected and unaffected lines: "
        "affected individuals have higher mean A (positive selection). "
        "C and E show no systematic selection since they are independent of liability "
        "threshold crossing conditional on total liability. "
        "Generation-specific prevalence annotated on x-axis."
    ),
    # -- Survival phenotype & censoring --
    "age_at_onset_death": (
        "Figure 15: Age-at-onset and death-age histograms.\n\n"
        "A 2\u00d72 grid, rows = traits 1 and 2. Left column shows density histograms "
        "of observed age-at-onset for affected individuals (\u03b4 = 1). Right column shows "
        "age-at-death histograms for death-censored unaffected individuals."
    ),
    "mortality": (
        "Figure 16: Mortality rate by decade.\n\n"
        "Two-panel figure. Left panel shows per-decade mortality rate "
        "(deaths in decade / alive at start of decade), averaged across replicates. "
        "Right panel shows cumulative mortality, "
        "with cumulative survival probability annotated above each bar."
    ),
    "cumulative_incidence.by_sex": (
        "Figure 17: Cumulative incidence by sex.\n\n"
        "Two-panel figure, one per trait. Green line = female (sex=0), blue line = male "
        "(sex=1) observed cumulative incidence. Legend shows sample size and prevalence "
        "per sex. Statistics computed on full (non-subsampled) data."
    ),
    "cumulative_incidence.by_sex.by_generation": (
        "Figure 18: Cumulative incidence by sex and generation.\n\n"
        "Grid: rows = traits, columns = generations. Each panel shows cumulative incidence "
        "curves for female (green) and male (blue) separately. Legend shows per-sex sample "
        "size and prevalence within each generation. Statistics computed on full "
        "(non-subsampled) data."
    ),
    "cumulative_incidence.phenotype": (
        "Figure 19: Cumulative incidence curves.\n\n"
        "Two-panel figure, one per trait. Blue solid line = observed cumulative incidence "
        "from censored data (with min-max band across replicates). Grey solid line = true "
        "cumulative incidence from uncensored event times. Grey dashed crosshairs mark the "
        "ages at which 25% (Q1), 50%, and 75% (Q3) of lifetime cases have occurred. "
        "Text box shows affected %, true prevalence %, and censored %."
    ),
    "censoring": (
        "Figure 20: Censoring windows by generation.\n\n"
        "Grid of panels: rows = traits, columns = generations. Grey line = true cumulative "
        "incidence, blue line = observed cumulative incidence. Text box shows affected %, "
        "left-censored %, right-censored %, and death-censored % per generation. Column "
        "titles show observation window [lo, hi]."
    ),
    "censoring_confusion": (
        "Figure 21: Censoring confusion matrix.\n\n"
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
        "Figure 22: Censoring cascade.\n\n"
        "Per-trait stacked bar chart decomposing true cases (event time < censor_age) "
        "by generation into four mutually exclusive fates: observed (green), "
        "death-censored (red), right-censored (purple), and left-truncated (orange). "
        "Total bar height equals true case count per generation. Sensitivity "
        "(observed / true) is annotated per generation; subplot titles show overall "
        "sensitivity. Only generations with non-degenerate observation windows are shown. "
        "Statistics computed on full (non-subsampled) data."
    ),
    "liability_vs_aoo": (
        "Figure 23: Liability vs. age-at-onset.\n\n"
        "Side-by-side joint plots, one per trait. Central scatter of liability (x) vs. "
        "observed age-at-onset (y) for affected individuals, with regression line and "
        "95% CI band. Annotations show slope \u00b1 SE, Pearson r, n, and p-value, "
        "averaged across replicates. Marginal histograms on top and right."
    ),
    # -- Within-trait correlations --
    "tetrachoric.phenotype": (
        "Figure 24: Tetrachoric correlations by relationship type (survival model).\n\n"
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
    "tetrachoric.phenotype.by_sex": (
        "Figure 25: Tetrachoric correlations by sex (survival model).\n\n"
        "2\u00d72 grid: rows = traits, columns = sex (female, male). Same encoding as "
        "Figure 23: coloured violins show observed tetrachoric correlations for "
        "same-sex pairs only (FF or MM). Black dashed = liability correlation, "
        "red dotted = parametric E[r]. Opposite-sex pairs are excluded."
    ),
    "tetrachoric.phenotype.by_generation": (
        "Figure 26: Tetrachoric correlations by generation (survival model).\n\n"
        "Grid: rows = traits, columns = generations. Same encoding as Figure 23 "
        "(violins = observed tetrachoric correlations, black dashed = true liability "
        "correlations, red dotted = parametric E[r], dots = per-replicate estimates), "
        "computed within each generation separately."
    ),
    # -- Cross-trait correlations --
    "cross_trait.phenotype": (
        "Figure 27: Cross-trait liability joint plots coloured by affected status (trait 1).\n\n"
        "Same 2\u00d72 layout as Figure 5, but with affected-status colouring based on trait 1. "
        "Blue points = unaffected, orange points = affected (trait 1). Marginal histograms stacked "
        "by affected status."
    ),
    "cross_trait.phenotype.t2": (
        "Figure 28: Cross-trait liability joint plots coloured by affected status (trait 2).\n\n"
        "Same 2\u00d72 layout as Figure 5, but with affected-status colouring based on trait 2. "
        "Blue points = unaffected, orange points = affected (trait 2). Marginal histograms stacked "
        "by affected status."
    ),
    "joint_affected.phenotype": (
        "Figure 29: Joint affected status heatmap (survival model).\n\n"
        "2\u00d72 heatmap of joint affected status across both traits. Cell annotations "
        "show proportion and count. Title shows cross-trait correlation estimates: "
        "'r_tet' = tetrachoric correlation on censored binary affected status; "
        "'r_frailty' = frailty-estimated liability correlation from uncensored survival "
        "data (oracle); 'stratified' = generation-stratified estimate that "
        "computes per-generation correlations and combines via inverse-variance "
        "weighting; 'naive' = unweighted pooled censored estimate. "
        "Statistics computed on full (non-subsampled) data."
    ),
    "cross_trait_tetrachoric": (
        "Figure 30: Cross-trait tetrachoric correlations.\n\n"
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
        "Figure 31: Prevalence by generation (threshold model).\n\n"
        "Bar chart comparing observed vs. configured prevalence per generation and trait. "
        "Configured values shown as reference markers."
    ),
    "cross_trait.simple_ltm": (
        "Figure 32: Cross-trait liability joint plot (threshold model).\n\n"
        "Scatter of trait 1 vs. trait 2 liability coloured by threshold affected status."
    ),
    "liability_violin.simple_ltm": (
        "Figure 33: Liability violin plots by affected status (threshold model).\n\n"
        "Split violins showing liability for affected vs. unaffected under the threshold "
        "model. Diamond mean markers with \u03bc annotations and prevalence text."
    ),
    "liability_violin.simple_ltm.by_generation": (
        "Figure 34: Liability violin plots by generation (threshold model).\n\n"
        "Per-generation split violins with configured prevalence annotated. Same encoding "
        "as Figure 11 but for the liability-threshold phenotype."
    ),
    "joint_affected.simple_ltm": (
        "Figure 35: Joint affected status heatmap (threshold model).\n\n"
        "2\u00d72 heatmap of joint affected status proportions and counts with tetrachoric "
        "correlation annotated. Statistics computed on full (non-subsampled) data."
    ),
    "tetrachoric.simple_ltm": (
        "Figure 36: Tetrachoric correlations by relationship type (threshold model).\n\n"
        "Violin plots of tetrachoric correlations for threshold affected status indicators. "
        "Same encoding as Figure 24: coloured violins show observed tetrachoric correlations, "
        "black dots are per-replicate estimates, black dashed "
        "lines are the ground-truth Pearson liability correlations, "
        "red dotted lines are the parametric E[r] from configured ACE components, "
        "and pair counts are annotated above each violin."
    ),
    "cross_trait_tetrachoric.simple_ltm": (
        "Figure 37: Cross-trait tetrachoric correlations (threshold model).\n\n"
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
    from simace.plotting.plot_pipeline import render_pipeline_figure

    fig = render_pipeline_figure(params, scenario=scenario)
    pdf.savefig(fig)
    plt.close(fig)


_PAGE_W, _PAGE_H = 11.69, 8.27  # A4 landscape (inches)
_TOP_MARGIN = 0.04  # figure-fraction margin at top of plot pages


def _render_section_page(
    pdf: PdfPages,
    title: str,
    subtitle: str = "",
    equations: list[str] | None = None,
) -> None:
    """Render a section divider page with centred title and optional equations."""
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))

    if equations:
        # Shift layout to accommodate equation lines
        title_y = 0.62
        fig.text(
            0.5,
            title_y,
            title,
            fontsize=28,
            fontweight="bold",
            fontfamily="sans-serif",
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
                fontfamily="sans-serif",
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
                fontfamily="sans-serif",
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
            fontfamily="sans-serif",
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
                fontfamily="sans-serif",
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
    from simace.plotting.plot_table1 import render_table1_figure

    fig = render_table1_figure(all_stats, params, scenario=scenario)
    pdf.savefig(fig)
    plt.close(fig)


def _render_inline_caption(
    fig,
    x: float,
    y: float,
    title: str,
    body: str,
    fontsize: int = 11,
    fontfamily: str = "sans-serif",
) -> None:
    """Render a caption with bold title inline with normal-weight body text.

    Measures the rendered bold title width via the figure renderer, then
    places the body text exactly 3 spaces after it.
    """
    import textwrap

    from matplotlib.font_manager import FontProperties

    line_h = 0.022
    page_w = fig.get_figwidth()
    usable_frac = 0.96 - x
    chars_per_line = int(page_w * usable_frac * 11.5)

    if not body:
        fig.text(
            x,
            y,
            title,
            fontsize=fontsize,
            fontweight="bold",
            fontfamily=fontfamily,
            verticalalignment="top",
            transform=fig.transFigure,
        )
        return

    # Render bold title and measure its width via the renderer
    title_text = fig.text(
        x,
        y,
        title,
        fontsize=fontsize,
        fontweight="bold",
        fontfamily=fontfamily,
        verticalalignment="top",
        transform=fig.transFigure,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = title_text.get_window_extent(renderer=renderer)
    title_width_fig = bb.width / fig.dpi / page_w

    # Measure width of 3 spaces in medium weight at same fontsize
    fp = FontProperties(family=fontfamily, size=fontsize, weight="medium")
    space_text = fig.text(0, 0, "   ", fontproperties=fp, transform=fig.transFigure)
    space_bb = space_text.get_window_extent(renderer=renderer)
    space_width_fig = space_bb.width / fig.dpi / page_w
    space_text.remove()

    # Position body text after title + 3-space gap
    body_x = x + title_width_fig + space_width_fig

    # Estimate how many chars fit on the first line after the title
    first_line_chars = int((0.96 - body_x) / usable_frac * chars_per_line)
    if first_line_chars < 15:
        # Not enough room; put body on the next line
        wrapped = textwrap.fill(body, width=chars_per_line)
        fig.text(
            x,
            y - line_h,
            wrapped,
            fontsize=fontsize,
            fontweight="medium",
            fontfamily=fontfamily,
            verticalalignment="top",
            transform=fig.transFigure,
        )
        return

    # Wrap body: first line shorter, subsequent lines full width
    words = body.split()
    first_line_words = []
    current_len = 0
    for word in words:
        if current_len + len(word) + (1 if first_line_words else 0) > first_line_chars:
            break
        first_line_words.append(word)
        current_len += len(word) + (1 if len(first_line_words) > 1 else 0)
    first_line_body = " ".join(first_line_words)
    remaining_body = body[len(first_line_body) :].lstrip()

    # Render first-line body text
    fig.text(
        body_x,
        y,
        first_line_body,
        fontsize=fontsize,
        fontweight="medium",
        fontfamily=fontfamily,
        verticalalignment="top",
        transform=fig.transFigure,
    )

    # Wrap and render remaining lines
    if remaining_body:
        remaining_lines = textwrap.wrap(remaining_body, width=chars_per_line)
        for i, line in enumerate(remaining_lines):
            fig.text(
                x,
                y - (i + 1) * line_h,
                line,
                fontsize=fontsize,
                fontweight="medium",
                fontfamily=fontfamily,
                verticalalignment="top",
                transform=fig.transFigure,
            )


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

            # A4 landscape page; reserve bottom for caption, top for margin
            page_w, page_h = _PAGE_W, _PAGE_H
            # Scale caption space by text length so long captions don't overflow
            if not caption:
                caption_frac = 0.0
            elif len(caption) < 300:
                caption_frac = 0.13
            elif len(caption) < 500:
                caption_frac = 0.18
            else:
                caption_frac = 0.24
            img_frac = 1.0 - caption_frac - _TOP_MARGIN

            fig = plt.figure(figsize=(page_w, page_h))

            # Image axes: below top margin, above caption
            ax = fig.add_axes([0.005, caption_frac + 0.005, 0.99, img_frac - 0.005])
            ax.imshow(img)
            ax.axis("off")

            # Thin hairline border around the figure image
            rect = plt.Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                linewidth=0.3,
                edgecolor="#cccccc",
                facecolor="none",
                clip_on=False,
            )
            ax.add_patch(rect)

            # Caption text in the lower portion — inline bold title + body
            if caption:
                caption_y = caption_frac - 0.015
                body_with_ref = f"{body}  [{rel}]" if body else f"[{rel}]"
                _render_inline_caption(
                    fig,
                    0.04,
                    caption_y,
                    title,
                    body_with_ref,
                    fontsize=11,
                    fontfamily="sans-serif",
                )

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    logger.info("Atlas saved to %s (%d plots)", output_path, len(plot_paths))
