"""Assemble individual plots into a multi-page PDF atlas with figure captions."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Caption text for each plot, keyed by basename (filename without extension).
# Content drawn from plots.md.
# ---------------------------------------------------------------------------

# Captions ordered to match _PHENOTYPE_BASENAMES in workflow/common.py:
# liability structure -> Weibull phenotype -> censoring -> correlations.
PHENOTYPE_CAPTIONS: dict[str, str] = {
    # -- Pedigree structure --
    "pedigree_counts.ped": (
        "Figure 1: Pedigree relationship pair counts (G_ped).\n\n"
        "Schematic multi-generational pedigree diagram showing the 10 relationship "
        "categories extracted from the full simulated pedigree spanning all G_ped "
        "generations. Colored arcs connect pairs of related individuals, with mean "
        "pair counts (averaged across replicates) superimposed. Node shapes follow "
        "standard pedigree conventions: squares = male, circles = female. "
        "Relationship categories: MZ twin, Full sib, Maternal half sib, "
        "Paternal half sib, Mother-offspring, Father-offspring, 1st cousin, "
        "Grandparent-grandchild, Avuncular, 2nd cousin."
    ),
    "pedigree_counts": (
        "Figure 2: Pedigree relationship pair counts (G_pheno).\n\n"
        "Same diagram as Figure 1, but restricted to the last G_pheno phenotyped "
        "generations. These are the individuals for whom survival times and affected "
        "status are observed, and whose relationship pairs are used for downstream "
        "correlation and heritability analyses."
    ),
    # -- Liability structure --
    "cross_trait": (
        "Figure 3: Cross-trait liability joint plots.\n\n"
        "2\u00d72 grid of joint plots for Liability, A (additive genetic), C (common environment), "
        "and E (unique environment). Central scatter of Trait 1 (x) vs. Trait 2 (y) with "
        "Pearson r annotation and marginal histograms."
    ),
    # -- Liability-scale heritability --
    "parent_offspring_liability.by_generation": (
        "Figure 4: Parent-offspring liability regressions.\n\n"
        "Grid: rows = traits, columns = last 3 non-founder generations. Scatter of "
        "midparent liability (x) vs. offspring liability (y) with regression line. "
        "Text box shows Pearson r and pair count n."
    ),
    "heritability.by_generation": (
        "Figure 5: Narrow-sense liability-scale heritability by generation.\n\n"
        "1\u00d72 figure, one panel per trait. Narrow-sense heritability "
        "h\u00b2 = Var(A) / (Var(A) + Var(C) + Var(E)) is computed from the "
        "per-generation variance components for each replicate. Blue dots show "
        "per-replicate h\u00b2 estimates. "
        "Orange dashed line marks the parametric heritability (A parameter). "
        "Stable h\u00b2 across generations confirms that the ACE variance "
        "decomposition is maintained through the simulation."
    ),
    "broad_heritability.by_generation": (
        "Figure 6: Additive genetic and shared environment by generation.\n\n"
        "1\u00d72 figure, one panel per trait. Combined proportion "
        "(Var(A) + Var(C)) / (Var(A) + Var(C) + Var(E)) is computed from "
        "the per-generation variance components for each replicate. Blue dots show "
        "per-replicate estimates. "
        "Orange dashed line marks the parametric value (A + C). "
        "Comparing with the narrow-sense h\u00b2 (Figure 5) isolates the "
        "contribution of shared environment to familial resemblance."
    ),
    # -- Liability by affected status --
    "cross_trait.frailty": (
        "Figure 7: Cross-trait liability joint plots coloured by affected status.\n\n"
        "Same 2\u00d72 layout as Figure 3, but with affected-status colouring based on trait 1. "
        "Blue points = unaffected, orange points = affected (trait 1). Marginal histograms stacked "
        "by affected status."
    ),
    "liability_violin.frailty": (
        "Figure 8: Liability violin plots by affected status (frailty model).\n\n"
        "Split violin plots, one per trait. Left half = unaffected, right half = affected. "
        "Diamond markers show mean liability for each group with \u03bc annotations. "
        "Prevalence annotated below each trait."
    ),
    "liability_violin.frailty.by_generation": (
        "Figure 9: Liability violin plots by generation (frailty model).\n\n"
        "Grid: rows = traits, columns = recorded generations. Split violins for affected vs. "
        "unaffected within each generation. Diamond markers and \u03bc annotations show per-group "
        "means. x-axis labels show observed generation-specific prevalence."
    ),
    # -- Frailty phenotype --
    "liability_vs_aoo": (
        "Figure 10: Liability vs. age-at-onset.\n\n"
        "Side-by-side joint plots, one per trait. Central scatter of liability (x) vs. "
        "observed age-at-onset (y) for affected individuals, with regression line and R\u00b2 "
        "annotation. Marginal histograms on top and right."
    ),
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
    "cumulative_incidence.frailty": (
        "Figure 13: Cumulative incidence curves.\n\n"
        "Two-panel figure, one per trait. Blue solid line = observed cumulative incidence "
        "from censored data (with min-max band across replicates). Grey solid line = true "
        "cumulative incidence from uncensored event times. Grey dashed crosshairs mark the "
        "age at which 50% of lifetime cases have occurred. Text box shows affected %, "
        "true prevalence %, and censored %."
    ),
    "cumulative_incidence.by_sex": (
        "Figure 14: Cumulative incidence by sex.\n\n"
        "Two-panel figure, one per trait. Blue line = female (sex=0), red line = male "
        "(sex=1) observed cumulative incidence. Legend shows sample size and prevalence "
        "per sex. Separation between curves indicates a sex-differential hazard rate "
        "driven by the beta_sex covariate."
    ),
    "cumulative_incidence.by_sex.by_generation": (
        "Figure 15: Cumulative incidence by sex and generation.\n\n"
        "Grid: rows = traits, columns = generations. Each panel shows cumulative incidence "
        "curves for female (blue) and male (red) separately. Legend shows per-sex sample "
        "size and prevalence within each generation. Useful for detecting generation-specific "
        "sex effects or interactions with censoring windows."
    ),
    # -- Censoring --
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
        "specificity (true negative rate). False negatives arise from left-truncation, "
        "right-censoring, or competing mortality. Only phenotyped generations "
        "(those with non-degenerate observation windows) are included."
    ),
    "censoring_cascade": (
        "Figure 18: Censoring cascade.\n\n"
        "Per-trait stacked bar chart decomposing true cases (event time < censor_age) "
        "by generation into four mutually exclusive fates: observed (green), "
        "death-censored (coral), right-censored (amber), and left-truncated (orange). "
        "Total bar height equals true case count per generation. Sensitivity "
        "(observed / true) is annotated per generation; subplot titles show overall "
        "sensitivity. Only generations with non-degenerate observation windows are shown."
    ),
    # -- Familial correlations --
    "joint_affected.frailty": (
        "Figure 19: Joint affected status heatmap (frailty model).\n\n"
        "2\u00d72 heatmap of joint affected status across both traits. Cell annotations "
        "show proportion and count. Title shows cross-trait correlation estimates: "
        "'r_tet' = tetrachoric correlation on censored binary affected status; "
        "'r_frailty' = frailty-estimated liability correlation from uncensored survival "
        "data (oracle); 'stratified' = generation-stratified estimate that "
        "computes per-generation correlations and combines via inverse-variance "
        "weighting, reducing bias from heterogeneous censoring across generations; "
        "'naive' = unweighted pooled censored estimate for comparison."
    ),
    "cross_trait_frailty.by_generation": (
        "Figure 20: Cross-trait frailty correlation by generation.\n\n"
        "Per-generation cross-trait liability correlation estimated from censored "
        "survival data. Blue dots = per-replicate per-generation estimates "
        "with 95% CI error bars; blue line = mean across replicates. "
        "Green dash-dot line = uncensored oracle (ground truth from raw event times). "
        "Orange dashed line = inverse-variance weighted mean across generations "
        "(stratified estimate). Dark orange dotted line = naive pooled estimate (biased). "
        "Generations with very low event rates may hit the boundary and be excluded."
    ),
    "tetrachoric.frailty": (
        "Figure 21: Tetrachoric correlations by relationship type (frailty model).\n\n"
        "Two-panel figure, one per trait. Coloured violins show the distribution of "
        "tetrachoric correlations (computed from censored binary affected status) across "
        "replicates for each relationship type. "
        "Black dots = individual per-replicate tetrachoric estimates. "
        "Black dashed lines = mean Pearson liability correlation (ground-truth correlation "
        "on the continuous latent liability, serving as the theoretical reference). "
        "Green dash-dot lines = mean uncensored frailty pairwise survival-time correlation "
        "(showing what the correlation would be without censoring distortion; present when "
        "available). N = mean pairs per replicate. The gap between violins and dashed lines "
        "reflects attenuation from censoring and dichotomization."
    ),
    "tetrachoric.frailty.by_generation": (
        "Figure 22: Tetrachoric correlations by generation (frailty model).\n\n"
        "Grid: rows = traits, columns = generations. Same encoding as Figure 21 "
        "(violins = observed tetrachoric correlations, black dashed = true liability "
        "correlations, dots = per-replicate estimates), computed within each generation "
        "separately."
    ),
    "cross_trait_tetrachoric": (
        "Figure 23: Cross-trait tetrachoric correlations.\n\n"
        "Two-panel figure measuring cross-trait association via tetrachoric "
        "correlation between affected1 and affected2. "
        "Left panel: same-person cross-trait r by generation (blue dots per rep, "
        "line = mean), with overall r (black dashed) and frailty oracle (green "
        "dash-dot) reference lines when available. "
        "Right panel: cross-person cross-trait r by relationship type "
        "(coloured violins + black dots), showing how genetic and environmental "
        "relatedness induces cross-trait association across individuals."
    ),
}

# Captions ordered to match _THRESHOLD_BASENAMES in workflow/common.py:
# prevalence -> liability -> correlations.
THRESHOLD_CAPTIONS: dict[str, str] = {
    "prevalence_by_generation": (
        "Figure 24: Prevalence by generation (threshold model).\n\n"
        "Bar chart comparing observed vs. configured prevalence per generation and trait. "
        "Configured values shown as reference markers."
    ),
    "cross_trait.threshold": (
        "Figure 25: Cross-trait liability joint plot (threshold model).\n\n"
        "Scatter of trait 1 vs. trait 2 liability coloured by threshold affected status."
    ),
    "liability_violin.threshold": (
        "Figure 26: Liability violin plots by affected status (threshold model).\n\n"
        "Split violins showing liability for affected vs. unaffected under the threshold "
        "model. Diamond mean markers with \u03bc annotations and prevalence text."
    ),
    "liability_violin.threshold.by_generation": (
        "Figure 27: Liability violin plots by generation (threshold model).\n\n"
        "Per-generation split violins with configured prevalence annotated. Same encoding "
        "as Figure 9 but for the liability-threshold phenotype."
    ),
    "joint_affected.threshold": (
        "Figure 28: Joint affected status heatmap (threshold model).\n\n"
        "2\u00d72 heatmap of joint affected status proportions and counts with tetrachoric "
        "correlation annotated."
    ),
    "tetrachoric.threshold": (
        "Figure 29: Tetrachoric correlations by relationship type (threshold model).\n\n"
        "Violin plots of tetrachoric correlations for threshold affected status indicators. "
        "Same encoding as Figure 21: coloured violins show observed tetrachoric correlations "
        "from binary affected status, black dots are per-replicate estimates, black dashed "
        "lines are the ground-truth Pearson liability correlations, and pair counts are "
        "annotated above each violin."
    ),
    "cross_trait_tetrachoric.threshold": (
        "Figure 30: Cross-trait tetrachoric correlations (threshold model).\n\n"
        "Same two-panel layout as the Weibull cross-trait tetrachoric figure. "
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
        "among parents with \u22651 child. Orange dashes mark configured Poisson \u03bb "
        "(fam_size)."
    ),
    "twin_rate": (
        "Figure 2: MZ twin rate.\n\n"
        "Observed MZ twin individual rate per replicate (blue dots) vs. configured "
        "p_mztwin (orange dashes)."
    ),
    "half_sib_proportions": (
        "Figure 3: Half-sibling proportions.\n\n"
        "Left panel: Observed maternal half-sibling pair proportion vs. expected "
        "1 \u2212 (1 \u2212 p_nonsocial)\u00b2. Right panel: Proportion of offspring with "
        "at least one half-sibling."
    ),
    "variance_components": (
        "Figure 4: Variance components.\n\n"
        "2\u00d73 grid, rows = traits 1 and 2, columns = A, C, E. Blue dots show "
        "observed founder-generation variance for each component per replicate. Orange "
        "dashes mark configured variance parameters."
    ),
    "correlations_A": (
        "Figure 5: A-component correlations.\n\n"
        "2\u00d72 grid. Panel 1: MZ twin A\u2081 correlation (expected = 1.0). "
        "Panel 2: DZ (full-sibling) A\u2081 correlation (expected = 0.5). "
        "Panel 3: Half-sibling A\u2081 correlation (expected = 0.25). "
        "Panel 4: Midparent-offspring A\u2081 R\u00b2 (expected = 0.5). "
        "Each panel shows blue dots with orange dashed reference line."
    ),
    "correlations_phenotype": (
        "Figure 6: Phenotype (liability) correlations.\n\n"
        "2\u00d72 grid. Expected values computed per-scenario from configured variance "
        "components. Panel 1: MZ twin liability\u2081 correlation (expected = A\u2081 + C\u2081). "
        "Panel 2: DZ sibling liability\u2081 correlation (expected = 0.5A\u2081 + C\u2081). "
        "Panel 3: Half-sibling liability\u2081 correlation (expected = 0.25A\u2081). "
        "Panel 4: Midparent-offspring liability\u2081 slope (expected = A\u2081)."
    ),
    "heritability_estimates": (
        "Figure 7: Heritability estimates.\n\n"
        "2\u00d72 grid, rows = traits 1 and 2. Left: Falconer's h\u00b2 vs. configured A. "
        "Right: Midparent-offspring liability slope vs. configured A."
    ),
    "cross_trait_correlations": (
        "Figure 8: Cross-trait correlations.\n\n"
        "1\u00d73 figure. Panel 1: Observed r_A vs. configured rA. Panel 2: Observed "
        "r_C vs. configured rC. Panel 3: Observed r_E with reference at 0 "
        "(theoretical independence)."
    ),
    "summary_bias": (
        "Figure 9: Summary bias.\n\n"
        "2\u00d73 grid of strip plots showing observed \u2212 expected for six metrics: "
        "A\u2081 bias, C\u2081 bias, E\u2081 bias, twin rate bias, DZ A\u2081 correlation "
        "bias (vs. 0.5), half-sibling A\u2081 correlation bias (vs. 0.25). Red dashed "
        "reference line at 0 (no bias)."
    ),
    "runtime": (
        "Figure 10: Simulation runtime.\n\n"
        "Log-log scatter of population size N (x) vs. simulation wall-clock seconds (y), "
        "coloured by scenario."
    ),
    "memory": (
        "Figure 11: Memory usage.\n\n"
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


def _render_section_page(pdf: PdfPages, title: str, subtitle: str = "") -> None:
    """Render a section divider page with centred title."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(
        0.5, 0.55, title,
        fontsize=28, fontweight="bold", fontfamily="serif",
        ha="center", va="center", transform=fig.transFigure,
    )
    if subtitle:
        fig.text(
            0.5, 0.45, subtitle,
            fontsize=16, fontfamily="serif", color="0.4",
            ha="center", va="center", transform=fig.transFigure,
        )
    pdf.savefig(fig)
    plt.close(fig)


def assemble_atlas(
    plot_paths: list[Path],
    captions: dict[str, str],
    output_path: Path,
    scenario_params: dict | None = None,
    section_breaks: dict[int, tuple[str, str]] | None = None,
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
        section_breaks: Map from plot index to (title, subtitle) pairs.
            A section divider page is inserted before the plot at each index.
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
        for idx, path in enumerate(plot_paths):
            # Insert section divider page if configured for this index
            if idx in section_breaks:
                sec_title, sec_subtitle = section_breaks[idx]
                _render_section_page(pdf, sec_title, sec_subtitle)
            path = Path(path)
            if not path.exists():
                logger.warning("Atlas: skipping missing plot %s", path)
                continue

            img = mpimg.imread(str(path))
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
            ax = fig.add_axes([0.02, caption_frac + 0.01, 0.96, img_frac - 0.02])
            ax.imshow(img)
            ax.axis("off")

            # Caption text in the lower portion
            if title:
                fig.text(
                    0.04, caption_frac - 0.02, title,
                    fontsize=14, fontweight="bold", fontfamily="serif",
                    verticalalignment="top",
                    transform=fig.transFigure,
                )
            if body:
                body += f"  [{rel}]"
                fig.text(
                    0.04, caption_frac - 0.05, body,
                    fontsize=12, fontfamily="serif",
                    verticalalignment="top",
                    wrap=True,
                    transform=fig.transFigure,
                )

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    logger.info("Atlas saved to %s (%d plots)", output_path, len(plot_paths))
