"""Assemble individual plots into a multi-page PDF atlas with figure captions."""

__all__ = [
    "assemble_atlas",
    "get_model_equation",
    "get_model_family",
]

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from simace.plotting.atlas_manifest import AtlasItem, PlotEntry, SectionBreak

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
# Page renderers
# ---------------------------------------------------------------------------


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
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    bb = title_text.get_window_extent(renderer=renderer)
    title_width_fig = bb.width / fig.dpi / page_w

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
    items: list[AtlasItem],
    plot_dir: Path,
    output_path: Path,
    *,
    plot_ext: str = "png",
    scenario_params: dict | None = None,
    stats_data: list[dict] | None = None,
) -> None:
    """Combine plots and section breaks into a multi-page PDF with captions.

    Walks ``items`` linearly. ``PlotEntry`` items render as a plot+caption
    page; the ``"Figure {N}: "`` prefix is derived from the running plot
    index (1-based, counting only :class:`PlotEntry` items). ``SectionBreak``
    items render as a section divider page.

    Args:
        items: Ordered atlas manifest, mixing
            :class:`~simace.plotting.atlas_manifest.PlotEntry` and
            :class:`~simace.plotting.atlas_manifest.SectionBreak`.
        plot_dir: Directory containing the plot image files; each
            ``PlotEntry.basename`` resolves to ``plot_dir / f"{basename}.{plot_ext}"``.
        output_path: Path for the combined PDF.
        plot_ext: Image extension (default ``"png"``).
        scenario_params: If provided, a dict with key ``"scenario"`` and
            parameter names. A title page with all parameters is rendered first.
        stats_data: If provided, a list of phenotype_stats dicts (one per rep).
            A Table 1 page is rendered after the title page.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = Path(plot_dir)
    atlas_dir = output_path.parent.resolve()

    n_plots = sum(1 for item in items if isinstance(item, PlotEntry))

    with PdfPages(str(output_path)) as pdf:
        # Optional title page with scenario parameters
        if scenario_params is not None:
            scenario_name = scenario_params.get("scenario", "unknown")
            _render_params_page(pdf, scenario_name, scenario_params)

            # Table 1 page (requires both params and stats)
            if stats_data:
                _render_table1_page(pdf, stats_data, scenario_name, scenario_params)

        plot_idx = 0
        for item in items:
            if isinstance(item, SectionBreak):
                _render_section_page(
                    pdf,
                    item.title,
                    item.subtitle,
                    equations=list(item.equations) if item.equations else None,
                )
                continue

            plot_idx += 1
            path = plot_dir / f"{item.basename}.{plot_ext}"
            if not path.exists():
                logger.warning("Atlas: skipping missing plot %s", path)
                continue

            try:
                rel = path.resolve().relative_to(atlas_dir)
            except ValueError:
                rel = path.name

            title = f"Figure {plot_idx}: {item.title}"
            body = item.body

            caption_len = len(title) + 2 + len(body)
            if caption_len < 300:
                caption_frac = 0.13
            elif caption_len < 500:
                caption_frac = 0.18
            else:
                caption_frac = 0.24
            img_frac = 1.0 - caption_frac - _TOP_MARGIN

            fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))

            ax = fig.add_axes([0.005, caption_frac + 0.005, 0.99, img_frac - 0.005])
            with Image.open(path) as img:
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

    logger.info("Atlas saved to %s (%d plots)", output_path, n_plots)
