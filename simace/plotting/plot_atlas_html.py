"""Assemble atlas manifest entries into a single-page HTML atlas."""

__all__ = ["assemble_html_atlas"]

import base64
import logging
import re
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from simace.plotting.atlas_manifest import AtlasItem, PlotEntry, SectionBreak
from simace.plotting.plot_table1 import Table1Row, Table1Section, Table1Summary, build_table1_summary

logger = logging.getLogger(__name__)

# Browser-displayable plot sources and their data-URI MIME types. Keys must
# stay in sync with _MIME_BY_EXT below (the HTML atlas embeds every plot inline).
_MIME_BY_EXT = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
    "svg": "image/svg+xml",
}
_BROWSER_IMAGE_EXTS = frozenset(_MIME_BY_EXT)


def _validate_plot_ext(plot_ext: str) -> str:
    """Return a normalised browser-displayable plot extension."""
    normalised = plot_ext.lower().lstrip(".")
    if normalised not in _BROWSER_IMAGE_EXTS:
        allowed = ", ".join(sorted(_BROWSER_IMAGE_EXTS))
        raise ValueError(
            "The HTML atlas embeds plots inline and needs a browser-displayable "
            f"plot source ({allowed}); got {plot_ext!r}. Use png or svg for the "
            "HTML atlas, or build the on-demand PDF atlas (atlas.pdf) when the "
            "plot source is pdf."
        )
    return normalised


def _data_uri(path: Path, ext: str) -> str:
    """Return a base64 ``data:`` URI embedding the plot file bytes.

    ``ext`` is a normalised extension from :func:`_validate_plot_ext`, so it is
    guaranteed to have a MIME mapping. Embedding keeps the atlas a single
    self-contained file with no sibling assets.
    """
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{_MIME_BY_EXT[ext]};base64,{encoded}"


def _slug(text: str, fallback: str) -> str:
    """Return a URL-safe slug for generated section anchors."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or fallback


def _toc_item(href: str, label: str, *, item_class: str = "") -> str:
    """Render one sidebar table-of-contents item."""
    class_attr = f' class="{escape(item_class, quote=True)}"' if item_class else ""
    return f'<li{class_attr}><a href="{escape(href, quote=True)}">{escape(label)}</a></li>'


def _html_page(title: str, nav: list[str], body: list[str]) -> str:
    """Wrap sidebar navigation and body cards into a complete HTML document."""
    title_html = escape(title)
    nav_html = "\n".join(nav)
    body_html = "\n".join(body)
    css = """
:root {
  color-scheme: light;
  --bg: #f5f7fb;
  --card: #ffffff;
  --ink: #1f2937;
  --muted: #667085;
  --line: #d0d5dd;
  --accent: #2457a6;
  --warn: #9a4b00;
  --warn-bg: #fff4e5;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.5;
}
.layout { display: grid; grid-template-columns: minmax(15rem, 19rem) minmax(0, 1fr); min-height: 100vh; }
.sidebar {
  position: sticky;
  top: 0;
  height: 100vh;
  overflow: auto;
  padding: 1.25rem 1rem;
  background: #ffffff;
  border-right: 1px solid var(--line);
}
.sidebar h2 { margin: 0 0 0.75rem; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.toc { list-style: none; margin: 0; padding: 0; }
.toc li { margin: 0.12rem 0; }
.toc li.section { margin-top: 0.75rem; font-weight: 700; }
.toc li.missing a { color: var(--warn); }
.toc a { display: block; padding: 0.22rem 0.35rem; border-radius: 0.35rem; color: var(--accent); text-decoration: none; }
.toc a:hover { background: #eef4ff; }
.content { max-width: 72rem; padding: 2rem clamp(1rem, 3vw, 2.75rem) 4rem; }
.page-header { margin-bottom: 1.5rem; }
.page-header .eyebrow { margin: 0 0 0.25rem; color: var(--muted); font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; }
h1 { margin: 0; font-size: clamp(2rem, 4vw, 3rem); line-height: 1.1; }
.card {
  margin: 0 0 1.5rem;
  padding: 1.25rem;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 0.8rem;
  box-shadow: 0 1px 2px rgb(16 24 40 / 0.06);
}
.card h2 { margin: 0 0 0.5rem; font-size: 1.45rem; }
.card p { margin: 0.35rem 0 0; color: var(--muted); }
.card img { display: block; max-width: 100%; height: auto; margin: 0.85rem auto 0; border-radius: 0.35rem; }
.equation-svg { margin: 0.85rem auto 0; }
.equation-svg svg { display: block; max-width: 100%; height: auto; margin: 0 auto; }
.overview-card > p { max-width: 58rem; }
.overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: 1rem; margin-top: 1rem; }
.overview-panel { padding: 1rem; border: 1px solid var(--line); border-radius: 0.65rem; background: #fbfcff; }
.overview-panel.wide { grid-column: 1 / -1; }
.overview-panel h3 { margin: 0 0 0.75rem; font-size: 1rem; color: var(--ink); }
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr)); gap: 0.65rem; }
.metric { padding: 0.65rem; border: 1px solid #e4e7ec; border-radius: 0.5rem; background: #ffffff; }
.metric .label { display: block; margin-bottom: 0.15rem; color: var(--muted); font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }
.metric .value { display: block; color: var(--ink); font-size: 1.05rem; font-weight: 750; }
.metric.missing .value { color: var(--muted); font-style: italic; font-weight: 600; }
.param-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
.param-table th, .param-table td { padding: 0.45rem 0.5rem; border-top: 1px solid #e4e7ec; text-align: left; vertical-align: top; }
.param-table th { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.param-table .subheader-row th { padding-top: 0.8rem; color: var(--accent); font-size: 0.82rem; }
.param-table td.value { font-weight: 650; }
.param-table td.missing { color: var(--muted); font-style: italic; }
.table1-card > p { margin-bottom: 1rem; }
.table1-section { margin-top: 1.1rem; }
.table1-section h3 { margin: 0 0 0.45rem; font-size: 1rem; color: var(--accent); }
.table1-scroll { overflow-x: auto; border: 1px solid #e4e7ec; border-radius: 0.55rem; }
.table1-table { width: 100%; min-width: 44rem; border-collapse: collapse; font-size: 0.9rem; background: #ffffff; }
.table1-table th, .table1-table td { padding: 0.48rem 0.6rem; border-top: 1px solid #e4e7ec; text-align: left; vertical-align: top; }
.table1-table thead th { border-top: 0; background: #f8fbff; color: var(--muted); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.04em; }
.table1-table tbody tr:nth-child(even) { background: #fbfcff; }
.table1-table tbody th { width: 34%; font-weight: 650; color: var(--ink); }
.table1-table td { font-variant-numeric: tabular-nums; }
.table1-table tr.muted th, .table1-table tr.muted td { color: var(--muted); }
.table1-table td.empty { color: var(--muted); }
.table1-footnotes { margin: 0.9rem 0 0; padding-left: 1.2rem; color: var(--muted); font-size: 0.88rem; }
.nested-table { width: auto; min-width: 16rem; border-collapse: collapse; margin: 0.15rem 0; font-size: 0.9rem; }
.nested-table th, .nested-table td { padding: 0.35rem 0.55rem; border-top: 1px solid #e4e7ec; text-align: left; }
.nested-table thead th { border-top: 0; color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
.trait-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr)); gap: 1rem; }
.trait-card { padding: 0.75rem; border: 1px solid #e4e7ec; border-radius: 0.5rem; background: #ffffff; }
.trait-card h4 { margin: 0 0 0.45rem; font-size: 0.95rem; }
.metadata-note { margin-top: 1rem; padding: 0.8rem; border-left: 0.3rem solid var(--warn); border-radius: 0.4rem; background: var(--warn-bg); color: var(--warn); }
.figure-card img { border: 1px solid var(--line); }
.section-card { border-left: 0.45rem solid var(--accent); }
.section-card .subtitle { color: var(--muted); }
figcaption { margin-top: 0.9rem; color: var(--ink); }
figcaption strong { font-weight: 800; }
.missing-placeholder {
  margin-top: 0.85rem;
  padding: 2rem;
  border: 2px dashed #e2a03f;
  border-radius: 0.65rem;
  background: var(--warn-bg);
  color: var(--warn);
  text-align: center;
  font-weight: 700;
}
.note { background: #f8fbff; }
@media (max-width: 900px) {
  .layout { display: block; }
  .sidebar { position: static; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }
  .content { padding-top: 1.25rem; }
}
""".strip()
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_html}</title>
<style>
{css}
</style>
</head>
<body>
<div class="layout">
<aside class="sidebar" aria-label="Atlas navigation">
<h2>Atlas</h2>
<ul class="toc">
{nav_html}
</ul>
</aside>
<main class="content">
<header class="page-header">
<p class="eyebrow">simACE scenario atlas</p>
<h1>{title_html}</h1>
</header>
{body_html}
</main>
</div>
</body>
</html>
"""


def _namespace_svg_ids(svg: str, prefix: str) -> str:
    """Prefix every id and intra-document reference in one SVG fragment.

    matplotlib reuses ids (``figure_1``, glyph ids like ``DejaVuSans-30``,
    clip-path ids) across figures. Inlining several equation SVGs into one
    page would otherwise produce duplicate ids; prefixing every id and its
    ``#`` / ``url(#…)`` references with a per-block ``prefix`` keeps the
    single-page document valid and each block self-contained.
    """
    svg = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', svg)
    svg = re.sub(r'\bhref="#([^"]+)"', lambda m: f'href="#{prefix}{m.group(1)}"', svg)
    return re.sub(r"url\(#([^)]+)\)", lambda m: f"url(#{prefix}{m.group(1)})", svg)


def _equations_svg(equations: tuple[str, ...], gid_prefix: str) -> str:
    """Render model mathtext equations to inline ``<svg>`` markup.

    Returns dependency-free SVG (no on-disk asset) with the XML prolog and
    DOCTYPE stripped so it can be inlined directly into the atlas body, with
    all ids namespaced by ``gid_prefix`` so multiple inlined equation blocks
    cannot collide. Equations stay crisp at any zoom.
    """
    height = max(1.1, 0.55 * len(equations) + 0.25)
    fig = plt.figure(figsize=(10.5, height))
    step = min(0.28, 0.8 / max(len(equations), 1))
    y = 0.5 + step * (len(equations) - 1) / 2
    for equation in equations:
        fig.text(
            0.5,
            y,
            equation,
            fontsize=18,
            fontfamily="sans-serif",
            ha="center",
            va="center",
            transform=fig.transFigure,
        )
        y -= step
    buf = BytesIO()
    try:
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.2)
    finally:
        plt.close(fig)
    svg = buf.getvalue().decode("utf-8")
    # Strip the XML prolog + DOCTYPE so the <svg> can be inlined directly.
    start = svg.find("<svg")
    if start != -1:
        svg = svg[start:]
    return _namespace_svg_ids(svg, gid_prefix)


def _render_table1_row(section: Table1Section, row: Table1Row) -> str:
    """Render one native HTML Table 1 row."""
    row_class = ' class="muted"' if row.muted else ""
    value_cells: list[str] = []
    if len(row.values) == 1 and len(section.columns) > 1:
        value_cells.append(f'<td colspan="{len(section.columns)}">{escape(row.values[0])}</td>')
    else:
        for idx in range(len(section.columns)):
            value = row.values[idx] if idx < len(row.values) else ""
            cell_class = ' class="empty"' if not value else ""
            value_cells.append(f"<td{cell_class}>{escape(value)}</td>")
    return f"""
<tr{row_class}>
<th scope="row">{escape(row.label)}</th>
{chr(10).join(value_cells)}
</tr>
""".strip()


def _render_table1_section(section: Table1Section) -> str:
    """Render one Table 1 section as a native HTML table."""
    col_headers = ["Characteristic", *section.columns]
    header_cells = "\n".join(f'<th scope="col">{escape(col)}</th>' for col in col_headers)
    rows = "\n".join(_render_table1_row(section, row) for row in section.rows)
    return f"""
<div class="table1-section">
<h3>{escape(section.title)}</h3>
<div class="table1-scroll">
<table class="table1-table">
<thead><tr>{header_cells}</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>
</div>
""".strip()


def _render_table1_card(summary: Table1Summary) -> str:
    """Render Table 1 as native HTML instead of a PNG asset."""
    sections = "\n".join(_render_table1_section(section) for section in summary.sections)
    footnotes = ""
    if summary.footnotes:
        items = "\n".join(f"<li>{escape(note)}</li>" for note in summary.footnotes)
        footnotes = f'<ol class="table1-footnotes">\n{items}\n</ol>'
    return f"""
<section id="table1" class="card table1-card">
<h2>Table 1</h2>
<p>{escape(summary.title)}</p>
{sections}
{footnotes}
</section>
""".strip()


_MISSING = object()
_OVERVIEW_EXPECTED_KEYS = (
    "N",
    "G_sim",
    "G_ped",
    "G_pheno",
    "A1",
    "C1",
    "E1",
    "A2",
    "C2",
    "E2",
    "phenotype_model1",
    "phenotype_model2",
    "censor_age",
    "death_scale",
    "death_rho",
    "N_sample",
    "dropout_rate",
    "case_ascertainment_ratio",
)


def _render_metadata_missing_note() -> str:
    """Render the overview placeholder used when scenario metadata is absent."""
    return """
<section id="overview" class="card note">
<h2>Overview</h2>
<p>Scenario metadata was not provided, so the parameter overview could not be rendered.</p>
</section>
""".strip()


def _dict_sort_key(item: tuple[Any, Any]) -> tuple[int, int | str]:
    """Return a stable sort key for dictionaries with generation-like keys."""
    key = item[0]
    try:
        return (0, int(key))
    except (TypeError, ValueError):
        return (1, str(key))


def _format_value(value: Any) -> str:
    """Format a parameter value for compact display in the overview."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if value.is_integer() and abs(value) >= 10:
            return f"{int(value):,}"
        return f"{value:.4g}"
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        parts = [f"{k}: {_format_value(v)}" for k, v in sorted(value.items(), key=_dict_sort_key)]
        if len(parts) > 6:
            parts = [*parts[:6], "…"]
        return "{" + ", ".join(parts) + "}"
    return str(value)


def _param_value(params: dict, key: str) -> tuple[str, bool]:
    """Return (display value, missing flag) for one parameter key."""
    if key not in params:
        return ("not recorded", True)
    return (_format_value(params[key]), False)


def _metric(label: str, value: str, *, missing: bool = False) -> str:
    """Render a labelled scalar metric tile."""
    cls = "metric missing" if missing else "metric"
    return f"""
<div class="{cls}">
<span class="label">{escape(label)}</span>
<span class="value">{escape(value)}</span>
</div>
""".strip()


def _metric_from_key(params: dict, key: str, label: str) -> str:
    """Render a metric tile directly from a scenario parameter key."""
    value, missing = _param_value(params, key)
    return _metric(label, value, missing=missing)


def _metric_panel(title: str, metrics: list[str], *, wide: bool = False) -> str:
    """Render a panel containing scalar metric tiles."""
    cls = "overview-panel wide" if wide else "overview-panel"
    return f"""
<div class="{cls}">
<h3>{escape(title)}</h3>
<div class="metric-grid">
{chr(10).join(metrics)}
</div>
</div>
""".strip()


def _table_row(label: str, value: str, *, missing: bool = False) -> str:
    """Render one two-column parameter table row."""
    value_cls = "value missing" if missing else "value"
    return f"""
<tr>
<th scope="row">{escape(label)}</th>
<td class="{value_cls}">{escape(value)}</td>
</tr>
""".strip()


def _param_table(rows: list[str]) -> str:
    """Wrap pre-rendered rows in a parameter table."""
    return f"""
<table class="param-table">
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
""".strip()


def _param_table_row(params: dict, key: str, label: str) -> str:
    """Render a parameter table row directly from a scenario parameter key."""
    value, missing = _param_value(params, key)
    return _table_row(label, value, missing=missing)


def _a_plus_c(params: dict, trait_num: int) -> tuple[str, bool]:
    """Return the derived A+C variance share for one trait when available."""
    a_key = f"A{trait_num}"
    c_key = f"C{trait_num}"
    if a_key not in params or c_key not in params:
        return ("not recorded", True)
    a_val = params[a_key]
    c_val = params[c_key]
    if isinstance(a_val, dict) or isinstance(c_val, dict):
        return ("generation-specific", False)
    try:
        return (_format_value(float(a_val) + float(c_val)), False)
    except (TypeError, ValueError):
        return ("not available", True)


def _ace_cell(params: dict, key: str) -> str:
    """Render one ACE table value cell."""
    value, missing = _param_value(params, key)
    cls = "value missing" if missing else "value"
    return f'<td class="{cls}">{escape(value)}</td>'


def _render_ace_panel(params: dict) -> str:
    """Render the additive/common/unique variance component summary."""
    rows = []
    for trait_num in (1, 2):
        ac_value, ac_missing = _a_plus_c(params, trait_num)
        ac_cls = "value missing" if ac_missing else "value"
        rows.append(
            f"""
<tr>
<th scope="row">Trait {trait_num}</th>
{_ace_cell(params, f"A{trait_num}")}
{_ace_cell(params, f"C{trait_num}")}
{_ace_cell(params, f"E{trait_num}")}
<td class="{ac_cls}">{escape(ac_value)}</td>
</tr>
""".strip()
        )
    rows.append(
        """
<tr class="subheader-row"><th colspan="5">Cross-trait correlations</th></tr>
""".strip()
    )
    rows.append(
        f"""
<tr>
<th scope="row">Trait 1 vs Trait 2</th>
{_ace_cell(params, "rA")}
{_ace_cell(params, "rC")}
{_ace_cell(params, "rE")}
<td class="missing">not applicable</td>
</tr>
""".strip()
    )
    return f"""
<div class="overview-panel wide">
<h3>ACE variance components</h3>
<table class="param-table">
<thead><tr><th>Trait</th><th>A</th><th>C</th><th>E</th><th>A + C</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</div>
""".strip()


def _render_trait_phenotype_card(params: dict, trait_num: int) -> str:
    """Render the phenotype model card for one trait."""
    rows = [
        _param_table_row(params, f"phenotype_model{trait_num}", "model"),
        _param_table_row(params, f"beta{trait_num}", "Beta"),
        _param_table_row(params, f"beta_sex{trait_num}", "Beta_sex"),
    ]
    params_key = f"phenotype_params{trait_num}"
    if params_key not in params:
        rows.append(_table_row("params", "not recorded", missing=True))
    elif isinstance(params[params_key], dict):
        for key, value in sorted(params[params_key].items(), key=_dict_sort_key):
            rows.append(_table_row(str(key), _format_value(value)))
    else:
        rows.append(_table_row("params", _format_value(params[params_key])))

    return f"""
<div class="trait-card">
<h4>Trait {trait_num}</h4>
{_param_table(rows)}
</div>
""".strip()


def _render_phenotype_panel(params: dict) -> str:
    """Render phenotype model and hazard/frailty parameter cards."""
    return f"""
<div class="overview-panel wide">
<h3>Phenotype models</h3>
<div class="trait-grid">
{_render_trait_phenotype_card(params, 1)}
{_render_trait_phenotype_card(params, 2)}
</div>
</div>
""".strip()


def _render_generation_windows_table(value: Any) -> str:
    """Render generation censoring windows as a low/high nested table."""
    if not isinstance(value, dict):
        return escape(_format_value(value))

    rows = []
    for gen, window in sorted(value.items(), key=_dict_sort_key):
        if isinstance(window, list | tuple) and len(window) == 2:
            low = _format_value(window[0])
            high = _format_value(window[1])
        else:
            low = _format_value(window)
            high = "not recorded"
        rows.append(
            f"""
<tr>
<th scope="row">G{escape(str(gen))}</th>
<td>{escape(low)}</td>
<td>{escape(high)}</td>
</tr>
""".strip()
        )

    return f"""
<table class="nested-table generation-windows">
<thead><tr><th>Generation</th><th>Low</th><th>High</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
""".strip()


def _generation_windows_row(value: Any) -> str:
    """Render the generation-window row with nested low/high columns."""
    return f"""
<tr>
<th scope="row">generation windows</th>
<td class="value">{_render_generation_windows_table(value)}</td>
</tr>
""".strip()


def _render_censoring_panel(params: dict) -> str:
    """Render censoring and mortality parameters."""
    rows = [
        _param_table_row(params, "censor_age", "maximum follow-up age"),
        _param_table_row(params, "death_scale", "death scale"),
        _param_table_row(params, "death_rho", "death shape rho"),
    ]
    if "gen_censoring" in params:
        rows.append(_generation_windows_row(params["gen_censoring"]))
    else:
        rows.append(_table_row("generation windows", "not recorded", missing=True))
    return f"""
<div class="overview-panel wide">
<h3>Censoring and mortality</h3>
{_param_table(rows)}
</div>
""".strip()


def _render_metadata_note(params: dict) -> str:
    """Render a warning note when expected overview fields are absent."""
    missing = [key for key in _OVERVIEW_EXPECTED_KEYS if key not in params]
    if not missing:
        return ""
    shown = ", ".join(missing[:8])
    suffix = "…" if len(missing) > 8 else ""
    return f"""
<p class="metadata-note">
Some expected scenario fields were not available in the atlas metadata ({escape(shown + suffix)}).
If this came from an old per-replicate params.yaml, regenerate with the current workflow or use the CLI fallback from the repo root so config defaults can be merged.
</p>
""".strip()


def _render_overview_card(params: dict | None) -> str:
    """Render a native HTML overview of scenario parameters."""
    if not params:
        return _render_metadata_missing_note()

    scenario = _format_value(params.get("scenario", "unknown"))
    panels = [
        _metric_panel(
            "Scenario and population",
            [
                _metric("Scenario", scenario),
                _metric_from_key(params, "seed", "Seed"),
                _metric_from_key(params, "replicates", "Replicates"),
                _metric_from_key(params, "N", "Population size per generation"),
                _metric_from_key(params, "G_sim", "Simulated generations"),
                _metric_from_key(params, "G_ped", "Recorded pedigree generations"),
                _metric_from_key(params, "G_pheno", "Phenotyped generations"),
                _metric_from_key(params, "standardize", "Liability standardization"),
            ],
            wide=True,
        ),
        _metric_panel(
            "Pedigree and mating",
            [
                _metric_from_key(params, "mating_model", "Mating model"),
                _metric_from_key(params, "mating_lambda", "Mating λ"),
                _metric_from_key(params, "p_mztwin", "MZ twin probability"),
                _metric_from_key(params, "assort1", "Assortment trait 1"),
                _metric_from_key(params, "assort2", "Assortment trait 2"),
            ],
        ),
        _render_ace_panel(params),
        _render_phenotype_panel(params),
        _render_censoring_panel(params),
        _metric_panel(
            "Ascertainment and analysis",
            [
                _metric_from_key(params, "N_sample", "Sample size (N_sample)"),
                _metric_from_key(params, "case_ascertainment_ratio", "Case weighting"),
                _metric_from_key(params, "dropout_rate", "Dropout rate"),
                _metric_from_key(params, "max_degree", "Relationship max degree"),
            ],
            wide=True,
        ),
    ]

    return f"""
<section id="overview" class="card overview-card">
<h2>Overview</h2>
<p>Parameters used to simulate, phenotype, censor, ascertain, and analyse the data shown in this atlas.</p>
<div class="overview-grid">
{chr(10).join(panels)}
</div>
{_render_metadata_note(params)}
</section>
""".strip()


def _render_section_break(anchor_id: str, item: SectionBreak, equation_svg: str | None = None) -> str:
    """Render a manifest section break as a semantic HTML section card.

    ``equation_svg`` is inline ``<svg>`` markup (not a link); it is embedded
    verbatim so the atlas stays a single self-contained file.
    """
    subtitle = f'<p class="subtitle">{escape(item.subtitle)}</p>' if item.subtitle else ""
    equation_block = ""
    if equation_svg:
        equation_block = f'<div class="equation-svg" role="img" aria-label="Model equations">{equation_svg}</div>'
    return f"""
<section id="{escape(anchor_id, quote=True)}" class="card section-card">
<h2>{escape(item.title)}</h2>
{subtitle}
{equation_block}
</section>
""".strip()


def _render_image_card(item: PlotEntry, plot_idx: int, src: str) -> str:
    """Render a normal figure card with an embedded plot image and caption."""
    return f"""
<figure id="figure-{plot_idx}" class="card figure-card">
<h2>Figure {plot_idx}: {escape(item.title)}</h2>
<img src="{escape(src, quote=True)}" alt="{escape(item.title, quote=True)}">
<figcaption><strong>Figure {plot_idx}: {escape(item.title)}</strong> {escape(item.body)}</figcaption>
</figure>
""".strip()


def _render_missing_plot_card(item: PlotEntry, plot_idx: int, filename: str) -> str:
    """Render a visible placeholder for a missing plot while preserving numbering."""
    return f"""
<figure id="figure-{plot_idx}" class="card figure-card missing-card">
<h2>Figure {plot_idx}: {escape(item.title)}</h2>
<div class="missing-placeholder">Missing plot image: {escape(filename)}</div>
<figcaption><strong>Figure {plot_idx}: {escape(item.title)}</strong> {escape(item.body)}</figcaption>
</figure>
""".strip()


def assemble_html_atlas(
    items: list[AtlasItem],
    plot_dir: Path,
    output_path: Path,
    *,
    plot_ext: str = "png",
    scenario_params: dict | None = None,
    stats_data: list[dict] | None = None,
) -> None:
    """Combine atlas manifest entries into a single-page HTML atlas.

    Args:
        items: Ordered atlas manifest, mixing :class:`PlotEntry` and
            :class:`SectionBreak` items.
        plot_dir: Directory containing plot image files.
        output_path: Destination path for the generated HTML file.
        plot_ext: Browser-displayable plot image extension.
        scenario_params: Optional scenario metadata used to render the
            pipeline/title companion card.
        stats_data: Optional per-replicate stats report views used to render
            the Table 1 companion card.
    """
    normalised_ext = _validate_plot_ext(plot_ext)
    plot_dir = Path(plot_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scenario_name = "unknown"
    if scenario_params:
        scenario_name = str(scenario_params.get("scenario", "unknown"))
    page_title = f"{scenario_name} atlas" if scenario_name != "unknown" else "Scenario atlas"

    nav: list[str] = [_toc_item("#overview", "Overview")]
    body: list[str] = []

    body.append(_render_overview_card(scenario_params))

    if stats_data and scenario_params:
        table1_summary = build_table1_summary(
            stats_data,
            scenario_params,
            scenario=str(scenario_params.get("scenario", "")),
        )
        nav.append(_toc_item("#table1", "Table 1"))
        body.append(_render_table1_card(table1_summary))

    plot_idx = 0
    section_idx = 0
    for item in items:
        if isinstance(item, SectionBreak):
            section_idx += 1
            anchor_id = f"section-{section_idx}-{_slug(item.title, 'section')}"
            equation_svg = _equations_svg(item.equations, f"eq{section_idx}-") if item.equations else None
            nav.append(_toc_item(f"#{anchor_id}", item.title, item_class="section"))
            body.append(_render_section_break(anchor_id, item, equation_svg))
            continue

        plot_idx += 1
        plot_path = plot_dir / f"{item.basename}.{normalised_ext}"
        nav_class = ""
        if plot_path.exists():
            card = _render_image_card(item, plot_idx, _data_uri(plot_path, normalised_ext))
        else:
            logger.warning("HTML atlas: missing plot %s", plot_path)
            nav_class = "missing"
            card = _render_missing_plot_card(item, plot_idx, plot_path.name)
        nav.append(_toc_item(f"#figure-{plot_idx}", f"Figure {plot_idx}: {item.title}", item_class=nav_class))
        body.append(card)

    output_path.write_text(_html_page(page_title, nav, body), encoding="utf-8")
    logger.info("HTML atlas saved to %s (%d plots)", output_path, plot_idx)
