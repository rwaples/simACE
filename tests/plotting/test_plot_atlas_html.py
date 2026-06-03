"""Unit tests for the self-contained HTML atlas."""

import logging
import re

import pytest

from simace.plotting.atlas_manifest import PlotEntry, SectionBreak
from simace.plotting.plot_atlas_html import assemble_html_atlas


def _touch_plot(plot_dir, basename: str, ext: str = "png"):
    path = plot_dir / f"{basename}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder image bytes")
    return path


def test_creates_html_and_embeds_existing_image(tmp_path):
    plot_dir = tmp_path / "plots"
    _touch_plot(plot_dir, "example")
    output = plot_dir / "atlas.html"

    assemble_html_atlas(
        [PlotEntry(basename="example", title="Example title", body="Caption body.")],
        plot_dir,
        output,
    )

    html = output.read_text(encoding="utf-8")
    assert output.exists()
    # Self-contained: the plot is base64-embedded, not linked, and no sibling
    # asset directory is written.
    assert 'src="data:image/png;base64,' in html
    assert 'src="example.png"' not in html
    assert not (plot_dir / "atlas_assets").exists()
    assert "<h2>Figure 1: Example title</h2>" in html
    assert "Figure 1: Example title" in html
    assert "Caption body." in html


@pytest.mark.parametrize(
    ("ext", "mime"),
    [
        ("png", "image/png"),
        ("jpg", "image/jpeg"),
        ("jpeg", "image/jpeg"),
        ("gif", "image/gif"),
        ("webp", "image/webp"),
        ("svg", "image/svg+xml"),
    ],
)
def test_embeds_correct_mime_per_plot_ext(tmp_path, ext, mime):
    plot_dir = tmp_path / "plots"
    _touch_plot(plot_dir, "example", ext=ext)
    output = plot_dir / "atlas.html"

    assemble_html_atlas(
        [PlotEntry(basename="example", title="T", body="B")],
        plot_dir,
        output,
        plot_ext=ext,
    )

    html = output.read_text(encoding="utf-8")
    assert f'src="data:{mime};base64,' in html


def test_missing_plot_placeholder_preserves_caption_and_warns(tmp_path, caplog):
    plot_dir = tmp_path / "plots"
    output = plot_dir / "atlas.html"

    with caplog.at_level(logging.WARNING):
        assemble_html_atlas(
            [PlotEntry(basename="missing", title="Missing title", body="Missing caption.")],
            plot_dir,
            output,
        )

    html = output.read_text(encoding="utf-8")
    assert "Missing plot image: missing.png" in html
    assert "Figure 1: Missing title" in html
    assert "Missing caption." in html
    assert "missing plot" in caplog.text.lower()


def test_html_escapes_caption_text(tmp_path):
    plot_dir = tmp_path / "plots"
    _touch_plot(plot_dir, "escape")
    output = plot_dir / "atlas.html"

    assemble_html_atlas(
        [
            PlotEntry(
                basename="escape",
                title="<script>alert(1)</script> & title",
                body="Body & <bad>",
            )
        ],
        plot_dir,
        output,
    )

    html = output.read_text(encoding="utf-8")
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; title" in html
    assert "Body &amp; &lt;bad&gt;" in html
    assert "<script>" not in html
    assert "Body & <bad>" not in html


def test_rejects_pdf_plot_extension(tmp_path):
    with pytest.raises(ValueError, match="browser-displayable"):
        assemble_html_atlas([], tmp_path, tmp_path / "atlas.html", plot_ext="pdf")


def test_section_equations_render_inline_svg(tmp_path):
    plot_dir = tmp_path / "plots"
    output = plot_dir / "atlas.html"

    assemble_html_atlas(
        [SectionBreak(title="Model Section", subtitle="Resolved model", equations=(r"$h_0(t) = \lambda$",))],
        plot_dir,
        output,
    )

    html = output.read_text(encoding="utf-8")
    # Equations are inlined as dependency-free SVG, not a companion PNG asset.
    assert 'class="equation-svg"' in html
    assert "<svg" in html
    assert "atlas_assets" not in html
    assert not (plot_dir / "atlas_assets").exists()
    # The mathtext is rendered into vector glyph paths.
    assert "<path" in html


def test_multiple_equation_sections_have_unique_svg_ids(tmp_path):
    plot_dir = tmp_path / "plots"
    output = plot_dir / "atlas.html"

    assemble_html_atlas(
        [
            SectionBreak(title="Model A", subtitle="a", equations=(r"$h_0(t) = \lambda$",)),
            SectionBreak(title="Model B", subtitle="b", equations=(r"$\theta > 0$",)),
        ],
        plot_dir,
        output,
    )

    html = output.read_text(encoding="utf-8")
    assert html.count("<svg") == 2
    # Per-block id namespacing keeps the single-page document free of the
    # duplicate ids matplotlib would otherwise emit across inlined SVGs.
    ids = re.findall(r'\bid="([^"]+)"', html)
    assert len(ids) == len(set(ids))
    assert any(i.startswith("eq1-") for i in ids)
    assert any(i.startswith("eq2-") for i in ids)


def test_overview_renders_native_parameter_dashboard(tmp_path):
    plot_dir = tmp_path / "plots"
    output = plot_dir / "atlas.html"

    assemble_html_atlas(
        [],
        plot_dir,
        output,
        scenario_params={
            "scenario": "small_test",
            "N": 10000,
            "G_sim": 8,
            "G_ped": 6,
            "G_pheno": 3,
            "A1": 0.5,
            "C1": 0.0,
            "E1": 0.5,
            "phenotype_model1": "frailty",
            "phenotype_params1": {"distribution": "weibull", "scale": 2160, "rho": 0.8},
            "gen_censoring": {0: [80, 80], 1: [40, 80], 2: [0, 45]},
        },
    )

    html = output.read_text(encoding="utf-8")
    assert "Parameters used to simulate, phenotype, censor, ascertain, and analyse" in html
    assert "Scenario and population" in html
    assert "Population size per generation" in html
    assert "ACE variance components" in html
    assert "Cross-trait correlations" in html
    assert "Phenotype models" in html
    assert "Beta" in html
    assert "Beta_sex" in html
    assert "Generation" in html
    assert "Low" in html
    assert "High" in html
    assert "G2" in html
    assert "45" in html
    assert "Sample size (N_sample)" in html
    assert "10,000" in html
    assert 'src="atlas_assets/pipeline.png"' not in html
    assert not (plot_dir / "atlas_assets" / "pipeline.png").exists()


def test_native_table1_renders_when_inputs_exist(tmp_path):
    plot_dir = tmp_path / "plots"
    output = plot_dir / "atlas.html"
    assemble_html_atlas(
        [],
        plot_dir,
        output,
        scenario_params={"scenario": "small_test", "censor_age": 80},
        stats_data=[{"n_individuals": 1, "n_generations": 1}],
    )

    html = output.read_text(encoding="utf-8")
    assert '<section id="table1" class="card table1-card">' in html
    assert '<table class="table1-table">' in html
    assert "A. Population" in html
    assert "Total phenotyped individuals, n" in html
    # Table 1 is native HTML; no companion asset directory is written.
    assert not (plot_dir / "atlas_assets").exists()


def test_no_stats_omits_table1_but_renders(tmp_path):
    plot_dir = tmp_path / "plots"
    output = plot_dir / "atlas.html"
    assemble_html_atlas([], plot_dir, output, scenario_params={"scenario": "small_test"}, stats_data=[])

    assert output.exists()
    assert not (plot_dir / "atlas_assets" / "table1.png").exists()
    assert "Table 1" not in output.read_text(encoding="utf-8")


def test_numbering_parity_missing_plot_consumes_figure_number(tmp_path):
    plot_dir = tmp_path / "plots"
    _touch_plot(plot_dir, "first")
    _touch_plot(plot_dir, "third")
    output = plot_dir / "atlas.html"
    items = [
        PlotEntry(basename="first", title="First", body="First caption."),
        PlotEntry(basename="missing", title="Missing", body="Missing caption."),
        PlotEntry(basename="third", title="Third", body="Third caption."),
    ]

    assemble_html_atlas(items, plot_dir, output)

    html = output.read_text(encoding="utf-8")
    assert "Figure 1: First" in html
    assert "Figure 2: Missing" in html
    assert "Figure 3: Third" in html
    assert 'id="figure-3"' in html
