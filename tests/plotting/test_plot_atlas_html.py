"""Unit tests for the HTML atlas prototype."""

import logging

import matplotlib
import matplotlib.pyplot as plt
import pytest

from simace.plotting.atlas_manifest import PlotEntry, SectionBreak
from simace.plotting.plot_atlas_html import assemble_html_atlas

matplotlib.use("Agg")


def _touch_plot(plot_dir, basename: str, ext: str = "png"):
    path = plot_dir / f"{basename}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder image bytes")
    return path


def _dummy_figure(*_args, **_kwargs):
    fig = plt.figure(figsize=(1, 1))
    fig.text(0.5, 0.5, "dummy", ha="center", va="center")
    return fig


def test_creates_html_and_links_existing_image(tmp_path):
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
    assert 'src="example.png"' in html
    assert "<h2>Figure 1: Example title</h2>" in html
    assert "Figure 1: Example title" in html
    assert "Caption body." in html


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


def test_section_equations_render_to_asset_png(tmp_path):
    plot_dir = tmp_path / "plots"
    output = plot_dir / "atlas.html"

    assemble_html_atlas(
        [SectionBreak(title="Model Section", subtitle="Resolved model", equations=(r"$h_0(t) = \lambda$",))],
        plot_dir,
        output,
    )

    assert (plot_dir / "atlas_assets" / "model_equations.png").exists()
    html = output.read_text(encoding="utf-8")
    assert 'src="atlas_assets/model_equations.png"' in html
    assert r"$h_0(t) = \lambda$" not in html


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


def test_companion_assets_written_when_inputs_exist(tmp_path, monkeypatch):
    import simace.plotting.plot_atlas_html as plot_atlas_html

    monkeypatch.setattr(plot_atlas_html, "render_table1_figure", _dummy_figure)

    plot_dir = tmp_path / "plots"
    output = plot_dir / "atlas.html"
    assemble_html_atlas(
        [],
        plot_dir,
        output,
        scenario_params={"scenario": "small_test"},
        stats_data=[{"n_individuals": 1}],
    )

    assert (plot_dir / "atlas_assets" / "table1.png").exists()
    html = output.read_text(encoding="utf-8")
    assert 'src="atlas_assets/table1.png"' in html


def test_no_stats_omits_table1_asset_but_renders(tmp_path, monkeypatch):
    import simace.plotting.plot_atlas_html as plot_atlas_html

    monkeypatch.setattr(
        plot_atlas_html,
        "render_table1_figure",
        lambda *_args, **_kwargs: pytest.fail("Table 1 should not render without stats"),
    )

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
