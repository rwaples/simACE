"""Unit tests for the render_atlas format-dispatch seam."""

import pytest

from simace.plotting import render_atlas as render_atlas_mod
from simace.plotting.atlas_manifest import PlotEntry
from simace.plotting.render_atlas import render_atlas

_ITEMS = [PlotEntry(basename="example", title="Example", body="Caption.")]


def test_html_suffix_routes_to_html_backend(tmp_path, monkeypatch):
    calls = {}

    def _spy(items, plot_dir, output_path, **kwargs):
        calls["backend"] = "html"
        calls["args"] = (items, plot_dir, output_path, kwargs)
        output_path.write_text("html")

    monkeypatch.setitem(render_atlas_mod._RENDERERS, ".html", _spy)

    render_atlas(_ITEMS, tmp_path, tmp_path / "atlas.html")

    assert calls["backend"] == "html"
    assert calls["args"][2].parent == tmp_path
    assert (tmp_path / "atlas.html").read_text() == "html"


def test_pdf_suffix_routes_to_pdf_backend(tmp_path, monkeypatch):
    calls = {}

    def _spy(items, plot_dir, output_path, **kwargs):
        calls["backend"] = "pdf"
        output_path.write_text("pdf")

    monkeypatch.setitem(render_atlas_mod._RENDERERS, ".pdf", _spy)

    render_atlas(_ITEMS, tmp_path, tmp_path / "atlas.pdf")

    assert calls["backend"] == "pdf"


def test_kwargs_forwarded_to_backend(tmp_path, monkeypatch):
    seen = {}

    def _spy(items, plot_dir, output_path, **kwargs):
        seen.update(kwargs)
        output_path.write_text("html")

    monkeypatch.setitem(render_atlas_mod._RENDERERS, ".html", _spy)

    render_atlas(
        _ITEMS,
        tmp_path,
        tmp_path / "atlas.html",
        plot_ext="svg",
        scenario_params={"scenario": "demo"},
        stats_data=[{"rep": 0}],
    )

    assert seen["plot_ext"] == "svg"
    assert seen["scenario_params"] == {"scenario": "demo"}
    assert seen["stats_data"] == [{"rep": 0}]


@pytest.mark.parametrize("bad", ["atlas.txt", "atlas.svg", "atlas", "atlas.PDFX"])
def test_unknown_suffix_raises(tmp_path, bad):
    with pytest.raises(ValueError, match="unsupported atlas extension"):
        render_atlas(_ITEMS, tmp_path, tmp_path / bad)


def test_suffix_dispatch_is_case_insensitive(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setitem(
        render_atlas_mod._RENDERERS,
        ".html",
        lambda _i, _d, out, **_k: (calls.setdefault("hit", True), out.write_text("x")),
    )

    render_atlas(_ITEMS, tmp_path, tmp_path / "atlas.HTML")

    assert calls.get("hit") is True


def test_failed_render_keeps_previous_atlas(tmp_path, monkeypatch):
    final = tmp_path / "atlas.html"
    final.write_text("previous")

    def _crash(items, plot_dir, output_path, **kwargs):
        output_path.write_text("half")
        raise RuntimeError("renderer died")

    monkeypatch.setitem(render_atlas_mod._RENDERERS, ".html", _crash)

    with pytest.raises(RuntimeError, match="renderer died"):
        render_atlas(_ITEMS, tmp_path, final)

    assert final.read_text() == "previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["atlas.html"]


def test_render_creates_missing_parent(tmp_path, monkeypatch):
    monkeypatch.setitem(render_atlas_mod._RENDERERS, ".html", lambda _i, _d, out, **_k: out.write_text("x"))

    render_atlas(_ITEMS, tmp_path, tmp_path / "new" / "atlas.html")

    assert (tmp_path / "new" / "atlas.html").read_text() == "x"


def test_empty_pdf_atlas_raises_and_keeps_previous(tmp_path):
    final = tmp_path / "atlas.pdf"
    final.write_text("previous")

    with pytest.raises(ValueError, match="PDF atlas has no pages: all 1 plots are missing"):
        render_atlas(_ITEMS, tmp_path, final)

    assert final.read_text() == "previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["atlas.pdf"]


def test_log_names_final_path(tmp_path, monkeypatch, caplog):
    monkeypatch.setitem(render_atlas_mod._RENDERERS, ".html", lambda _i, _d, out, **_k: out.write_text("x"))

    with caplog.at_level("INFO", logger="simace.plotting.render_atlas"):
        render_atlas(_ITEMS, tmp_path, tmp_path / "atlas.html")

    assert f"Atlas saved to {tmp_path / 'atlas.html'} (1 plots)" in caplog.text
