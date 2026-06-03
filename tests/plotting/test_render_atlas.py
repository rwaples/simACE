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

    monkeypatch.setitem(render_atlas_mod._RENDERERS, ".html", _spy)

    render_atlas(_ITEMS, tmp_path, tmp_path / "atlas.html")

    assert calls["backend"] == "html"
    assert calls["args"][2] == tmp_path / "atlas.html"


def test_pdf_suffix_routes_to_pdf_backend(tmp_path, monkeypatch):
    calls = {}

    def _spy(items, plot_dir, output_path, **kwargs):
        calls["backend"] = "pdf"

    monkeypatch.setitem(render_atlas_mod._RENDERERS, ".pdf", _spy)

    render_atlas(_ITEMS, tmp_path, tmp_path / "atlas.pdf")

    assert calls["backend"] == "pdf"


def test_kwargs_forwarded_to_backend(tmp_path, monkeypatch):
    seen = {}

    def _spy(items, plot_dir, output_path, **kwargs):
        seen.update(kwargs)

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
    monkeypatch.setitem(render_atlas_mod._RENDERERS, ".html", lambda *a, **k: calls.setdefault("hit", True))

    render_atlas(_ITEMS, tmp_path, tmp_path / "atlas.HTML")

    assert calls.get("hit") is True
