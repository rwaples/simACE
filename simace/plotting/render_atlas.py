"""Format-dispatching seam for atlas rendering.

A single entry point that routes an atlas manifest to the right backend based
on the requested output extension: ``.html`` renders the primary, self-contained
HTML atlas; ``.pdf`` renders the on-demand multi-page PDF atlas (ADR 0010).

Both backends share an identical signature, so every consumer can stay
format-agnostic — flipping an atlas to HTML-primary is a change of output
extension, nothing more.

Kept in its own module to avoid a circular import between :mod:`plot_atlas`
(PDF backend) and :mod:`plot_atlas_html` (HTML backend).
"""

__all__ = ["render_atlas"]

from pathlib import Path

from simace.plotting.atlas_manifest import AtlasItem
from simace.plotting.plot_atlas import assemble_atlas
from simace.plotting.plot_atlas_html import assemble_html_atlas

_RENDERERS = {
    ".html": assemble_html_atlas,
    ".pdf": assemble_atlas,
}


def render_atlas(
    items: list[AtlasItem],
    plot_dir: Path,
    output_path: Path,
    *,
    plot_ext: str = "png",
    scenario_params: dict | None = None,
    stats_data: list[dict] | None = None,
) -> None:
    """Render an atlas manifest, dispatching on the ``output_path`` extension.

    ``.html`` → :func:`~simace.plotting.plot_atlas_html.assemble_html_atlas`
    (the primary, default-built artifact); ``.pdf`` →
    :func:`~simace.plotting.plot_atlas.assemble_atlas` (on-demand export).
    Any other suffix raises :class:`ValueError`.

    Args:
        items: Ordered atlas manifest, mixing
            :class:`~simace.plotting.atlas_manifest.PlotEntry` and
            :class:`~simace.plotting.atlas_manifest.SectionBreak`.
        plot_dir: Directory containing the plot image files.
        output_path: Destination path; its suffix selects the renderer.
        plot_ext: Image extension passed through to the backend (default ``"png"``).
        scenario_params: Optional scenario metadata (title/overview, Table 1).
        stats_data: Optional per-replicate stats report views (Table 1).
    """
    suffix = Path(output_path).suffix.lower()
    renderer = _RENDERERS.get(suffix)
    if renderer is None:
        supported = ", ".join(sorted(_RENDERERS))
        raise ValueError(
            f"render_atlas cannot render {output_path!r}: unsupported atlas "
            f"extension {suffix!r}. Supported extensions: {supported}."
        )
    renderer(
        items,
        plot_dir,
        output_path,
        plot_ext=plot_ext,
        scenario_params=scenario_params,
        stats_data=stats_data,
    )
