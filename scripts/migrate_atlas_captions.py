r"""One-shot migration: emit ``simace/plotting/atlas_manifest.py``.

Reads:
  * ``simace.plotting.plot_atlas.PHENOTYPE_CAPTIONS``
  * ``simace.plotting.plot_atlas.VALIDATION_CAPTIONS``
  * ``workflow.common._PHENOTYPE_BASENAMES``
  * ``workflow.common._VALIDATION_BASENAMES``
  * Section-break definitions from
    ``workflow/scripts/simace/assemble_atlas.py`` (indices 10 / 14 / 22 / 25
    in the phenotype atlas; the 10-th break is the dynamic model section
    and is emitted as ``MODEL_SECTION`` sentinel).

Strips ``r"^Figure \d+:\s*"`` from each caption title; the assembler will
prepend ``f"Figure {N}: "`` at render time. Asserts every basename has a
caption and every caption is consumed.

Output: ``simace/plotting/atlas_manifest.py``. Reviewable as code.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = REPO_ROOT / "simace" / "plotting" / "atlas_manifest.py"

# --- Phenotype section breaks (preserved from
#     workflow/scripts/simace/assemble_atlas.py:66 ff). The dict key is the
#     index in _PHENOTYPE_BASENAMES; the break is inserted *before* that
#     plot. ``None`` for the break body emits the MODEL_SECTION sentinel.
_PHENOTYPE_BREAKS: dict[int, dict | None] = {
    10: None,  # MODEL_SECTION sentinel — resolved at render against scenario_params.
    14: {
        "title": "Age of Onset & Censoring",
        "subtitle": "Age-at-onset, mortality, cumulative incidence, and censoring analysis",
        "equations": (),
    },
    22: {
        "title": "Within-Trait Correlations",
        "subtitle": "Familial tetrachoric correlations",
        "equations": (),
    },
    25: {
        "title": "Cross-Trait Correlations",
        "subtitle": "Cross-trait correlation by generation and relationship type",
        "equations": (),
    },
}


_FIGURE_PREFIX = re.compile(r"^Figure\s+\d+:\s*")


def _split_caption(caption: str) -> tuple[str, str]:
    """Split a caption into (title, body); strip the leading 'Figure N:'."""
    head, _, body = caption.partition("\n")
    title = _FIGURE_PREFIX.sub("", head.strip())
    body = body.lstrip("\n").rstrip()
    return title, body


def _format_string_literal(s: str, indent: int) -> str:
    r"""Format a string as one or more Python implicit-concat lines.

    Single-line if short; otherwise split into ``\n``-separated logical
    lines and wrap each long logical line into 80-char chunks via
    ``textwrap.wrap``. Output is wrapped in parentheses for adjacent-string
    concatenation.
    """
    pad = " " * indent
    if "\n" not in s and len(repr(s)) + indent <= 100:
        return repr(s)

    pieces: list[str] = []
    logical_lines = s.split("\n")
    for li, line in enumerate(logical_lines):
        is_last_logical = li == len(logical_lines) - 1
        suffix = "" if is_last_logical else "\\n"
        if not line:
            pieces.append(repr(suffix))
            continue
        chunks = textwrap.wrap(line, width=80, drop_whitespace=False, break_long_words=False)
        if not chunks:
            chunks = [line]
        for ci, chunk in enumerate(chunks):
            is_last_chunk = ci == len(chunks) - 1
            if is_last_chunk:
                pieces.append(repr(chunk + suffix))
            else:
                pieces.append(repr(chunk))
    if len(pieces) == 1:
        return pieces[0]
    inner = ("\n" + pad).join(pieces)
    return f"(\n{pad}{inner}\n{' ' * (indent - 4)})"


def _emit_plot_entry(basename: str, title: str, body: str) -> str:
    body_lit = _format_string_literal(body, indent=12)
    return (
        f"    PlotEntry(\n        basename={basename!r},\n        title={title!r},\n        body={body_lit},\n    ),\n"
    )


def _emit_section_break(brk: dict) -> str:
    eqs = brk["equations"]
    if eqs:
        eq_str = "(" + ", ".join(repr(e) for e in eqs) + (",)" if len(eqs) == 1 else ")")
        return (
            "    SectionBreak(\n"
            f"        title={brk['title']!r},\n"
            f"        subtitle={brk['subtitle']!r},\n"
            f"        equations={eq_str},\n"
            "    ),\n"
        )
    return f"    SectionBreak(\n        title={brk['title']!r},\n        subtitle={brk['subtitle']!r},\n    ),\n"


def _emit_model_section() -> str:
    return "    MODEL_SECTION,\n"


_MODULE_HEADER = '''"""Atlas manifest: ordered registry of plots and section breaks.

Each entry is either a :class:`PlotEntry` (a figure to include in the
atlas) or a :class:`SectionBreak` (a divider page). The assembler walks
the manifest linearly and derives ``Figure N`` numbers from the running
:class:`PlotEntry` index — inserting or reordering plots is a single
edit here, with no figure-number cascade.

To add a phenotype plot: write the rendering function in the appropriate
``simace/plotting/plot_*.py`` module so it produces ``<basename>.png``,
then add a :class:`PlotEntry` to :data:`PHENOTYPE_ATLAS` at the
right position. The Snakemake workflow imports
:func:`phenotype_basenames` to declare expected output paths, so adding
the entry automatically wires the new plot into the build.

The :data:`MODEL_SECTION` sentinel is resolved at render time by
:func:`build_phenotype_atlas`: it is replaced with a model-aware
:class:`SectionBreak` carrying equations from
``simace.plotting.plot_atlas.get_model_family`` /
``get_model_equation`` against the scenario params.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlotEntry:
    """One plot in an atlas: filename basename and caption text.

    Attributes:
        basename: Plot filename without extension. Must match what the
            corresponding ``simace.plotting.plot_*`` rendering function writes.
        title: Short caption title (no ``"Figure N:"`` prefix; the
            assembler prepends one at render time).
        body: Multi-line caption body.
    """

    basename: str
    title: str
    body: str


@dataclass(frozen=True)
class SectionBreak:
    """A section divider page between plots.

    Attributes:
        title: Bold heading on the divider page.
        subtitle: Sub-heading below the title.
        equations: Optional LaTeX strings rendered below the subtitle.
    """

    title: str
    subtitle: str
    equations: tuple[str, ...] = ()


AtlasItem = PlotEntry | SectionBreak

# Sentinel for the model-aware phenotype section break. Resolved at
# render time by ``build_phenotype_atlas`` against scenario params; the
# placeholder values below are never displayed.
MODEL_SECTION = SectionBreak(title="<MODEL>", subtitle="<MODEL>")


# ---------------------------------------------------------------------------
# Phenotype atlas
# ---------------------------------------------------------------------------

PHENOTYPE_ATLAS: tuple[AtlasItem, ...] = (
'''


_VALIDATION_HEADER = """
# ---------------------------------------------------------------------------
# Validation atlas
# ---------------------------------------------------------------------------

VALIDATION_ATLAS: tuple[AtlasItem, ...] = (
"""


_FOOTER = '''

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def phenotype_basenames() -> list[str]:
    """Ordered phenotype plot basenames (excluding section breaks).

    Used by ``workflow.common`` to declare the Snakemake outputs.
    """
    return [e.basename for e in PHENOTYPE_ATLAS if isinstance(e, PlotEntry)]


def validation_basenames() -> list[str]:
    """Ordered validation plot basenames (excluding section breaks)."""
    return [e.basename for e in VALIDATION_ATLAS if isinstance(e, PlotEntry)]


def build_phenotype_atlas(params: dict[str, Any] | None) -> list[AtlasItem]:
    """Return ``PHENOTYPE_ATLAS`` with ``MODEL_SECTION`` resolved.

    When ``params`` is ``None`` the model section is omitted (no scenario
    context to derive the title from). Otherwise the sentinel is replaced
    with a :class:`SectionBreak` whose title / subtitle / equations come
    from ``get_model_family`` and ``get_model_equation`` in
    ``simace.plotting.plot_atlas``.
    """
    from simace.plotting.plot_atlas import get_model_equation, get_model_family

    out: list[AtlasItem] = []
    for item in PHENOTYPE_ATLAS:
        if item is MODEL_SECTION:
            if params is None:
                continue
            name, desc = get_model_family(params)
            equations = tuple(get_model_equation(params))
            out.append(SectionBreak(title=name, subtitle=desc, equations=equations))
        else:
            out.append(item)
    return out
'''


def _build_phenotype_block() -> str:
    sys.path.insert(0, str(REPO_ROOT))
    from simace.plotting.plot_atlas import PHENOTYPE_CAPTIONS
    from workflow.common import _PHENOTYPE_BASENAMES

    consumed: set[str] = set()
    out = []
    for idx, basename in enumerate(_PHENOTYPE_BASENAMES):
        if idx in _PHENOTYPE_BREAKS:
            brk = _PHENOTYPE_BREAKS[idx]
            out.append(_emit_model_section() if brk is None else _emit_section_break(brk))
        if basename not in PHENOTYPE_CAPTIONS:
            raise SystemExit(f"phenotype basename {basename!r} has no caption")
        title, body = _split_caption(PHENOTYPE_CAPTIONS[basename])
        out.append(_emit_plot_entry(basename, title, body))
        consumed.add(basename)
    leftover = set(PHENOTYPE_CAPTIONS) - consumed
    if leftover:
        raise SystemExit(f"phenotype captions never consumed: {sorted(leftover)}")
    return "".join(out)


def _build_validation_block() -> str:
    sys.path.insert(0, str(REPO_ROOT))
    from simace.plotting.plot_atlas import VALIDATION_CAPTIONS

    # Authoritative validation order is the local list in
    # ``simace/plotting/plot_validation.py`` (lines 543–556), not the stale
    # 12-entry copy in ``workflow/common.py`` — that copy is missing
    # ``consanguineous_matings``, an actually-rendered plot. The manifest
    # adopts the authoritative 13-entry order; the workflow shim
    # picks it up automatically.
    validation_basenames = [
        "family_size",
        "twin_rate",
        "half_sib_proportions",
        "consanguineous_matings",
        "variance_components",
        "correlations_A",
        "correlations_phenotype",
        "heritability_estimates",
        "cross_trait_correlations",
        "summary_bias",
        "runtime",
        "memory",
    ]
    consumed: set[str] = set()
    out = []
    for basename in validation_basenames:
        if basename not in VALIDATION_CAPTIONS:
            raise SystemExit(f"validation basename {basename!r} has no caption")
        title, body = _split_caption(VALIDATION_CAPTIONS[basename])
        out.append(_emit_plot_entry(basename, title, body))
        consumed.add(basename)
    leftover = set(VALIDATION_CAPTIONS) - consumed
    if leftover:
        raise SystemExit(f"validation captions never consumed: {sorted(leftover)}")
    return "".join(out)


def main() -> int:
    """Generate atlas_manifest.py."""
    body = (
        _MODULE_HEADER
        + _build_phenotype_block()
        + ")\n"
        + _VALIDATION_HEADER
        + _build_validation_block()
        + ")\n"
        + _FOOTER
    )
    TARGET_PATH.write_text(body)
    print(f"wrote {TARGET_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
