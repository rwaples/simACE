# ADR 0010: HTML as the Primary Atlas Rendering

## Status

Accepted. Permanentizes the prototype tracked in
`notes/html_atlas_prototype_compromises.md` and
`notes/html_atlas_prototype_implementation_plan.md`.

## Context

The plot **atlas** (see CONTEXT.md) was historically a multi-page PDF: every
atlas consumer in simACE and fitACE calls one shared seam,
`simace.plotting.plot_atlas.assemble_atlas(items, plot_dir, output_path, ...)`,
which walks an ordered manifest of `PlotEntry` / `SectionBreak` items into a
`PdfPages` document. An HTML rendering, `assemble_html_atlas`, was added as a
CLI-only prototype with the *identical* signature; it has since grown a native
HTML parameter overview and a native Table 1, surpassing the PDF in formatting,
navigation (sticky TOC), and responsiveness. Only the model equations still
rendered to a companion PNG.

The PDF atlas is purely human-facing: no rule embeds it and fitACE does not
read it. So which rendering is the *default* artifact is a free product choice,
not a pipeline constraint. Six atlases ride the shared seam — the per-scenario
atlas, the per-folder validation atlas, and the fitACE-side EPIMIGHT,
EPIMIGHT-bias, and onset-censoring atlases — so any decision about the default
rendering applies uniformly across three repos via that one seam. (The PA-FGRS
atlas builds its own bespoke `PdfPages` and is **not** on the seam.)

## Decision

The **HTML atlas is the primary, always-built rendering**; the PDF atlas is an
on-demand export.

- **Dispatch seam.** A thin `render_atlas(items, plot_dir, output_path, ...)` in
  `simace.plotting` dispatches on `output_path` suffix: `.html` →
  `assemble_html_atlas`, `.pdf` → `assemble_atlas`, anything else → `ValueError`.
  `assemble_atlas` / `assemble_html_atlas` remain as backends (additive — their
  contracts do not change). Every consumer becomes format-agnostic; flipping an
  atlas to HTML-primary is a change of output extension.
- **Workflow shape.** Each flipped atlas gets two rules sharing inputs: an
  `atlas.html` rule (in the default target — e.g. `scenario.done`) and a sibling
  `atlas.pdf` rule (built only when requested, e.g. `snakemake .../atlas.pdf`).
  `get_scenario_sim_outputs` appends `atlas.html`.
- **Self-contained.** `atlas.html` embeds every plot as a base64 data URI and
  renders equations as **inline SVG** (matplotlib mathtext → SVG). One portable
  file; no `atlas_assets/` directory; no JavaScript, CDN, or vendored libraries
  (dependency-free). Equations stay crisp at any zoom.
- **Scope.** All on-seam atlases flip: scenario, validation, EPIMIGHT,
  EPIMIGHT-bias, onset-censoring. PA-FGRS stays PDF (off-seam; an HTML rewrite
  of its bespoke pages is a separate effort).
- **Missing plots.** Official builds declare every figure as a rule input, so a
  missing plot cannot occur. In the CLI / partial-run path the HTML atlas keeps
  a visible placeholder card preserving figure numbering — a deliberate
  divergence from the PDF path, which silently skips missing plots.

## Consequences

- The default human-facing deliverable across simACE and fitACE_epimight becomes
  `atlas.html`. Docs (output-structure, quickstart, examples, the CLAUDE.md
  force-rebuild idiom) point at `atlas.html`; the PDF is documented as an
  on-demand export.
- `scenario.done` now requires `atlas.html`, not `atlas.pdf` — a contract change
  for anything keying off scenario outputs.
- Embedding duplicates plot bytes into the HTML (~4–8 MB per scenario atlas),
  trading per-scenario storage for a portable single file.
- One render seam means a future third rendering (or a change to the
  default-vs-export policy) is a localized change, not a sweep across six call
  sites.
