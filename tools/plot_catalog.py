"""Write docs/user-guide/plot-catalog.md from the atlas manifest.

Usage: ``pixi run python tools/plot_catalog.py``

The catalog lists every plot the scenario, validation, and effective-size
atlases contain, with the basename and caption from
``simace.plotting.atlas_manifest``. Rerun after editing the manifest.
"""

from __future__ import annotations

import sys
from pathlib import Path

from simace.plotting import atlas_manifest as m

OUT = Path(__file__).resolve().parent.parent / "docs" / "user-guide" / "plot-catalog.md"

HEADER = """\
# Plot catalog

Generated from `simace/plotting/atlas_manifest.py` by
`tools/plot_catalog.py`. Do not edit by hand. Rerun the script after
changing the manifest:

```bash
pixi run python tools/plot_catalog.py
```

Each basename gets the extension set by `plot_format`, `png` by default.
Scenario plots live in `results/{folder}/{scenario}/plots/`. Validation
plots live in `results/{folder}/plots/`. Effective-size plots live in the
scenario plots directory when the `effective_size` rule runs.
"""


def _caption(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _section(title: str, items: list[m.AtlasItem]) -> list[str]:
    lines = [f"## {title}", ""]
    in_table = False
    for item in items:
        if isinstance(item, m.SectionBreak):
            if in_table:
                lines.append("")
            heading = "Phenotype model" if item is m.MODEL_SECTION else item.title.rstrip(".")
            lines += [f"### {heading}", ""]
            in_table = False
            continue
        if not in_table:
            lines += ["| File | Caption |", "|---|---|"]
            in_table = True
        lines.append(f"| `{item.basename}` | **{_caption(item.title).rstrip('.')}.** {_caption(item.body)} |")
    lines.append("")
    return lines


def main() -> int:
    """Write the catalog and return the process exit code."""
    lines = [HEADER]
    lines += _section("Scenario atlas", list(m.PHENOTYPE_ATLAS))
    lines += _section("Validation atlas", list(m.VALIDATION_ATLAS))
    lines += _section("Effective-size atlas", list(m.EFFECTIVE_SIZE_ATLAS))
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
