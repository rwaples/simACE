"""Keep the configuration reference synchronized with its source data."""

from pathlib import Path

import yaml

from simace.config import _HIERARCHICAL_TO_FLAT

REPO_ROOT = Path(__file__).parent.parent
REFERENCE_PATH = REPO_ROOT / "docs" / "user-guide" / "configuration.md"
DEFAULTS_PATH = REPO_ROOT / "config" / "_default.yaml"


def _between_markers(text: str, name: str) -> str:
    """Return the text inside a named pair of HTML comment markers."""
    start = f"<!-- {name}:start -->"
    end = f"<!-- {name}:end -->"
    assert text.count(start) == 1, f"expected one {start} marker"
    assert text.count(end) == 1, f"expected one {end} marker"
    return text.split(start, 1)[1].split(end, 1)[0]


def _markdown_rows(block: str, width: int) -> list[list[str]]:
    """Parse code-keyed rows from the Markdown tables in ``block``."""
    rows = []
    for line in block.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert len(cells) == width, f"expected {width} cells in row: {line}"
        rows.append(cells)
    return rows


def _unquote_code_cell(cell: str) -> str:
    """Remove the single pair of backticks around a Markdown table cell."""
    assert cell.startswith("`"), f"expected a code cell, got {cell!r}"
    assert cell.endswith("`"), f"expected a code cell, got {cell!r}"
    return cell[1:-1]


def _flatten_defaults(value, prefix: tuple[str, ...] = ()) -> dict[str, object]:
    """Flatten defaults to the paths used by the reference tables."""
    if isinstance(value, dict) and prefix != ("censoring", "gen_censoring"):
        flattened = {}
        for key, child in value.items():
            flattened.update(_flatten_defaults(child, (*prefix, str(key))))
        return flattened
    return {".".join(prefix): value}


def test_documented_scenario_defaults_match_default_yaml():
    """Document every scenario default once and with its current value."""
    reference = REFERENCE_PATH.read_text()
    block = _between_markers(reference, "scenario-defaults")
    rows = _markdown_rows(block, width=4)

    documented = {}
    for parameter_cell, _type_cell, default_cell, _description_cell in rows:
        parameter = _unquote_code_cell(parameter_cell)
        assert parameter not in documented, f"duplicate documented default: {parameter}"
        documented[parameter] = yaml.safe_load(_unquote_code_cell(default_cell))

    raw = yaml.safe_load(DEFAULTS_PATH.read_text())
    expected = _flatten_defaults(raw["defaults"])
    assert documented == expected


def test_documented_legacy_aliases_match_loader_mapping():
    """Document every loader alias once and reject stale aliases."""
    reference = REFERENCE_PATH.read_text()
    block = _between_markers(reference, "legacy-aliases")
    rows = _markdown_rows(block, width=2)

    documented = {}
    for sectioned_cell, flat_cell in rows:
        sectioned = _unquote_code_cell(sectioned_cell)
        assert sectioned not in documented, f"duplicate documented alias: {sectioned}"
        documented[sectioned] = _unquote_code_cell(flat_cell)

    expected = {".".join(path): flat for path, flat in _HIERARCHICAL_TO_FLAT.items()}
    assert documented == expected
