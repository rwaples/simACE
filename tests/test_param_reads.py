"""Only the boundary modules read the assortment, mating-model, burn-in, C transmission and inbreeding keys (#86).

Past a boundary, code takes the parameter records of
``simace.simulation.params``, and ``ty`` checks it. What ``ty`` cannot see is
a new raw read such as ``params.get("assort1", 0)``, whose own default is how
the predicates came to disagree (#93). This test fails on one outside the
modules listed here.
"""

import re
from pathlib import Path

SIMACE = Path(__file__).resolve().parents[1] / "simace"

_KEYS = r"mating_model|assort1|assort2|assort_matrix|Ctm_\w+|G_sim|inbreeding_mode|inbreeding_depth"
_RAW_READ = re.compile(rf"""(\.get\(|\[)["']({_KEYS})["']""")

#: Modules allowed to read those keys from a mapping, and why.
_BOUNDARIES = {
    "config.py": "config load: parses the scenario records",
    "simulation/params.py": "the records' own parsers",
    "simulation/emit_params.py": "writes params.yaml",
    "cli/stages.py": "builds stage command lines from the flat parameters",
    "analysis/gather.py": "writes the gathered TSV from report.yaml",
    "plotting/plot_pipeline.py": "prints the configured values on the atlas title page",
}


def test_record_owned_keys_are_read_only_at_the_boundaries():
    offenders = [
        f"{path.relative_to(SIMACE)}:{n}: {line.strip()}"
        for path in sorted(SIMACE.rglob("*.py"))
        if str(path.relative_to(SIMACE)) not in _BOUNDARIES
        for n, line in enumerate(path.read_text().splitlines(), start=1)
        if _RAW_READ.search(line)
    ]
    assert offenders == [], "read these through the parameter records:\n" + "\n".join(offenders)


def test_every_listed_boundary_exists():
    assert all((SIMACE / module).is_file() for module in _BOUNDARIES)
