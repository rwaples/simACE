"""Discipline test: per-job packages do not import ``scipy.stats`` at module level.

Every Snakemake job in the per-replicate chain (simulate, phenotype, censor,
ascertainment) and the EPIMIGHT emitter starts a fresh interpreter, so module
import time is paid once per replicate per stage. ``scipy.stats`` alone costs
about 0.5 s to import; moving its two single-caller imports inside their
functions took the phenotype job from 1.10 s to 0.68 s. A module-level
``from scipy.stats import ...`` anywhere on these import paths silently gives
that back, so this test asserts the module stays unloaded after importing what
each job script imports.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# What the workflow/scripts/simace job wrappers import at module level, plus
# the analysis entry points the downstream stats jobs start from.
_JOB_MODULES = [
    "simace.simulation.simulate",
    "simace.phenotype",
    "simace.censoring.censor",
    "simace.ascertainment",
    "simace.core.parquet",
    "simace.core.snakemake_adapter",
]


@pytest.mark.parametrize("module", _JOB_MODULES)
def test_job_module_does_not_import_scipy_stats(module: str) -> None:
    """Importing *module* in a fresh interpreter leaves ``scipy.stats`` unloaded."""
    code = f"import sys; import {module}; sys.exit(1 if 'scipy.stats' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"{module} pulls scipy.stats in at import time (about 0.5 s per job); "
        f"import it inside the function that needs it instead.\n{result.stderr}"
    )
