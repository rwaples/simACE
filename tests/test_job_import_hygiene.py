"""Discipline test: per-job packages do not import the heavy scipy subpackages at module level.

Every stage in the per-replicate chain (simulate, phenotype, censor,
ascertain) runs as its own ``python -m simace <stage>`` process, and the
EPIMIGHT emitter starts a fresh interpreter too, so module
import time is paid once per replicate per stage. ``scipy.stats`` alone costs
about 0.5 s to import; moving its two single-caller imports inside their
functions took the phenotype job from 1.10 s to 0.68 s; ``scipy.special``
(~110 ms, three phenotype callers) and ``scipy.spatial`` (~160 ms, the
assortative-mating tree) followed. A module-level import of any of them on
these paths silently gives the time back, so this test asserts each stays
unloaded after importing what each job script imports.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# What a `python -m simace <stage>` process imports: the dispatcher, then the
# stage's own module.
_JOB_MODULES = [
    "simace.__main__",
    "simace.cli",
    "simace.simulation.simulate",
    "simace.phenotype.runner",
    "simace.censoring.censor",
    "simace.ascertainment.runner",
    "simace.core.parquet",
    "simace.core.publish",
]


_HEAVY_SCIPY = ["scipy.stats", "scipy.special", "scipy.spatial"]


@pytest.mark.parametrize("module", _JOB_MODULES)
def test_job_module_does_not_import_heavy_scipy(module: str) -> None:
    """Importing *module* in a fresh interpreter leaves the heavy scipy subpackages unloaded."""
    code = f"import sys; import {module}; print(' '.join(m for m in {_HEAVY_SCIPY!r} if m in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = result.stdout.split()
    assert not loaded, (
        f"{module} pulls {loaded} in at import time (0.1-0.5 s per job each); "
        f"import them inside the functions that need them instead."
    )
