"""R cross-check for Ne_I (optiSel) and Ne_iΔF (purgeR), gated behind rpy2.

The expected steady state is "skipped" — neither rpy2 nor the two R
packages are part of the simACE conda environment, and the master plan
explicitly states that R is not added to project deps.  This file is a
structural placeholder: it loads the R toolchain when present, builds a
small fixture pedigree, and stops at a TODO where the optiSel /
purgeR API calls would go.

Install instructions for someone wanting to actually run this:

    conda install -c conda-forge rpy2
    R -e 'install.packages(c("optiSel", "purgeR"), repos="https://cran.r-project.org")'

Then drop the ``pytest.skip`` markers below and fill in the API-specific
assertions, asserting agreement with the simace estimators to within
1 % (tolerance per master plan).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip the entire module if rpy2 isn't installed.
_robjects = pytest.importorskip("rpy2.robjects", reason="rpy2 not installed")
_pkg_module = pytest.importorskip("rpy2.robjects.packages", reason="rpy2.robjects.packages unavailable")

from pedigree_graph import (  # noqa: E402  — after importorskip
    PedigreeGraph,
    ne_inbreeding,
    ne_individual_delta_f,
)


@pytest.fixture(scope="module")
def r_packages():
    """Load optiSel and purgeR; skip if either R package is missing."""
    try:
        optiSel = _pkg_module.importr("optiSel")
        purgeR = _pkg_module.importr("purgeR")
    except Exception as exc:  # rpy2 surfaces R errors as generic exceptions
        pytest.skip(f"R packages unavailable: {exc}")
    return optiSel, purgeR


@pytest.fixture(scope="module")
def fixture_pedigree() -> tuple[pd.DataFrame, PedigreeGraph]:
    """Small deterministic random-mating pedigree for cross-checking."""
    rng = np.random.default_rng(2026)
    n, n_gens = 50, 6
    rows: list[dict] = [
        {
            "id": i,
            "sex": 1 if i % 2 == 0 else 0,
            "generation": 0,
            "mother": -1,
            "father": -1,
            "twin": -1,
        }
        for i in range(n)
    ]
    next_id = n
    for g in range(1, n_gens + 1):
        prev_start = (g - 1) * n
        male_ids = np.arange(prev_start, prev_start + n, 2)
        female_ids = np.arange(prev_start + 1, prev_start + n, 2)
        f_pick = rng.choice(male_ids, size=n)
        m_pick = rng.choice(female_ids, size=n)
        for i in range(n):
            rows.append(
                {
                    "id": next_id,
                    "sex": 1 if i % 2 == 0 else 0,
                    "generation": g,
                    "mother": int(m_pick[i]),
                    "father": int(f_pick[i]),
                    "twin": -1,
                }
            )
            next_id += 1
    df = pd.DataFrame(rows)
    return df, PedigreeGraph(df)


def test_ne_inbreeding_matches_optiSel(r_packages, fixture_pedigree):
    """Compare pedigree-graph ``ne_inbreeding`` to optiSel's Ne estimator."""
    _df, pg = fixture_pedigree
    sim_ne = ne_inbreeding(pg).ne
    pytest.skip(
        "TODO: invoke optiSel::Ne() on the fixture pedigree, extract its inbreeding-rate "
        "estimate, and assert ``abs(sim/optiSel - 1) < 0.01`` against "
        f"sim_ne={sim_ne!r}.  Requires familiarity with optiSel's pedigree input "
        "format (Indiv, Sire, Dam columns)."
    )


def test_ne_individual_delta_f_matches_purgeR(r_packages, fixture_pedigree):
    """Compare pedigree-graph ``ne_individual_delta_f`` to purgeR's Ne estimator."""
    _df, pg = fixture_pedigree
    sim_ne = ne_individual_delta_f(pg).ne
    pytest.skip(
        "TODO: invoke purgeR::ip_F() and Ne() on the fixture pedigree and assert "
        f"``abs(sim/purgeR - 1) < 0.01`` against sim_ne={sim_ne!r}.  Requires "
        "purgeR's expected pedigree column layout."
    )
