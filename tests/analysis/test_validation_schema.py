"""Contract tests for METRIC_REGISTRY ↔ the curated v2 report.

The registry in `simace.analysis.validation_schema` is the single source of
truth for the report-summary TSV columns. These tests run a small,
parameter-tuned coverage simulation through `run_validation`, assemble it into a
v2 report, and assert that **every** registered path resolves to a non-None
value. If a producer-side rename in `validate.py` / `report.py` empties a
registry-tracked column, this suite fails — the silent-drop bug class this
refactor closes. Every registry path lives in ``truth`` / ``estimators``, both
derived from the validation report, so the stats half can be empty here.
"""

from __future__ import annotations

import pytest

from simace.analysis.gather import _get_nested, extract_metrics
from simace.analysis.report import assemble_report
from simace.analysis.validate import run_validation
from simace.analysis.validation_schema import METRIC_REGISTRY
from simace.core.yaml_io import dump_yaml, load_yaml
from simace.simulation.simulate import run_simulation

# Coverage parameters mirror config/test.yaml::coverage_scenario. Tuned so
# every registered metric is populated: half-sib pairs (mating_lambda=0.5,
# N=2000), MZ twins (p_mztwin=0.05), and meaningfully non-zero mate
# liability correlation (assort1/2=0.3).
_COVERAGE_PARAMS = dict(
    seed=1234,
    N=2000,
    G_ped=4,
    G_sim=4,
    mating_lambda=0.5,
    p_mztwin=0.05,
    A1=0.5,
    C1=0.2,
    E1=0.3,
    A2=0.4,
    C2=0.2,
    E2=0.4,
    rA=0.0,
    rC=0.0,
    rE=0.0,
    assort1=0.3,
    assort2=0.3,
)


@pytest.fixture(scope="session")
def coverage_report(tmp_path_factory) -> dict:
    """Run a coverage simulation, assemble a v2 report, and round-trip it.

    The registry reads only ``truth`` / ``estimators`` (both derived from the
    validation report), so ``stats_report`` is empty here. Round-trips through
    `dump_yaml` / `load_yaml` so the test sees what `gather` reads off disk.
    """
    work = tmp_path_factory.mktemp("coverage_scenario")
    pedigree = run_simulation(**_COVERAGE_PARAMS)
    pedigree_path = work / "pedigree.full.parquet"
    pedigree.to_parquet(pedigree_path)

    params_path = work / "params.yaml"
    dump_yaml(_COVERAGE_PARAMS, params_path)

    validation = run_validation(str(pedigree_path), str(params_path))
    report, _payload = assemble_report(
        replicate={"folder": "test", "scenario": "coverage", "rep": 1, "seed": 1234},
        params=_COVERAGE_PARAMS,
        case_ascertainment_ratio=1.0,
        validation_report=validation,
        stats_report={},
        prevalence_phenotyped={},
        scope_counts={},
    )
    out_path = work / "report.yaml"
    dump_yaml(report, out_path)
    return load_yaml(out_path)


def test_unique_columns():
    columns = [spec.column for spec in METRIC_REGISTRY]
    duplicates = {c for c in columns if columns.count(c) > 1}
    assert not duplicates, f"Duplicate columns in METRIC_REGISTRY: {sorted(duplicates)}"


def test_every_registry_path_resolves(coverage_report):
    """Each registered path must hit a non-None leaf in the coverage report."""
    missing = []
    for spec in METRIC_REGISTRY:
        value = _get_nested(coverage_report, *spec.path)
        if value is None:
            missing.append((spec.column, "/".join(spec.path)))
    assert not missing, (
        "Registry paths that did not resolve in the coverage report "
        "(producer must emit these keys, or registry must be updated): " + repr(missing)
    )


def test_extract_metrics_populates_registry_columns(tmp_path, coverage_report):
    """End-to-end: extract_metrics returns a non-None value for every registered column."""
    val_dir = tmp_path / "results" / "test" / "coverage_scenario" / "rep1"
    val_dir.mkdir(parents=True)
    report_path = val_dir / "report.yaml"
    dump_yaml(coverage_report, report_path)

    row = extract_metrics(str(report_path))
    missing = [spec.column for spec in METRIC_REGISTRY if row.get(spec.column) is None]
    assert not missing, f"extract_metrics returned None for registry columns: {missing}"
