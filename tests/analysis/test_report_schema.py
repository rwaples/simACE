"""Contract tests for the v2 report schema helpers."""

import pytest

from simace.analysis.report_schema import (
    DENSE_ARRAY_KEYS,
    REPORT_SCHEMA_NAME,
    REPORT_SCHEMA_VERSION,
    REPORT_TOP_LEVEL_GROUPS,
    assert_report_contract,
    find_dense_keys,
    partition_dense,
)


def _minimal_report() -> dict:
    return {group: {} for group in REPORT_TOP_LEVEL_GROUPS} | {
        "schema": {"name": REPORT_SCHEMA_NAME, "version": REPORT_SCHEMA_VERSION},
    }


class TestPartitionDense:
    def test_splits_leaf_arrays_from_scalars(self):
        node = {
            "trait1": {"ages": [1, 2, 3], "observed_values": [0.1, 0.2, 0.3], "half_target_age": 40.0},
            "n": 5,
        }
        scalar, dense = partition_dense(node)
        assert scalar == {"trait1": {"half_target_age": 40.0}, "n": 5}
        assert dense == {"trait1": {"ages": [1, 2, 3], "observed_values": [0.1, 0.2, 0.3]}}

    def test_empty_branches_dropped(self):
        scalar, dense = partition_dense({"only": {"ages": [1]}})
        assert scalar == {}
        assert dense == {"only": {"ages": [1]}}

    def test_dense_key_names_are_recognized(self):
        assert "ages" in DENSE_ARRAY_KEYS
        assert "aj_survival" in DENSE_ARRAY_KEYS
        assert "censoring_ages" in DENSE_ARRAY_KEYS


class TestFindDenseKeys:
    def test_reports_dotted_paths(self):
        node = {"observed": {"analysis_sample": {"cumulative_incidence": {"trait1": {"ages": [1]}}}}}
        assert list(find_dense_keys(node)) == ["observed.analysis_sample.cumulative_incidence.trait1.ages"]

    def test_clean_report_has_none(self):
        assert list(find_dense_keys(_minimal_report())) == []


class TestAssertReportContract:
    def test_accepts_minimal_valid_report(self):
        assert_report_contract(_minimal_report())

    def test_rejects_wrong_schema_version(self):
        report = _minimal_report()
        report["schema"] = {"name": REPORT_SCHEMA_NAME, "version": 99}
        with pytest.raises(ValueError, match="schema"):
            assert_report_contract(report)

    def test_rejects_missing_group(self):
        report = _minimal_report()
        del report["truth"]
        with pytest.raises(ValueError, match="missing top-level groups"):
            assert_report_contract(report)

    def test_rejects_dense_array_in_report(self):
        report = _minimal_report()
        report["observed"] = {"analysis_sample": {"cumulative_incidence": {"trait1": {"ages": [1, 2]}}}}
        with pytest.raises(ValueError, match="dense plot-array"):
            assert_report_contract(report)
