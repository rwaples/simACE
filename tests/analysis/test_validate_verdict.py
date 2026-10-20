"""Verdicts of calibrated tests across reps (ADR 0028 §2–5), against hand-computed cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from simace.analysis.validate.on_demand import ON_DEMAND_CHECKS
from simace.analysis.validate.verdict import (
    SCHEMA,
    Coverage,
    MeanZ,
    RepsOverBound,
    SdZ,
    decide,
    folder_family,
    load_verdict,
    overlapping_seed_ranges,
    records,
    scenario_family,
    write_verdict,
)

TESTS = {"onset_tau": ON_DEMAND_CHECKS["onset_tau"]}
CELLS = (*(f"trait{t}_{code}" for t in (1, 2) for code in ("MZ", "FS", "MHS", "PHS", "MO", "FO")), "within_person")


def _section(z: dict[str, float | None], z_bound: float = 2.89) -> dict:
    """One rep's onset-τ section: a cell with z None is information."""
    cells = {
        name: {"status": "scored", "z": value} if value is not None else {"status": "information", "reason": "n_eff"}
        for name, value in z.items()
    }
    scored = [abs(v) for v in z.values() if v is not None]
    return {"cells": cells, "summary": {"max_abs_z": max(scored) if scored else None, "z_bound": z_bound}}


def _reps(n: int, seed: int = 0, cells: tuple[str, ...] = CELLS) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [_section(dict(zip(cells, rng.standard_normal(len(cells)).tolist(), strict=True))) for _ in range(n)]


def test_mean_z_bound_is_the_normal_quantile_over_root_n() -> None:
    passed, record = MeanZ(np.array([1.0, -1.0, 0.5, 0.5])).decide(0.05)
    assert passed
    assert record["mean"] == 0.25
    assert record["bound"] == pytest.approx(1.959963984540054 / 2)
    assert not MeanZ(np.array([2.0, 2.0, 2.0, 2.0])).decide(0.05)[0]
    assert MeanZ(np.array([0.3])).decide(0.05) == (False, {"n": 1, "mean": None, "bound": None, "passed": False})


def test_sd_z_tests_the_chi_square_of_n_minus_1() -> None:
    # SD² (n − 1) = 4/3 · 3 = 4, inside χ²₃'s 2.5% and 97.5% quantiles, 0.2158 and 9.3484.
    passed, record = SdZ(np.array([1.0, -1.0, 1.0, -1.0])).decide(0.05)
    assert passed
    assert record["sd"] == pytest.approx(math.sqrt(4 / 3))
    assert record["lo"] == pytest.approx(math.sqrt(0.2158 / 3), abs=1e-4)
    assert record["hi"] == pytest.approx(math.sqrt(9.3484 / 3), abs=1e-4)
    assert not SdZ(np.array([0.0, 0.0, 0.0, 0.001])).decide(0.05)[0]
    assert not SdZ(np.array([10.0, -10.0, 10.0, -10.0])).decide(0.05)[0]


def test_reps_over_bound_allows_the_binomial_quantile() -> None:
    alpha = 0.05 / 161

    def tail(k: int) -> float:  # P(X > k), X ~ binomial(100, 0.05)
        return sum(math.comb(100, j) * 0.05**j * 0.95 ** (100 - j) for j in range(k + 1, 101))

    allowed = next(k for k in range(101) if tail(k) <= alpha)
    passed, record = RepsOverBound(allowed, 100, 0.05).decide(alpha)
    assert passed
    assert record == {"reps": 100, "over_bound": allowed, "allowed": allowed, "passed": True}
    assert not RepsOverBound(allowed + 1, 100, 0.05).decide(alpha)[0]


def test_coverage_needs_95_percent_of_reps_and_is_not_counted() -> None:
    assert Coverage(19, 20, 0.95).decide(0.05)[0]
    assert not Coverage(18, 20, 0.95).decide(0.05)[0]
    assert decide([])["n_tests"] == 0
    family = scenario_family({"onset_tau": _reps(20)}, TESTS)
    assert sum(not e.statistic.counted for e in family) == len(CELLS)


def test_family_sizes_match_adr_0027s_acceptance() -> None:
    """Five scenarios of 13 cells: 27 tests per scenario, 161 in the folder (2·65 + 2·13 + 5)."""
    by_scenario = {f"s{i}": {"onset_tau": _reps(20, seed=i)} for i in range(5)}
    scenario = decide(scenario_family(by_scenario["s0"], TESTS))
    assert scenario["n_tests"] == 27
    assert scenario["alpha_each"] == pytest.approx(0.05 / 27)
    assert set(scenario["tests"]["onset_tau"]) == {"cells", "reps_over_bound"}
    assert set(scenario["tests"]["onset_tau"]["cells"]["trait1_FS"]) == {"coverage", "mean_z", "sd_z"}
    folder = decide(folder_family(by_scenario, TESTS))
    assert folder["n_tests"] == 161
    assert folder["alpha_each"] == pytest.approx(0.05 / 161)
    assert set(folder) == {"verdict", "n_tests", "alpha_each", "scenarios", "pooled"}
    assert folder["pooled"]["onset_tau"]["cells"]["trait1_FS"]["mean_z"]["n"] == 100


def test_standard_normal_z_pass() -> None:
    assert decide(scenario_family({"onset_tau": _reps(100)}, TESTS))["verdict"] == "pass"


def test_a_shifted_cell_fails_its_mean_test() -> None:
    reps = _reps(100)
    for section in reps:
        section["cells"]["trait1_FS"]["z"] += 1.0
    decided = decide(scenario_family({"onset_tau": reps}, TESTS))
    assert decided["verdict"] == "fail"
    assert not decided["tests"]["onset_tau"]["cells"]["trait1_FS"]["mean_z"]["passed"]
    assert decided["tests"]["onset_tau"]["cells"]["trait1_MZ"]["mean_z"]["passed"]


def test_a_rarely_scored_cell_fails_coverage() -> None:
    reps = _reps(20)
    for section in reps[:2]:
        section["cells"]["within_person"] = {"status": "information", "reason": "ties"}
    decided = decide(scenario_family({"onset_tau": reps}, TESTS))
    coverage = decided["tests"]["onset_tau"]["cells"]["within_person"]["coverage"]
    assert coverage == {"scored": 18, "reps": 20, "share": 0.9, "passed": False}
    assert decided["verdict"] == "fail"
    assert decided["n_tests"] == 27


def test_not_applicable_reps_leave_the_coverage_count() -> None:
    """ADR 0029 §4: a degenerate cell is neither scored nor counted against coverage."""
    reps = _reps(20)
    for section in reps[:2]:
        section["cells"]["within_person"] = {"status": "not_applicable", "reason": "zero variance"}
    decided = decide(scenario_family({"onset_tau": reps}, TESTS))
    cell = decided["tests"]["onset_tau"]["cells"]["within_person"]
    assert cell["coverage"] == {"scored": 18, "reps": 18, "share": 1.0, "passed": True}
    assert cell["mean_z"]["n"] == 18


def test_a_cell_not_applicable_in_every_rep_has_no_statistics() -> None:
    reps = _reps(20)
    for section in reps:
        section["cells"]["within_person"] = {"status": "not_applicable", "reason": "zero variance"}
    decided = decide(scenario_family({"onset_tau": reps}, TESTS))
    assert "within_person" not in decided["tests"]["onset_tau"]["cells"]
    assert decided["n_tests"] == 25
    assert decided["verdict"] == "pass"


def test_a_test_not_applicable_to_the_scenario_drops_out_of_its_family() -> None:
    """ADR 0029 §4: C configured 0 on both traits gives household_draw no cell, which must not block a pass."""
    tests = {**TESTS, "household_draw": ON_DEMAND_CHECKS["household_draw"]}
    absent = [{"cells": {}, "summary": {"max_abs_z": None, "z_bound": None}, "not_applicable": "C"}] * 30
    decided = decide(scenario_family({"onset_tau": _reps(30), "household_draw": absent}, tests))
    assert decided["verdict"] == "pass"
    assert "household_draw" not in decided["tests"]
    assert decided["n_tests"] == 27


def test_reps_over_their_bound_are_counted_from_max_abs_z() -> None:
    reps = _reps(20, cells=("trait1_FS",))
    for section in reps[:10]:
        section["summary"]["max_abs_z"] = 3.0
    tail = decide(scenario_family({"onset_tau": reps}, TESTS))["tests"]["onset_tau"]["reps_over_bound"]
    assert tail["over_bound"] == 10 + sum(abs(s["cells"]["trait1_FS"]["z"]) > 2.89 for s in reps[10:])
    assert not tail["passed"]


def test_fewer_than_min_reps_is_insufficient_never_pass() -> None:
    decided = decide(scenario_family({"onset_tau": _reps(19)}, TESTS))
    assert decided["verdict"] == "insufficient"
    assert decided["n_tests"] == 0
    assert decided["alpha_each"] is None
    assert decided["tests"]["onset_tau"] == {"insufficient": "19 rep(s); the verdict needs 20"}


def test_a_folder_pools_only_when_every_scenario_has_min_reps_and_a_failure_outranks_it() -> None:
    shifted = _reps(20, seed=1)
    for section in shifted:
        section["cells"]["trait1_FS"]["z"] += 2.0
    decided = decide(folder_family({"a": {"onset_tau": _reps(5)}, "b": {"onset_tau": shifted}}, TESTS))
    assert decided["pooled"]["onset_tau"] == {"insufficient": "a below 20 rep(s)"}
    assert decided["scenarios"]["a"]["onset_tau"] == {"insufficient": "5 rep(s); the verdict needs 20"}
    assert decided["n_tests"] == 27
    assert decided["verdict"] == "fail"


def test_pooling_skips_cells_never_scored_and_keeps_cells_scored_anywhere() -> None:
    a = _reps(20, seed=1, cells=("trait1_FS", "trait1_MZ"))
    b = _reps(20, seed=2, cells=("trait1_FS", "trait1_MZ"))
    for section in a:
        section["cells"]["trait1_MZ"] = {"status": "information", "reason": "n_eff"}
    for section in a + b:
        section["cells"]["within_person"] = {"status": "information", "reason": "ties"}
    pooled = decide(folder_family({"a": {"onset_tau": a}, "b": {"onset_tau": b}}, TESTS))["pooled"]["onset_tau"]
    assert set(pooled["cells"]) == {"trait1_FS", "trait1_MZ"}
    assert pooled["cells"]["trait1_MZ"]["mean_z"]["n"] == 20


def test_overlapping_seed_ranges() -> None:
    ranges = {"a": (0, 99), "b": (100, 199), "c": (150, 160), "d": (200, 299)}
    assert overlapping_seed_ranges(ranges) == [("b", "c")]


def test_the_verdict_file_carries_its_schema_and_is_read_back(tmp_path) -> None:
    path = tmp_path / "validation_summary.yaml"
    decided = decide(scenario_family({"onset_tau": _reps(20)}, TESTS))
    write_verdict(path, {"family": "scenario"}, decided)
    data = yaml.safe_load(path.read_text())
    assert data["schema"] == SCHEMA
    assert load_verdict(path) == data
    path.write_text(yaml.safe_dump({"schema": {"kind": "on_demand_checks", "version": 1}, "verdict": "pass"}))
    assert load_verdict(path) is None


def test_onset_tau_tail_rate_is_its_per_rep_familywise_level() -> None:
    """The rule's expected rate of reps over their bound is the level that sets each rep's ``z_bound``."""
    from simace.analysis.validate.calibrated import FAMILYWISE_ALPHA

    assert ON_DEMAND_CHECKS["onset_tau"].rule.rep_rate == FAMILYWISE_ALPHA
    assert ON_DEMAND_CHECKS["onset_tau"].min_reps == 20


def test_a_scenario_with_no_cell_is_insufficient_not_pass() -> None:
    """Non-frailty traits report no cell (ADR 0027); an untested calibration must not pass (#66)."""
    reps = [_section({}) for _ in range(20)]
    decided = decide(scenario_family({"onset_tau": reps}, TESTS))
    assert decided["verdict"] == "insufficient"
    assert decided["n_tests"] == 0
    assert decided["tests"]["onset_tau"] == {"insufficient": "no rep reports a cell"}


def test_refused_pooling_leaves_the_scenario_tests_and_marks_pooled_insufficient() -> None:
    by_scenario = {f"s{i}": {"onset_tau": _reps(20, seed=i)} for i in range(2)}
    decided = decide(folder_family(by_scenario, TESTS, pooling_refused="rep seed ranges overlap: s0 and s1"))
    assert decided["n_tests"] == 54
    assert decided["pooled"]["onset_tau"] == {"insufficient": "rep seed ranges overlap: s0 and s1"}
    assert decided["verdict"] == "insufficient"


def test_records_walk_every_statistic_of_a_written_family_with_its_path() -> None:
    """What the atlas and gather read back: each record by path, the header left out."""
    shifted = _reps(20, seed=1)
    for section in shifted:
        section["cells"]["trait1_FS"]["z"] += 2.0
    decided = decide(folder_family({"a": {"onset_tau": _reps(5)}, "b": {"onset_tau": shifted}}, TESTS))
    found = dict(records({"schema": dict(SCHEMA), "family": "folder", "reps_by_scenario": {}, **decided}))
    assert found[("scenarios", "a", "onset_tau")] == {"insufficient": "5 rep(s); the verdict needs 20"}
    assert found[("pooled", "onset_tau")] == {"insufficient": "a below 20 rep(s)"}
    assert found[("scenarios", "b", "onset_tau", "cells", "trait1_FS", "mean_z")]["passed"] is False
    assert ("scenarios", "b", "onset_tau", "reps_over_bound") in found
    assert len(found) == 2 + 13 * 3 + 1
    assert all(path[0] in {"scenarios", "pooled"} for path in found)


def test_an_unchanged_verdict_is_not_rewritten(tmp_path) -> None:
    path = tmp_path / "validation_summary.yaml"
    decided = decide(scenario_family({"onset_tau": _reps(20)}, TESTS))
    write_verdict(path, {"family": "scenario"}, decided)
    written = path.stat().st_mtime_ns
    write_verdict(path, {"family": "scenario"}, decided)
    assert path.stat().st_mtime_ns == written
    write_verdict(path, {"family": "scenario", "reps": "1-20"}, decided)
    assert path.stat().st_mtime_ns != written


def test_record_names_drop_grouping_keys_by_position_not_by_name() -> None:
    from simace.analysis.validate.verdict import record_name

    assert record_name(("tests", "onset_tau", "cells", "trait1_FS", "mean_z")) == "onset_tau/trait1_FS/mean_z"
    assert record_name(("scenarios", "tests", "onset_tau", "cells", "cells", "sd_z")) == "tests/onset_tau/cells/sd_z"
    assert record_name(("pooled", "onset_tau", "cells", "within_person", "mean_z")) == (
        "pooled/onset_tau/within_person/mean_z"
    )
    assert record_name(("scenarios", "a", "onset_tau", "reps_over_bound")) == "a/onset_tau/reps_over_bound"
    assert record_name(("tests", "onset_tau")) == "onset_tau"
