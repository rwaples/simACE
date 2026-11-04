"""The inbreeding resolver: handoff, engine choice, refusal and tail layout (ADR 0033 §2-§5)."""

from __future__ import annotations

import pytest

from simace.simulation import inbreeding_plan
from simace.simulation.inbreeding_plan import InbreedingPlan, InbreedingRefused, predicted_seconds, resolve


def test_off_is_the_no_op_plan() -> None:
    assert resolve("off", 6, 1_000_000, 100) == InbreedingPlan("off", "none")


@pytest.mark.parametrize(("mode", "depth"), [("none", 6), (False, 6), ("approx", 0), ("approx", 6.0), ("approx", True)])
def test_bad_mode_or_depth_is_rejected(mode, depth) -> None:
    with pytest.raises(ValueError, match="inbreeding"):
        resolve(mode, depth, 1000, 10)


@pytest.mark.parametrize("G", [6, 7])
def test_handoff_memory_boundary(G) -> None:
    # 8 N^2 <= 2^30 holds up to N = 11,585; at G <= 7, N^2 G stays below 1e9 there.
    assert resolve("approx", 3, 11_585, G).mode == "exact"
    assert resolve("approx", 3, 11_586, G) == InbreedingPlan("approx", "xexact", 3, True)


def test_handoff_n2g_boundary() -> None:
    assert resolve("approx", 6, 10_000, 10).mode == "exact"  # N^2 G = 1e9 exactly
    assert resolve("approx", 6, 10_000, 11).mode == "approx"


@pytest.mark.parametrize("depth", [20, 21, 50])
def test_depth_at_or_beyond_g_sim_runs_exact(depth) -> None:
    assert resolve("approx", depth, 1_000_000, 12) == resolve("exact", 6, 1_000_000, 12)


def test_engine_choice_follows_the_cost_models() -> None:
    assert resolve("exact", 6, 10_000, 100) == InbreedingPlan("exact", "densep")
    assert resolve("exact", 6, 30_000, 16) == InbreedingPlan("exact", "walkd2")
    assert resolve("exact", 6, 1_000_000, 12) == InbreedingPlan("exact", "walkd2")


def test_densep_needs_16_gib() -> None:
    # 8 N^2 <= 16 GiB up to N = 46,340; G 100 would favour densep on time alone.
    assert predicted_seconds("densep", 46_341, 30) < predicted_seconds("walkd2", 46_341, 30)
    assert resolve("exact", 6, 46_340, 30).engine == "densep"
    assert resolve("exact", 6, 46_341, 30).engine == "walkd2"


def test_deep_large_exact_is_refused_naming_the_prediction() -> None:
    assert predicted_seconds("walkd2", 1_000_000, 20) == pytest.approx(4_770, rel=1e-3)
    with pytest.raises(InbreedingRefused, match=r"walkd2 is predicted to take 4,767 s.*use mode: approx"):
        resolve("exact", 6, 1_000_000, 20)
    with pytest.raises(InbreedingRefused, match=r"lower depth below G_sim \(20; default 6\)"):
        resolve("approx", 20, 1_000_000, 20)


def test_refusal_boundary_is_twice_the_prediction_above_an_hour(monkeypatch) -> None:
    assert not inbreeding_plan._refused(1_800.0)
    assert inbreeding_plan._refused(1_800.0 + 1e-9)
    monkeypatch.setattr(inbreeding_plan, "predicted_seconds", lambda engine, N, G: 1_800.0)
    assert resolve("exact", 6, 1_000_000, 20).engine == "walkd2"
    monkeypatch.setattr(inbreeding_plan, "predicted_seconds", lambda engine, N, G: 1_800.5)
    with pytest.raises(InbreedingRefused):
        resolve("exact", 6, 1_000_000, 20)


@pytest.mark.parametrize(("N", "cap"), [(1_000_000, 9), (5_000_000, 7), (10_000_000, 6)])
def test_packing_cap(N, cap) -> None:
    assert resolve("approx", cap, N, 100).packed
    assert not resolve("approx", cap + 1, N, 100).packed


def test_record_is_the_run_yaml_shape() -> None:
    assert resolve("approx", 6, 1_000_000, 20).record() == {
        "mode": "approx",
        "engine": "xexact",
        "depth": 6,
        "packed": True,
    }


def test_deep_small_runs_resolve_without_overflow() -> None:
    # 2.0**k overflows from k = 1024 (Codex review): the walkd2 visit model must saturate instead.
    assert resolve("exact", 6, 100, 1100) == InbreedingPlan("exact", "densep")
    assert resolve("approx", 6, 100, 1100).mode == "exact"
