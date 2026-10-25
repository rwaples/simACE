"""Shared fixtures of the inbreeding tests (ADR 0033 §10): ancestry capture, oracles and plan forcing."""

from __future__ import annotations

from typing import Any

import numpy as np
from pedigree_graph import PedigreeGraph

import simace.simulation.params as params_mod
import simace.simulation.simulate as sim
from simace.simulation.inbreeding import Inbreeding
from simace.simulation.inbreeding_plan import InbreedingPlan

BASE: dict[str, Any] = dict(
    seed=7,
    N=300,
    G_ped=3,
    G_sim=12,
    mating_lambda=0.5,
    p_mztwin=0.05,
    A1=0.5,
    C1=0.2,
    E1=0.3,
    A2=0.4,
    C2=0.1,
    E2=0.5,
    rA=0.3,
    rC=0.0,
    rE=0.0,
)

DENSEP = InbreedingPlan("exact", "densep")


class Capture:
    """Every mating iteration's ``(parents, twins)`` and parental sex, recorded from inside ``run_simulation``."""

    def __init__(self) -> None:
        self.gens: list[tuple[np.ndarray, np.ndarray]] = []
        self.sex: list[np.ndarray] = []

    def ancestry(self, last_sex: np.ndarray) -> PedigreeGraph:
        """The whole simulated ancestry as a graph: founders, then every iteration's offspring, ``N`` rows each."""
        N = len(self.sex[0])
        G = len(self.gens)
        ids = np.arange(N * (G + 1))
        mother = np.full(len(ids), -1)
        father = np.full(len(ids), -1)
        twin = np.full(len(ids), -1)
        for i, (parents, twins) in enumerate(self.gens):
            lo = (i + 1) * N
            mother[lo : lo + N] = parents[:, 0] + i * N
            father[lo : lo + N] = parents[:, 1] + i * N
            if len(twins):
                twin[lo + twins[:, 0]] = lo + twins[:, 1]
                twin[lo + twins[:, 1]] = lo + twins[:, 0]
        sex = np.concatenate([*self.sex, last_sex])
        return PedigreeGraph.from_arrays(ids=ids, mother_ids=mother, father_ids=father, twin_ids=twin, sex=sex)


def install_capture(monkeypatch) -> Capture:
    """Record every mating call of the ``run_simulation`` calls that follow."""
    cap = Capture()
    for name in ("_mating_standard", "_mating_wf"):
        original = getattr(sim, name)

        def wrapped(rng, sex, *args, _original=original, **kwargs):
            res = _original(rng, sex, *args, **kwargs)
            cap.gens.append((np.asarray(res[0]).astype(np.int64), np.asarray(res[1]).astype(np.int64).reshape(-1, 2)))
            cap.sex.append(np.asarray(sex).copy())
            return res

        monkeypatch.setattr(sim, name, wrapped)
    return cap


def force_plan(monkeypatch, plan: InbreedingPlan) -> None:
    """Make ``SimulationParams`` resolve every non-off mode to ``plan``, past the handoff and the cost rule."""
    resolve = params_mod.resolve
    monkeypatch.setattr(params_mod, "resolve", lambda mode, *a: resolve(mode, *a) if mode == "off" else plan)


class Dense64:
    """Dense float64 kinship stepped one generation at a time (``notes/issue-76-am/am76.py:72-92``)."""

    def __init__(self, N: int) -> None:
        self.K = np.zeros((N, N))
        self.K[np.diag_indices(N)] = 0.5

    def step(self, parents: np.ndarray, twins: np.ndarray) -> np.ndarray:
        m, f = parents[:, 0], parents[:, 1]
        F = self.K[m, f].copy()
        Kp = self.K[m] + self.K[f]
        self.K = (Kp[:, m] + Kp[:, f]) * 0.25
        self.K[np.diag_indices(len(m))] = 0.5 * (1 + F)
        if len(twins):
            t0, t1 = twins[:, 0], twins[:, 1]
            self.K[t0, t1] = self.K[t1, t0] = self.K[t0, t0]
        return F


def run_engine(plan: InbreedingPlan, gens: list[tuple[np.ndarray, np.ndarray]], mother_sorted: bool) -> np.ndarray:
    """F of every iteration's offspring, shape ``(G, N)``, from the stepper under ``plan``."""
    N = len(gens[0][0])
    inb = Inbreeding(plan, N, len(gens), mother_sorted=mother_sorted)
    return np.stack([inb.step(p, t) for p, t in gens])


def dense64(gens: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    ref = Dense64(len(gens[0][0]))
    return np.stack([ref.step(p, t) for p, t in gens])
