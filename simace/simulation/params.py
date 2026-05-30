"""Validated, normalized parameter record for :func:`run_simulation`.

``run_simulation`` doubles as a public test API, so it cannot rely on
config-load validation having already run. :class:`SimulationParams` owns the
up-front input validation and normalization that used to sit inline at the top
of ``run_simulation``:

  * range / type checks for the scalar variance components, correlations, and
    population/generation counts;
  * standard-model-only checks (``mating_lambda``, ``p_mztwin``, assortment
    ranges) that are no-ops under Wright-Fisher;
  * ``assort_matrix`` resolution — shape/symmetry validation plus extraction of
    the diagonal into ``assort1`` / ``assort2`` and the matrix into ``R_mf``.

Per-generation *resolution* (forward-filling dict-valued params) and all
assortative-mating planning live elsewhere (``resolve_per_gen_param`` and
:class:`simace.simulation.assortment.AssortmentPlan`); this class only
validates and normalizes the raw inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationParams:
    """Validated run_simulation parameters with assort_matrix already resolved.

    Field values are post-normalization: ``G_sim`` defaults to ``G_ped`` when
    unset, and when an ``assort_matrix`` was supplied, ``assort1`` / ``assort2``
    hold its diagonal and ``R_mf`` the matrix itself (else ``R_mf`` is ``None``).
    """

    seed: int
    N: int
    G_ped: int
    G_sim: int
    mating_model: str
    mating_lambda: float
    p_mztwin: float
    A1: float
    C1: float | dict[int, float]
    A2: float
    C2: float | dict[int, float]
    E1: float | dict[int, float]
    E2: float | dict[int, float]
    rA: float
    rC: float
    rE: float
    assort1: float | dict[int, float]
    assort2: float | dict[int, float]
    R_mf: np.ndarray | None

    @classmethod
    def create(
        cls,
        *,
        seed: int,
        N: int,
        G_ped: int,
        mating_lambda: float,
        p_mztwin: float,
        A1: float,
        C1: float,
        A2: float,
        C2: float,
        rA: float,
        rC: float,
        rE: float,
        E1: float | dict[int, float],
        E2: float | dict[int, float],
        G_sim: int | None = None,
        assort1: float | dict[int, float] = 0.0,
        assort2: float | dict[int, float] = 0.0,
        assort_matrix: list[list[float]] | np.ndarray | None = None,
        mating_model: str = "standard",
    ) -> SimulationParams:
        """Validate and normalize raw run_simulation inputs.

        Raises:
            ValueError: if any parameter is outside its valid range, or the
                ``assort_matrix`` is malformed / incompatible with per-generation
                assortment.
        """
        if G_sim is None:
            G_sim = G_ped

        # --- mating_model allowed-value check (gates standard-only validation
        # below — config-load enforces this too, but run_simulation is also a
        # public API exercised by tests) ---
        from simace.config import VALID_MATING_MODELS

        if mating_model not in VALID_MATING_MODELS:
            raise ValueError(f"mating_model must be one of {sorted(VALID_MATING_MODELS)}, got {mating_model!r}")

        # --- Input validation ---
        for name, val in [("A1", A1), ("C1", C1), ("A2", A2), ("C2", C2)]:
            if not (isinstance(val, (int, float)) and val >= 0):
                raise ValueError(f"{name} must be a non-negative scalar, got {val}")

        if not (int(N) == N and N > 0):
            raise ValueError(f"N must be a positive integer, got {N}")
        if not (G_ped == int(G_ped) and G_ped >= 1):
            raise ValueError(f"G_ped must be an integer >= 1, got {G_ped}")
        if not (-1 <= rA <= 1):
            raise ValueError(f"rA must be in [-1, 1], got {rA}")
        if not (-1 <= rC <= 1):
            raise ValueError(f"rC must be in [-1, 1], got {rC}")
        if not (-1 <= rE <= 1):
            raise ValueError(f"rE must be in [-1, 1], got {rE}")

        # Standard-model-only validation: mating_lambda, p_mztwin, assort* are
        # no-ops under WF (each offspring picks parents independently → no ZTP,
        # no twin-eligible matings, no AM by construction). Skip these checks
        # so callers can pass inherited defaults under WF without pre-zeroing.
        R_mf = None
        if mating_model == "standard":
            if not (mating_lambda > 0):
                raise ValueError(f"mating_lambda must be > 0, got {mating_lambda}")
            if not (0 <= p_mztwin < 1):
                raise ValueError(f"p_mztwin must be in [0, 1), got {p_mztwin}")

            # assort1/assort2 may be scalar or per-gen dict (raw-iter keyed,
            # forward-filled). Validate each before resolving.
            def _validate_assort_value(name: str, value: float | dict[int, float]) -> None:
                if isinstance(value, dict):
                    for k, v in value.items():
                        if not (-1 <= float(v) <= 1):
                            raise ValueError(f"{name}[{k}] must be in [-1, 1], got {v}")
                elif not (-1 <= float(value) <= 1):
                    raise ValueError(f"{name} must be in [-1, 1], got {value}")

            _validate_assort_value("assort1", assort1)
            _validate_assort_value("assort2", assort2)

            # Per-gen assort dicts are incompatible with a fixed `assort_matrix`
            # because the matrix specifies one off-diagonal cross-AM, while per-gen
            # AM implies the cross-AM also varies per generation. v1 rejects this
            # combination — users wanting a fixed cross-AM must use scalar
            # assort1/assort2.
            if assort_matrix is not None and (isinstance(assort1, dict) or isinstance(assort2, dict)):
                raise ValueError(
                    "assort_matrix is incompatible with per-generation assort1/assort2 "
                    "(dict-valued). Pass scalar assort1/assort2 with assort_matrix, or "
                    "use per-gen dicts and let cross-AM auto-compute from rho_w."
                )

            # Resolve assort_matrix (standard-only — WF has no AM)
            if assort_matrix is not None:
                R_mf = np.asarray(assort_matrix, dtype=np.float64)
                if R_mf.shape != (2, 2):
                    raise ValueError(f"assort_matrix must be 2x2, got shape {R_mf.shape}")
                if abs(R_mf[0, 1] - R_mf[1, 0]) > 1e-10:
                    raise ValueError(f"assort_matrix must be symmetric: got [{R_mf[0, 1]}, {R_mf[1, 0]}]")
                # Scalar assort1/assort2 override from the matrix diagonal.
                assort1 = float(R_mf[0, 0])
                assort2 = float(R_mf[1, 1])
                if not (-1 <= assort1 <= 1):
                    raise ValueError(f"assort_matrix[0,0] must be in [-1, 1], got {assort1}")
                if not (-1 <= assort2 <= 1):
                    raise ValueError(f"assort_matrix[1,1] must be in [-1, 1], got {assort2}")

        if G_sim < G_ped:
            raise ValueError(f"G_sim ({G_sim}) must be >= G_ped ({G_ped})")

        return cls(
            seed=seed,
            N=N,
            G_ped=G_ped,
            G_sim=G_sim,
            mating_model=mating_model,
            mating_lambda=mating_lambda,
            p_mztwin=p_mztwin,
            A1=A1,
            C1=C1,
            A2=A2,
            C2=C2,
            E1=E1,
            E2=E2,
            rA=rA,
            rC=rC,
            rE=rE,
            assort1=assort1,
            assort2=assort2,
            R_mf=R_mf,
        )
