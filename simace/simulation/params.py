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
from typing import TYPE_CHECKING, Any

import numpy as np

from simace.core.relationships import transmission_coefficients
from simace.simulation.inbreeding_plan import DEFAULT_DEPTH, OFF, resolve

if TYPE_CHECKING:
    from collections.abc import Mapping

    from simace.simulation.inbreeding_plan import InbreedingPlan


@dataclass(frozen=True)
class CTransmission:
    """Vertical transmission coefficients of the common-environment component (ADR 0025).

    ``maternal[t]`` and ``paternal[t]`` are trait ``t+1``'s ``b_m`` and
    ``b_f``: an offspring's ``C`` is ``b_m * C_mother + b_f * C_father``
    plus a household draw. All zero (the default) is today's fresh draw.
    """

    maternal: tuple[float, float] = (0.0, 0.0)
    paternal: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> CTransmission:
        """Read the flat ``Ctm_maternal{t}`` / ``Ctm_paternal{t}`` keys, each absent one as 0 (pre-ADR-0025 params)."""
        (bm1, bf1), (bm2, bf2) = (transmission_coefficients(params, t) for t in (1, 2))
        return cls(maternal=(bm1, bm2), paternal=(bf1, bf2))

    @property
    def active(self) -> bool:
        """Whether any coefficient is nonzero."""
        return any(self.maternal) or any(self.paternal)

    def validate(self, C1: float, C2: float, rC: float) -> None:
        """Reject coefficients outside the supported scope or infeasible for the configured C.

        Each coefficient must lie in [0, 1] and ``b_m**2 + b_f**2 <= 1`` per
        trait (a scope restriction, ADR 0025). Then the random-mating
        population residual ``U = S - B_m S B_m - B_f S B_f`` must be positive
        semidefinite, where ``S`` is the configured C covariance matrix
        (``run_simulation`` accepts scalar C only, so ``S`` is constant).

        Raises:
            ValueError: naming the offending trait or residual.
        """
        for side, coefs in (("maternal", self.maternal), ("paternal", self.paternal)):
            for t, b in enumerate(coefs, start=1):
                if not 0.0 <= b <= 1.0:
                    raise ValueError(f"Ctm_{side}{t} must be in [0, 1], got {b} (ADR 0025)")
        for t, (bm, bf) in enumerate(zip(self.maternal, self.paternal, strict=True), start=1):
            if bm**2 + bf**2 > 1.0 + 1e-12:
                raise ValueError(
                    f"Ctm_maternal{t}**2 + Ctm_paternal{t}**2 = {bm**2 + bf**2:.4g} exceeds 1, "
                    "outside the supported scope (ADR 0025)"
                )
        if not self.active:
            return
        B_m, B_f = np.diag(self.maternal), np.diag(self.paternal)
        S = c_covariance(C1, C2, rC)
        residual = S - B_m @ S @ B_m - B_f @ S @ B_f
        lowest = float(np.linalg.eigvalsh(residual)[0])
        if lowest < -1e-12:
            raise ValueError(
                f"C transmission is infeasible: the fresh C draw would need a covariance with eigenvalue "
                f"{lowest:.4g} < 0 (residual {residual.round(6).tolist()}); lower the coefficients or raise C "
                "(ADR 0025)"
            )


def am_transmitted_share(b_m: float, b_f: float, assort: float, C: float, V_P: float) -> float:
    """First-generation transmitted share of C under assortative mating: ``b_m² + b_f² + 2 b_m b_f r_CC``.

    Mates paired at liability correlation ``assort`` have C correlated at
    ``r_CC = assort * C / V_P`` in the first generation (C is the regression
    of liability's C part on liability, slope ``C / V_P``). Above 1 the
    configured C cannot be the total (Rao, Morton & Cloninger 1979, eq. 2):
    ``_transmitted_c`` stops the run once the mates' C covariance is past
    sampling. Transmission builds up A-C covariance over generations, so the
    equilibrium share is larger still.
    """
    r_cc = assort * C / V_P if V_P > 0 else 0.0
    return b_m**2 + b_f**2 + 2.0 * b_m * b_f * r_cc


def c_covariance(C1: float, C2: float, rC: float) -> np.ndarray:
    """The 2x2 covariance matrix of (C1, C2) for variances ``C1``, ``C2`` and correlation ``rC``."""
    off = rC * np.sqrt(C1 * C2)
    return np.array([[C1, off], [off, C2]])


@dataclass(frozen=True)
class SimulationParams:
    """Validated run_simulation parameters with assort_matrix already resolved.

    Field values are post-normalization: ``G_sim`` defaults to ``G_ped`` when
    unset, and when an ``assort_matrix`` was supplied, ``assort1`` / ``assort2``
    hold its diagonal and ``R_mf`` the matrix itself (else ``R_mf`` is ``None``).
    ``inbreeding`` is the resolved plan of the inbreeding mode (ADR 0033).
    """

    seed: int
    N: int
    G_ped: int
    G_sim: int
    mating_model: str
    mating_lambda: float
    p_mztwin: float
    A1: float
    C1: float
    A2: float
    C2: float
    E1: float | dict[int, float]
    E2: float | dict[int, float]
    rA: float
    rC: float
    rE: float
    assort1: float | dict[int, float]
    assort2: float | dict[int, float]
    R_mf: np.ndarray | None
    c_transmission: CTransmission
    inbreeding: InbreedingPlan = OFF

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
        c_transmission: CTransmission | None = None,
        inbreeding_mode: str = "off",
        inbreeding_depth: int = DEFAULT_DEPTH,
    ) -> SimulationParams:
        """Validate and normalize raw run_simulation inputs.

        Raises:
            ValueError: if any parameter is outside its valid range, or the
                ``assort_matrix`` is malformed / incompatible with per-generation
                assortment.
            InbreedingRefused: when an exact inbreeding run is predicted to
                take too long (ADR 0033 §4).
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

        c_transmission = c_transmission or CTransmission()
        c_transmission.validate(C1, C2, rC)

        inbreeding = resolve(inbreeding_mode, inbreeding_depth, N, G_sim)
        if inbreeding.mode != "off" and N * (G_sim + 1) > np.iinfo(np.int32).max:
            raise ValueError(
                f"Inbreeding history N * (G_sim + 1) = {N * (G_sim + 1):,} exceeds int32 max "
                f"({np.iinfo(np.int32).max:,}). Reduce N or G_sim."
            )

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
            c_transmission=c_transmission,
            inbreeding=inbreeding,
        )
