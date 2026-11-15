"""The scenario parameter records, each the one owner of its keys' reading.

Parameters reach simACE through several boundaries: config load, a rep's
``params.yaml``, ``report.yaml``'s ``inputs.parameters``, a gathered TSV row,
and fitACE's Snakemake projection. Each boundary parses its mapping once into
the records below, and code past the boundary takes records, never the
mapping:

  * :class:`Span`: the recorded and simulated generation counts;
  * :class:`VarianceComponents`: A, C and E per trait and the cross-trait
    correlations;
  * :class:`Assortment`: the mating model and the per-iteration assortment
    schedule it mates on, with the only assortative-mating predicates;
  * :class:`CTransmission`: the C transmission coefficients (ADR 0025);
  * :class:`SimulationParams`: the whole simulator input, composed of the
    four and validated up front. ``params.yaml`` is this record plus
    provenance, so the analysis parses it with
    :meth:`SimulationParams.from_mapping`.

A record validates its own invariants when built. ``from_mapping`` reads the
flat keys and supplies the defaults of keys an older ``params.yaml`` lacks
(:data:`LEGACY_DEFAULTS`); every other key is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np

from simace.core.relationships import transmission_key
from simace.simulation.inbreeding_plan import DEFAULT_DEPTH, OFF, resolve

if TYPE_CHECKING:
    from collections.abc import Mapping

    from simace.simulation.inbreeding_plan import InbreedingPlan

MatingModel = Literal["standard", "wright_fisher"]

#: Allowed values for ``pedigree.mating_model``.
VALID_MATING_MODELS: frozenset[str] = frozenset(get_args(MatingModel))

#: A per-generation parameter: a scalar, or a map from generation (mating
#: iteration) to value, forward-filled (:func:`resolve_per_gen_param`).
PerGeneration = float | dict[int, float]

#: Defaults of the keys a ``params.yaml`` written before them lacks: the
#: mating model (ADR 0002), C transmission (ADR 0025) and inbreeding (ADR 0033).
LEGACY_DEFAULTS: dict[str, Any] = {
    "mating_model": "standard",
    **{transmission_key(side, t): 0.0 for t in (1, 2) for side in ("maternal", "paternal")},
    "inbreeding_mode": "off",
    "inbreeding_depth": DEFAULT_DEPTH,
}


def _legacy(m: Mapping[str, Any], key: str) -> Any:
    value = m.get(key)
    return LEGACY_DEFAULTS[key] if value is None else value


@dataclass(frozen=True)
class Span:
    """The generations of a simulation: ``G_ped`` recorded, ``G_sim`` simulated, the first ``burnin`` discarded."""

    G_ped: int
    G_sim: int

    def __post_init__(self) -> None:
        if not (self.G_ped == int(self.G_ped) and self.G_ped >= 1):
            raise ValueError(f"G_ped must be an integer >= 1, got {self.G_ped}")
        if self.G_sim < self.G_ped:
            raise ValueError(f"G_sim ({self.G_sim}) must be >= G_ped ({self.G_ped})")

    @classmethod
    def create(cls, G_ped: int, G_sim: int | None = None) -> Span:
        """``G_sim`` unset means no burn-in."""
        return cls(G_ped, G_ped if G_sim is None else G_sim)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> Span:
        """Read ``G_ped`` and ``G_sim`` (null or absent: ``G_ped``)."""
        G_sim = m.get("G_sim")
        return cls.create(int(m["G_ped"]), None if G_sim is None else int(G_sim))

    @property
    def burnin(self) -> int:
        """Mating iterations before the first recorded generation's: iteration ``i`` records generation ``i - burnin``."""
        return self.G_sim - self.G_ped


def _is_scalar(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


@dataclass(frozen=True)
class VarianceComponents:
    """Trait ``t``'s variances ``A[t-1]``, ``C[t-1]``, ``E[t-1]`` and the cross-trait correlations.

    A and C are scalars (the simulator holds them constant); E may be a
    per-generation map.
    """

    A: tuple[float, float]
    C: tuple[float, float]
    E: tuple[PerGeneration, PerGeneration]
    rA: float
    rC: float
    rE: float

    def __post_init__(self) -> None:
        for name, values in (("A", self.A), ("C", self.C)):
            for t, val in enumerate(values, start=1):
                if not (_is_scalar(val) and val >= 0):
                    raise ValueError(f"{name}{t} must be a non-negative scalar, got {val}")
        for t, e in enumerate(self.E, start=1):
            resolve_per_gen_param(e, 1, name=f"E{t}")
        for name, r in (("rA", self.rA), ("rC", self.rC), ("rE", self.rE)):
            if not (-1 <= r <= 1):
                raise ValueError(f"{name} must be in [-1, 1], got {r}")

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> VarianceComponents:
        """Read ``A{t}``, ``C{t}``, ``E{t}``, ``rA``, ``rC`` and ``rE``."""
        return cls(
            A=(m["A1"], m["A2"]),
            C=(m["C1"], m["C2"]),
            E=(m["E1"], m["E2"]),
            rA=m["rA"],
            rC=m["rC"],
            rE=m["rE"],
        )

    def E_per_gen(self, t: int, G: int) -> list[float]:
        """Trait ``t``'s E in each of ``G`` generations."""
        return resolve_per_gen_param(self.E[t - 1], G, name=f"E{t}")

    def scalar_E(self, t: int) -> bool:
        """Whether trait ``t``'s E is one value for every generation."""
        return not isinstance(self.E[t - 1], dict)

    def V_P(self, t: int) -> float | None:
        """Trait ``t``'s liability variance ``A + C + E``; None for a per-generation E, which has no single one."""
        E = self.E[t - 1]
        if isinstance(E, dict):
            return None
        return float(self.A[t - 1]) + float(self.C[t - 1]) + float(E)


def _validate_assort(name: str, value: PerGeneration) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            if not (-1 <= float(v) <= 1):
                raise ValueError(f"{name}[{k}] must be in [-1, 1], got {v}")
    elif not (-1 <= float(value) <= 1):
        raise ValueError(f"{name} must be in [-1, 1], got {value}")


@dataclass(frozen=True)
class Assortment:
    """What the simulator mates on: its mating model and the assortment of each of its ``G_sim`` mating iterations.

    ``schedule[t-1][i]`` is the trait-``t`` mate liability correlation of
    iteration ``i``. A set ``assort_matrix`` replaces ``assort1`` and
    ``assort2`` by its diagonal, and Wright-Fisher pairs parents at random
    whatever assortment it inherits, so its schedule is all zero and it has no
    matrix. ``matrix`` is the validated ``assort_matrix`` the standard model
    mates on (None when unset).
    """

    mating_model: MatingModel
    schedule: tuple[tuple[float, ...], tuple[float, ...]]
    matrix: tuple[tuple[float, float], tuple[float, float]] | None = None

    @classmethod
    def create(
        cls,
        span: Span,
        assort1: PerGeneration = 0.0,
        assort2: PerGeneration = 0.0,
        assort_matrix: list[list[float]] | np.ndarray | None = None,
        mating_model: str = "standard",
    ) -> Assortment:
        """Validate the configured assortment and resolve the schedule mated on.

        Raises:
            ValueError: for an unknown mating model, an assortment outside
                [-1, 1], a malformed ``assort_matrix``, or a matrix combined
                with a per-generation map. Wright-Fisher ignores assortment,
                so it validates none of it.
        """
        if mating_model not in VALID_MATING_MODELS:
            raise ValueError(f"mating_model must be one of {sorted(VALID_MATING_MODELS)}, got {mating_model!r}")
        if mating_model != "standard":
            return cls("wright_fisher", ((0.0,) * span.G_sim, (0.0,) * span.G_sim))
        _validate_assort("assort1", assort1)
        _validate_assort("assort2", assort2)
        if assort_matrix is None:
            return cls(
                "standard",
                (
                    tuple(resolve_per_gen_param(assort1, span.G_sim, name="assort1", allow_negative=True)),
                    tuple(resolve_per_gen_param(assort2, span.G_sim, name="assort2", allow_negative=True)),
                ),
            )
        # The matrix fixes one cross-trait term, which per-generation assortment would vary.
        if isinstance(assort1, dict) or isinstance(assort2, dict):
            raise ValueError(
                "assort_matrix is incompatible with per-generation assort1/assort2 "
                "(dict-valued). Pass scalar assort1/assort2 with assort_matrix, or "
                "use per-gen dicts and let cross-AM auto-compute from rho_w."
            )
        R = np.asarray(assort_matrix, dtype=np.float64)
        if R.shape != (2, 2):
            raise ValueError(f"assort_matrix must be 2x2, got shape {R.shape}")
        if abs(R[0, 1] - R[1, 0]) > 1e-10:
            raise ValueError(f"assort_matrix must be symmetric: got [{R[0, 1]}, {R[1, 0]}]")
        for t in (0, 1):
            if not (-1 <= R[t, t] <= 1):
                raise ValueError(f"assort_matrix[{t},{t}] must be in [-1, 1], got {float(R[t, t])}")
        matrix = ((float(R[0, 0]), float(R[0, 1])), (float(R[1, 0]), float(R[1, 1])))
        return cls("standard", ((matrix[0][0],) * span.G_sim, (matrix[1][1],) * span.G_sim), matrix)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any], span: Span) -> Assortment:
        """Read ``mating_model`` (absent: standard), ``assort1``, ``assort2`` and ``assort_matrix``."""
        return cls.create(
            span,
            m["assort1"],
            m["assort2"],
            m.get("assort_matrix"),
            _legacy(m, "mating_model"),
        )

    def active(self, t: int) -> bool:
        """Whether trait ``t`` assorts in some mating iteration."""
        return any(self.schedule[t - 1])

    @property
    def any_active(self) -> bool:
        """Whether either trait assorts in some mating iteration; False is random mating throughout."""
        return self.active(1) or self.active(2)

    def constant(self, t: int) -> bool:
        """Whether trait ``t`` assorts by one value in every mating iteration."""
        return len(set(self.schedule[t - 1])) <= 1

    @property
    def mated_matrix(self) -> np.ndarray | None:
        """The ``assort_matrix`` the simulator mates on, as an array; None when unset."""
        return None if self.matrix is None else np.array(self.matrix)


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
    def from_mapping(cls, m: Mapping[str, Any]) -> CTransmission:
        """Read the flat ``Ctm_maternal{t}`` / ``Ctm_paternal{t}`` keys, each absent one as 0 (pre-ADR-0025 params)."""
        bm1, bf1, bm2, bf2 = (
            float(_legacy(m, transmission_key(side, t))) for t in (1, 2) for side in ("maternal", "paternal")
        )
        return cls(maternal=(bm1, bm2), paternal=(bf1, bf2))

    def coefficients(self, t: int) -> tuple[float, float]:
        """Trait ``t``'s ``(b_m, b_f)``."""
        return self.maternal[t - 1], self.paternal[t - 1]

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
    """The validated input of one simulation.

    ``mating_lambda`` and ``p_mztwin`` are validated under the standard model
    only: Wright-Fisher ignores them. ``inbreeding`` is the resolved plan of
    the inbreeding mode (ADR 0033).
    """

    seed: int
    N: int
    span: Span
    mating_lambda: float
    p_mztwin: float
    vc: VarianceComponents
    assortment: Assortment
    c_transmission: CTransmission
    inbreeding: InbreedingPlan = OFF

    @property
    def mating_model(self) -> MatingModel:
        """The mating model, ``standard`` or ``wright_fisher``."""
        return self.assortment.mating_model

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
        E1: PerGeneration,
        E2: PerGeneration,
        G_sim: int | None = None,
        assort1: PerGeneration = 0.0,
        assort2: PerGeneration = 0.0,
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
        if mating_model not in VALID_MATING_MODELS:
            raise ValueError(f"mating_model must be one of {sorted(VALID_MATING_MODELS)}, got {mating_model!r}")
        vc = VarianceComponents(A=(A1, A2), C=(C1, C2), E=(E1, E2), rA=rA, rC=rC, rE=rE)
        if not (int(N) == N and N > 0):
            raise ValueError(f"N must be a positive integer, got {N}")
        span = Span.create(G_ped, G_sim)
        # Wright-Fisher has no ZTP mating counts and no twins, so these are no-ops there.
        if mating_model == "standard":
            if not (mating_lambda > 0):
                raise ValueError(f"mating_lambda must be > 0, got {mating_lambda}")
            if not (0 <= p_mztwin < 1):
                raise ValueError(f"p_mztwin must be in [0, 1), got {p_mztwin}")
        assortment = Assortment.create(span, assort1, assort2, assort_matrix, mating_model)

        c_transmission = c_transmission or CTransmission()
        c_transmission.validate(C1, C2, rC)

        inbreeding = resolve(inbreeding_mode, inbreeding_depth, N, span.G_sim)
        if inbreeding.mode != "off" and N * (span.G_sim + 1) > np.iinfo(np.int32).max:
            raise ValueError(
                f"Inbreeding history N * (G_sim + 1) = {N * (span.G_sim + 1):,} exceeds int32 max "
                f"({np.iinfo(np.int32).max:,}). Reduce N or G_sim."
            )
        return cls(seed, N, span, mating_lambda, p_mztwin, vc, assortment, c_transmission, inbreeding)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> SimulationParams:
        """Parse a flat parameter mapping: a resolved scenario, a ``params.yaml``, or a report's ``inputs.parameters``.

        Keys an older ``params.yaml`` lacks take :data:`LEGACY_DEFAULTS`.
        """
        return cls.create(
            seed=m["seed"],
            N=m["N"],
            G_ped=m["G_ped"],
            G_sim=m.get("G_sim"),
            mating_lambda=m["mating_lambda"],
            p_mztwin=m["p_mztwin"],
            A1=m["A1"],
            C1=m["C1"],
            A2=m["A2"],
            C2=m["C2"],
            rA=m["rA"],
            rC=m["rC"],
            rE=m["rE"],
            E1=m["E1"],
            E2=m["E2"],
            assort1=m["assort1"],
            assort2=m["assort2"],
            assort_matrix=m.get("assort_matrix"),
            mating_model=_legacy(m, "mating_model"),
            c_transmission=CTransmission.from_mapping(m),
            inbreeding_mode=_legacy(m, "inbreeding_mode"),
            inbreeding_depth=_legacy(m, "inbreeding_depth"),
        )


def resolve_per_gen_param(
    value: PerGeneration,
    G: int,
    name: str = "param",
    allow_negative: bool = False,
) -> list[float]:
    """Resolve a variance-component parameter to a per-generation list.

    Args:
        value: scalar (constant across all generations) or dict mapping
               generation index → value.  Missing generation keys are
               forward-filled from the most recent earlier key.
        G: total number of generations to resolve for (indices 0..G-1).
        name: parameter name, used in error messages.
        allow_negative: if False (default), values must be ≥ 0. Set True for
            parameters with a natural negative range (e.g. assortative-mating
            correlations).

    Returns:
        List of length *G* with the resolved value for each generation.

    Raises:
        ValueError: if any resolved value is negative (and allow_negative is
            False) or if a dict has no key <= 0 (so generation 0 would be
            undefined).
    """
    if isinstance(value, (int, float)):
        v = float(value)
        if not allow_negative and v < 0:
            raise ValueError(f"{name} must be >= 0, got {v}")
        return [v] * G

    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a scalar or dict, got {type(value).__name__}")

    if not value:
        raise ValueError(f"{name} dict must not be empty")

    # Sort keys and validate
    sorted_keys = sorted(value)
    if sorted_keys[0] > 0:
        raise ValueError(
            f"{name} dict must have a key <= 0 so generation 0 is defined; smallest key is {sorted_keys[0]}"
        )
    if not allow_negative:
        for k in sorted_keys:
            v = float(value[k])
            if v < 0:
                raise ValueError(f"{name}[{k}] must be >= 0, got {v}")

    # Forward-fill
    result = [0.0] * G
    key_idx = 0
    current_val = float(value[sorted_keys[0]])
    for gen in range(G):
        # Advance to the latest key <= gen
        while key_idx + 1 < len(sorted_keys) and sorted_keys[key_idx + 1] <= gen:
            key_idx += 1
            current_val = float(value[sorted_keys[key_idx]])
        result[gen] = current_val
    return result
