"""PhenotypeModel ABC.

Each phenotype model family is a frozen dataclass that owns:

  * its typed parameter fields (validated in ``__post_init__``);
  * a ``_from_config(phenotype_params, trait_num, beta=, beta_sex=)`` hook
    that converts an already-checked copy of ``phenotype_params{trait_num}``
    into an instance. Callers use the inherited, final ``from_config(params,
    trait_num)``, which rejects keys the model does not accept (its
    ``config_keys``) before the hook runs;
  * a ``simulate(...)`` method (the actual phenotype draw).

Validation error messages are prefixed with the trait number by ``from_config``,
so ``_from_config`` raises unprefixed errors; the dataclass
itself is trait-agnostic so it can be unit-tested in isolation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Self, final

if TYPE_CHECKING:
    import numpy as np

    from simace.phenotype.hazards import StandardizeMode

__all__ = [
    "PhenotypeModel",
    "check_finite_beta",
    "emit_standardize_hazard",
    "validate_standardize_hazard",
    "wrap_trait_error",
]


def check_finite_beta(beta: float) -> None:
    """Raise ``ValueError`` if beta is non-finite (NaN or +/-inf)."""
    if not math.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta}")


def validate_standardize_hazard(value: StandardizeMode | None) -> None:
    """Validate the per-trait ``standardize_hazard`` override (``None`` = inherit)."""
    if value is None:
        return
    from simace.phenotype.hazards import coerce_standardize_mode

    coerce_standardize_mode(value)


def emit_standardize_hazard(out: dict[str, Any], value: StandardizeMode | None) -> None:
    """Write ``standardize_hazard`` into ``out`` only when set (non-``None``)."""
    if value is not None:
        out["standardize_hazard"] = value


class PhenotypeModel(ABC):
    """Abstract base class for phenotype model families.

    Subclasses are frozen dataclasses; their fields are the model's typed
    parameters. Validation happens in ``__post_init__``.
    """

    name: ClassVar[str]
    config_keys: ClassVar[frozenset[str]]
    """The ``phenotype_params{N}`` keys this model accepts."""

    @classmethod
    def _accepted_keys(cls, phenotype_params: dict[str, Any]) -> frozenset[str]:
        """The keys ``phenotype_params`` may hold; override when they depend on its values."""
        return cls.config_keys

    @final
    @classmethod
    def from_config(cls, params: dict[str, Any], trait_num: int) -> Self:
        """Build an instance from the simulation parameter dict for trait ``trait_num``.

        Reads ``phenotype_params{trait_num}``, ``beta{trait_num}``, and
        ``beta_sex{trait_num}`` (default 0.0) from ``params``.
        """
        with wrap_trait_error(trait_num):
            phenotype_params = dict(params.get(f"phenotype_params{trait_num}", {}))
            allowed = cls._accepted_keys(phenotype_params)
            unknown = set(phenotype_params) - allowed
            if unknown:
                raise ValueError(
                    f"phenotype_params{trait_num} for model {cls.name!r} has unknown key(s) "
                    f"{sorted(unknown)}; valid: {sorted(allowed)}"
                )
            if f"beta{trait_num}" not in params:
                raise ValueError(f"beta{trait_num} is not set")
            return cls._from_config(
                phenotype_params,
                trait_num,
                beta=params[f"beta{trait_num}"],
                beta_sex=params.get(f"beta_sex{trait_num}", 0.0),
            )

    @classmethod
    @abstractmethod
    def _from_config(cls, phenotype_params: dict[str, Any], trait_num: int, *, beta: float, beta_sex: float) -> Self:
        """Convert a key-checked copy of ``phenotype_params{trait_num}`` into an instance.

        ``phenotype_params`` holds only keys this model accepts and may be
        mutated. Raise ``ValueError`` or ``TypeError`` without trait context;
        ``from_config`` adds it.
        """

    @abstractmethod
    def simulate(
        self,
        liability: np.ndarray,
        *,
        seed: int,
        standardize: StandardizeMode | bool,
        sex: np.ndarray,
        generation: np.ndarray,
    ) -> np.ndarray:
        """Simulate event times for one trait.

        ``standardize`` is the global liability-standardization mode
        (``"none" | "global" | "per_generation"``); legacy ``True``/``False``
        bools are accepted via the same shim used at config-load.

        Concrete subclasses route ``standardize`` differently:

        * Threshold-style models (``threshold``, ``adult.ltm``) consume it
          directly to standardize liability before the threshold comparison.
        * Hazard-bearing models (``frailty``, ``cure_frailty``,
          ``first_passage``, ``adult.cox``) carry an additional per-trait
          field ``standardize_hazard`` (defaulting to ``None`` → inherit) and
          resolve it via :func:`simace.phenotype.hazards.resolve_hazard_mode`
          for the hazard step. ``cure_frailty`` is the only model that
          honors both knobs (threshold step + hazard step).
        """


def wrap_trait_error(trait_num: int):
    """Context manager that prefixes ``ValueError``/``TypeError`` with trait context.

    Usage::

        with wrap_trait_error(trait_num):
            return cls(distribution=..., ...)

    Catches errors raised by dataclass ``__post_init__`` (and ``__init__``
    type errors from missing required fields) and reraises them with a
    ``"phenotype.trait{trait_num}: ..."`` prefix so the originating trait
    is visible in the message.
    """
    return _TraitErrorContext(trait_num)


class _TraitErrorContext:
    __slots__ = ("trait_num",)

    def __init__(self, trait_num: int) -> None:
        self.trait_num = trait_num

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        if issubclass(exc_type, (ValueError, TypeError)):
            raise ValueError(f"phenotype.trait{self.trait_num}: {exc_val}") from exc_val
        return False
