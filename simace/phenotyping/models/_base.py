"""PhenotypeModel ABC.

Each phenotype model family is a frozen dataclass that owns:

  * its typed parameter fields (validated in ``__post_init__``);
  * a ``from_config(params, trait_num)`` constructor (reads the per-trait
    config dict ``phenotype_params{trait_num}`` plus shared ``beta{N}`` /
    ``beta_sex{N}``);
  * a ``from_cli(args, trait)`` constructor (reads namespaced argparse
    flags and rejects flags belonging to other model families);
  * an ``add_cli_args(parser, trait)`` classmethod (declares its flags);
  * a ``simulate(...)`` method (the actual phenotype draw);
  * a ``to_params_dict()`` method (back to the dict shape Snakemake stores).

CLI flag naming convention:

  * Model-specific flags use the family name as a kebab-case prefix and
    the trait number as a numeric suffix with no separating dash:
    ``--frailty-rho1``, ``--cure-frailty-distribution2``,
    ``--first-passage-drift1``.
  * Multi-token parameter names retain a separating dash before the trait
    number for readability: ``--adult-cip-x0-1``, ``--adult-cip-k-1``.
  * Un-namespaced shared flags (used by every model): ``--phenotype-model{N}``,
    ``--beta{N}``, ``--beta-sex{N}``. Defined directly in ``phenotype.cli``.
  * Prevalence is namespaced because only adult / cure_frailty accept it
    (``--adult-prevalence{N}``, ``--cure-frailty-prevalence{N}``).

Validation error messages are prefixed with the trait number by ``from_config``
and ``from_cli``; the dataclass itself is trait-agnostic so it can be unit-tested
in isolation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Self

if TYPE_CHECKING:
    import argparse

    import numpy as np

    from simace.phenotyping.hazards import StandardizeMode

__all__ = ["PhenotypeModel", "check_finite_beta", "check_no_foreign_flags", "wrap_trait_error"]


def check_finite_beta(beta: float) -> None:
    """Raise ``ValueError`` if beta is non-finite (NaN or +/-inf)."""
    if not math.isfinite(beta):
        raise ValueError(f"beta must be finite, got {beta}")


class PhenotypeModel(ABC):
    """Abstract base class for phenotype model families.

    Subclasses are frozen dataclasses; their fields are the model's typed
    parameters. Validation happens in ``__post_init__``.
    """

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def from_config(cls, params: dict[str, Any], trait_num: int) -> Self:
        """Build an instance from the simulation parameter dict for trait ``trait_num``."""

    @classmethod
    @abstractmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser, trait: int) -> None:
        """Register this model's CLI flags on ``parser`` for the given trait."""

    @classmethod
    @abstractmethod
    def from_cli(cls, args: argparse.Namespace, trait: int) -> Self:
        """Build an instance from parsed CLI args for the given trait.

        Must raise ``ValueError`` (with trait context) if flags belonging
        to another family are populated alongside this model's selection,
        or if a required flag for this model is missing.
        """

    @classmethod
    @abstractmethod
    def cli_flag_attrs(cls, trait: int) -> set[str]:
        """Return the argparse attribute names this model owns for the given trait.

        Used by sibling models' ``from_cli`` to detect foreign-flag conflicts.
        """

    @abstractmethod
    def to_params_dict(self) -> dict[str, Any]:
        """Round-trip back to the per-trait ``phenotype_params{N}`` dict shape."""

    @abstractmethod
    def simulate(
        self,
        liability: np.ndarray,
        *,
        seed: int,
        standardize: StandardizeMode | bool,
        sex: np.ndarray | None,
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
          resolve it via :func:`simace.phenotyping.hazards.resolve_hazard_mode`
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


def check_no_foreign_flags(cls: type[PhenotypeModel], args: argparse.Namespace, trait: int) -> None:
    """Raise ``ValueError`` if any other model's CLI flags are populated for this trait.

    Foreign flags are detected by intersecting populated argparse attributes
    against every sibling model's ``cli_flag_attrs(trait)`` (minus this
    model's own attrs to allow shared flags). Default value is ``None``; any
    flag that ``argparse`` left as ``None`` is treated as not supplied.
    """
    from simace.phenotyping.models import MODELS  # local to avoid import cycle

    own = cls.cli_flag_attrs(trait)
    for other in MODELS.values():
        if other is cls:
            continue
        for attr in other.cli_flag_attrs(trait) - own:
            if getattr(args, attr, None) is not None:
                raise ValueError(
                    f"phenotype.trait{trait}: --{attr.replace('_', '-')} belongs to "
                    f"--phenotype-model{trait}={other.name} but selected model is {cls.name!r}"
                )


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
