"""Liability standardization modes.

Kept free of numba and numpy so config resolution can validate
``standardize`` without importing the phenotype stack.
"""

from __future__ import annotations

__all__ = ["STANDARDIZE_CHOICES", "StandardizeMode", "coerce_standardize_mode"]

from typing import Literal, cast

STANDARDIZE_CHOICES: tuple[str, ...] = ("none", "global", "per_generation")
StandardizeMode = Literal["none", "global", "per_generation"]
_VALID_STD_MODES: frozenset[str] = frozenset(STANDARDIZE_CHOICES)


def coerce_standardize_mode(value: object) -> StandardizeMode:
    """Resolve a user-supplied standardize value to one of the canonical modes.

    Accepts the legacy bool form (``True`` → ``"global"``, ``False`` → ``"none"``)
    or one of the three string modes. Raises ``ValueError`` otherwise.
    """
    if isinstance(value, bool):
        return "global" if value else "none"
    if isinstance(value, str) and value in _VALID_STD_MODES:
        return cast("StandardizeMode", value)
    raise ValueError(f"standardize must be one of {sorted(_VALID_STD_MODES)} or bool; got {value!r}")
