"""The records every quality-check family returns, and the report rows built from them.

A family returns a :class:`FamilyResult`: the checks it asserts, as
:class:`QualityCheck` records, and the values it measures on the recorded
pedigree, as a family-specific measurements record. The report's
``quality_checks`` rows come from the checks; its ``truth`` and
``estimators`` come from the measurements (ADR 0008).
"""

from __future__ import annotations

__all__ = ["CheckValue", "FamilyResult", "QualityCheck", "check_rows"]

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

#: An observed, expected or tolerance value: a scalar, or one per parent / generation.
type CheckValue = int | float | list[int] | list[float] | None


@dataclass(frozen=True)
class QualityCheck:
    """One quality check: a pass/fail verdict on the recorded pedigree (GLOSSARY "Quality check").

    Values a check reports but does not assert are measurements, not checks:
    each family keeps them on its own measurements record.

    Attributes:
        name: The check's name within its family (the report id is ``family.name``).
        passed: Whether the rep satisfies the check.
        message: One line saying what was compared, with the numbers.
        observed: The value measured on the rep, when the check compares one.
        expected: The value it is compared against.
        tolerance: The allowed distance, when the check states one.
    """

    name: str
    passed: bool
    message: str
    observed: CheckValue = None
    expected: CheckValue = None
    tolerance: CheckValue = None

    def __post_init__(self) -> None:
        # NumPy comparisons yield np.bool_; the record holds a Python bool.
        object.__setattr__(self, "passed", bool(self.passed))


@dataclass(frozen=True)
class FamilyResult[M]:
    """One check family's result: its measurements and the checks it asserts, in check order."""

    measurements: M
    checks: tuple[QualityCheck, ...]

    def find(self, name: str) -> QualityCheck | None:
        """The check called ``name``, or ``None`` when the family did not assert it on this rep."""
        return next((c for c in self.checks if c.name == name), None)

    def check(self, name: str) -> QualityCheck:
        """The check called ``name``; raises ``KeyError`` when the family did not assert it."""
        found = self.find(name)
        if found is None:
            raise KeyError(f"no {name!r} check in {[c.name for c in self.checks]}")
        return found


def check_rows(family: str, checks: Iterable[QualityCheck]) -> list[dict[str, Any]]:
    """The report's ``quality_checks.checks`` rows for one family, in check order."""
    return [
        {
            "id": f"{family}.{c.name}",
            "scope": "recorded_pedigree",
            "severity": "error",
            "status": "pass" if c.passed else "fail",
            "observed": c.observed,
            "expected": c.expected,
            "tolerance": c.tolerance,
            "message": c.message,
        }
        for c in checks
    ]
