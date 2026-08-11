"""Pytest plugin: rehearse pandas 3 semantics while still on pandas 2.x.

Enables the two forward-compatibility options that make pandas 2.x behave like
pandas 3 for everything this family does with DataFrames:

``mode.copy_on_write = True``
    Full Copy-on-Write. Arrays returned by ``.values`` / ``.to_numpy()`` become
    **read-only**, and chained assignment no longer propagates to the parent.

    ``"warn"`` is *not* equivalent and must not be substituted: it retains
    legacy 2.x semantics and only warns on chained assignment, leaving the
    read-only view behavior untested. With ~570 ``.values``/``.to_numpy()``
    call sites across the family, that is the dominant pandas 3 risk surface.

``future.infer_string = True``
    Inferred string columns use the dedicated ``str`` dtype backed by PyArrow,
    so ``Series.values`` returns an extension array and Parquet writes
    ``large_string`` rather than ``string``.

Usage, from any family repo root::

    PYTHONPATH=/data/Documents/simACE/tools pytest -p pandas3_rehearsal

The options are applied at import time *and* again in ``pytest_configure`` so
that xdist workers (several repos use ``-n auto``) are covered. A session-scoped
autouse fixture asserts both are live and errors the run if either is not, so a
green suite can never mean "the switch silently did not apply".

This module is a validation harness for the pandas 3 migration
(``plans/pandas-3-migration.md``, Phase 2). It is not imported by any package
and has no effect unless explicitly requested with ``-p``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import pytest

#: Options this plugin forces on, and the value each must hold.
REHEARSAL_OPTIONS: dict[str, object] = {
    "mode.copy_on_write": True,
    "future.infer_string": True,
}


def _apply() -> None:
    """Set every rehearsal option. Safe to call more than once."""
    for option, value in REHEARSAL_OPTIONS.items():
        pd.set_option(option, value)


def _live() -> dict[str, object]:
    """Return the current value of each rehearsal option."""
    return {option: pd.get_option(option) for option in REHEARSAL_OPTIONS}


# Applied at import so the options are live before conftest collection creates
# any DataFrame; reapplied in pytest_configure for xdist workers.
_apply()


def pytest_configure(config: pytest.Config) -> None:
    """Reapply the options in the master process and in each xdist worker."""
    _apply()


def pytest_report_header() -> str:
    """Record the active configuration in the run log, for auditability."""
    live = " ".join(f"{name}={value!r}" for name, value in _live().items())
    return f"pandas3-rehearsal: pandas={pd.__version__} {live}"


try:  # pragma: no cover - fixture definition, exercised by every run
    import pytest

    @pytest.fixture(scope="session", autouse=True)
    def _pandas3_rehearsal_active() -> None:
        """Fail the session if a rehearsal option is not actually in effect.

        Guards the case where the plugin loads in the master process but not in
        an xdist worker: without this, those tests would run under legacy
        pandas 2 semantics and pass misleadingly.
        """
        stale = {
            name: (value, REHEARSAL_OPTIONS[name])
            for name, value in _live().items()
            if value != REHEARSAL_OPTIONS[name]
        }
        if stale:
            detail = ", ".join(f"{n}: got {got!r}, want {want!r}" for n, (got, want) in stale.items())
            raise AssertionError(f"pandas3_rehearsal options not in effect ({detail})")
except ImportError:  # pragma: no cover - pytest always present when plugin loads
    pass
