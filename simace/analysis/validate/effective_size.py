"""Effective-size (Ne) observed-vs-expected validation."""

from typing import Any

from ._common import _result

_NE_TOLERANCE = 0.20  # ±20 % per master plan


def validate_effective_size(ne_stats: dict[str, Any] | None, params: dict[str, Any]) -> dict[str, Any]:
    """Validate Ne observed-vs-expected for the eight estimators.

    Each estimator entry in ``ne_stats`` (as written by
    :func:`simace.analysis.stats.compute_effective_size`) supplies an
    ``expected`` field (``None`` under non-standard configs) and a
    scalar ``ne``.  A check passes when either ``expected`` is ``None``
    (vacuous), or ``abs(ne / expected − 1) < 0.20``.

    Args:
        ne_stats: Loaded ``effective_size.yaml`` dict (estimator-keyed
            mapping).  Returns an empty dict when input is ``None`` or
            empty.
        params: Per-rep params.yaml dict (unused, accepted for parity
            with other validators).

    Returns:
        Dict keyed on estimator name with ``passed`` / ``expected`` /
        ``observed`` / ``details`` fields.
    """
    del params  # accepted for API parity
    es = ne_stats
    if not es:
        return {}
    out: dict[str, Any] = {}
    for name, entry in es.items():
        if not isinstance(entry, dict):
            continue
        expected = entry.get("expected")
        observed = entry.get("ne")
        if expected is None:
            out[name] = _result(
                True,
                f"{name}: no theoretical expectation under this config",
                expected=None,
                observed=None if observed is None else float(observed),
            )
            continue
        if observed is None:
            out[name] = _result(
                False,
                f"{name}: expected {expected:.3g} but observed is None",
                expected=float(expected),
                observed=None,
            )
            continue
        rel_err = abs(observed / expected - 1.0)
        passed = rel_err < _NE_TOLERANCE
        out[name] = _result(
            passed,
            f"{name}: observed {observed:.3g} vs expected {expected:.3g} (rel err {rel_err:.3f}, tol {_NE_TOLERANCE})",
            expected=float(expected),
            observed=float(observed),
            relative_error=float(rel_err),
        )
    return out
