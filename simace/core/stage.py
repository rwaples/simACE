"""Decorator that lifts the schema-pair contract onto a stage function.

Pipeline stage functions share a contract: ``(DataFrame, **kwargs) ->
DataFrame`` with a known input schema and output schema.  This decorator
brackets the stage body with ``assert_schema`` on its input and its output.

Scope: DataFrame stages only.  Non-DataFrame helpers (e.g. ``emit_params``)
are not decorated.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING

from simace.core.schema import assert_schema

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import polars as pl

    # Polars-only (ADR 0015). assert_schema
    # enforces both frame type and eagerness at every stage boundary.
    type _Frame = pl.DataFrame

__all__ = ["stage"]


def stage(
    *,
    reads: Mapping[str, str] | None,
    writes: Mapping[str, str] | None,
) -> Callable[[Callable[..., _Frame]], Callable[..., _Frame]]:
    """Wrap a stage function with input/output schema assertions.

    Args:
        reads: schema the first positional argument must satisfy.  ``None``
            skips input validation (e.g. ``run_simulation`` has no input
            DataFrame).
        writes: schema the return value must satisfy.  ``None`` skips
            output validation.

    Errors name the stage by the wrapped function's ``__name__`` with a
    leading ``run_`` stripped (``run_phenotype`` → ``"phenotype"``).
    ``functools.wraps`` preserves the wrapped function's signature for
    callers that introspect keyword-only parameters.
    """

    def decorate(fn: Callable[..., _Frame]) -> Callable[..., _Frame]:
        stage_name = fn.__name__.removeprefix("run_")  # ty: ignore[unresolved-attribute]
        first_param = next(iter(inspect.signature(fn).parameters), None) if reads is not None else None

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if reads is not None:
                if args:
                    df = args[0]
                elif first_param is not None and first_param in kwargs:
                    df = kwargs[first_param]
                else:
                    raise TypeError(f"{stage_name}: stage with reads={reads!r} called with no input DataFrame")
                assert_schema(df, reads, where=f"{stage_name} input")
            result = fn(*args, **kwargs)
            if writes is not None:
                assert_schema(result, writes, where=f"{stage_name} output")
            return result

        return wrapper

    return decorate
