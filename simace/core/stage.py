"""Decorator that lifts the schema-pair contract onto a stage function.

Pipeline stage functions share a contract: ``(DataFrame, **kwargs) ->
DataFrame`` with a known input schema and output schema.  Today every
stage opens with ``assert_schema(input, ...)`` and closes with
``assert_schema(output, ...)`` by hand.  This decorator bracketed the
two asserts and attaches the schema-pair as queryable metadata on the
wrapper, so downstream tooling (DAG diagrams, doc generation, future
stage registry) can read the contract without re-reading the body.

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

    import pandas as pd

__all__ = ["stage"]


def stage(
    *,
    reads: Mapping[str, str] | None,
    writes: Mapping[str, str] | None,
    name: str | None = None,
) -> Callable[[Callable[..., pd.DataFrame]], Callable[..., pd.DataFrame]]:
    """Wrap a stage function with input/output schema assertions and metadata.

    Args:
        reads: schema the first positional argument must satisfy.  ``None``
            skips input validation (e.g. ``run_simulation`` has no input
            DataFrame).
        writes: schema the return value must satisfy.  ``None`` skips
            output validation.
        name: stage label used in ``where=`` strings.  Defaults to the
            wrapped function's ``__name__`` with a leading ``run_``
            stripped (e.g. ``run_phenotype`` → ``"phenotype"``).

    The wrapper exposes ``fn.reads``, ``fn.writes``, and ``fn.stage_name``
    as queryable metadata.  ``functools.wraps`` preserves the wrapped
    function's signature so :func:`simace.core.snakemake_adapter.run_wrapper`
    can still introspect keyword-only parameters.
    """

    def decorate(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        stage_name = name if name is not None else fn.__name__.removeprefix("run_")
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

        wrapper.reads = reads
        wrapper.writes = writes
        wrapper.stage_name = stage_name
        return wrapper

    return decorate
