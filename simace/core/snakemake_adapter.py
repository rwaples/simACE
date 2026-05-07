"""Signature-introspection bridge between Snakemake script wrappers and domain functions.

Snakemake injects a magic ``snakemake`` object into the script's module-level
globals.  Hand-rolled wrappers used to rebuild a ``param_dict`` from
``snakemake.params`` and pass it to a domain function with a ``params: dict``
seam.  After flattening domain signatures to keyword-only parameters, this
adapter introspects the function and pulls each kwarg from
``snakemake.params`` by name — wrappers shrink to one ``run_wrapper(...)``
call plus a ``cli_or_snakemake(...)`` dispatch.

Strict by design: every keyword-only parameter in the domain signature must
be present in ``snakemake.params``.  Drift between a rule's ``params:`` block
and the function's signature fails fast with ``AttributeError``.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from simace import _snakemake_tag, setup_logging

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import pandas as pd

__all__ = ["cli_or_snakemake", "run_wrapper", "write_parquet_plain"]


def write_parquet_plain(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame as parquet without dtype narrowing or compression tweaks.

    Mirrors the prior wrapper behavior of ``df.to_parquet(path, index=False)``,
    distinct from :func:`simace.core.utils.save_parquet` which also calls
    :func:`optimize_dtypes` and uses zstd.
    """
    df.to_parquet(path, index=False)


def run_wrapper(
    snakemake: Any,
    domain_fn: Callable,
    *,
    inputs: dict[str, Callable[[str], Any] | None],
    output: str,
    writer: Callable[[Any, str], None],
) -> None:
    """Run a DataFrame-in/DataFrame-out domain function from a Snakemake script.

    Sets up logging from ``snakemake.log[0]`` tagged with the rule's wildcards,
    loads each input via the supplied loader, introspects the domain
    function's keyword-only parameters and pulls each from
    ``snakemake.params`` by name, then writes the return value with
    ``writer``.

    Args:
        snakemake: the Snakemake-injected magic object from the wrapper script.
        domain_fn: the domain function to call.  Its first positional
            parameters must be the inputs (in the order they appear in
            ``inputs``); remaining parameters must be keyword-only.
        inputs: ``{param_name: loader}`` mapping.  ``loader`` is called on
            ``snakemake.input.<param_name>`` to produce the value passed to
            ``domain_fn``.  A loader of ``None`` passes the raw path string.
        output: name of the ``snakemake.output`` attribute that ``writer``
            should target.
        writer: callable ``(result, path)`` that writes the domain function's
            return value to ``path``.
    """
    setup_logging(log_file=snakemake.log[0], tag=_snakemake_tag(snakemake.wildcards))

    loaded: dict[str, Any] = {}
    for name, loader in inputs.items():
        path = getattr(snakemake.input, name)
        loaded[name] = path if loader is None else loader(path)

    sig = inspect.signature(domain_fn)
    kwargs: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param_name in inputs:
            continue
        if param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[param_name] = getattr(snakemake.params, param_name)

    result = domain_fn(**loaded, **kwargs)
    writer(result, getattr(snakemake.output, output))


def cli_or_snakemake(cli_fn: Callable, wrapper_fn: Callable, caller_globals: dict[str, Any]) -> None:
    """Dispatch to ``wrapper_fn`` under Snakemake or ``cli_fn`` from the shell.

    Snakemake injects a module-level ``snakemake`` global into the wrapper
    script; if absent, the script was launched as a CLI.  Replaces the
    five-line ``try/except NameError`` block at the bottom of every wrapper.

    Args:
        cli_fn: the domain module's ``cli()`` entry point.
        wrapper_fn: the script's ``_run()`` function (calls ``run_wrapper``).
        caller_globals: pass ``globals()`` from the wrapper script.  Required
            because ``snakemake`` is injected into the *script's* namespace,
            not the adapter's.
    """
    if "snakemake" in caller_globals:
        wrapper_fn()
    elif caller_globals.get("__name__") == "__main__":
        cli_fn()
