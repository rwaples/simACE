"""Reading ``simace validate``'s files: each carries a required ``schema`` block, and a file without it is refused.

``validation.yaml`` (per rep) and ``validation_summary.yaml`` (per scenario and
folder) both carry ``schema: {kind, version}`` (ADR 0028 §5). ``results/``
still holds ``validation.yaml`` files from before ADR 0008 and from ADR 0027's
first onset-τ files, whose ``summary.passed`` would otherwise read as a
verdict; :func:`load_schemad` keeps them out of every reader.
"""

from __future__ import annotations

__all__ = ["load_schemad", "one_warning_for_refused"]

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from simace.core.yaml_io import load_yaml

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

#: The refused files of the innermost :func:`one_warning_for_refused` by file name, or None outside one.
_refused: ContextVar[dict[str, list[Path]] | None] = ContextVar("_refused", default=None)


def load_schemad(path: Path, schema: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    """The YAML file at ``path``, or None when it does not exist or its ``schema`` block is not ``schema``.

    ``name`` is the file's name in the results layout, for the messages. A
    refused file is logged by name, so a reader that skips it says why;
    inside :func:`one_warning_for_refused` it is counted instead.
    """
    if not path.exists():
        return None
    data = load_yaml(path)
    found = data.get("schema") if isinstance(data, dict) else None
    if found == schema:
        return data
    refused = _refused.get()
    if refused is None:
        logger.warning("Ignoring %s: schema %s is not %s; rerun `simace validate`", path, found, dict(schema))
    else:
        refused.setdefault(f"{name} file(s) without schema {dict(schema)}", []).append(path)
    return None


@contextmanager
def one_warning_for_refused() -> Generator[None]:
    """Log one warning per file name for every file :func:`load_schemad` refuses inside: their count and the first."""
    refused: dict[str, list[Path]] = {}
    token = _refused.set(refused)
    try:
        yield
    finally:
        _refused.reset(token)
        for what, paths in refused.items():
            logger.warning("Ignoring %d %s, first %s; rerun `simace validate`", len(paths), what, paths[0])
