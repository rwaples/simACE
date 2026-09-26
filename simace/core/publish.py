"""Atomic publication of stage outputs."""

from __future__ import annotations

__all__ = ["TMP_SUFFIX", "publish"]

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

TMP_SUFFIX = ".tmp"


@contextmanager
def publish(*paths: str | Path) -> Iterator[tuple[Path, ...]]:
    """Yield a ``<path>.tmp`` for each path, renamed onto the path on success.

    A stage writes its outputs to the yielded temporary paths. When the block
    exits normally every temporary is ``os.replace``d onto its final path, so
    no single output is ever seen half written. The renames happen one after
    another, so a set of outputs is not replaced as a unit; a rep's
    ``run.yaml`` is what marks the whole set complete. When the block raises,
    or leaves a temporary unwritten, nothing is renamed and the temporaries are
    removed. A process killed mid-write leaves only ``.tmp`` files, which
    ``simace run`` clears before recomputing a rep.

    Raises:
        FileNotFoundError: the block exited without writing every temporary.

    Args:
        paths: final output paths. Parent directories are created.

    Yields:
        The temporary paths, in the order given.
    """
    finals = [Path(p) for p in paths]
    temps = tuple(p.with_name(p.name + TMP_SUFFIX) for p in finals)
    for final in finals:
        final.parent.mkdir(parents=True, exist_ok=True)
    try:
        yield temps
    except BaseException:
        for temp in temps:
            temp.unlink(missing_ok=True)
        raise
    missing = [temp for temp in temps if not temp.exists()]
    if missing:
        for temp in temps:
            temp.unlink(missing_ok=True)
        raise FileNotFoundError(f"stage exited without writing {', '.join(map(str, missing))}")
    for temp, final in zip(temps, finals, strict=True):
        os.replace(temp, final)
