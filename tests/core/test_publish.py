from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from simace.core.publish import publish

if TYPE_CHECKING:
    from pathlib import Path


def test_publish_renames_every_temporary_on_success(tmp_path: Path) -> None:
    a, b = tmp_path / "sub" / "a.parquet", tmp_path / "b.yaml"
    with publish(a, b) as (ta, tb):
        assert ta.name == "a.parquet.tmp"
        ta.write_text("A")
        tb.write_text("B")
        assert not a.exists()
    assert a.read_text() == "A"
    assert b.read_text() == "B"
    assert sorted(p.name for p in tmp_path.rglob("*.tmp")) == []


def _failing_stage(out: Path) -> None:
    with publish(out) as (tmp,):
        tmp.write_text("half")
        raise RuntimeError("stage failed")


def test_publish_leaves_no_output_when_the_block_raises(tmp_path: Path) -> None:
    out = tmp_path / "out.yaml"
    out.write_text("old")
    with pytest.raises(RuntimeError):
        _failing_stage(out)
    assert out.read_text() == "old"
    assert list(tmp_path.iterdir()) == [out]


def test_publish_renames_nothing_when_a_temporary_was_not_written(tmp_path: Path) -> None:
    a, b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    a.write_text("old a")
    with pytest.raises(FileNotFoundError, match=r"b\.yaml\.tmp"), publish(a, b) as (ta, _tb):
        ta.write_text("new a")
    assert a.read_text() == "old a"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.yaml"]
