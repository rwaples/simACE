"""Code-policy guardrails that ruff cannot express.

- Every polars CSV/TSV read passes ``null_values``: without it an ``NA``
  cell reads as the string ``"NA"`` and the column silently becomes text
  (ADR 0015, the one silent-data-loss trap).
- Parquet is written only by ``simace.core.parquet.save_parquet``, which
  applies the NaN-to-null contract (ADR 0015 §3). ruff's banned-api rule
  cannot see ``df.write_parquet(...)``, an instance method.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_CSV_READS = {"read_csv", "scan_csv", "read_csv_batched"}
_POLARS_NAMES = {"pl", "polars"}
_PARQUET_WRITES = {"write_parquet", "to_parquet"}
_PYARROW_PARQUET = {"pq", "pyarrow.parquet"}


def _tracked(*pathspecs: str) -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *pathspecs],
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line.endswith(".py")]


def _calls(source: str) -> list[ast.Call]:
    return [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]


def csv_reads_without_null_values(source: str) -> list[int]:
    """Line numbers of polars CSV reads that pass no ``null_values`` (a ``**`` splat counts as passing)."""
    lines = []
    for call in _calls(source):
        func = call.func
        if func.attr in _CSV_READS and isinstance(func.value, ast.Name) and func.value.id in _POLARS_NAMES:
            keywords = {k.arg for k in call.keywords}
            if "null_values" not in keywords and None not in keywords:
                lines.append(call.lineno)
    return lines


def raw_parquet_writes(source: str) -> list[int]:
    """Line numbers of parquet writes that bypass ``save_parquet``."""
    lines = []
    for call in _calls(source):
        func = call.func
        if func.attr in _PARQUET_WRITES or (func.attr == "write_table" and ast.unparse(func.value) in _PYARROW_PARQUET):
            lines.append(call.lineno)
    return lines


def _violations(check, paths: list[Path], exempt: set[str] = frozenset()) -> list[str]:
    found = []
    for path in paths:
        rel = str(path.relative_to(REPO_ROOT))
        if rel not in exempt:
            found += [f"{rel}:{line}" for line in check(path.read_text(encoding="utf-8"))]
    return found


def test_tsv_reads_pass_null_values() -> None:
    assert _violations(csv_reads_without_null_values, _tracked("simace", "tests")) == []


def test_parquet_writes_go_through_save_parquet() -> None:
    exempt = {"simace/core/parquet.py"}
    assert _violations(raw_parquet_writes, _tracked("simace"), exempt) == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('pl.read_csv(p, separator="\\t")', [1]),
        ('pl.scan_csv(p, separator="\\t", null_values=["NA", ""])', []),
        ("pl.read_csv(p, **kwargs)", []),
        ("csv.read_csv(p)", []),
    ],
)
def test_csv_check_flags_reads_without_null_values(source: str, expected: list[int]) -> None:
    assert csv_reads_without_null_values(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("df.write_parquet(p)", [1]),
        ("df.to_parquet(p)", [1]),
        ("pq.write_table(t, p)", [1]),
        ("save_parquet(df, p)", []),
        ("writer.write_table(t)", []),
    ],
)
def test_parquet_check_flags_raw_writes(source: str, expected: list[int]) -> None:
    assert raw_parquet_writes(source) == expected
