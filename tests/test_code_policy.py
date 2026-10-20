"""Code-policy guardrails that ruff cannot express.

- Every polars CSV/TSV read passes ``null_values``: without it an ``NA``
  cell reads as the string ``"NA"`` and the column silently becomes text
  (ADR 0015, the one silent-data-loss trap).
- Parquet is written only by ``simace.core.parquet.save_parquet``, which
  applies the NaN-to-null contract (ADR 0015 §3). ruff's banned-api rule
  cannot see ``df.write_parquet(...)``, an instance method.
- Data outputs are written through ``simace.core.publish`` (ADR 0020): a
  module that opens a file for writing must also call ``publish``, so the
  write lands on a temporary that is renamed into place. Plots and the atlas
  are exempt (rebuilt on every run), as are the few non-output writes below.
  The check is per module, not per write: simACE's idiom is a domain
  function that writes the path it is given, called by its ``cli()`` with a
  ``publish`` temporary, so one ``publish`` call in a module clears it.
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
_TEXT_WRITES = {"write_text", "write_bytes", "write_csv", "write_json", "write_ndjson", "dump_yaml"}

# Files that write without publish, and why.
_RAW_WRITE_ALLOWED = {
    "simace/cli/launch.py": "stage logs and timing.tsv rows are appended as the stage runs",
    "simace/cli/cgroups.py": "writes cgroup control files in /sys/fs/cgroup",
    "simace/core/yaml_io.py": "dump_yaml writes whatever path its caller passes, a publish temporary",
}


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


def raw_writes(source: str) -> list[int]:
    """Line numbers of calls that open or write a file for output."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in _TEXT_WRITES:
            lines.append(node.lineno)
        elif name == "open":
            # open(p, "w") and gzip.open(p, "w") take the mode second; Path.open("w") takes it first.
            modes = [*node.args[:2], *(k.value for k in node.keywords if k.arg == "mode")]
            if any(
                isinstance(m, ast.Constant) and isinstance(m.value, str) and set(m.value) & set("wax+") for m in modes
            ):
                lines.append(node.lineno)
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


def test_data_writes_go_through_publish() -> None:
    found = []
    for path in _tracked("simace"):
        rel = str(path.relative_to(REPO_ROOT))
        if rel.startswith("simace/plotting/") or rel in _RAW_WRITE_ALLOWED:
            continue
        source = path.read_text(encoding="utf-8")
        if "publish(" not in source:
            found += [f"{rel}:{line}" for line in raw_writes(source)]
    assert found == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('open(p, "w")', [1]),
        ('gzip.open(p, mode="wb")', [1]),
        ("p.write_text(s)", [1]),
        ("df.write_csv(p)", [1]),
        ('path.open("w")', [1]),
        ('open(p, "r+")', [1]),
        ("dump_yaml(d, p)", [1]),
        ("open(p)", []),
        ('open(p, "rb")', []),
    ],
)
def test_write_check_flags_output_writes(source: str, expected: list[int]) -> None:
    assert raw_writes(source) == expected


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
