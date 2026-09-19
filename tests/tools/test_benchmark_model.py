from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from tools.benchmark.model import (
    MANIFEST_SCHEMA,
    RESULTS_SCHEMA,
    SCHEMA_VERSION,
    BenchmarkError,
    build_summaries,
    read_run,
    write_json,
)

if TYPE_CHECKING:
    from pathlib import Path


def _execution(scenario: str, wall: float, rss: int, rule_wall: float) -> dict:
    return {
        "phase": "measured",
        "status": "complete",
        "scenario": scenario,
        "wall_seconds": wall,
        "peak_summed_rss_kb": rss,
        "max_individual_rss_kb": rss - 10,
        "gnu_time_max_rss_kb": rss - 5,
        "rules": {
            "simulate": {
                "peak_rss_kb": rss // 2,
                "snakemake": [{"wall_seconds": rule_wall, "max_rss_mb": 10.0}],
            }
        },
    }


def test_build_summaries_uses_only_complete_measured_executions():
    executions = [
        _execution("small", 10.0, 1000, 2.0),
        _execution("small", 12.0, 1200, 4.0),
        _execution("small", 11.0, 1100, 3.0),
        {**_execution("small", 99.0, 9999, 99.0), "phase": "warmup"},
        {**_execution("small", 99.0, 9999, 99.0), "status": "failed"},
    ]

    pipeline, rule = build_summaries(executions)

    assert pipeline["wall_seconds"] == {"n": 3, "median": 11.0, "min": 10.0, "max": 12.0}
    assert pipeline["peak_rss_kb"]["median"] == 1100
    assert rule["rule"] == "simulate"
    assert rule["wall_seconds"]["median"] == 3.0
    assert rule["peak_rss_kb"]["n"] == 3


def test_read_run_rejects_schema_mismatch(tmp_path: Path):
    write_json(
        tmp_path / "manifest.json",
        {"schema": {"name": MANIFEST_SCHEMA, "version": 99}, "run_id": "run"},
    )
    write_json(
        tmp_path / "results.json",
        {
            "schema": {"name": RESULTS_SCHEMA, "version": SCHEMA_VERSION},
            "run_id": "run",
            "status": "complete",
            "executions": [],
            "summaries": [],
        },
    )

    with pytest.raises(BenchmarkError, match="unsupported schema"):
        read_run(tmp_path)


def test_write_json_replaces_document_atomically(tmp_path: Path):
    path = tmp_path / "value.json"
    write_json(path, {"value": 1})
    write_json(path, {"value": 2})

    assert json.loads(path.read_text()) == {"value": 2}
    assert list(tmp_path.iterdir()) == [path]
