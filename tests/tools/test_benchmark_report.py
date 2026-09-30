from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import pytest

from tools.benchmark.__main__ import main
from tools.benchmark.model import MANIFEST_SCHEMA, RESULTS_SCHEMA, SCHEMA_VERSION, BenchmarkError, read_run, write_json
from tools.benchmark.report import compare_runs

if TYPE_CHECKING:
    from pathlib import Path


def _manifest(run_id: str) -> dict:
    return {
        "schema": {"name": MANIFEST_SCHEMA, "version": SCHEMA_VERSION},
        "run_id": run_id,
        "inputs": {
            "pixi_lock_sha256": "lock",
            "config_files_sha256": {"config/test.yaml": "config"},
            "scenarios": {"small_test": {"parameters_sha256": "params"}},
        },
        "host": {
            "system": "Linux",
            "machine": "x86_64",
            "cpu_model": "test cpu",
            "physical_cores": 4,
            "logical_cores": 8,
        },
        "tools": {"pixi": "1", "python": "3.13", "simace": "1"},
        "execution": {
            "folder": "test",
            "scenarios": ["small_test"],
            "jobs": 1,
            "cache_mode": "warm",
            "sample_interval_seconds": 0.25,
            "thread_environment": {},
            "command": ["python", "-m", "simace", "run", "<scenario>", "--force", "--jobs", "1"],
        },
    }


def _results(run_id: str, wall: float, rss: float, *, status: str = "complete") -> dict:
    return {
        "schema": {"name": RESULTS_SCHEMA, "version": SCHEMA_VERSION},
        "run_id": run_id,
        "status": status,
        "executions": [],
        "summaries": [
            {
                "kind": "pipeline",
                "scenario": "small_test",
                "rule": None,
                "wall_seconds": {"n": 3, "median": wall, "min": wall - 1, "max": wall + 1},
                "peak_rss_kb": {"n": 3, "median": rss, "min": rss - 10, "max": rss + 10},
            }
        ],
    }


def _write_run(path: Path, wall: float, rss: float):
    path.mkdir()
    write_json(path / "manifest.json", _manifest(path.name))
    write_json(path / "results.json", _results(path.name, wall, rss))
    return read_run(path)


def test_compare_fails_when_candidate_exceeds_default_gate(tmp_path: Path):
    baseline = _write_run(tmp_path / "baseline", 100.0, 1000.0)
    candidate = _write_run(tmp_path / "candidate", 106.0, 1040.0)

    comparison = compare_runs(
        baseline,
        candidate,
        time_threshold_percent=5.0,
        memory_threshold_percent=5.0,
        allow_incompatible=False,
    )

    assert comparison.regressions == 1
    assert [row["metric"] for row in comparison.rows if row["regression"]] == ["wall_seconds"]


def test_compare_rejects_incompatible_provenance(tmp_path: Path):
    baseline = _write_run(tmp_path / "baseline", 100.0, 1000.0)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    manifest = deepcopy(_manifest("candidate"))
    manifest["execution"]["jobs"] = 8
    write_json(candidate_dir / "manifest.json", manifest)
    write_json(candidate_dir / "results.json", _results("candidate", 100.0, 1000.0))
    candidate = read_run(candidate_dir)

    with pytest.raises(BenchmarkError, match=r"execution\.jobs"):
        compare_runs(
            baseline,
            candidate,
            time_threshold_percent=5.0,
            memory_threshold_percent=5.0,
            allow_incompatible=False,
        )

    comparison = compare_runs(
        baseline,
        candidate,
        time_threshold_percent=5.0,
        memory_threshold_percent=5.0,
        allow_incompatible=True,
    )
    assert comparison.incompatibilities


def test_compare_rejects_incomplete_run(tmp_path: Path):
    baseline = _write_run(tmp_path / "baseline", 100.0, 1000.0)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    write_json(candidate_dir / "manifest.json", _manifest("candidate"))
    write_json(candidate_dir / "results.json", _results("candidate", 100.0, 1000.0, status="failed"))

    with pytest.raises(BenchmarkError, match="status 'failed'"):
        compare_runs(
            baseline,
            read_run(candidate_dir),
            time_threshold_percent=5.0,
            memory_threshold_percent=5.0,
            allow_incompatible=False,
        )


def test_compare_cli_returns_one_for_regression(tmp_path: Path):
    _write_run(tmp_path / "baseline", 100.0, 1000.0)
    _write_run(tmp_path / "candidate", 106.0, 1000.0)

    status = main(["compare", str(tmp_path / "baseline"), str(tmp_path / "candidate")])

    assert status == 1


def test_compare_cli_rejects_negative_threshold(tmp_path: Path):
    _write_run(tmp_path / "baseline", 100.0, 1000.0)
    _write_run(tmp_path / "candidate", 100.0, 1000.0)

    status = main(
        [
            "compare",
            str(tmp_path / "baseline"),
            str(tmp_path / "candidate"),
            "--time-threshold-percent",
            "-1",
        ]
    )

    assert status == 2
