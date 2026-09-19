from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tools.benchmark import runner
from tools.benchmark.model import MANIFEST_SCHEMA, SCHEMA_VERSION, BenchmarkError, read_run
from tools.benchmark.runner import CommandFailed, RunConfig, _create_output, _parse_time_file, scenario_orders

if TYPE_CHECKING:
    from pathlib import Path


def _config(output: Path) -> RunConfig:
    return RunConfig(
        folder="test",
        scenarios=("one", "two", "three"),
        profile=None,
        repeats=3,
        cores=4,
        cache_mode="warm",
        order_seed=7,
        sample_interval_seconds=0.25,
        output=output,
        cli_argv=("benchmark", "run"),
    )


def test_scenario_order_is_seeded_and_rotated():
    warmup, measured = scenario_orders(("one", "two", "three"), 3, 7)

    assert measured[0] == warmup
    assert measured[1] == warmup[1:] + warmup[:1]
    assert measured[2] == warmup[2:] + warmup[:2]


def test_output_path_must_not_exist(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(BenchmarkError, match="already exists"):
        _create_output(_config(output), {"short_commit": "abc"})


@pytest.mark.parametrize(("elapsed", "expected"), [("8:15.44", 495.44), ("1:02:03", 3723.0)])
def test_parse_gnu_elapsed_time(tmp_path: Path, elapsed: str, expected: float):
    path = tmp_path / "time.txt"
    path.write_text(
        f"\tElapsed (wall clock) time (h:mm:ss or m:ss): {elapsed}\n\tMaximum resident set size (kbytes): 1234\n",
        encoding="utf-8",
    )

    parsed = _parse_time_file(path)

    assert parsed["elapsed_seconds"] == pytest.approx(expected)
    assert parsed["max_rss_kb"] == 1234


def test_failed_command_leaves_schema_valid_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / "failed-run"
    config = _config(output)
    planned = [{"phase": "measured", "repetition": 1, "order": 1, "scenario": "one"}]
    manifest = {
        "schema": {"name": MANIFEST_SCHEMA, "version": SCHEMA_VERSION},
        "run_id": output.name,
        "execution": {"planned": planned},
    }
    monkeypatch.setattr(runner, "_preflight", lambda _config: None)
    monkeypatch.setattr(runner, "_git_provenance", lambda: {"short_commit": "abc"})
    monkeypatch.setattr(runner, "_resolve_inputs", lambda _config: ({}, {}))
    monkeypatch.setattr(runner, "scenario_orders", lambda *_args: ([], []))
    monkeypatch.setattr(runner, "_build_manifest", lambda *_args: manifest)
    monkeypatch.setattr(
        runner,
        "_run_execution",
        lambda *_args, **_kwargs: {
            "phase": "measured",
            "scenario": "one",
            "status": "failed",
            "returncode": 9,
            "artifacts": {"log": "runs/failed.log"},
        },
    )

    with pytest.raises(CommandFailed) as error:
        runner.run_benchmark(config)

    assert error.value.returncode == 9
    result = read_run(output).results
    assert result["status"] == "failed"
    assert result["error"]["type"] == "CommandFailed"
