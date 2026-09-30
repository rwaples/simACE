"""The ``simace`` umbrella dispatches to each command's ``cli(argv, prog)``."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys

import pandas as pd
import pytest

from simace.cli import COMMANDS, main


@pytest.mark.parametrize("name", sorted(COMMANDS))
def test_every_command_resolves_to_a_cli_taking_argv_and_prog(name: str) -> None:
    command = COMMANDS[name]
    entry = getattr(importlib.import_module(command.module), command.func)
    assert list(inspect.signature(entry).parameters) == ["argv", "prog"]


def test_help_lists_every_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert all(name in out for name in COMMANDS)


def test_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["no-such-stage"])
    assert exc.value.code == 2


def test_dispatch_passes_the_remaining_arguments(tmp_path) -> None:
    pq = tmp_path / "in.parquet"
    pd.DataFrame({"id": [1, 2], "x": [0.5, 1.5]}).to_parquet(pq)
    main(["parquet-to-tsv", str(pq), "--no-gzip"])
    assert (tmp_path / "in.tsv").read_text().splitlines()[0] == "id\tx"


def test_python_dash_m_simace_reports_the_command_prog() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "simace", "censor", "--help"], capture_output=True, text=True, check=True
    )
    assert result.stdout.startswith("usage: simace censor")
