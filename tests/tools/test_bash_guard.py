"""``tools/claude_hooks/bash_guard.py``: refuse self-matching ``pkill -f``/``pgrep -f``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/claude_hooks"))

from bash_guard import self_matching

SCRIPT = Path(__file__).resolve().parents[2] / "tools/claude_hooks/bash_guard.py"


@pytest.mark.parametrize(
    "command",
    [
        'pkill -f "run_big.sh"',
        "pgrep -f bench_pair_kinship.py",
        "sleep 1; pkill -9 -f run_big.sh && echo done",
        "until ! pgrep -af pg08_wheel_gate.sh; do sleep 5; done",
        "kill $(pgrep -f scratchpad/depthscale.sh)",
        "pkill --full gate",
        'bash -c "pkill -f run_big.sh"',
        "timeout 5 sh -c 'pkill -f x'",
    ],
)
def test_blocks_self_matching_patterns(command: str) -> None:
    assert self_matching(command) is not None


@pytest.mark.parametrize(
    "command",
    [
        "pkill -f '[r]un_big.sh'",
        'until ! pgrep -f "[p]g08_wheel_gate.sh"; do sleep 5; done',
        "pkill run_big",
        "pgrep -u ryanw python",
        "kill 12345",
        "grep -f patterns.txt file",
        "python3 - <<'EOF'\nnote = 'kill $(pgrep -f <pattern>)'\nEOF\necho ok",
        "cat > a.md <<EOF\npkill -f x\nEOF",
    ],
)
def test_passes_other_commands(command: str) -> None:
    assert self_matching(command) is None


@pytest.mark.parametrize(
    "prefix",
    [
        "cat > a.md <<'EOF'\ntext\nEOF",
        "x=$((1<<2))",
        "x=$((x<<n))",
        "grep x <<< foo",
        'echo "x <<EOF"',
        "echo 'x <<EOF'",
        "echo x  # cat <<EOF",
        "cat <<EOF",
    ],
)
def test_checks_commands_after_a_heredoc_or_lookalike(prefix: str) -> None:
    assert self_matching(f"{prefix}\npkill -f run_big.sh") is not None


def test_hook_exit_codes() -> None:
    def run(command: str) -> int:
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
        return subprocess.run([sys.executable, "-I", str(SCRIPT)], input=payload, text=True, check=False).returncode

    assert run('pkill -f "run_big.sh"') == 2
    assert run("pkill -f '[r]un_big.sh'") == 0
