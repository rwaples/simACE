#!/usr/bin/env python3
"""Claude Code PreToolUse hook: refuse Bash commands that match their own command line.

``pkill -f PAT`` and ``pgrep -f PAT`` search every process's full command line,
including the shell the Bash tool runs the command in, which contains ``PAT``
too. ``pkill`` then kills that shell (exit 144) and a ``pgrep`` waiter never
sees its target finish. The pattern ``'[r]un_big.sh'`` still matches
``run_big.sh`` but not itself, so a pattern with a ``[`` passes. Heredoc
bodies are skipped: they are file contents or another program's input, and
agents write such text into docs and scripts.

Reads the hook JSON on stdin; exit 2 with a reason on stderr blocks the call.
Wired in ``.claude/settings.json``; see docs/agents/harnesses.md.
"""

from __future__ import annotations

import json
import re
import shlex
import sys

#: A heredoc operator; group 2 is its terminator word. Not a here-string (``<<<``), and the word
#: starts with a letter or ``_``, so an arithmetic shift (``$((1<<2))``) does not match.
HEREDOC_RE = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_]\w*)\1")

#: A pkill/pgrep invocation up to the end of its simple command. A leading quote opens a
#: ``sh -c '...'`` script, whose text is also on the outer shell's command line.
CALL_RE = re.compile(r"(?:^|[\s;&|(`$'\"])(pkill|pgrep)\s+([^;&|()`\n]*)")


def full_match_flag(args: list[str]) -> bool:
    """Whether ``args`` turn on full-command-line matching (``-f``, ``-af``, ``--full``)."""
    return any(a == "--full" or (a.startswith("-") and not a.startswith("--") and "f" in a) for a in args)


def code_positions(line: str, quote: str | None) -> tuple[list[bool], str | None]:
    """Which characters of ``line`` are shell code, not quoted text or a comment.

    ``quote`` is the quote left open by the previous line; the second value is the one this line leaves open.
    """
    code = [False] * len(line)
    i = 0
    while i < len(line):
        c = line[i]
        if quote == "'":
            quote = None if c == "'" else quote
        elif quote == '"':
            if c == "\\":
                i += 1
            elif c == '"':
                quote = None
        elif c == "\\":
            i += 1
        elif c in "'\"":
            quote = c
        elif c == "#" and (i == 0 or line[i - 1] in " \t;&|("):
            break
        else:
            code[i] = True
        i += 1
    return code, quote


def strip_heredocs(command: str) -> str:
    """``command`` without its heredoc bodies (the lines after a ``<<WORD`` line, through ``WORD``).

    Only an unquoted, uncommented ``<<WORD`` whose ``WORD`` line follows counts, so the guard
    fails closed on lookalikes such as ``echo "x <<EOF"`` or an unterminated heredoc.
    """
    lines = command.split("\n")
    kept: list[str] = []
    terminators: list[str] = []
    quote: str | None = None
    for n, line in enumerate(lines):
        if terminators:
            if line.strip() == terminators[0]:
                terminators.pop(0)
            continue
        kept.append(line)
        code, quote = code_positions(line, quote)
        # A shift by a variable (``$((x<<n))``) sits inside an open ``((``.
        terminators = [
            m.group(2)
            for m in HEREDOC_RE.finditer(line)
            if code[m.start()]
            and line[: m.start()].count("((") <= line[: m.start()].count("))")
            and any(rest.strip() == m.group(2) for rest in lines[n + 1 :])
        ]
    return "\n".join(kept)


def self_matching(command: str) -> str | None:
    """The first ``pkill -f``/``pgrep -f`` call in ``command`` whose pattern would match itself, else ``None``."""
    for m in CALL_RE.finditer(strip_heredocs(command)):
        try:
            args = shlex.split(m.group(2))
        except ValueError:
            args = m.group(2).split()
        patterns = [a for a in args if not a.startswith("-")]
        if full_match_flag(args) and patterns and "[" not in patterns[-1]:
            return f"{m.group(1)} {m.group(2).strip()}"
    return None


def main() -> int:
    """Hook entry point."""
    command = json.load(sys.stdin).get("tool_input", {}).get("command", "")
    call = self_matching(command)
    if call is None:
        return 0
    print(
        f"`{call}` matches this Bash call's own command line, so it would kill or wait on its own shell. "
        "Bracket one character of the pattern ('[r]un_big.sh'), or kill the pid recorded from $! at launch.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
