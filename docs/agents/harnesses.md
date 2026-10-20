# LLM coding-agent harnesses

simACE supports exactly three coding-agent harnesses: **Claude Code**, **Codex**, and
**Pi**. Configuration for any other harness (Gemini, Kiro, Qoder, OpenCode, Copilot,
VS Code MCP) is unsupported and should not be added.

All harness configuration is machine-local and gitignored except the two files every
harness reads: `CLAUDE.md` and its tracked symlink `AGENTS.md`.

## What each harness reads

| Location | Claude Code | Codex | Pi | Notes |
|---|---|---|---|---|
| `CLAUDE.md` / `AGENTS.md` | yes | yes (`AGENTS.md`) | yes (`AGENTS.md`) | Single source of truth; `AGENTS.md -> CLAUDE.md` symlink is tracked. |
| `.agents/skills/<name>/SKILL.md` | via symlinks | yes | yes | Canonical skill tree. Author skills here. |
| `.claude/skills/<name>` | yes | – | – | Symlinks into `.agents/skills/`; `ln -s ../../.agents/skills/<name> .claude/skills/<name>` after adding a skill. |
| `.claude/settings.json`, `settings.local.json` | yes | – | – | Permissions, and the `PreToolUse` Bash guard `tools/claude_hooks/bash_guard.py` (refuses a self-matching `pkill -f`/`pgrep -f`). |
| `.pi/extensions/plan-mode/` | – | – | yes | Project-local read-only plan mode (`/plan`, `--plan`). |

Codex needs no project-local directory: its settings live in `~/.codex/config.toml`
and it discovers `AGENTS.md` and `.agents/skills/` on its own.

## Conventions

- Git hooks are tracked in `tools/githooks/` (pre-commit lint, format, ty and lock
  gates; commit-msg trailer ban; pre-push docs build). Every family checkout's
  `core.hooksPath` points at the main checkout's copy:
  `git -C <checkout> config core.hooksPath /data/Documents/simACE/tools/githooks`.
  Lint runs at commit, not after each edit.
- Skills are authored once in `.agents/skills/` and never duplicated per harness.
- A project skill wanted in other projects too gets a user-level symlink
  (`ln -s <simACE>/.agents/skills/<name> ~/.agents/skills/<name>`), never a copy.
  On a name clash Claude Code runs the personal skill and Codex the project one, so
  a copy silently diverges between harnesses.
- `grill-me`, `grill-with-docs`, `handoff`, and `improve-codebase-architecture` began
  as `mattpocock/skills` and are now maintained here as forks. Don't update them
  with the `skills` CLI; upstream has been restructured and would overwrite them.
  `codebase-design` and `domain-modeling` were split out of those forks, `smelly`
  was reworked from upstream `code-review`, and `research`,
  `to-questionnaire`, `wait-what`, and `claude-handoff` (Claude Code only) adapt their
  upstream namesakes; `writing-for-agents` is an unmodified copy.
  Last compared against upstream `dd400c3` on 2026-10-07; diff a fork against that
  commit to separate local edits from upstream drift. `retro` was added later from
  upstream `5beb35b` (2026-10-20), adapted to this repo's steering files, session
  logs, guardrails, and reviewers; its Codex `agents/openai.yaml` was dropped.
- No MCP servers are configured at the project level. The former `code-review-graph`
  server and its hooks were removed in September 2026; do not reintroduce `.mcp.json`.
- Repo-wide rules for agents (issue tracker, triage labels, domain docs) live alongside
  this file in `docs/agents/`.
