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
| `.claude/settings.json`, `settings.local.json` | yes | – | – | Permissions and the post-edit `ruff check` hook. |
| `.pi/extensions/plan-mode/` | – | – | yes | Project-local read-only plan mode (`/plan`, `--plan`). |
| `skills-lock.json` | – | – | – | Provenance for the `mattpocock/skills` entries in `.agents/skills/`; used by the `skills` CLI to update them. |

Codex needs no project-local directory: its settings live in `~/.codex/config.toml`
and it discovers `AGENTS.md` and `.agents/skills/` on its own.

## Conventions

- Skills are authored once in `.agents/skills/` and never duplicated per harness.
- No MCP servers are configured at the project level. The former `code-review-graph`
  server and its hooks were removed in September 2026; do not reintroduce `.mcp.json`.
- Repo-wide rules for agents (issue tracker, triage labels, domain docs) live alongside
  this file in `docs/agents/`.
