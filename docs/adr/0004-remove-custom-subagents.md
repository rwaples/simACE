# ADR 0004: Remove Custom Subagent Definitions

## Status

Accepted.

## Context

`.claude/agents/` contained six custom subagent definitions intended to
specialize Claude Code's work on this repo: `reviewer`, `agent-review`,
`analyst`, `critic`, `pipeline`, and `test`.

Usage telemetry from 92 session logs in
`~/.claude/projects/-data-Documents-simACE/` showed almost none of them were
being invoked:

| Agent | Invocations | Purpose |
|-------|-------------|---------|
| `reviewer` | 12 | Code review for statistical correctness in pedigree/variance code |
| `agent-review` | 0 | Meta-agent that audits other agents for staleness |
| `analyst` | 0 | Validate pipeline outputs (heritability, prevalence, censoring) |
| `critic` | 0 | Scientific critique of methodology assumptions |
| `pipeline` | 0 | Run/debug Snakemake pipeline |
| `test` | 0 | Run pytest, ruff, snakefmt |

Built-in agents over the same period: `Explore` (108), `general-purpose`
(67), `Plan` (14). The specialized agents were being bypassed.

Across all six definitions, the "ACE conda env is always active" instruction
had become stale (the active env is `simACE`), and file-path references in
`analyst`, `pipeline`, and `test` had drifted with the recent
`simace/{core,simulation,phenotype,...}` sub-package refactor.

## Decision

Remove all six agent files. Future work is delegated to the built-in
`general-purpose`, `Explore`, and `Plan` agents, or handled by the main
thread directly.

Per-agent rationale:

- **`reviewer`** — the only one with non-zero usage (12 calls). Removed
  anyway because its specialised value (domain knowledge of kinship
  formulas, multiplicity-before-booleanisation bugs, cross-package coupling
  to `fit_ace`) belongs in `CLAUDE.md` (now migrated under "Code review
  gotchas"), not duplicated in an agent prompt. Subagent isolation also
  fragments review context from the implementing thread.
- **`agent-review`** — meta-agent whose job is to audit other agents. With
  no other custom agents to audit, it has nothing to do.
- **`analyst`** — pipeline-output validation. The trigger condition
  ("after a pipeline run") rarely occurs in interactive Claude sessions;
  when output inspection does happen, the main thread does it directly
  with Read/grep.
- **`critic`** — academic-reviewer-style methodology critique. Trigger
  ("evaluating a new analysis method") is too narrow for ordinary code
  changes. Domain knowledge of ACE assumptions belongs in `CONTEXT.md` /
  ADRs, not an agent prompt.
- **`pipeline`** — Snakemake operations. The main thread runs them
  directly; delegating to a subagent loses streaming output and adds
  isolation overhead with no benefit.
- **`test`** — `pytest`/`ruff`/`snakefmt`. Quick one-shots that the main
  thread runs directly; delegation adds latency and isolates failure
  context from the implementing thread.

## Consequences

- Code review work continues via `general-purpose` or the main thread.
- The domain knowledge from `reviewer.md` (kinship formulas, known bug
  patterns, cross-package coupling notes) has been migrated into CLAUDE.md
  under "Code review gotchas (statistical correctness)".
- No automation of "audit the agents" — irrelevant now since there are no
  custom agents.
- Stale `ACE`-vs-`simACE` env references and outdated file paths in agent
  prompts can no longer drift further.

## Non-goals

- No removal of skills (`.claude/skills/`) or commands (`.claude/commands/`).
- No removal of built-in or third-party agent integrations.
