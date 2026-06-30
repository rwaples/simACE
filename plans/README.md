# plans/ — working plan drafts

Scratch space for plans and design-interview output. **Everything in this
directory except this README is gitignored** — drafts live here without
polluting git history.

## Conventions

- One plan per file, kebab-case slug: `plans/<slug>.md`.
- **Never overwrite an existing plan file** — if the slug exists, add a `-v2`
  (etc.) suffix or confirm before replacing.
- When a plan is written, state its **absolute path** in chat.
- Promote a *finalized* plan to `docs/plans/` (tracked, published via mkdocs)
  when it should become part of the record; an ADR in `docs/adr/` is the home
  for locked architectural decisions.
