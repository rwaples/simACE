# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --json title,body,state,labels,comments`.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone. Pass `-R <owner>/<repo>` anyway for a nested repo: the agent's shell often resets to the simACE root, and an issue meant for pedigree-graph once landed in simACE.

`gh` comes from `pixi global install gh` (`~/.pixi/bin/gh`). The apt build at `/usr/bin/gh` (2.45) fails `gh issue view` and `gh pr edit` with "Projects (classic) is being deprecated"; if `gh --version` shows 2.45, fix PATH rather than working around it.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --json title,body,state,labels,comments`.
