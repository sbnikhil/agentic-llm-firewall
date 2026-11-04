# Rules for working in this repo

## Scope
- docs/ROADMAP.md is project context, not a task list. Do only the task given in the current message. If something from a later phase seems needed, stop and ask instead of doing it.
- One task per session. Before editing, propose a plan and wait for my approval.

## Git
- Work only on main. Never create branches, commit, push, force-push, rebase, or delete branches. I do all git operations.
- When a task is done, list the changed files and suggest a commit message: one line, imperative, under 72 characters.
- Never add Co-Authored-By, Claude-Session, or "Generated with Claude Code" lines anywhere.

## Reporting
- After finishing, write a short report: what changed, why, commands run and their output, anything you could not do.
- Explain every non-trivial change in plain language so I can explain it in a code review.

## Code changes
- Refactors must not change behavior. Characterization tests must pass before and after.
- Never delete, skip, or weaken a test to make the suite pass. Never edit expected values, thresholds, golden data, eval data, or detector rules to make a test pass. If a test fails, stop and explain why.
- No new dependencies without asking me first.

## Tests
- Unit tests must run offline: mock databases, model servers, and network model downloads.
- Known bugs are recorded as tests marked `xfail(strict=True)` describing the correct behavior, with a reason string. Do not fix them unless the task says so.

## Security
- Never print or commit secrets. Report only whether a variable is set.
- Security decisions (auth, block rules, thresholds) are mine. You may write tests and scaffolding for them; ask before implementing the logic.
