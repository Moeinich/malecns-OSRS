---
description: Delegate an implementation to a subagent — pick the route, write the task, check the result
---

Delegate this task: $ARGUMENTS

Implementation is not written by whoever planned it. See `## Implementation is delegated` in
`AGENTS.md` for when the rule applies to you and what is exempt.

## 1. Pick the route

The harness subagent is the only route: `coder` when the work is scoped and has to be written,
`debugger` when a failure's cause is unknown. `Explore` and `Plan` are read-only built-ins for
search and design questions — they take no `FILES:` and they are not lanes.

Pick the model and the effort per lane and state both: `opus` where there is reasoning, `sonnet`
for mechanical work with a clear spec, `haiku` for lookups. The per-call model overrides the
agent file's default.

## 2. Write the task

```
[PLAN: <absolute path>] | FILES: <disjoint list> | SCOPE: <what> | GOAL: <acceptance criterion>
```

`FILES:` is the **transitive** closure — everything the lane may touch. Two lanes in the same file
overwrite each other; there is no isolation.

`PLAN:` is optional and carries an **absolute** path — the plan-mode file under
`~/.claude/plans/<slug>.md` when the session has one. It exists so context shared by several lanes
sits in one place instead of being retyped into each task. One lane and no plan file: leave it out
and put the context in `SCOPE:`. Writing the same paragraph into a second lane is the signal that
you needed it. The file is outside the repo, so open it yourself before spawning lanes — the
subagent may need read permission for that path — and remember it is not versioned and will not
appear in any diff.

Boilerplate belongs in `AGENTS.md` and in the agent definition, not in the delegation.

## 3. Run the lane

At most **three** concurrent lanes and at most **two** follow-up rounds. If the lane is not there
by then, close it and either fix it yourself or rewrite the task.

Resume/SendMessage to a finished agent hangs — a correction is a fresh agent with the new context.

## 4. Check — always

When a lane reports done, you verify. Every time, whatever `VERIFIED:` claims:

1. `git diff -- <the lane's FILES:>` — read it. Did it touch anything outside its list?
   `git status --porcelain` catches files it created *beside* the ones it should have changed.
2. `git log --oneline -3` — did the lane commit? Nothing prevents it; only this check catches it.
3. Re-run the lane's `VERIFIED:` command yourself.

   > **TODO(stack) — there is no gate yet.** No build, test, lint or format command exists in this
   > repo. Until the stack lands, steps 1–2 are the whole gate. Fill in the real commands here and
   > in `AGENTS.md` (Common rules → Verification), then delete both `TODO(stack)` markers.
   > Do not invent a command.

4. Commit the lane by itself. **The agent never commits.**
