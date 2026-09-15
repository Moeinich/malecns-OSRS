# AGENTS.md

Injected into **every** agent session. Only what applies to all roles lives here — role knowledge
is in `docs/agents/claude/*.md`, command rules in `docs/commands/*.md`. `CLAUDE.md` symlinks here.

## Core Principles

The shortest solution that actually holds. Walk the ladder and **stop at the first rung that
carries:**

1. **Should this exist at all?** Speculative need → drop it, say so in one line.
2. **Does it already exist here?** A helper, type or pattern already in the repo → reuse it.
3. **Can the language, the platform or an installed dependency do it?** Never add a dependency
   for a few lines.
4. **Can it be one line?** Then one line. Only then: the smallest amount of code that works.

**Find the root cause** — no temporary patches — and touch **only what is necessary**.
**No comments that restate the code**; comment only a non-obvious _why_. Default to zero.
**Maximum brevity**, always.

## Structure

`docs/` is canonical. `.claude/agents` and `.claude/commands` are committed symlinks into it —
edit the files under `docs/`, never through the harness path, and never put anything canonical
in `.claude/`.

- `docs/agents/claude/` — subagent definitions, read through `.claude/agents`
- `docs/commands/` — slash commands, read through `.claude/commands`

<!-- TODO(stack): no source code yet. When the stack lands, describe the layout here — the
     directories, where domain code goes, and what we deliberately do not have. -->

## Implementation is delegated

With `task` access you do not write implementation code. You plan, delegate, and **read the diff
every time a lane reports done**, whatever it claims under `VERIFIED:`. You verify and commit; the
agent never commits.

- **Delegate to a named role.** `coder` or `debugger` from `docs/agents/claude/`; they carry the
  role definition so the delegation does not have to. Lane format and the verify loop:
  `docs/commands/delegate.md`.
- **You pick the model and the effort per lane.** The agent file pins a default; the per-call
  override wins. `opus` for logic and anything needing judgment, `sonnet` for mechanical
  multi-file work with a clear spec, `haiku` for lookups and one-file chores. Effort high only
  where the lane has to reason, low for the rest. State both in the delegation, and always:
  maximum brevity.
- **Resume/SendMessage to a finished agent hangs.** Start a fresh agent with the new context.
- **Hung ≠ a quiet transcript** (it buffers). Check source mtimes and running build/test
  processes. Never kill on transcript silence alone.
- **Nothing enforces "the agent never commits."** This repo has no hooks. `debugger` cannot edit —
  its tool list says so, and that is real. `coder` can run `git commit` and it will succeed. The
  rule is the only thing stopping it, and a lane that commits folds itself into whatever else is
  in the shared tree. Step 4 of `/delegate` checks `git log` for exactly this reason.

**Exempt:** one-liners, typos, docs you fix yourself. Logic or more than one file → delegate.

## Common rules

**Working tree.** One shared tree on `main` — touch only your own `FILES:`; missing one →
`BLOCKED`.

**Push.** Only `git push origin main`, and only when asked.

**Verification.**

> **TODO(stack) — there is no gate yet.** This repo has no build, test, lint or format command.
> Until the stack lands, the gate is: read the diff, and nothing else.
> Fill in: typecheck/build, the test command and how to run a single test, lint, format. Then
> replace this block and step 4 of `docs/commands/delegate.md`, and delete both `TODO(stack)`
> markers.
> Do not invent a command. A lane reporting `VERIFIED:` with a command that does not exist is
> worse than one reporting `VERIFIED: none`.

**Leave the area better than you found it.** Small cleanups inside your own `FILES:` — fold them
in. Bigger ones — report them. A lane that quietly grows is worse than one that names debt.

## Language

**Everything written into the repo is English** — identifiers, file and folder names, comments,
types, test names, commit messages, docs.
