# Agent roles

`docs/agents/claude/` holds the Claude Code subagent definitions. `.claude/agents` is a symlink
to this folder — edit the files here, never through the symlink path.

## The rules

**The body lives here, inline, in one place.** There is no second roster and no longer role file
to point at. If one ever appears, this file's body becomes a pointer to it and is never copied:
two copies of a rule drift, and the day they disagree nobody knows which one holds.

**`description` fields share one token budget across every agent** — each one is in the context
of every session that can delegate. Write **when** the role is called, not what it is. One or two
lines.

**Knowledge that belongs to a task rather than a role does not go in a role file.** Knowledge
that applies to every role goes in `AGENTS.md`.

## The roster

| Agent      | When                                                        |
| ---------- | ----------------------------------------------------------- |
| `coder`    | scoped work that has to be written                          |
| `debugger` | a failure whose cause is unknown; read-only by construction |

The built-ins cover the rest: `Explore` for search, `Plan` for design questions. Both are
read-only, neither takes a `FILES:` list, and neither is a lane.

**No `qa` role.** There is no gate for it to run yet. When the stack lands and there is a test and
lint command, that is the moment a `qa` role becomes something other than a name.
