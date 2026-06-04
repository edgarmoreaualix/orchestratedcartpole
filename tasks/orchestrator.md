# Orchestrator — dispatch protocol

The `tasks/` directory is the command bus for the orchestrated build. Five specialized agents each have a `tasks/<agent>.md` file; the orchestrator (this terminal) appends task blocks at the bottom of each, the agent executes the latest TODO, then sets its STATUS to DONE.

## Agents, ownership, and worktrees

| Agent                | Owns                                       | Branch                       | Worktree (working directory)                                              |
|----------------------|--------------------------------------------|------------------------------|---------------------------------------------------------------------------|
| Environment Agent    | `src/cartpole/env.py`, `tests/test_env.py` | `feat/env-agent`             | `/Users/edgarmoreau/rl/orchestratedcartpole-env-agent`                    |
| Algorithm Agent      | `src/cartpole/algorithm.py`, `tests/test_algorithm.py` | `feat/algorithm-agent` | `/Users/edgarmoreau/rl/orchestratedcartpole-algorithm-agent`        |
| Training Agent       | `src/cartpole/train.py`, `tests/test_train.py` | `feat/training-agent`    | `/Users/edgarmoreau/rl/orchestratedcartpole-training-agent`               |
| Evaluation Agent     | `src/cartpole/eval.py`, `tests/test_eval.py`   | `feat/evaluation-agent`  | `/Users/edgarmoreau/rl/orchestratedcartpole-evaluation-agent`             |
| Documentation Agent  | `README.md`, `docs/architecture.md`        | `feat/documentation-agent`   | `/Users/edgarmoreau/rl/orchestratedcartpole-documentation-agent`          |

Each worktree shares the same underlying `.git` database but holds its own checkout of its branch, so the five agents can edit files in parallel without contending for the working tree.

## Round 2 branches

Each round uses its own feature branch suffix so the git log shows round-by-round progression. Round 2 branches (Algorithm Agent sits out this round):

| Agent                | Round 2 branch                  |
|----------------------|----------------------------------|
| Environment Agent    | `feat/env-agent-r2`              |
| Training Agent       | `feat/training-agent-r2`         |
| Evaluation Agent     | `feat/evaluation-agent-r2`       |
| Documentation Agent  | `feat/documentation-agent-r2`    |

Round 2 also adds explicit runtime dependencies between agents (Eval needs Training's release, Docs needs Eval's GIF). When a dep is missing, the dependent agent posts `BLOCKED: waiting for <what>` on its PR and stops; the operator re-fires it once the dependency lands.

## Cross-agent rules

- An agent only edits its own files (the path it owns and its own task file).
- An agent never touches `docs/INTERFACES.md`. If a signature change is needed, the agent opens a PR titled `interface: <reason>` that touches only `docs/INTERFACES.md`, and the orchestrator coordinates.
- Every agent commits under its own author identity (configured locally per shell, see each agent file).
- Every agent's PR title uses a conventional-commits prefix: `env:`, `algo:`, `train:`, `eval:`, `docs:`.

## Identity norm (Round 2+) — per-commit, not per-config

Round 1 demonstrated that `git config user.name` writes to the shared `.git/config` across worktrees and silently corrupts attribution. From Round 2 onward, every agent uses the per-commit override pattern. Set the identity once at shell start as bash variables, then use the helper for every commit:

```bash
AGENT_NAME="<Agent Name>"
AGENT_EMAIL="<slug>@cartpole.dev"
agent_commit() {
  git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit "$@"
}
```

Use `agent_commit -m "..."` instead of `git commit -m "..."`. The `-c key=value` form sets configuration only for the duration of that single git invocation. It does not touch any persistent config file and therefore cannot bleed across worktrees. See [`POSTMORTEM.md`](../POSTMORTEM.md) for the full root cause.

## Autonomy norm (applies to every agent on every task)

Agents run end-to-end without asking the operator for confirmation. Specifically:

1. **Decide reasonable defaults silently.** When the spec leaves a choice open (file structure inside your owned path, naming of helper variables, choice between two equally-valid library calls), pick one and move on. Document the choice in the PR body, not in a question to the operator.
2. **Do not ask permission to commit, push, or open the PR.** The task file already authorises all three; just do them.
3. **Auto-rebase after any upstream merge.** If `git pull --rebase origin main` shows new commits that conflict with your branch (most commonly `uv.lock` or `pyproject.toml`), resolve in favour of `main` for shared infra files and re-apply your code changes. Force-push your branch (`git push --force-with-lease`) and post a one-line comment on your PR: `rebased onto <short-sha>`.
4. **Stop only on true blockers.** Real blockers are: a failing test you cannot fix without changing another agent's owned file; an interface change you cannot make unilaterally; a missing system dependency you cannot install. Anything else, decide and proceed.
5. **When you stop, report.** Push whatever partial work exists, post a single comment on the PR starting with `BLOCKED:` followed by exactly what is needed to unblock. Do not loop.

## Dispatch protocol (orchestrator side)

1. Append a new `### Task N — <title> (YYYY-MM-DD)` block at the bottom of `tasks/<agent>.md`, status `TODO`.
2. Commit and push the task file on `main` (or batch multiple).
3. Tell the operator (Edgar) which agent terminals to fire.
4. When the agent's PR lands, review the diff, merge (squash) into `main`.
5. The agent picks up its next task on its next invocation.

## Merge order for Round 1

Even though work is parallel, merges respect dependencies:

```
env  →  algorithm  →  training  →  evaluation  →  documentation
```

PRs that land out of order wait. Rebases are the agent's responsibility, coordinated by the orchestrator.
