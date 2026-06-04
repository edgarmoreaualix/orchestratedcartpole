# Orchestrator — dispatch protocol

The `tasks/` directory is the command bus for the orchestrated build. Five specialized agents each have a `tasks/<agent>.md` file; the orchestrator (this terminal) appends task blocks at the bottom of each, the agent executes the latest TODO, then sets its STATUS to DONE.

## Agents and ownership

| Agent                | Owns                                       | Branch                       |
|----------------------|--------------------------------------------|------------------------------|
| Environment Agent    | `src/cartpole/env.py`, `tests/test_env.py` | `feat/env-agent`             |
| Algorithm Agent      | `src/cartpole/algorithm.py`, `tests/test_algorithm.py` | `feat/algorithm-agent` |
| Training Agent       | `src/cartpole/train.py`, `tests/test_train.py` | `feat/training-agent`    |
| Evaluation Agent     | `src/cartpole/eval.py`, `tests/test_eval.py`   | `feat/evaluation-agent`  |
| Documentation Agent  | `README.md`, `docs/architecture.md`        | `feat/documentation-agent`   |

## Cross-agent rules

- An agent only edits its own files (the path it owns and its own task file).
- An agent never touches `docs/INTERFACES.md`. If a signature change is needed, the agent opens a PR titled `interface: <reason>` that touches only `docs/INTERFACES.md`, and the orchestrator coordinates.
- Every agent commits under its own author identity (configured locally per shell, see each agent file).
- Every agent's PR title uses a conventional-commits prefix: `env:`, `algo:`, `train:`, `eval:`, `docs:`.

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
