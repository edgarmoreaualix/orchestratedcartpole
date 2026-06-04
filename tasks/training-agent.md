# Training Agent — Task Bus

You are the **Training Agent**. You own `src/cartpole/train.py` and `tests/test_train.py`.

## Working directory and identity

Set the per-commit identity helper once per shell — do NOT use `git config user.name` (see [`POSTMORTEM.md`](../POSTMORTEM.md)):

```bash
cd /Users/edgarmoreau/rl/orchestratedcartpole-training-agent
AGENT_NAME="Training Agent"
AGENT_EMAIL="training@cartpole.dev"
agent_commit() {
  git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit "$@"
}
```

Use `agent_commit -m "..."` for every commit you make.

## Workflow

Same shape as `tasks/env-agent.md`, on branch `feat/training-agent`.

**Dependency note:** Your code imports from `src/cartpole/env.py` and `src/cartpole/algorithm.py`. While Env Agent and Algorithm Agent are still working on theirs in parallel, code against `docs/INTERFACES.md` — call `make_env(...)` and `get_ppo_config(...)` as if they're real. Your smoke test will fail until those PRs are merged into `main`; that's expected. Mark the smoke test with `@pytest.mark.slow` and tag it `xfail(reason="env+algo not merged yet")` so CI stays green.

---

## Tasks

### Task 1 — Implement training entrypoint (2026-06-04)

**STATUS:** DONE — PR https://github.com/edgarmoreaualix/orchestratedcartpole/pull/4

**Scope:** Implement `src/cartpole/train.py:main(num_envs, max_iterations, device, seed)` per `docs/INTERFACES.md`. Wires env + algo into rsl_rl's `OnPolicyRunner` and runs `max_iterations` of PPO.

**Implementation outline:**
```python
import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from cartpole.env import make_env
from cartpole.algorithm import get_ppo_config

def main(num_envs=4096, max_iterations=200, device=None, seed=1):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = make_env(num_envs=num_envs, device=device, play=False)
    agent_cfg = get_ppo_config(experiment_name="cartpole")
    agent_cfg.max_iterations = max_iterations
    agent_cfg.seed = seed
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.learn(num_learning_iterations=max_iterations)
```

Add a `tyro.cli` entry under `if __name__ == "__main__":` (already in the stub).

**Tests to add** in `tests/test_train.py`:

- `test_main_runs_one_iteration_cpu` — call `main(num_envs=4, max_iterations=1, device="cpu")`; assert no exception, assert `logs/cartpole/` directory created.
  - Use `pytest.mark.slow` decorator (so CI can skip; we still run locally).

**Definition of done:**
- [ ] `main` runs end-to-end for 1 iteration on CPU with 4 envs.
- [ ] Smoke test passes locally: `uv run pytest tests/test_train.py -v -m slow`.
- [ ] PR title `train: implement PPO training loop on rsl_rl`. Body has summary + smoke-test output.
- [ ] STATUS flipped to DONE, PR URL appended.

### Task 2 — Train a real cartpole policy + ship release (2026-06-04, Round 2)

**STATUS:** TODO

**Branch:** `feat/training-agent-r2` (already checked out in your worktree).

**Scope (three commits):**

**Commit 1 — remove the `xfail` marker** on `test_main_runs_one_iteration_cpu` in `tests/test_train.py`. Env and Algorithm are now merged into `main`, so the smoke test should pass cleanly. Run `uv run pytest tests/test_train.py -v -m slow` to confirm (this will execute the 1-iteration smoke training).

**Commit 2 — run a real training to convergence on CPU and capture metrics.** Choose `num_envs` and `max_iterations` that fit a ~5-minute wall-clock budget on this Mac (no GPU). Sensible starting point: `num_envs=128`, `max_iterations=100`. If your run finishes faster than 5 min and isn't converged (mean episodic return not saturating), bump `max_iterations` and re-run. If it's too slow, reduce `num_envs`.

```bash
uv run python -m cartpole.train --num-envs=128 --max-iterations=100 --device=cpu --seed=1
```

Capture stdout to a file: `uv run python -m cartpole.train ... 2>&1 | tee training.log`. Note the final mean episode reward.

**Commit 3 — ship the checkpoint as a GitHub release.** Find the last checkpoint under `logs/cartpole/<timestamp>/model_<N>.pt`. Create release `v0.2.0`:

```bash
gh release create v0.2.0 logs/cartpole/<timestamp>/model_<N>.pt \
  --title "Round 2 cartpole checkpoint" \
  --notes "Trained on CPU, num_envs=<...>, max_iterations=<...>. Final mean episode reward: <...>. See PR for full training log."
```

Do NOT commit the checkpoint to git (`.gitignore` already excludes `*.pt`). The release is the canonical artifact.

**Definition of done:**
- [ ] `tests/test_train.py` no longer has `xfail`; smoke test passes.
- [ ] A training run completed; `logs/cartpole/<run>/model_<N>.pt` exists.
- [ ] GitHub release `v0.2.0` exists with the checkpoint attached. Verify: `gh release view v0.2.0`.
- [ ] PR title: `train: real training run + release v0.2.0 checkpoint`. Body: hyperparameters used, final mean episode reward, link to release.
- [ ] STATUS above flipped to DONE, PR URL appended.

**If you encounter a true blocker** (e.g. training diverges, OOM, gh auth missing), push partial work and post a single `BLOCKED:` comment on the PR with the exact issue.
