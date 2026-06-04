# Training Agent — Task Bus

You are the **Training Agent**. You own `src/cartpole/train.py` and `tests/test_train.py`.

## Working directory and identity

```bash
cd /Users/edgarmoreau/rl/orchestratedcartpole-training-agent
git config user.name "Training Agent"
git config user.email "training@cartpole.dev"
```

## Workflow

Same shape as `tasks/env-agent.md`, on branch `feat/training-agent`.

**Dependency note:** Your code imports from `src/cartpole/env.py` and `src/cartpole/algorithm.py`. While Env Agent and Algorithm Agent are still working on theirs in parallel, code against `docs/INTERFACES.md` — call `make_env(...)` and `get_ppo_config(...)` as if they're real. Your smoke test will fail until those PRs are merged into `main`; that's expected. Mark the smoke test with `@pytest.mark.slow` and tag it `xfail(reason="env+algo not merged yet")` so CI stays green.

---

## Tasks

### Task 1 — Implement training entrypoint (2026-06-04)

**STATUS:** TODO

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
