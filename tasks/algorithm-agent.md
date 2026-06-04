# Algorithm Agent — Task Bus

You are the **Algorithm Agent**. You own `src/cartpole/algorithm.py` and `tests/test_algorithm.py`.

## Identity

```bash
cd /Users/edgarmoreau/rl/orchestratedcartpole
git config user.name "Algorithm Agent"
git config user.email "algo@cartpole.dev"
```

## Branch

`feat/algorithm-agent`.

## Workflow

Same as `tasks/env-agent.md` workflow, on your own branch.

---

## Tasks

### Task 1 — Implement PPO config for cartpole (2026-06-04)

**STATUS:** TODO

**Scope:** Implement `src/cartpole/algorithm.py:get_ppo_config(experiment_name)` per `docs/INTERFACES.md`. Return an `mjlab.rl.RslRlOnPolicyRunnerCfg` tuned for cartpole.

**Sensible defaults to start with** (these are well-known good PPO hyperparameters for cartpole-class tasks; you may justify tweaks in the PR body):

- Policy: MLP `[64, 64]` for both actor and critic, `tanh` activation.
- PPO: clip 0.2, value-loss coeff 0.5, entropy coeff 0.01, GAE λ=0.95, γ=0.99.
- Optimization: Adam lr 3e-4, 4 mini-epochs, 4 mini-batches per epoch.
- Rollout: 16 steps per env.
- Runner: `max_iterations=200`, `save_interval=50`, `experiment_name=experiment_name`.

**Implementation hints:** Look at how the SMP project at `/Users/edgarmoreau/rl/smp/src/smp/rl/rl_cfg.py` builds its `RslRlOnPolicyRunnerCfg` for shape, but **do not import from that repo** — write a self-contained config here. Reuse only `mjlab.rl` classes (`RslRlOnPolicyRunnerCfg`, `RslRlPpoActorCriticCfg`, `RslRlPpoAlgorithmCfg`).

**Tests to add** in `tests/test_algorithm.py`:

- `test_get_ppo_config_returns_runner_cfg` — call with default args, assert it's a `RslRlOnPolicyRunnerCfg`.
- `test_get_ppo_config_respects_experiment_name` — pass `experiment_name="my-run"`, assert `cfg.experiment_name == "my-run"`.
- `test_get_ppo_config_hyperparameters_sensible` — assert clip in (0, 1), lr > 0, gae lambda in (0, 1].

**Definition of done:**
- [ ] `get_ppo_config` returns a valid `RslRlOnPolicyRunnerCfg`.
- [ ] All three tests pass: `uv run pytest tests/test_algorithm.py -v`.
- [ ] PR title `algo: implement PPO config for cartpole`. Body has summary + pytest output.
- [ ] STATUS flipped to DONE, PR URL appended.
