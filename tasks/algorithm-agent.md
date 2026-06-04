# Algorithm Agent — Task Bus

You are the **Algorithm Agent**. You own `src/cartpole/algorithm.py` and `tests/test_algorithm.py`.

## Working directory and identity

Set the per-commit identity helper once per shell — do NOT use `git config user.name` (see [`POSTMORTEM.md`](../POSTMORTEM.md)):

```bash
cd /Users/edgarmoreau/rl/orchestratedcartpole-algorithm-agent
AGENT_NAME="Algorithm Agent"
AGENT_EMAIL="algo@cartpole.dev"
agent_commit() {
  git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit "$@"
}
```

Use `agent_commit -m "..."` for every commit you make.

## Workflow

Same shape as `tasks/env-agent.md`: `git fetch origin && git rebase origin/main`, implement, `uv run pytest tests/test_algorithm.py -v`, push to your current branch, open PR, flip STATUS to DONE.

---

## Tasks

### Task 1 — Implement PPO config for cartpole (2026-06-04)

**STATUS:** DONE — PR https://github.com/edgarmoreaualix/orchestratedcartpole/pull/3

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

### Task 2 — Tune hyperparameters for actual convergence (2026-06-04, Round 3)

**STATUS:** DONE — https://github.com/edgarmoreaualix/orchestratedcartpole/pull/8

**Branch:** `feat/algorithm-agent-r3`.

**Context:** Round 2's training run achieved mean episode reward 12 / 1000 — the policy did not balance the pole. Visual inspection confirmed the cart wiggles but the pole falls. Diagnosis points at three config issues in the existing PPO config that hurt convergence on this task:

| Field | Current | New | Why |
|---|---|---|---|
| `actor.obs_normalization` | `False` | `True` | mjlab cartpole observations have different scales (positions vs velocities). Without normalization PPO spends iterations learning the scale before it can learn the policy. |
| `critic.obs_normalization` | `False` | `True` | Same reason for the value function. |
| `actor.distribution_cfg["init_std"]` | `1.0` | `0.5` | Initial exploration was too noisy for a 1-D continuous action; the policy was effectively random for the first ~50 iters. |
| `algorithm.entropy_coef` | `0.01` | `0.005` | The entropy bonus was strong enough to keep the policy diffuse even after it discovered useful actions; lowering it lets the policy commit. |

**Scope (one commit):** Update `src/cartpole/algorithm.py` to apply the four field changes above. Do not change anything else.

**Tests to update** in `tests/test_algorithm.py`:

- Add `test_get_ppo_config_obs_normalization_enabled` — assert `cfg.actor.obs_normalization is True` and same for `critic`.
- Add `test_get_ppo_config_init_std_is_05` — assert `cfg.actor.distribution_cfg["init_std"] == 0.5`.
- Add `test_get_ppo_config_entropy_coef_is_005` — assert `cfg.algorithm.entropy_coef == 0.005`.

**Definition of done:**
- [ ] Config updated with the four changes.
- [ ] All six tests pass: `uv run pytest tests/test_algorithm.py -v` (3 existing + 3 new).
- [ ] PR title: `algo: tune hyperparameters for cartpole convergence`. Body: a short table of the changes, a one-line rationale per change.
- [ ] STATUS above flipped to DONE, PR URL appended.

### Task 3 — Adopt mjlab's reference cartpole PPO config (2026-06-04, Round 4)

**STATUS:** TODO

**Branch:** `feat/algorithm-agent-r4`.

**Context:** Round 3's hand-tuned config (Task 2) trained to mean reward **32 / 1000** — the policy collapsed (action std → 0.02) before finding a balancing solution. Investigation of mjlab's source (`mjlab.tasks.cartpole.cartpole_env_cfg.cartpole_ppo_runner_cfg`) revealed mjlab ships a **canonical PPO config tested on this exact env** that we were diverging from on multiple dimensions — most critically the KL-adaptive LR schedule, which our `schedule="fixed"` was missing. Round 4 aligns our config with the upstream reference.

**Scope (one commit):** Replace the body of `get_ppo_config` in `src/cartpole/algorithm.py` with the values below, which match mjlab's `cartpole_ppo_runner_cfg()` verbatim except that `experiment_name` remains parameterized for our use case.

```python
def get_ppo_config(experiment_name: str = "cartpole") -> RslRlOnPolicyRunnerCfg:
  """Return the PPO RslRlOnPolicyRunnerCfg. See docs/INTERFACES.md.

  Hyperparameters match mjlab's upstream reference cartpole_ppo_runner_cfg()
  (mjlab.tasks.cartpole.cartpole_env_cfg.cartpole_ppo_runner_cfg). Round 3's
  hand-tuned config diverged from this reference and failed to converge;
  Round 4 aligns with the upstream-tested values.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    num_steps_per_env=32,
    max_iterations=500,
    save_interval=50,
    experiment_name=experiment_name,
  )
```

**Tests to update** in `tests/test_algorithm.py`:

The three Round 3 tests (`test_get_ppo_config_obs_normalization_enabled`, `test_get_ppo_config_init_std_is_05`, `test_get_ppo_config_entropy_coef_is_005`) assert values that Round 4 reverts. Replace them with tests that assert the Round 4 values aligned with mjlab's reference:

- `test_get_ppo_config_obs_normalization_disabled` — `cfg.actor.obs_normalization is False` and same for `critic`.
- `test_get_ppo_config_uses_elu_activation` — `cfg.actor.activation == "elu"`.
- `test_get_ppo_config_uses_adaptive_lr_schedule` — `cfg.algorithm.schedule == "adaptive"` and `cfg.algorithm.desired_kl == 0.01`.
- `test_get_ppo_config_uses_value_loss_clipping` — `cfg.algorithm.use_clipped_value_loss is True` and `cfg.algorithm.max_grad_norm == 1.0`.
- Keep the three pre-existing Round 1 tests untouched (signature, experiment_name, hyperparameter sanity).

**Definition of done:**
- [ ] `get_ppo_config` body matches the reference above.
- [ ] All seven tests pass: `uv run pytest tests/test_algorithm.py -v` (3 Round 1 + 4 Round 4).
- [ ] PR title: `algo: adopt mjlab's reference cartpole PPO config`. Body: link to `mjlab.tasks.cartpole.cartpole_env_cfg.cartpole_ppo_runner_cfg`, a brief note on why we diverged (Round 3 hand-tuning failed to converge), and the test output.
- [ ] STATUS above flipped to DONE, PR URL appended.
