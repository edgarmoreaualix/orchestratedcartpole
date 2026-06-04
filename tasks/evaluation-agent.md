# Evaluation Agent — Task Bus

You are the **Evaluation Agent**. You own `src/cartpole/eval.py` and `tests/test_eval.py`.

## Working directory and identity

```bash
cd /Users/edgarmoreau/rl/orchestratedcartpole-evaluation-agent
git config user.name "Evaluation Agent"
git config user.email "eval@cartpole.dev"
```

## Workflow

Same shape as `tasks/env-agent.md`, on branch `feat/evaluation-agent`.

**Dependency note:** Your code imports `make_env` from `env.py` and reuses the PPO inference loader. You need a checkpoint to actually test end-to-end, so the integration test is gated on `pytest.mark.requires_checkpoint` and skipped if none is present. The unit-level tests run anyway.

---

## Tasks

### Task 1 — Implement evaluation entrypoint (2026-06-04)

**STATUS:** TODO

**Scope:** Implement `src/cartpole/eval.py:main(checkpoint, num_envs, num_episodes, device, video) -> dict` per `docs/INTERFACES.md`. Returns `{"mean_return": float, "std_return": float, "mean_episode_length": float}`.

**Implementation outline:**
```python
import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from cartpole.env import make_env
from cartpole.algorithm import get_ppo_config

def main(checkpoint, num_envs=1, num_episodes=10, device=None, video=False):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = make_env(num_envs=num_envs, device=device, play=True)
    agent_cfg = get_ppo_config(experiment_name="cartpole-eval")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    returns, lengths = [], []
    obs, _ = env.reset()
    ep_return = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    done_count = 0
    while done_count < num_episodes:
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += reward
        ep_len += 1
        done = terminated | truncated
        for i in done.nonzero(as_tuple=False).flatten().tolist():
            returns.append(ep_return[i].item())
            lengths.append(ep_len[i].item())
            ep_return[i] = 0
            ep_len[i] = 0
            done_count += 1
    import statistics
    return {
        "mean_return": statistics.fmean(returns),
        "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "mean_episode_length": statistics.fmean(lengths),
    }
```

If `video=True`, additionally wrap the env with `mjlab.utils.wrappers.VideoRecorder` and write to `videos/eval/`. Tag this part with a comment so it's reviewable.

**Tests to add** in `tests/test_eval.py`:

- `test_main_signature` — `import cartpole.eval; assert "checkpoint" in inspect.signature(cartpole.eval.main).parameters`.
- `test_main_returns_metric_dict` (mark `requires_checkpoint`) — calls `main(<path>, num_episodes=2)`, asserts the returned dict has all three keys with the right types.

**Definition of done:**
- [ ] `main` returns the metric dict per the contract.
- [ ] `test_main_signature` passes: `uv run pytest tests/test_eval.py -v`.
- [ ] PR title `eval: implement policy eval + metrics`. Body has summary + test output.
- [ ] STATUS flipped to DONE, PR URL appended.
