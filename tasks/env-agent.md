# Environment Agent — Task Bus

You are the **Environment Agent** for the orchestratedcartpole project. You own `src/cartpole/env.py` and `tests/test_env.py`. Do not edit other agents' files.

## Identity (configure once per shell)

```bash
cd /Users/edgarmoreau/rl/orchestratedcartpole
git config user.name "Environment Agent"
git config user.email "env@cartpole.dev"
```

## Branch

`feat/env-agent`, cut from `origin/main`.

## Workflow

1. `git fetch origin && git switch -c feat/env-agent origin/main` (first time) — afterwards `git switch feat/env-agent && git rebase origin/main`.
2. Read the bottom-most `STATUS: TODO` task below.
3. Implement it in the files you own.
4. Run `uv run pytest tests/test_env.py -v` until green.
5. `git push -u origin feat/env-agent`.
6. `gh pr create --base main --head feat/env-agent --title "<title>" --body "<scope + DoD checklist + pytest output>"`.
7. Edit this file to flip the task's STATUS to `DONE`, append the PR URL. Commit + push the task file on the same branch (the orchestrator merges everything together).

---

## Tasks

### Task 1 — Implement mjlab cartpole env factory (2026-06-04)

**STATUS:** TODO

**Scope:** Implement `src/cartpole/env.py:make_env(num_envs, device, play)` per `docs/INTERFACES.md`. Use mjlab's existing `Mjlab-Cartpole-Balance` task — do not redefine it.

**Implementation outline:**
```python
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
import mjlab.tasks  # registers built-in tasks

def make_env(num_envs=1, device="cpu", play=False):
    cfg = load_env_cfg("Mjlab-Cartpole-Balance", play=play)
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg=cfg, device=device)
```

**Tests to add** in `tests/test_env.py`:

- `test_make_env_returns_managerbasedrlenv` — build with `num_envs=2, device="cpu"`. Assert `env.action_space.shape[0] == 2`.
- `test_make_env_steps` — `env.reset()`, then `env.step(zero_action)` 5 times, assert no exception, assert reward tensor has shape `(2,)`.
- `test_make_env_play_mode` — build with `play=True`, assert `env.unwrapped.cfg.scene.num_envs == 1` (override still applied).

**Definition of done:**
- [ ] `make_env` no longer raises NotImplementedError; returns a real `ManagerBasedRlEnv`.
- [ ] All three tests pass on CPU: `uv run pytest tests/test_env.py -v`.
- [ ] PR opened with title `env: implement mjlab cartpole env factory` and body containing (a) a 2-sentence summary and (b) the pytest output pasted in a fenced block.
- [ ] STATUS above flipped to DONE, PR URL appended.
