# Environment Agent — Task Bus

You are the **Environment Agent** for the orchestratedcartpole project. You own `src/cartpole/env.py` and `tests/test_env.py`. Do not edit other agents' files.

## Working directory and identity

You work in a dedicated git worktree on your current branch (Round 2: `feat/env-agent-r2`). Set the per-commit identity helper once per shell — do NOT use `git config user.name` (it bleeds across worktrees; see [`POSTMORTEM.md`](../POSTMORTEM.md)):

```bash
cd /Users/edgarmoreau/rl/orchestratedcartpole-env-agent
AGENT_NAME="Environment Agent"
AGENT_EMAIL="env@cartpole.dev"
agent_commit() {
  git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit "$@"
}
```

Use `agent_commit -m "..."` for every commit you make.

## Workflow

1. `git fetch origin && git rebase origin/main` — pull the latest task brief from main.
2. Read the bottom-most `STATUS: TODO` task below.
3. Implement it in the files you own (`src/cartpole/env.py`, `tests/test_env.py`).
4. `uv sync` if `.venv/` does not yet exist; then `uv run pytest tests/test_env.py -v` until green.
5. Commit and push: `git push -u origin feat/env-agent`.
6. Open a PR: `gh pr create --base main --head feat/env-agent --title "<title>" --body "<summary + DoD checklist + pytest output>"`.
7. Edit this file to flip the task's STATUS to `DONE` and append the PR URL. Commit and push the task file on the same branch (the orchestrator will see it on the PR).

---

## Tasks

### Task 1 — Implement mjlab cartpole env factory (2026-06-04)

**STATUS:** DONE — https://github.com/edgarmoreaualix/orchestratedcartpole/pull/5

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

### Task 2 — Add `scripts/view.py` + fix indent style (2026-06-04, Round 2)

**STATUS:** TODO

**Branch:** `feat/env-agent-r2` (already checked out in your worktree).

**Scope (two independent commits):**

**Commit 1 — reformat `src/cartpole/env.py` to 2-space indent.** Round 1 shipped this file with 4-space indent, inconsistent with `pyproject.toml`'s `indent-width = 2`. Reformat by hand (don't run `ruff format` on the whole repo — only touch this file).

**Commit 2 — add `scripts/view.py`.** A project-owned visualization entrypoint so users no longer need `-m mjlab.scripts.play`. Implement it to support zero, random, and trained policies, with the native MuJoCo viewer.

Implementation outline for `scripts/view.py`:

```python
"""View the cartpole env in a viewer.

Usage:
  uv run scripts/view.py                          # zero policy, native viewer
  uv run scripts/view.py --agent random           # random policy
  uv run scripts/view.py --agent trained --checkpoint path/to/model.pt
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import tyro

from cartpole.env import make_env


@dataclass(frozen=True)
class ViewConfig:
  agent: Literal["zero", "random", "trained"] = "zero"
  checkpoint: str | None = None
  num_envs: int = 1
  viewer: Literal["native", "viser"] = "native"


def main(cfg: ViewConfig) -> None:
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

  from cartpole.algorithm import get_ppo_config

  env = make_env(num_envs=cfg.num_envs, device="cpu", play=True)
  env.cfg.terminations = {}  # always disable for ad-hoc viewing

  agent_cfg = get_ppo_config()
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  action_shape = env.unwrapped.action_space.shape

  if cfg.agent == "zero":
    policy = lambda obs: torch.zeros(action_shape, device="cpu")
  elif cfg.agent == "random":
    policy = lambda obs: 2 * torch.rand(action_shape, device="cpu") - 1
  else:
    if cfg.checkpoint is None:
      raise ValueError("--agent=trained requires --checkpoint")
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cpu")
    runner.load(cfg.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cpu")
    policy = runner.get_inference_policy(device="cpu")

  if cfg.viewer == "native":
    NativeMujocoViewer(env, policy).run()
  else:
    ViserPlayViewer(env, policy).run()


if __name__ == "__main__":
  tyro.cli(main)
```

**Tests to add** in `tests/test_view.py`:

- `test_view_module_imports` — `import scripts.view` (you'll need `tests/conftest.py` to insert the project root onto sys.path; check if one exists from eval and extend it, or add a minimal `tests/conftest.py` that does `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))`).
- `test_view_config_defaults` — `ViewConfig()` returns a config with `agent="zero"`, `viewer="native"`, `num_envs=1`.

**Definition of done:**
- [ ] `src/cartpole/env.py` uses 2-space indent.
- [ ] `scripts/view.py` exists and matches the spec.
- [ ] Both new tests pass: `uv run pytest tests/test_view.py tests/test_env.py -v`.
- [ ] Run the script once locally with `--agent=zero` to sanity-check that the viewer opens (close the window after seeing it). Do not commit any output.
- [ ] PR title: `env: add scripts/view.py + fix indent style`. Body: summary, pytest output, and a note that the visualization command is now `uv run mjpython scripts/view.py --agent=zero`.
- [ ] STATUS above flipped to DONE, PR URL appended.
