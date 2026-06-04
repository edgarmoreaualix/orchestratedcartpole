"""Environment factory — owned by Environment Agent.

Contract: see `docs/INTERFACES.md`.
"""

from __future__ import annotations

import mjlab.tasks  # registers built-in tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def make_env(num_envs: int = 1, device: str = "cpu", play: bool = False) -> ManagerBasedRlEnv:
  """Build the mjlab Mjlab-Cartpole-Balance env. See docs/INTERFACES.md."""
  cfg = load_env_cfg("Mjlab-Cartpole-Balance", play=play)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=device)
