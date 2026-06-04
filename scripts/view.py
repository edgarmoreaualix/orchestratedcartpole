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
