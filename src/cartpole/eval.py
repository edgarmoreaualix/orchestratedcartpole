"""Evaluation entrypoint — owned by Evaluation Agent.

Contract: see `docs/INTERFACES.md`.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict

import torch


def main(
  checkpoint: str,
  num_envs: int = 1,
  num_episodes: int = 10,
  device: str | None = None,
  video: bool = False,
) -> dict:
  """Load a checkpoint, run eval episodes, return metrics. See docs/INTERFACES.md."""
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

  from cartpole.algorithm import get_ppo_config
  from cartpole.env import make_env

  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env = make_env(num_envs=num_envs, device=device, play=True)
  agent_cfg = get_ppo_config(experiment_name="cartpole-eval")

  # video recording — wrap before the vecenv wrapper so frames are captured
  if video:
    from mjlab.utils.wrappers import VideoRecorder

    env = VideoRecorder(env, output_dir="videos/eval/")

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

  return {
    "mean_return": statistics.fmean(returns),
    "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
    "mean_episode_length": statistics.fmean(lengths),
  }


if __name__ == "__main__":
  import tyro

  tyro.cli(main)
