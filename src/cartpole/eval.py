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
  # play=False keeps episode_length_s=50s and time_out terminations active.
  # play=True sets episode_length_s=1e10, so time_out never fires and the
  # eval loop hangs forever — that was the Round 2 root cause.
  env = make_env(num_envs=num_envs, device=device, play=False)
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
  # Hard cap: 1500 steps per episode (mjlab cartpole horizon is 1000).
  # Protects against env configurations where dones never fires.
  MAX_TOTAL_STEPS = num_episodes * 1500
  step = 0
  while done_count < num_episodes and step < MAX_TOTAL_STEPS:
    action = policy(obs)
    # RslRlVecEnvWrapper.step returns (obs, rew, dones, extras) — 4-tuple.
    obs, reward, dones, _ = env.step(action)
    ep_return += reward
    ep_len += 1
    step += 1
    done = dones.bool()
    for i in done.nonzero(as_tuple=False).flatten().tolist():
      returns.append(ep_return[i].item())
      lengths.append(ep_len[i].item())
      ep_return[i] = 0
      ep_len[i] = 0
      done_count += 1

  # If the hard cap fired before all episodes completed, record in-flight episodes.
  if done_count < num_episodes:
    for i in range(num_envs):
      if ep_len[i] > 0:
        returns.append(ep_return[i].item())
        lengths.append(ep_len[i].item())

  metrics = {
    "mean_return": statistics.fmean(returns),
    "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
    "mean_episode_length": statistics.fmean(lengths),
  }
  print(metrics)
  return metrics


if __name__ == "__main__":
  import tyro

  tyro.cli(main)
