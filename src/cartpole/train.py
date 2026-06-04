"""Training entrypoint — owned by Training Agent.

Contract: see `docs/INTERFACES.md`.
"""

from __future__ import annotations


def main(
  num_envs: int = 4096,
  max_iterations: int = 200,
  device: str | None = None,
  seed: int = 1,
) -> None:
  """Train PPO on cartpole. See docs/INTERFACES.md."""
  from dataclasses import asdict

  import torch
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

  from cartpole.algorithm import get_ppo_config
  from cartpole.env import make_env

  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env = make_env(num_envs=num_envs, device=device, play=False)
  agent_cfg = get_ppo_config(experiment_name="cartpole")
  agent_cfg.max_iterations = max_iterations
  agent_cfg.seed = seed
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.learn(num_learning_iterations=max_iterations)


if __name__ == "__main__":
  import tyro

  tyro.cli(main)
