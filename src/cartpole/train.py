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
  raise NotImplementedError(
    "Training Agent: implement in feat/training-agent. "
    "See tasks/training-agent.md."
  )


if __name__ == "__main__":
  import tyro

  tyro.cli(main)
