"""Evaluation entrypoint — owned by Evaluation Agent.

Contract: see `docs/INTERFACES.md`.
"""

from __future__ import annotations


def main(
  checkpoint: str,
  num_envs: int = 1,
  num_episodes: int = 10,
  device: str | None = None,
  video: bool = False,
) -> dict:
  """Load a checkpoint, run eval episodes, return metrics. See docs/INTERFACES.md."""
  raise NotImplementedError(
    "Evaluation Agent: implement in feat/evaluation-agent. "
    "See tasks/evaluation-agent.md."
  )


if __name__ == "__main__":
  import tyro

  tyro.cli(main)
