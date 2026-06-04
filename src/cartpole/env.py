"""Environment factory — owned by Environment Agent.

Contract: see `docs/INTERFACES.md`.
"""

from __future__ import annotations


def make_env(num_envs: int = 1, device: str = "cpu", play: bool = False):
  """Build the mjlab Mjlab-Cartpole-Balance env. See docs/INTERFACES.md."""
  raise NotImplementedError(
    "Environment Agent: implement in feat/env-agent. See tasks/env-agent.md."
  )
