"""PPO algorithm config — owned by Algorithm Agent.

Contract: see `docs/INTERFACES.md`.
"""

from __future__ import annotations


def get_ppo_config(experiment_name: str = "cartpole"):
  """Return the PPO RslRlOnPolicyRunnerCfg. See docs/INTERFACES.md."""
  raise NotImplementedError(
    "Algorithm Agent: implement in feat/algorithm-agent. "
    "See tasks/algorithm-agent.md."
  )
