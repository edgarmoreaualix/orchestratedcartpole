"""PPO algorithm config — owned by Algorithm Agent.

Contract: see `docs/INTERFACES.md`.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def get_ppo_config(experiment_name: str = "cartpole") -> RslRlOnPolicyRunnerCfg:
  """Return the PPO RslRlOnPolicyRunnerCfg. See docs/INTERFACES.md."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="tanh",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="tanh",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      clip_param=0.2,
      value_loss_coef=0.5,
      entropy_coef=0.01,
      lam=0.95,
      gamma=0.99,
      learning_rate=3e-4,
      num_learning_epochs=4,
      num_mini_batches=4,
      schedule="fixed",
    ),
    num_steps_per_env=16,
    max_iterations=200,
    save_interval=50,
    experiment_name=experiment_name,
  )
