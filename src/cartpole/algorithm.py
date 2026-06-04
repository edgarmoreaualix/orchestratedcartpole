"""PPO algorithm config — owned by Algorithm Agent.

Contract: see `docs/INTERFACES.md`.
"""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def get_ppo_config(experiment_name: str = "cartpole") -> RslRlOnPolicyRunnerCfg:
  """Return the PPO RslRlOnPolicyRunnerCfg. See docs/INTERFACES.md.

  Hyperparameters match mjlab's upstream reference cartpole_ppo_runner_cfg()
  (mjlab.tasks.cartpole.cartpole_env_cfg.cartpole_ppo_runner_cfg). Round 3's
  hand-tuned config diverged from this reference and failed to converge;
  Round 4 aligns with the upstream-tested values.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    num_steps_per_env=32,
    max_iterations=500,
    save_interval=50,
    experiment_name=experiment_name,
  )
