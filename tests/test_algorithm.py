"""Tests for src/cartpole/algorithm.py."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from cartpole.algorithm import get_ppo_config


def test_get_ppo_config_returns_runner_cfg():
  cfg = get_ppo_config()
  assert isinstance(cfg, RslRlOnPolicyRunnerCfg)


def test_get_ppo_config_respects_experiment_name():
  cfg = get_ppo_config(experiment_name="my-run")
  assert cfg.experiment_name == "my-run"


def test_get_ppo_config_hyperparameters_sensible():
  cfg = get_ppo_config()
  assert 0 < cfg.algorithm.clip_param < 1
  assert cfg.algorithm.learning_rate > 0
  assert 0 < cfg.algorithm.lam <= 1


def test_get_ppo_config_obs_normalization_disabled():
  cfg = get_ppo_config()
  assert cfg.actor.obs_normalization is False
  assert cfg.critic.obs_normalization is False


def test_get_ppo_config_uses_elu_activation():
  cfg = get_ppo_config()
  assert cfg.actor.activation == "elu"


def test_get_ppo_config_uses_adaptive_lr_schedule():
  cfg = get_ppo_config()
  assert cfg.algorithm.schedule == "adaptive"
  assert cfg.algorithm.desired_kl == 0.01


def test_get_ppo_config_uses_value_loss_clipping():
  cfg = get_ppo_config()
  assert cfg.algorithm.use_clipped_value_loss is True
  assert cfg.algorithm.max_grad_norm == 1.0
