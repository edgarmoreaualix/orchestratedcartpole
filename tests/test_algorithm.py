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


def test_get_ppo_config_obs_normalization_enabled():
  cfg = get_ppo_config()
  assert cfg.actor.obs_normalization is True
  assert cfg.critic.obs_normalization is True


def test_get_ppo_config_init_std_is_05():
  cfg = get_ppo_config()
  assert cfg.actor.distribution_cfg["init_std"] == 0.5


def test_get_ppo_config_entropy_coef_is_005():
  cfg = get_ppo_config()
  assert cfg.algorithm.entropy_coef == 0.005
