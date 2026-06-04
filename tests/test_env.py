"""Tests for src/cartpole/env.py — owned by Environment Agent."""

import torch
import pytest

from cartpole.env import make_env


@pytest.fixture(scope="module")
def env_2():
    env = make_env(num_envs=2, device="cpu")
    yield env
    env.close()


def test_make_env_returns_managerbasedrlenv(env_2):
    from mjlab.envs import ManagerBasedRlEnv

    assert isinstance(env_2, ManagerBasedRlEnv)
    assert env_2.action_space.shape[0] == 2


def test_make_env_steps(env_2):
    env_2.reset()
    zero_action = torch.zeros(env_2.action_space.shape)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env_2.step(zero_action)
    assert reward.shape == (2,)


def test_make_env_play_mode():
    env = make_env(num_envs=1, device="cpu", play=True)
    try:
        assert env.unwrapped.cfg.scene.num_envs == 1
    finally:
        env.close()
