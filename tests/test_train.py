"""Tests for src/cartpole/train.py — owned by Training Agent."""

import pathlib

import pytest

from cartpole import train


@pytest.mark.slow
def test_main_runs_one_iteration_cpu():
  train.main(num_envs=4, max_iterations=1, device="cpu")
  assert pathlib.Path("logs/cartpole").is_dir()
