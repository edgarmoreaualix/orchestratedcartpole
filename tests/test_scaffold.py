"""Scaffold smoke tests — verify all modules import.

Each component agent replaces / extends these with real tests for their own
module in Round 1.
"""


def test_env_module_imports():
  from cartpole import env

  assert callable(env.make_env)


def test_algorithm_module_imports():
  from cartpole import algorithm

  assert callable(algorithm.get_ppo_config)


def test_train_module_imports():
  from cartpole import train

  assert callable(train.main)


def test_eval_module_imports():
  from cartpole import eval as cp_eval

  assert callable(cp_eval.main)
