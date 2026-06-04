"""Tests for scripts/view.py — owned by Environment Agent."""


def test_view_module_imports():
  import scripts.view

  assert hasattr(scripts.view, "main")


def test_view_config_defaults():
  from scripts.view import ViewConfig

  cfg = ViewConfig()
  assert cfg.agent == "zero"
  assert cfg.viewer == "native"
  assert cfg.num_envs == 1
