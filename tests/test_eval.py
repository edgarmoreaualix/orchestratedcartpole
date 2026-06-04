import inspect

import pytest


def test_main_signature():
  import cartpole.eval

  assert "checkpoint" in inspect.signature(cartpole.eval.main).parameters


@pytest.mark.requires_checkpoint
def test_main_returns_metric_dict(tmp_path):
  import cartpole.eval

  import glob
  checkpoints = glob.glob("logs/**/*.pt", recursive=True)
  checkpoint = checkpoints[0]

  result = cartpole.eval.main(checkpoint, num_episodes=2)
  assert isinstance(result, dict)
  assert set(result.keys()) == {"mean_return", "std_return", "mean_episode_length"}
  assert isinstance(result["mean_return"], float)
  assert isinstance(result["std_return"], float)
  assert isinstance(result["mean_episode_length"], float)
