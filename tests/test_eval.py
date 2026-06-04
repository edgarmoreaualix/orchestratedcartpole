import inspect
import subprocess

import pytest


def test_main_signature():
  import cartpole.eval

  assert "checkpoint" in inspect.signature(cartpole.eval.main).parameters


@pytest.fixture
def v03_checkpoint(tmp_path):
  """Download the v0.3.0 checkpoint from the GitHub release.

  Skips if the release does not exist yet (Training Agent hasn't shipped it).
  """
  result = subprocess.run(
    ["gh", "release", "view", "v0.3.0"],
    capture_output=True,
  )
  if result.returncode != 0:
    pytest.skip("v0.3.0 release not found — waiting for Training Agent Round 3")

  subprocess.run(
    ["gh", "release", "download", "v0.3.0", "--pattern", "model_*.pt", "--dir", str(tmp_path)],
    check=True,
  )
  checkpoints = list(tmp_path.glob("model_*.pt"))
  assert checkpoints, "No checkpoint downloaded"
  return str(checkpoints[0])


def test_main_returns_metric_dict(v03_checkpoint):
  import cartpole.eval

  result = cartpole.eval.main(v03_checkpoint, num_episodes=2)
  assert isinstance(result, dict)
  assert set(result.keys()) == {"mean_return", "std_return", "mean_episode_length"}
  assert isinstance(result["mean_return"], float)
  assert isinstance(result["std_return"], float)
  assert isinstance(result["mean_episode_length"], float)
