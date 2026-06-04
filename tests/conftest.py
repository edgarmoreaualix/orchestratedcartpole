import os
import pathlib
import sys

import pytest

# Allow `import scripts.view` from tests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def pytest_configure(config):
  config.addinivalue_line(
    "markers",
    "requires_checkpoint: skip test when no checkpoint file is available",
  )


def pytest_runtest_setup(item):
  if item.get_closest_marker("requires_checkpoint"):
    # Look for any .pt file under logs/ as a proxy for a trained checkpoint
    checkpoint_exists = any(
      True
      for root, _, files in os.walk("logs")
      for f in files
      if f.endswith(".pt")
    ) if os.path.isdir("logs") else False
    if not checkpoint_exists:
      pytest.skip("no checkpoint found — skipping integration test")
