# Interface contracts

The five specialized agents code against the function signatures below. Each agent owns one Python module and depends on this document for the shape of what they must implement. If a signature here changes, every dependent agent rebases.

## `src/cartpole/env.py` — Environment Agent

```python
from mjlab.envs import ManagerBasedRlEnv

def make_env(num_envs: int = 1, device: str = "cpu", play: bool = False) -> ManagerBasedRlEnv:
    """Build the mjlab Mjlab-Cartpole-Balance env at the requested scale.

    Args:
      num_envs: number of parallel envs.
      device:   torch device ("cpu", "cuda:0", "mps").
      play:     if True, configure env for deterministic playback.

    Returns:
      An instantiated `ManagerBasedRlEnv` ready for `.reset()` / `.step()`.
    """
```

## `src/cartpole/algorithm.py` — Algorithm Agent

```python
from mjlab.rl import RslRlOnPolicyRunnerCfg

def get_ppo_config(experiment_name: str = "cartpole") -> RslRlOnPolicyRunnerCfg:
    """Return the PPO config (actor/critic, hyperparameters) for cartpole.

    Args:
      experiment_name: name used for the log directory.

    Returns:
      A `RslRlOnPolicyRunnerCfg` consumed by `train.main`.
    """
```

## `src/cartpole/train.py` — Training Agent

```python
def main(
    num_envs: int = 4096,
    max_iterations: int = 200,
    device: str | None = None,
    seed: int = 1,
) -> None:
    """Train PPO on cartpole.

    Loads env via `env.make_env`, config via `algorithm.get_ppo_config`,
    runs `rsl_rl.OnPolicyRunner`, writes checkpoints under `logs/cartpole/`.
    """
```

## `src/cartpole/eval.py` — Evaluation Agent

```python
def main(
    checkpoint: str,
    num_envs: int = 1,
    num_episodes: int = 10,
    device: str | None = None,
    video: bool = False,
) -> dict:
    """Load a checkpoint, run eval episodes, return metrics.

    Returns:
      {"mean_return": float, "std_return": float, "mean_episode_length": float}
    """
```

## Versioning

If you need to change a signature, open a PR titled `interface: <reason>` against this file first and tag every dependent agent. Do not change a signature inside an implementation PR.
