# orchestratedcartpole

A PPO policy trained on the mjlab CartPole-Balance task, built by a team of five specialized component agents.

---

## What is CartPole?

CartPole-Balance is a canonical reinforcement learning benchmark: a pole is mounted on a cart that slides along a one-dimensional track, and the policy must apply left/right forces to keep the pole upright. The task is widely used because it has a compact observation space (cart position, cart velocity, pole angle, pole angular velocity) and a binary action space (push left, push right), yet still requires the policy to discover a non-trivial balancing strategy. Successful policies sustain the pole indefinitely without exceeding track limits.

## What is mjlab?

mjlab is a GPU-batched parallel simulation framework built on MuJoCo and NVIDIA Warp, positioned as a sibling to Isaac Lab. It exposes a `ManagerBasedRlEnv` interface compatible with rsl_rl runners and supports thousands of simultaneous environment instances on a single GPU. Physics fidelity is provided by MuJoCo; NVIDIA Warp handles the batching and device-side data management, enabling fast vectorized rollout collection for on-policy algorithms.

---

## System architecture

Five specialized agents built this codebase, each owning one layer of the stack:

| Agent | Module | Contribution |
|---|---|---|
| Environment Agent | `src/cartpole/env.py` | `make_env` factory — wraps mjlab Cartpole-Balance with `RslRlVecEnvWrapper` |
| Algorithm Agent | `src/cartpole/algorithm.py` | `get_ppo_config` — actor/critic network sizes and PPO hyperparameters |
| Training Agent | `src/cartpole/train.py` | `train.main` — drives `MjlabOnPolicyRunner` to convergence, writes checkpoints |
| Evaluation Agent | `src/cartpole/eval.py` | `eval.main` — loads a checkpoint, runs rollouts, returns return metrics |
| Documentation Agent | `README.md`, `docs/architecture.md` | project documentation |

See [docs/architecture.md](docs/architecture.md) for the data-flow diagram and design rationale.

---

## Install

```bash
uv sync
uv sync --group dev   # also installs pytest and ruff
```

Requires Python 3.10–3.13 and a CUDA-capable GPU for GPU-batched sim.

---

## Train

```bash
uv run python -m cartpole.train --num-envs 4096 --max-iterations 200
```

Checkpoints are written under `logs/cartpole/<run>/model_<iteration>.pt`.

---

## Evaluate

```bash
uv run python -m cartpole.eval --checkpoint logs/cartpole/<run>/model_200.pt --num-episodes 10
```

Prints mean return, return std-dev, and mean episode length.

---

## Repository layout

```
src/
  cartpole/
    __init__.py          — package init
    env.py               — make_env factory (Environment Agent)
    algorithm.py         — get_ppo_config (Algorithm Agent)
    train.py             — training entrypoint (Training Agent)
    eval.py              — evaluation entrypoint (Evaluation Agent)
tests/
  __init__.py            — test package init
  test_scaffold.py       — smoke tests for imports and interface contracts
docs/
  INTERFACES.md          — canonical function signatures; owned by orchestrator
  architecture.md        — data-flow diagram and design rationale
tasks/
  orchestrator.md        — orchestrator task bus
  documentation-agent.md — documentation agent task bus
  env-agent.md           — environment agent task bus
  algorithm-agent.md     — algorithm agent task bus
  training-agent.md      — training agent task bus
  evaluation-agent.md    — evaluation agent task bus
.github/
  CODEOWNERS             — per-module ownership rules
  workflows/ci.yml       — lint, type-check, and test pipeline
```

---

## License

MIT — see [LICENSE](LICENSE).
