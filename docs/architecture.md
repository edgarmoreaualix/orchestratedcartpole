# Architecture

## Data flow

```
make_env(num_envs, device)
        │
        ▼
RslRlVecEnvWrapper          ← wraps ManagerBasedRlEnv with the rsl_rl gym-compatible interface
        │
        ▼
MjlabOnPolicyRunner         ← drives the PPO training loop; config supplied by get_ppo_config()
  ├── collect_rollouts()    ← steps the wrapped env; fills rollout buffer
  ├── compute_returns()     ← GAE advantage estimation
  └── update_policy()       ← PPO clipped surrogate objective, value loss, entropy bonus
        │
        ▼
checkpoint (model_N.pt)     ← serialized actor/critic weights written under logs/cartpole/<run>/
        │
        ▼
eval.main(checkpoint, ...)  ← loads weights, runs num_episodes rollouts, returns metrics dict
```

---

## Module interface signatures

Full, versioned signatures live in [`docs/INTERFACES.md`](INTERFACES.md). Summary:

| Symbol | Module | Signature |
|---|---|---|
| `make_env` | `env.py` | `(num_envs=1, device="cpu", play=False) → ManagerBasedRlEnv` |
| `get_ppo_config` | `algorithm.py` | `(experiment_name="cartpole") → RslRlOnPolicyRunnerCfg` |
| `train.main` | `train.py` | `(num_envs=4096, max_iterations=200, device=None, seed=1) → None` |
| `eval.main` | `eval.py` | `(checkpoint, num_envs=1, num_episodes=10, device=None, video=False) → dict` |

Any signature change requires a PR against `docs/INTERFACES.md` before the implementation PR. See the versioning section in `INTERFACES.md`.

---

## PPO hyperparameter rationale

Hyperparameters are defined in `algorithm.get_ppo_config` (see the Algorithm Agent's PR on `feat/algorithm-agent`). Key choices:

- **Large batch (4 096 envs × rollout horizon):** CartPole-Balance has low observation dimensionality and a short effective episode length. A large parallel batch reduces gradient variance without requiring many sequential environment steps, keeping wall-clock training time low.
- **Clipping ε = 0.2:** Standard PPO clipping range. CartPole's reward landscape is smooth, so a conservative clip is sufficient; aggressive clipping would slow convergence unnecessarily.
- **GAE λ = 0.95:** High λ biases toward lower-variance returns at the cost of slightly higher bias — acceptable here because the value function converges quickly on a task this simple.
- **Entropy coefficient:** A small entropy bonus prevents the policy from collapsing to a deterministic boundary action early in training, when the value function is still poorly calibrated.
- **200 iterations:** Empirically sufficient for convergence on CartPole-Balance at 4 096 envs. A linear learning-rate schedule decays toward zero over these iterations.

---

## Sim-to-real notes

CartPole-Balance is among the simplest sim-to-real transfer cases:

- **No friction estimation required.** The cart–track interaction is a single scalar; its exact value barely affects the learned policy because PPO generalizes over the reward signal, not the dynamics model.
- **No actuator dynamics.** The action is a direct force applied to the cart. Real linear actuators introduce time constants and torque limits, but for a balancing task these can be absorbed by a proportional deadband without retraining.
- **Observation noise tolerance.** The four-dimensional observation (cart position, cart velocity, pole angle, pole angular velocity) is directly measurable from encoders and an IMU. Sensor noise at the scale of typical MEMS devices does not destabilize a well-trained PPO policy.

What would change at humanoid scale:
- **Contact-rich dynamics** (feet, ground) require friction and restitution estimation that cartpole does not exercise.
- **Actuator delay** (typically 1–3 control steps on hydraulic or Series Elastic Actuators) must be modeled explicitly or randomized during training to avoid policy degradation on hardware.
- **Domain randomization** over link masses, center-of-mass offsets, and joint damping becomes load-bearing; cartpole's low dimensionality makes it robust without randomization.
