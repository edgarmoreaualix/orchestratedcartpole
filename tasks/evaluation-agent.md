# Evaluation Agent — Task Bus

You are the **Evaluation Agent**. You own `src/cartpole/eval.py` and `tests/test_eval.py`.

## Working directory and identity

Set the per-commit identity helper once per shell — do NOT use `git config user.name` (see [`POSTMORTEM.md`](../POSTMORTEM.md)):

```bash
cd /Users/edgarmoreau/rl/orchestratedcartpole-evaluation-agent
AGENT_NAME="Evaluation Agent"
AGENT_EMAIL="eval@cartpole.dev"
agent_commit() {
  git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit "$@"
}
```

Use `agent_commit -m "..."` for every commit you make.

## Workflow

Same shape as `tasks/env-agent.md`, on branch `feat/evaluation-agent`.

**Dependency note:** Your code imports `make_env` from `env.py` and reuses the PPO inference loader. You need a checkpoint to actually test end-to-end, so the integration test is gated on `pytest.mark.requires_checkpoint` and skipped if none is present. The unit-level tests run anyway.

---

## Tasks

### Task 1 — Implement evaluation entrypoint (2026-06-04)

**STATUS:** DONE — PR https://github.com/edgarmoreaualix/orchestratedcartpole/pull/2

**Scope:** Implement `src/cartpole/eval.py:main(checkpoint, num_envs, num_episodes, device, video) -> dict` per `docs/INTERFACES.md`. Returns `{"mean_return": float, "std_return": float, "mean_episode_length": float}`.

**Implementation outline:**
```python
import torch
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from cartpole.env import make_env
from cartpole.algorithm import get_ppo_config

def main(checkpoint, num_envs=1, num_episodes=10, device=None, video=False):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = make_env(num_envs=num_envs, device=device, play=True)
    agent_cfg = get_ppo_config(experiment_name="cartpole-eval")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    returns, lengths = [], []
    obs, _ = env.reset()
    ep_return = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, dtype=torch.long, device=device)
    done_count = 0
    while done_count < num_episodes:
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += reward
        ep_len += 1
        done = terminated | truncated
        for i in done.nonzero(as_tuple=False).flatten().tolist():
            returns.append(ep_return[i].item())
            lengths.append(ep_len[i].item())
            ep_return[i] = 0
            ep_len[i] = 0
            done_count += 1
    import statistics
    return {
        "mean_return": statistics.fmean(returns),
        "std_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "mean_episode_length": statistics.fmean(lengths),
    }
```

If `video=True`, additionally wrap the env with `mjlab.utils.wrappers.VideoRecorder` and write to `videos/eval/`. Tag this part with a comment so it's reviewable.

**Tests to add** in `tests/test_eval.py`:

- `test_main_signature` — `import cartpole.eval; assert "checkpoint" in inspect.signature(cartpole.eval.main).parameters`.
- `test_main_returns_metric_dict` (mark `requires_checkpoint`) — calls `main(<path>, num_episodes=2)`, asserts the returned dict has all three keys with the right types.

**Definition of done:**
- [ ] `main` returns the metric dict per the contract.
- [ ] `test_main_signature` passes: `uv run pytest tests/test_eval.py -v`.
- [ ] PR title `eval: implement policy eval + metrics`. Body has summary + test output.
- [ ] STATUS flipped to DONE, PR URL appended.

### Task 2 — Run eval against the v0.2.0 checkpoint and ship a GIF (2026-06-04, Round 2)

**STATUS:** TODO

**Branch:** `feat/evaluation-agent-r2` (already checked out in your worktree).

**Dependency:** This task requires the GitHub release `v0.2.0` to exist (created by the Training Agent in `feat/training-agent-r2`). If `gh release view v0.2.0` errors with "not found", STOP and post a PR comment `BLOCKED: waiting for v0.2.0 release`.

**Scope (three commits):**

**Commit 1 — remove the `requires_checkpoint` skip marker** on `test_main_returns_metric_dict` in `tests/test_eval.py`. We now have a checkpoint, so the integration test should run. Add a fixture that downloads the v0.2.0 checkpoint to a tmp path at test time.

**Commit 2 — run eval with video recording, convert to GIF.**

```bash
# Download checkpoint
mkdir -p artifacts
gh release download v0.2.0 --pattern "model_*.pt" --dir artifacts/

# Run eval with video
uv run python -m cartpole.eval --checkpoint artifacts/model_<N>.pt --num-envs=1 --num-episodes=3 --video=True

# Find the mp4
ls videos/eval/*.mp4
```

Capture the metrics dict returned by `eval.main` (you may need to add a small print statement at the bottom of `eval.py` for stdout visibility, or call eval programmatically from a tiny `scripts/run_eval.py` you add).

Convert the resulting mp4 to a high-quality GIF using ffmpeg (install with `brew install ffmpeg` if missing). Use the two-pass palettegen technique:

```bash
ffmpeg -y -i videos/eval/<input>.mp4 \
  -vf "fps=15,scale=480:-1:flags=lanczos,palettegen=stats_mode=full" \
  /tmp/palette.png
ffmpeg -y -i videos/eval/<input>.mp4 -i /tmp/palette.png \
  -filter_complex "fps=15,scale=480:-1:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer" \
  assets/cartpole-trained.gif
```

Target: GIF under 2 MB, framerate ~15 fps, width 480 px.

**Commit 3 — commit the GIF + the metrics-output script** at:
- `assets/cartpole-trained.gif`
- `scripts/run_eval.py` (a tiny entrypoint that loads the checkpoint, runs eval, prints metrics dict; useful for the README's "reproduce these numbers" instructions).

Add `videos/` to `.gitignore` (it should already be there but verify). Do NOT commit the raw mp4 or the checkpoint.

**Definition of done:**
- [ ] `tests/test_eval.py` no longer has `requires_checkpoint` skip on `test_main_returns_metric_dict`; it downloads + tests against v0.2.0 checkpoint.
- [ ] `assets/cartpole-trained.gif` exists and is < 2 MB.
- [ ] `scripts/run_eval.py` exists; running it prints the eval metrics dict.
- [ ] PR title: `eval: run against v0.2.0 + ship GIF`. Body: the metrics dict (mean_return, std_return, mean_episode_length) and a note about ffmpeg setup if it had to be installed.
- [ ] STATUS above flipped to DONE, PR URL appended.

### Task 3 — Fix the eval hang + ship GIF against v0.3.0 (2026-06-04, Round 3)

**STATUS:** TODO

**Branch:** `feat/evaluation-agent-r3`.

**Dependency:** This task requires release `v0.3.0` (Training Agent Round 3). If `gh release view v0.3.0` errors, STOP and post `BLOCKED: waiting for v0.3.0 release`.

**Context — what went wrong in Round 2:** The Round 2 eval against v0.2.0 hung indefinitely. Two parallel runs were stuck at 99% CPU for >20 minutes with the video output directory completely empty. Root cause: the eval `while done_count < num_episodes` loop only exits on `dones` being True, but under the combination of (a) `play=True` env config and (b) the `VideoRecorder` wrapper, the `dones` signal never fires. The VideoRecorder also makes each step ~1000× slower than no-video, so any infinite loop also pegs CPU.

**Scope (two commits):**

**Commit 1 — fix the eval loop.** In `src/cartpole/eval.py`, add a hard maximum-steps escape so the loop can never run forever, regardless of env configuration:

```python
# Hard cap: 1500 steps per episode (mjlab cartpole horizon is 1000)
# Protects against env configurations where dones never fires.
MAX_TOTAL_STEPS = num_episodes * 1500
step = 0
while done_count < num_episodes and step < MAX_TOTAL_STEPS:
    action = policy(obs)
    obs, reward, dones, _ = env.step(action)
    ep_return += reward
    ep_len += 1
    step += 1
    done = dones.bool()
    for i in done.nonzero(as_tuple=False).flatten().tolist():
        ...
        done_count += 1

# If the hard cap was hit before num_episodes were observed,
# record whatever is in-flight as a truncated episode.
if done_count < num_episodes:
    for i in range(num_envs):
        if ep_len[i] > 0:
            returns.append(ep_return[i].item())
            lengths.append(ep_len[i].item())
```

Additionally, investigate WHY `dones` never fired in Round 2. Possible causes worth one comment each:
- `play=True` in `make_env` strips terminations (likely root cause).
- The VideoRecorder wrapper consumes the dones signal.
- mjlab cartpole has a time-limit termination only by total sim-time, not step count.

Pick the simplest fix that makes the eval finish in reasonable time: either pass `play=False` to `make_env` (which keeps terminations active) or explicitly re-enable terminations on the env after construction. Document your fix choice in the PR body.

Update `tests/test_eval.py`:
- Update the integration test to download v0.3.0 instead of v0.2.0.
- The integration test should now complete in under 2 minutes wall-clock.

**Commit 2 — record the GIF against v0.3.0.**

```bash
mkdir -p artifacts
gh release download v0.3.0 --pattern "model_*.pt" --dir artifacts/
uv run python -m cartpole.eval --checkpoint artifacts/model_<N>.pt --num-envs=1 --num-episodes=3 --video=True 2>&1 | tee eval-r3.log
# verify videos/eval/*.mp4 exists
ls -la videos/eval/
# convert to gif using the existing two-pass palettegen recipe (see Task 2 above)
ffmpeg -y -i videos/eval/<input>.mp4 ...palette commands... assets/cartpole-trained.gif
```

**Replace** the existing `assets/cartpole-trained.gif` (do not commit a second file). Verify the GIF is under 2 MB and the policy is actually balancing (cart making small corrective moves, pole staying near vertical).

**Definition of done:**
- [ ] `src/cartpole/eval.py` has a max-steps escape and can no longer hang.
- [ ] `eval-r3.log` committed with the final metrics line visible.
- [ ] `assets/cartpole-trained.gif` updated; visually shows balancing.
- [ ] Integration test updated to v0.3.0; passes in < 2 min.
- [ ] PR title: `eval: fix hang + record GIF against v0.3.0`. Body: root-cause analysis of the hang, the fix chosen, the new metrics dict (mean_return, std_return, mean_episode_length).
- [ ] STATUS above flipped to DONE, PR URL appended.
