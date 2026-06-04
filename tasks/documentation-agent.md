# Documentation Agent — Task Bus

You are the **Documentation Agent**. You own `README.md` and `docs/architecture.md`. You do **not** touch `docs/INTERFACES.md` — that file is owned by the orchestrator.

## Working directory and identity

Set the per-commit identity helper once per shell — do NOT use `git config user.name` (see [`POSTMORTEM.md`](../POSTMORTEM.md)):

```bash
cd /Users/edgarmoreau/rl/orchestratedcartpole-documentation-agent
AGENT_NAME="Documentation Agent"
AGENT_EMAIL="docs@cartpole.dev"
agent_commit() {
  git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit "$@"
}
```

Use `agent_commit -m "..."` for every commit you make.

## Workflow

Same shape as `tasks/env-agent.md`, on branch `feat/documentation-agent`. You don't need `uv sync` since your task is markdown only.

---

## Tasks

### Task 1 — Write the system README and architecture doc (2026-06-04)

**STATUS:** DONE — https://github.com/edgarmoreaualix/orchestratedcartpole/pull/1

**Scope:** Replace the placeholder `README.md` and `docs/architecture.md` with real documentation describing **the system, not the orchestration process**. The README is the project's front page: someone landing on it should learn what the project does, how to install it, how to train, how to evaluate, and how the codebase is structured.

**README must contain, in this order:**

1. **Title and one-sentence pitch.** `# orchestratedcartpole` then "A PPO policy trained on the mjlab CartPole-Balance task, built by a team of five specialized component agents."
2. **What is cartpole?** ~80 words: the classic balancing benchmark, why it's a canonical RL task, what the agent learns, what the action and observation spaces are.
3. **What is mjlab?** ~60 words: a MuJoCo-based RL environment framework on top of NVIDIA Warp, sibling to Isaac Lab, designed for GPU-batched parallel sim.
4. **System architecture.** A table of the five component agents and what each contributed. Link to `docs/architecture.md` for the full design.
5. **Install.** Exact `uv sync` commands.
6. **Train.** Exact `uv run python -m cartpole.train --num-envs 4096 --max-iterations 200` command.
7. **Evaluate.** Exact `uv run python -m cartpole.eval --checkpoint logs/cartpole/<run>/model_200.pt --num-episodes 10` command.
8. **Repository layout.** Tree of `src/`, `tests/`, `docs/`, `tasks/`, `.github/` with a one-line description of each file.
9. **License.** MIT, reference to LICENSE.

**`docs/architecture.md` must contain:**

- ASCII diagram of the data flow: `make_env` → `RslRlVecEnvWrapper` → `MjlabOnPolicyRunner` → checkpoint → `eval.main`.
- Per-module interface signatures (link to `docs/INTERFACES.md`).
- Rationale for the chosen PPO hyperparameters (cite Algorithm Agent's PR).
- Sim-to-real notes: why cartpole's domain is trivially close to real (no friction estimation, no actuator dynamics) and what would change for humanoid scale.

**Tone:** technical and concise. No first person. No mention of agents "learning" or "teaching" anything — the agents are software components that built a software system. Read like the docs of a polished open-source project, not a tutorial.

**Definition of done:**
- [ ] `README.md` replaced (no longer a scaffold).
- [ ] `docs/architecture.md` replaced.
- [ ] Both pass `uv run ruff check` (no Python in these files, but check anyway).
- [ ] PR title `docs: replace scaffold README and architecture doc`.
- [ ] STATUS flipped to DONE, PR URL appended.

### Task 2 — Add a Results section with GIF + metrics (2026-06-04, Round 2)

**STATUS:** TODO

**Branch:** `feat/documentation-agent-r2` (already checked out in your worktree).

**Dependency:** This task requires the GitHub release `v0.2.0` (Training Agent) and `assets/cartpole-trained.gif` on main (Evaluation Agent). If either is missing, STOP and post `BLOCKED: waiting for <what>`.

**Scope (single commit):**

Update `README.md` and `docs/architecture.md` with:

1. **In `README.md`, insert a new `## Results` section** after the existing `## Evaluate` section (before `## Repository layout`):

```markdown
## Results

A PPO policy trained on `Mjlab-Cartpole-Balance` (`num_envs=<from release>`, `max_iterations=<from release>`, CPU on Apple Silicon) achieves:

| Metric | Value |
|---|---|
| Mean episode return | <from eval> |
| Std episode return | <from eval> |
| Mean episode length | <from eval> |

![Trained policy balancing the pole](assets/cartpole-trained.gif)

The checkpoint is published as release [`v0.2.0`](https://github.com/edgarmoreaualix/orchestratedcartpole/releases/tag/v0.2.0). Reproduce by running:

\`\`\`bash
gh release download v0.2.0 --pattern "model_*.pt" --dir artifacts/
uv run python scripts/run_eval.py --checkpoint artifacts/model_<N>.pt
\`\`\`
```

(Backticks escaped above for this task file; do not escape them in the actual README.)

Pull the exact numbers from the Evaluation Agent's PR body (`feat/evaluation-agent-r2`). Pull the hyperparameters from the Training Agent's PR body (`feat/training-agent-r2`).

2. **Append to `README.md` after Results, before License**, a small note:

```markdown
## Engineering history

The repository was built by five specialized agents (Environment, Algorithm, Training, Evaluation, Documentation) coordinated by an orchestrator. See [`POSTMORTEM.md`](POSTMORTEM.md) for the Round 1 author-identity bleed bug we hit and fixed.
```

3. **In `docs/architecture.md`**, add a short "## Trained policy notes" section at the bottom citing the v0.2.0 hyperparameters and the observed mean return; note that CPU training was small-scale and GPU training at `num_envs=4096` would converge ~10-30× faster.

**Definition of done:**
- [ ] `README.md` has a Results section with the GIF, metrics table, and reproduce-command. The GIF renders on GitHub's rendered README view.
- [ ] `README.md` has an Engineering history section linking POSTMORTEM.md.
- [ ] `docs/architecture.md` has a Trained policy notes section.
- [ ] PR title: `docs: add Results section with GIF + v0.2.0 metrics`.
- [ ] STATUS above flipped to DONE, PR URL appended.

### Task 3 — Update Results with v0.3.0 + add iteration notes (2026-06-04, Round 3)

**STATUS:** TODO

**Branch:** `feat/documentation-agent-r3`.

**Dependency:** Evaluation Agent Round 3 PR must be merged (new `assets/cartpole-trained.gif` on `main`, release v0.3.0 live). If either is missing, STOP and post `BLOCKED:`.

**Scope (one commit):**

Update `README.md`:

1. **Replace the Results section's metrics table** with v0.3.0 numbers pulled from the Evaluation Agent's PR body (`feat/evaluation-agent-r3`).
2. **Replace the release link** from v0.2.0 to v0.3.0. Keep the reproduce-command pointing at v0.3.0.
3. **Insert a new `### Engineering iteration notes` subsection inside Results** with the following text (substitute the real numbers):

```markdown
### Engineering iteration notes

The first training run (release [`v0.2.0`](https://github.com/edgarmoreaualix/orchestratedcartpole/releases/tag/v0.2.0)) used the default PPO config and achieved a mean episode reward of approximately 12 / 1000 — the policy moved the cart but did not balance the pole. The eval pipeline simultaneously hung due to an infinite-loop bug when terminations were inactive under the `VideoRecorder` wrapper.

The second iteration ([`v0.3.0`](https://github.com/edgarmoreaualix/orchestratedcartpole/releases/tag/v0.3.0)) addressed both: the Algorithm Agent enabled observation normalization, lowered the initial action standard deviation, and reduced the entropy coefficient; the Training Agent re-ran with five times the iteration count; the Evaluation Agent added a hard step cap to the eval loop and re-enabled terminations during eval. The GIF above is the v0.3.0 policy.

See [`POSTMORTEM.md`](POSTMORTEM.md) for the Round 1 author-identity bug. See PR history for the Round 2 → Round 3 iteration trail.
```

4. In `docs/architecture.md`, append a one-paragraph "## Convergence and hyperparameter sensitivity" section noting that cartpole is sensitive to obs normalization and initial action standard deviation, citing the v0.2.0 → v0.3.0 jump.

**Definition of done:**
- [ ] `README.md` Results section reflects v0.3.0.
- [ ] `Engineering iteration notes` subsection added with the v0.2.0 → v0.3.0 narrative.
- [ ] `docs/architecture.md` has the convergence note.
- [ ] PR title: `docs: update Results to v0.3.0 + iteration notes`.
- [ ] STATUS above flipped to DONE, PR URL appended.
