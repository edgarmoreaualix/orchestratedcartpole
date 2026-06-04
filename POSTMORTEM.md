# Round 1 post-mortem — author identity rewrite

## Symptom

After Round 1 merged, every implementation commit on `main` was attributed to the orchestrator account (`Edgar Moreau <edgarmoreaualix@gmail.com>`) rather than to the agent that authored the code. The synthetic agent identities (`Environment Agent`, `Algorithm Agent`, etc.) were lost.

## Root cause

Two factors compounded:

1. **Per-worktree `git config user.name` writes to the shared `.git/config`.** Git worktrees do not get their own `[user]` section by default. When each agent's task file ran `git config user.name "<Agent>"` in its worktree, the value clobbered the previous agent's value across all worktrees. Whichever agent ran the command last determined the identity for every uncommitted-yet commit, regardless of which worktree the commit was made from.
2. **GitHub squash-merges author the squashed commit under the merger's identity, not the original branch authors.** Even when an agent's original branch commits had the correct author metadata, `gh pr merge --squash` rewrote that to the GitHub-API caller (the orchestrator's account).

Either factor alone would have corrupted attribution. Together they made it inevitable.

## Fix

This was repaired in a single rewrite step:

```bash
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --env-filter '
  msg=$(git log -1 --format=%s $GIT_COMMIT)
  case "$msg" in
    "env: "*)   export GIT_AUTHOR_NAME="Environment Agent";    export GIT_AUTHOR_EMAIL="env@cartpole.dev" ;;
    "algo: "*)  export GIT_AUTHOR_NAME="Algorithm Agent";      export GIT_AUTHOR_EMAIL="algo@cartpole.dev" ;;
    "train: "*) export GIT_AUTHOR_NAME="Training Agent";       export GIT_AUTHOR_EMAIL="training@cartpole.dev" ;;
    "eval: "*)  export GIT_AUTHOR_NAME="Evaluation Agent";     export GIT_AUTHOR_EMAIL="eval@cartpole.dev" ;;
    "docs: "*)  export GIT_AUTHOR_NAME="Documentation Agent";  export GIT_AUTHOR_EMAIL="docs@cartpole.dev" ;;
  esac
' -- main
git push --force-with-lease origin main
```

The dispatch is by conventional-commits prefix in the subject line, which is reliably present because the task files mandated it. `chore:` commits remain authored by the orchestrator, which is correct.

This was a destructive `--force` push to `main`. It is safe in this repository because (a) the repository has no other collaborators, and (b) the feature branches were already deleted on the remote.

## Prevention for Round 2

The next dispatch will replace the per-worktree `git config user.name` pattern with **per-commit identity** via inline overrides:

```bash
git -c user.name="Training Agent" \
    -c user.email="training@cartpole.dev" \
    commit -m "..."
```

The `-c key=value` form sets configuration only for the duration of that single git invocation. It does not touch any persistent config file and therefore cannot bleed across worktrees.

Additionally, every agent's Round 2 task file will define a small shell function the agent invokes for every commit, so the identity is impossible to forget:

```bash
agent_commit() {
  git -c user.name="$AGENT_NAME" -c user.email="$AGENT_EMAIL" commit "$@"
}
```

The squash-merge author-rewrite step is preserved as an orchestrator-side step: after every Round, the orchestrator runs `git filter-branch` once to normalize attribution on the merge commits.
