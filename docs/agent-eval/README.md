# Agent guide evaluation

This folder measures whether the agent-facing docs (`AGENTS.md`, `skills/`, the JS app guide, `docs/source/SDK/`) take a fresh AI coding agent from a plain user request to a good app, **using the docs alone**. It evaluates the docs, not the agents.

## How it works

- **The requests:** 10 written as a non-technical user would phrase them. They describe the top Reachy Mini apps on the Hub, and none names an app type or the original app.
- **The builders:** one fresh agent per request. Each gets a docs snapshot at a git ref as its only source, no robot and no web search, and must not publish. Each returns a structured report:
  - the app type it built and when it decided;
  - every reason, tagged `DOC path:line`, `OWN_KNOWLEDGE` or `INFERENCE`;
  - friction, with `path:line` and its cost;
  - its confidence that the app works on a real robot;
  - its top 3 doc changes.
- **The synthesis:** an app-type table checked against the expected type for each request (expected types come from the guide's own criteria), friction counts by theme, and a comparison with a baseline file.
- **What the agents aren't told:** the JS-vs-Python split is the main metric. Telling them would bias every agent toward JS.

## Running it

Run it from a Claude Code session with the Workflow tool:

```
Workflow({
  scriptPath: "docs/agent-eval/guide-eval.workflow.js",
  args: {
    ref: "origin/<branch-under-test>",
    repo: "/path/to/a/reachy_mini/clone",
    workDir: "/path/to/an/empty/scratch/dir",
    baselinePath: "docs/agent-eval/results/<latest>.md",
    only: ["04", "09"]            // optional subset
  }
})
```

It spawns 12 agents: 1 for setup, 10 builders and 1 for the synthesis. A full run takes about 15–25 min.

When it finishes, save the returned `synthesis` as `results/<date>-<ref>-<sha>.md`. Afterwards, remove the snapshot worktree with `git worktree remove <workDir>/reachy_mini`.

## Reading results

- **App type:** a JS choice is only as good as its reasons. Check that the reasons cite docs, and look out for "X isn't possible" claims tagged `OWN_KNOWLEDGE`. That was the failure that started this evaluation: an agent claimed from stale memory that JS apps couldn't use the robot's microphone.
- **Controls:** compare the themes the change targeted with themes it didn't touch. If the untouched themes move too, the difference is run-to-run noise. Each prompt runs once, so treat a single verdict as a hint.

## History

| Results file | Docs ref | Outcome |
|---|---|---|
| [`2026-10-20-main-24f74f0`](results/2026-10-20-main-24f74f0.md) | main after #1481 | 8 JS / 2 Python; 10/10 hit the plan.md contradiction, 8/8 JS agents assumed "Wireless only" (stale) |
| [`2026-10-20-pr1482-cd3cebe`](results/2026-10-20-pr1482-cd3cebe.md) | #1482 | targeted themes → 0; controls unchanged; Python cases now decided from the Constraints column |
