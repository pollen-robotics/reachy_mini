# Reachy Mini SDK Contributor Guide for AI Agents

This guide is for agents working **inside this repo** on the Reachy Mini SDK itself: the Python package under `src/reachy_mini/`, the daemon, the JS client at `js/`, the docs, and the test suite.

> If you are helping a user build an *app* on top of Reachy Mini (a Hugging Face Space, a Python control script using `reachy-mini-app-assistant`, etc.), read [`docs/contributing/building_apps.md`](docs/contributing/building_apps.md) instead. That is the user-facing app-development guide.

---

## Read first

Before any non-trivial change, skim these in order:

1. [`docs/contributing/architecture.md`](docs/contributing/architecture.md) — structural map of the codebase: what each package does, where the SDK ↔ daemon ↔ JS boundaries are, and a "where to look for X" lookup table.
2. [`docs/contributing/contributing.md`](docs/contributing/contributing.md) — environment setup, ruff, mypy, pytest, pre-commit, AI-assisted commit conventions.
3. [`docs/contributing/generate_docs.md`](docs/contributing/generate_docs.md) — building the Hugging Face docs and regenerating `docs/source/API/openapi.json`.

## Repo layout at a glance

| Path | What lives there |
|---|---|
| `src/reachy_mini/` | The Python SDK package (see architecture.md for a per-subpackage breakdown). |
| `src/reachy_mini/daemon/` | FastAPI daemon (REST + WebSocket at `:8000/api`), three backends (real robot, MuJoCo, mockup). |
| `src/reachy_mini/media/` | Audio/video. `AudioBase` + `CameraBase` abstractions, GStreamer pipelines, three backends (`LOCAL`, `WEBRTC`, `NO_MEDIA`). |
| `src/reachy_mini/motion/` | `goto_target` interpolation, `set_target` real-time path, recorded-move playback. |
| `src/reachy_mini/io/` | Protocol + WebSocket transport between SDK and daemon. |
| `src/reachy_mini/apps/` | `ReachyMiniApp` scaffold + the `reachy-mini-app-assistant` CLI. |
| `src/reachy_mini/{kinematics,descriptions,tools,utils}/` | Kinematics / URDFs / motor reflash tool / helpers. |
| `js/reachy-mini.js` | Browser ES-module SDK (WebRTC client for HF Spaces). |
| `tests/{unit_tests,integration_tests}/` | Pytest suites. Hardware-bound tests use markers (`audio`, `video`, `wireless`, `ipc_resolution`). |
| `examples/` | Runnable demo scripts. |
| `docs/source/` | Hugging Face doc-builder source. `_toctree.yml` is the navigation. |
| `docs/source/API/` | Generated API reference (per-module `*.mdx` + `openapi.json`). |
| `docs/contributing/` | These contributor docs. |
| `scripts/` | Maintenance scripts (e.g. `generate_openapi.py`). |

## API surface — grep before adding

Before introducing a new public function, helper, or REST endpoint, look at what already exists:

- **Python SDK** — `docs/source/API/` contains `reachymini.mdx`, `motion.mdx`, `media.mdx`, `apps.mdx`, `tools.mdx`, `utils.mdx`. Each lists the public functions/classes for that subpackage.
- **Daemon REST** — `docs/source/API/openapi.json` is the canonical list of routes (also browsable at `:8000/docs` when the daemon runs). `docs/source/API/rest-api.mdx` and `daemon.mdx` give a curated overview.

Grepping `docs/source/API/` first avoids reinventing helpers that are already exported.

## Working conventions

- **Pre-commit.** Install once per clone: `pre-commit install`. Hooks defined in `.pre-commit-config.yaml` run ruff-check (with `--select I --select D --ignore D203 --ignore D213` — import order + docstrings) and ruff-format on staged files. Don't bypass hooks.
- **Lint + types.** Run `ruff check .` and `mypy` before declaring work done. Both gates run in CI; fix all errors locally first.
- **OpenAPI regeneration.** After touching anything under `src/reachy_mini/daemon/app/routers/` or related Pydantic models, run `uv run python scripts/generate_openapi.py` and commit the updated `docs/source/API/openapi.json`. CI fails if the spec drifts.
- **Tests.** `pytest -m "not audio and not video and not wireless and not ipc_resolution"` is the no-hardware run that mirrors CI. Mark new hardware-dependent tests with the matching pytest marker so CI skips them.
- **Abstractions.** When adding to `media/` or `daemon/backend/`, extend the existing `AudioBase` / `CameraBase` / `Backend` interfaces rather than duplicating logic per backend.
- **Docs.** When adding a new `docs/source/*.md` page, register it in `docs/source/_toctree.yml` (otherwise `doc-builder preview` won't see it). Files outside `docs/source/` (like these contributor docs) are not in the toctree.

## AI-assisted commits

AI assistance is welcome and explicitly accepted on this repo. Follow the [Linux kernel coding-assistants convention](https://github.com/torvalds/linux/blob/master/Documentation/process/coding-assistants.rst).

Append a single trailer to commits where an AI helped:

```
Assisted-by: Claude:claude-opus-4-7
```

Use the exact model ID from the assistant's session context. If specialized analysis tools were used (coccinelle, sparse, mypy plugins, etc.), append them space-separated. Do **not** use `Co-Authored-By:`. `Signed-off-by:` is for humans only — it certifies the DCO and an AI cannot certify it.

## Reusable skills

The [`skills/`](skills/) folder is primarily aimed at agents helping users build apps, but several are generic enough to be useful when working on the SDK itself. Read them as needed:

| Skill | Why it helps SDK work |
|---|---|
| [`skills/deep-dive-docs.md`](skills/deep-dive-docs.md) | Pointers into the SDK source layout — complementary to `architecture.md`. |
| [`skills/motion-philosophy.md`](skills/motion-philosophy.md) | Explains the design intent behind `goto_target` vs `set_target`; useful before touching `motion/`. |
| [`skills/control-loops.md`](skills/control-loops.md) | Real-time control patterns using `set_target()`; useful when changing control timing or the daemon loop. |
| [`skills/symbolic-motion.md`](skills/symbolic-motion.md) | Mathematical motion definition; useful when adding interpolation curves or motion primitives. |
| [`skills/debugging.md`](skills/debugging.md) | Daemon lifecycle, log inspection, connectivity checks. |
| [`skills/rest-api.md`](skills/rest-api.md) *(mixed)* | Endpoint reference — useful when adding/refactoring routers. |
| [`skills/safe-torque.md`](skills/safe-torque.md) *(mixed)* | Motor enable/disable semantics — relevant for daemon and motor-control code. |
| [`skills/setup-environment.md`](skills/setup-environment.md) *(mixed)* | Reference setup for example apps; useful when validating SDK changes against example code. |

The remaining skills (`ai-integration.md`, `create-app.md`, `interaction-patterns.md`, `testing-apps.md`) are app-builder-specific and can be skipped for SDK work.

## PR hygiene

- Keep PRs scoped — one concern per PR. If a refactor surfaces during a feature change, prefer a follow-up PR.
- Add tests next to the code: a unit test under `tests/unit_tests/` for pure-Python logic, an integration test (with the right marker) for hardware paths.
- Update `docs/source/_toctree.yml` only when adding a brand-new doc page.
- If you change daemon routes/models, regenerate `openapi.json` in the same PR.

## When in doubt

- Architecture questions → `docs/contributing/architecture.md`.
- Setup / lint / test questions → `docs/contributing/contributing.md`.
- Doc build / API regeneration → `docs/contributing/generate_docs.md`.
- App-builder questions (the user is *using* the SDK, not editing it) → `docs/contributing/building_apps.md`.
- Existing public API → grep `docs/source/API/`.
- **Domain / user-facing documentation** → `docs/source/` has hardware references, media architecture, platform guides, troubleshooting, and SDK tutorials that often explain *why* the code looks the way it does. Useful entry points:
  - `docs/source/SDK/core-concept.md` — coordinate frames, motion model, safety limits.
  - `docs/source/SDK/media-architecture.md` — how the WebRTC / IPC / GStreamer pieces fit together at the user level.
  - `docs/source/SDK/gstreamer-installation.md`, `docs/source/SDK/installation.md` — system dependencies.
  - `docs/source/platforms/{reachy_mini,reachy_mini_lite,simulation}/` — per-platform hardware and runtime behavior (Wireless CM4, Lite USB, MuJoCo).
  - `docs/source/troubleshooting.md` — known failure modes; check here before assuming a bug is new.
