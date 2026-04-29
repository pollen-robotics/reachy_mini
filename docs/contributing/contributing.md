# Contributing

Audience: contributors to this repo. For an entry-point map of the codebase, see [`architecture.md`](architecture.md). For doc-building, see [`generate_docs.md`](generate_docs.md).

## Requirements

Follow the [installation instructions](https://huggingface.co/docs/reachy_mini/SDK/installation) to install the SDK. Linux users have to [install gstreamer manually](https://huggingface.co/docs/reachy_mini/SDK/gstreamer-installation).

## Dev environment

All linting, typing, and testing tools live in the `dev` dependency group. To match CI exactly (so any local pass/fail mirrors what CI sees):

```bash
uv sync --all-extras --group dev
```

If you only need the lint/test tools and not the optional extras:

```bash
uv sync --group dev
```

## Before you commit

> **You are strongly invited to fix all linter and test errors locally before committing.** CI will reject PRs that don't pass, and quick local checks save round-trips.

The recommended workflow:

1. **Install the pre-commit hook once per clone.** It runs ruff-check (with `--select I --select D --ignore D203 --ignore D213`, i.e. import order and docstring rules) and ruff-format on staged files automatically. Configuration lives in [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml).

   ```bash
   pre-commit install
   ```

2. **Run ruff against the whole tree** to catch anything the staged-only hook missed:

   ```bash
   ruff check .
   ```

3. **Run mypy** for type errors. mypy results depend on what's installed — the CI runs it with the full uv install on Linux. If you see a discrepancy, run `uv sync --all-extras --group dev` first.

   ```bash
   mypy
   ```

4. **Run the no-hardware test pass** that mirrors CI:

   ```bash
   pytest -m "not audio and not video and not wireless and not ipc_resolution"
   ```

   The full hardware-marker matrix is documented in `pyproject.toml`. To run hardware-specific tests when you have the device, e.g.:

   ```bash
   pytest -m "audio"
   ```

CI workflows: [`.github/workflows/lint.yml`](../../.github/workflows/lint.yml) (ruff + mypy) and [`.github/workflows/pytest.yml`](../../.github/workflows/pytest.yml) (the no-hardware pytest run).

## AI-assisted contributions

AI assistance is welcome and explicitly accepted on this repo. Follow the Linux kernel coding-assistants convention: <https://github.com/torvalds/linux/blob/master/Documentation/process/coding-assistants.rst>.

When an AI helped author a commit, append a single trailer:

```
Assisted-by: Claude:claude-opus-4-7
```

- `<model-id>` is the exact model ID from the assistant's session context (e.g. `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`).
- If specialized analysis tools were used (coccinelle, sparse, mypy plugins, …), append them space-separated.
- Do **not** use `Co-Authored-By:` — the kernel convention replaces it.
- `Signed-off-by:` is for humans only — it certifies the DCO and an AI cannot certify it.

## Where do I make my change?

If you are looking for the right file or subpackage to edit, see [`architecture.md`](architecture.md) — it has a "Where to look for X" lookup table covering the common cases (REST endpoints, audio parameters, motion curves, JS API, backends).

## Keep the contributor docs in sync

If your change touches the **architecture or main concepts** of the SDK — adding/removing a subpackage, introducing a new backend, changing the daemon ↔ SDK ↔ JS connection model, renaming a top-level entry point, shifting a public abstraction (`AudioBase`, `CameraBase`, `Backend`, …), or changing the lookup table on the right-hand side of architecture.md — update the contributor docs in the same PR:

- [`architecture.md`](architecture.md) — the codebase map, the connection diagram, and the "Where to look for X" table.
- [`../../AGENTS.md`](../../AGENTS.md) — the "Repo layout at a glance" table and any working convention that no longer holds.

Stale architecture docs are worse than missing ones — agents and human contributors alike act on them. Treat the docs as part of the change, not paperwork after the fact.
