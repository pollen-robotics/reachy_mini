# Releasing `reachy_mini`

Releases are driven by the **Release** workflow (`.github/workflows/wheels.yml`),
a single `workflow_dispatch` with a `release_type` dropdown. The version lives
statically in `pyproject.toml` (single source of truth): `main` carries
`X.Y.Z.dev0`; each `vX.Y-release` branch carries its released `X.Y.Z`.

> **Note:** `main` intentionally carries a [PEP 440](https://peps.python.org/pep-0440/)
> `.dev0` version (e.g. `1.10.0.dev0`), which is **not** valid semver. Any tooling that
> reads the version out of `pyproject.toml` must handle non-semver PEP 440 forms
> (`.devN`, `rcN`, etc.) — e.g. the npm-publish CI normalizes the version to semver
> before calling `npm version`.

## The three modes

| Mode | Trigger from | What it does |
|------|--------------|--------------|
| `dry-run` | anywhere | Read-only preflight: checks every secret/var is set, that `RELEASE_PAT` can push both repos, that the `pypi` environment exists, that OpenCode + the model + the HF token work (tiny live call), and previews the versions each mode would produce. Tags nothing, publishes nothing, opens no PR. Run this first. |
| `minor-prerelease` | `main` | Reads `X.Y.Z.dev0` → cuts `X.Y.Zrc<N>` (RC starts at 1), creates/reuses `vX.Y-release`, tags, publishes to PyPI, AI-drafts a **draft** GitHub release, and opens an RC-test PR in `reachy_mini_conversation_app`. |
| `minor-release` | `main` | Promotes the latest RC → `X.Y.Z`, publishes to PyPI, promotes the draft release to the final tag (marked *latest*), triggers the docs build, and opens a PR bumping `main` to `X.(Y+1).0.dev0`. |
| `patch-release` | `vX.Y-release` | Bumps the patch (`X.Y.Z+1`), tags, publishes. Not marked *latest*. |

Typical flow: `minor-prerelease` → validate the RC (its PR CI in the conversation app,
plus manual testing) → re-run `minor-prerelease` for more RCs if needed → `minor-release`.
Bugfix on a shipped minor: cherry-pick onto `vX.Y-release`, then `patch-release` from it.

Always run `dry-run` first to confirm secrets/vars/access are set.

## Job graph & recovery

```
prepare ──▶ publish-pypi ──┬──▶ release-notes
                           ├──▶ test-downstream   (prerelease only)
                           └──▶ post-release       (minor-release only)
```

`prepare` pushes the version-bump commit **and the tag before** publishing, so the tag
already exists once publishing starts. Everything after `prepare` is gated on a
successful `publish-pypi`: if the publish fails, no GitHub release is drafted/promoted,
no RC-test PR and no bump PR are opened — the pipeline stops.

**If `publish-pypi` fails:** do **not** re-trigger the whole workflow (`prepare` would
error on the now-existing tag). Instead **re-run the `publish-pypi` job from the Actions
UI** ("Re-run failed jobs"); once it succeeds, `release-notes` and the rest run off it.
No re-tagging needed. If the tag/branch are wrong and you must start over, delete the tag
(and the `vX.Y-release` branch if it was just created) before re-triggering.

## One-time setup

- **PyPI Trusted Publisher** for project `reachy-mini`: repo `pollen-robotics/reachy_mini`,
  workflow `wheels.yml`. (OIDC — no token stored.) The workflow is intentionally named
  `wheels.yml` to reuse the pre-existing Trusted Publisher (which has no environment
  restriction), so no PyPI change is needed. If you later rename it to `release.yml`, add a
  matching Trusted Publisher entry on PyPI first.
- **`pypi` GitHub Environment** in repo settings; add required reviewers to gate publishing.
- **Secret `RELEASE_PAT`** — PAT or GitHub App token with `contents:write` +
  `pull_requests:write` on **both** `reachy_mini` and `reachy_mini_conversation_app`
  (used to open the RC-test PR and the post-release bump PR so their CI runs).
- **Secret `RELEASE_NOTES_HF_TOKEN`** — HF token scoped to Inference Providers.
- **Var `RELEASE_NOTES_MODEL`** — e.g. `huggingface/zai-org/GLM-5.2`.
- **Var `OPENCODE_VERSION`** — pinned OpenCode version installed in CI.
- Add a `rc-testing` label in `reachy_mini_conversation_app` (optional; the PR still
  opens without it).

## Release notes (AI-drafted)

`utils/release_notes/` ports huggingface_hub's "trust-but-verify" generator:

1. `fetch_prs.py` — lists PRs merged since the previous tag (ground-truth manifest).
2. OpenCode drafts notes via the `.opencode/skills/reachy-mini-release-notes` skill.
3. `validate_notes.py` — checks every manifest PR appears and no extras leaked; the
   orchestrator loops to fix discrepancies (up to 3 iterations).

Run locally to preview:

```bash
export GITHUB_TOKEN=...            # repo read
export HF_TOKEN=...                # Inference Providers
export RELEASE_NOTES_MODEL=huggingface/zai-org/GLM-5.2
python -m utils.release_notes.generate_release_notes --since v1.9.0 --minor
# → .release-notes/RELEASE_NOTES_v1.10.0.md
```

The draft GitHub release is editable before you run `minor-release` — polish tone there.
