# Reachy Mini — JS workspace

This directory is an npm workspace housing the two published JS packages for [Reachy Mini](https://github.com/pollen-robotics/reachy_mini):

| Package | Folder | Purpose |
|---|---|---|
| [`@pollen-robotics/reachy-mini-sdk`](./sdk/README.md) | [`sdk/`](./sdk/) | Browser SDK — low-level `ReachyMini` class for direct robot control over WebRTC. |
| [`@pollen-robotics/reachy-mini-host`](./host/README.md) | [`host/`](./host/) | Host shell rendered around Reachy Mini apps deployed as Hugging Face Spaces: OAuth, robot picker, session lifecycle, and parent/iframe postMessage bridge. |

## Local development

```bash
cd js
npm ci              # installs both workspaces, hoisted into js/node_modules
npm run build       # builds the host bundles into host/dist/
```

## Publishing

Versions are managed by CI (`.github/workflows/npm-publish.yml`):

- **GitHub Release** → publishes both packages at `<pyproject.toml version>` with the `latest` dist-tag.
- **Push to `main`** (when `js/**` changes) → publishes both at `<pyproject.toml version>-main.<short-sha>` with the `main` dist-tag.

The `version` field in each subpackage `package.json` is a placeholder (`0.0.0-managed-by-ci`); CI overrides it before publish. The single source of truth for the released version is `pyproject.toml` at the repo root.

CI uses [npm trusted publishing](https://docs.npmjs.com/trusted-publishers) — no `NPM_TOKEN` secret. Each published package needs a trusted publisher configured on npmjs.com pointing at this workflow.
