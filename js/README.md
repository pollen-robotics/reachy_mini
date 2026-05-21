# Reachy Mini — JS package

This directory hosts the single npm package that ships every JS
surface for [Reachy Mini](https://github.com/pollen-robotics/reachy_mini):

| Subpath | Purpose |
|---|---|
| [`@pollen-robotics/reachy-mini-sdk`](./sdk/README.md) | Browser SDK — low-level `ReachyMini` class for direct robot control over WebRTC. Plain JS, no build step. |
| `@pollen-robotics/reachy-mini-sdk/host` | Host shell rendered around Reachy Mini apps deployed as Hugging Face Spaces: OAuth, robot picker, session lifecycle, and parent/iframe postMessage bridge. React + MUI, optional. |
| `@pollen-robotics/reachy-mini-sdk/host/auto` | CDN bundle pre-wired for the standalone `<script type="module">` flow. |
| `@pollen-robotics/reachy-mini-sdk/host/embed` | CDN bundle for the embedded app inside the host iframe. Vanilla TS, no React. |
| `@pollen-robotics/reachy-mini-sdk/host/protocol` | Protocol types + helpers (handy for app authors writing custom dispatchers). |

The package source lives under [`sdk/`](./sdk/):

```
sdk/
├── reachy-mini-sdk.js          # SDK runtime (plain JS)
├── reachy-mini-sdk.d.ts        # SDK type declarations (companion .d.ts)
├── README.md                   # Package README, surfaced on npmjs.com
├── host/
│   ├── README.md               # Host shell quickstart
│   ├── SPEC.md                 # Host ↔ embed protocol spec
│   ├── APP_AUTHOR_GUIDE.md     # Guide for app authors
│   ├── CHANGELOG.md
│   ├── src/                    # React + MUI sources (built to host/dist/)
│   ├── vite.config.ts
│   └── tsconfig*.json
└── package.json                # @pollen-robotics/reachy-mini-sdk
```

## Local development

```bash
cd js/sdk
npm install
npm run build       # bundles host/src/ → host/dist/
npm run typecheck
npm run dev         # vite dev server for the host
```

## Publishing

Versions are managed by CI (`.github/workflows/npm-publish.yml`):

- **GitHub Release** → publishes at `<pyproject.toml version>` with the `latest` dist-tag.
- **Push to `main`** (when `js/**` changes) → publishes at `<pyproject.toml version>-main.<short-sha>` with the `main` dist-tag.

The `version` field in `sdk/package.json` is a placeholder (`0.0.0-managed-by-ci`); CI overrides it before publish. The single source of truth for the released version is `pyproject.toml` at the repo root.

CI uses [npm trusted publishing](https://docs.npmjs.com/trusted-publishers) — no `NPM_TOKEN` secret. The package needs a trusted publisher configured on npmjs.com pointing at this workflow.

## Migration from `@pollen-robotics/reachy-mini-host`

The host package has been folded into the SDK. Update your app:

```diff
- "@pollen-robotics/reachy-mini-host": "^1.x"
- "@pollen-robotics/reachy-mini-sdk":  "^1.x"
+ "@pollen-robotics/reachy-mini-sdk":  "^1.x"
```

```diff
- import { mountHost } from '@pollen-robotics/reachy-mini-host/auto';
- import { connectToHost } from '@pollen-robotics/reachy-mini-host/embed';
- import { PROTOCOL_VERSION } from '@pollen-robotics/reachy-mini-host/protocol';
+ import { mountHost } from '@pollen-robotics/reachy-mini-sdk/host/auto';
+ import { connectToHost } from '@pollen-robotics/reachy-mini-sdk/host/embed';
+ import { PROTOCOL_VERSION } from '@pollen-robotics/reachy-mini-sdk/host/protocol';
```

The host bundles now import the SDK directly, so apps no longer
need to load `reachy-mini-sdk.js` from a CDN script tag or wire
`window.ReachyMini` themselves — the import side effect handles it.

`@pollen-robotics/reachy-mini-host` on npm will remain available
for one minor release at its current version to give consumers
time to switch, then be deprecated with a pointer to the new
subpath.
