# Changelog

All notable changes to `@pollen-robotics/reachy-mini-sdk` are
documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions
are kept in lock-step with the Reachy Mini Python daemon and are
driven from `pyproject.toml` at the repo root (the npm publish CI
overrides the `version` placeholder in `package.json`).

The host shell shares this package's version (single source of
truth); the wire protocol is versioned separately in
`PROTOCOL_VERSION` (see [host/SPEC.md §11](./host/SPEC.md#11-backlog)).

## Unreleased — succeeds 1.7.3

> **Breaking change.** Every app currently using
> `@pollen-robotics/reachy-mini-host` has to update its imports
> when it upgrades. Existing `1.7.x` installs keep working against
> the legacy `reachy-mini-host@1.7.x` tarball that stays on npm;
> only consumers who upgrade past `1.7.3` need to migrate. The
> target version is intentionally left unset in this PR — it will
> be picked at release time when `pyproject.toml` is bumped from
> `1.7.3` to the next number (the npm publish CI mirrors that bump
> into the package manifest).

**Single-package release: `@pollen-robotics/reachy-mini-host` is folded
into `@pollen-robotics/reachy-mini-sdk`.** App authors install,
version, and import from one entry point. The old package stays
available on npm at `1.7.x` for one minor cycle for graceful
migration and will then be `npm deprecate`d with a pointer to the
new subpaths.

### Migration

```diff
- "@pollen-robotics/reachy-mini-host": "^1.7.x",
- "@pollen-robotics/reachy-mini-sdk":  "^1.7.x"
+ "@pollen-robotics/reachy-mini-sdk":  "^<next-release>"
```

```diff
- import { mountHost }        from '@pollen-robotics/reachy-mini-host/auto';
- import { connectToHost }    from '@pollen-robotics/reachy-mini-host/embed';
- import { PROTOCOL_VERSION } from '@pollen-robotics/reachy-mini-host/protocol';
+ import { mountHost }        from '@pollen-robotics/reachy-mini-sdk/host/auto';
+ import { connectToHost }    from '@pollen-robotics/reachy-mini-sdk/host/embed';
+ import { PROTOCOL_VERSION } from '@pollen-robotics/reachy-mini-sdk/host/protocol';
```

The host CDN bundles also moved:

```diff
- https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-host@1/dist/entry/auto.js
+ https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/auto.js
```

### Added

- New subpath exports on `@pollen-robotics/reachy-mini-sdk`:
  - `./host`           — `mountHost`, `connectToHost`, types
  - `./host/auto`      — CDN auto bundle for standalone apps
  - `./host/embed`     — CDN embed bundle for the iframe side
  - `./host/protocol`  — `PROTOCOL_VERSION`, `decodeCredsFromHash`, etc.
- `reachy-mini-sdk.d.ts` ships next to `reachy-mini-sdk.js` as the
  canonical SDK type surface. The host re-exports from there
  (`host/src/lib/sdk-types.ts` is now a thin barrel), removing the
  earlier duplication TODO.

### Changed

- The host bundles (`./host`, `./host/auto`, `./host/embed`) now
  import the SDK runtime directly and self-assign
  `window.ReachyMini` (when unset) at module-load time, dispatching
  `reachymini:ready`. App `index.html` files no longer need a
  separate `<script type="module">` to load the SDK on the global,
  and `useSdk` + `connectToHost` no longer race against a
  late-arriving global. Apps that still set `window.ReachyMini`
  themselves are unaffected — we only assign when it's missing.
- Log-prefix tag harmonised to `[reachy-mini-sdk/host]` /
  `[reachy-mini-sdk/host/embed]` for greppability.
- Internal repo layout: `js/sdk/*` collapsed into `js/*` now that
  there is only one package. The host source lives at `js/host/`,
  the SDK runtime at `js/reachy-mini-sdk.js`, the package manifest
  at `js/package.json`. No effect on consumers; only the
  `repository.directory` field and the npm-publish CI working
  directory changed.

### Removed

- Workspace coordinator `js/package.json` (was an npm workspace
  root) and `js/package-lock.json` at the workspace level. The
  single package's lockfile lives at `js/package-lock.json`.
- `@pollen-robotics/reachy-mini-host` package manifest — the host
  ships under `@pollen-robotics/reachy-mini-sdk/host*` now. The
  legacy npm package stays on `1.7.x` for one minor cycle for
  graceful migration, then will be `npm deprecate`d.

### SDK changes drafted between 1.7.3 and the package merge

The following landed on `main` after `1.7.3` was published but
were never released on their own; they ship with this release:

- **Breaking (SDK)**: `wakeUp()` and `gotoSleep()` now return
  `Promise<void>` (previously `boolean`). The promise resolves on
  the daemon's `{command, completed: true}` response, after the
  trajectory player has fully landed. Apps that previously relied
  on the boolean return are unaffected in practice; apps that
  *want* to await trajectory completion (e.g. to chain
  `setMotorMode("disabled")` after a `gotoSleep`) can now do so
  without racing.
- **SDK**: both motion helpers take an optional `{ timeoutMs }`
  (default 8000 ms). The promise rejects on session teardown
  (`stopSession` / `disconnect`) so consumers never wait forever
  on an interrupted trajectory.
- **SDK**: `POST /send` responses with a 4xx / 5xx status now
  produce a `console.warn` carrying the rejected message `type`
  and the response body, making racy `setPeerStatus` /
  `endSession` failures easier to diagnose.
- **Host types**: `src/lib/sdk-types.ts` refreshed to cover the
  full SDK public surface (`autoConnect`, `gotoTarget`,
  `setMotorTorque`, `subscribeLogs`, `requestState`, version /
  hardware-id helpers, `robotState`, `isEmbedded`, jitter buffer
  option, `autoStartFromUrl`). The motion helpers now type as
  `Promise<void>` to match the SDK.

## 0.3.0 - 2026-05-16 (unreleased)

**This release is a full rewrite of the package against
[SPEC.md v1.0](./host/SPEC.md).** The public surface and the wire
protocol both change in incompatible ways.

### Breaking changes

- **API**: `mountHost()` options reduced to the documented surface
  (`appName`, `appIconUrl`, `appEmoji`, `enableMicrophone`,
  `clientId`, `devToken`, `target`). The following are removed:
  - `theme: { light, dark }` (host owns its bundled MUI theme).
  - `skipTheme` (no consumer-driven theming).
  - `skipAuth` (host owns OAuth).
- **API**: `ConnectedHandle` reduced to the documented surface.
  The following are removed:
  - `sendCustom()` / `onCustom()` (no free-form bidi channel).
  - `requestConfigUpdate()` (apps don't push config upstream).
- **API**: `devToken` payload renamed `{ token, username }` →
  `{ token, userName }` for camelCase consistency with `appName`
  and `hostName`.
- **API**: `ConnectedHandle.username` renamed to `ConnectedHandle.userName`.
- **Protocol**: `host:custom` / `embed:custom` removed.
- **Protocol**: `embed:request-config-update` removed.
- **Host shell**: removed the legacy `?app=<owner>/<space>`
  branding-swap behaviour. The host renders exactly the app it
  ships with, never another. Multi-app routing now lives in the
  mobile catalog only.

### Added

- `SPEC.md` (v1.0): behavioural contract, state machines,
  4 engineering invariants (single SDK per tab, token hygiene,
  bundle pinning, React Strict Mode safety).
- `APP_AUTHOR_GUIDE.md`: step-by-step guide for building a new
  app on top of the host.
- `REBUILD_PLAN.md`: project plan tracking the rewrite (deleted
  once 0.3.0 ships).
- `LICENSE` (MIT).

### Changed

- **Documentation**: replaced the legacy `SPEC.md` (823 lines,
  ambitious-future spec) with a focused v1.0 spec (~620 lines)
  matching what we actually want to ship.
- **Package metadata**: `repository.url` now points at the
  host-only repo. `homepage` and `bugs` added.
- **Versioning policy**: stays permissive pre-1.0 (see SPEC §11
  "Versioning policy"); each minor release between 0.x.y may
  ship breaking changes. Semver-strict starts at 1.0.

### Why a rewrite instead of a refactor

See `REBUILD_PLAN.md` § "Why a rebuild, not a refactor". TL;DR:
the API surface shrinks too much, and the previous code carried
half-implemented features (protocol-version negotiation,
heartbeat, custom channels) that were cheaper to retype than to
selectively remove.

## 0.2.0 - 2026-05-15

Initial extraction of the host shell into a standalone package.
Predates the SPEC.md v1.0 rewrite; deprecated in favour of 0.3.0.
