# Changelog

All notable changes to the host shell — now shipped under
`@pollen-robotics/reachy-mini-sdk/host*` subpath exports — are
documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to a permissive pre-1.0 semver (see
[SPEC.md §11](./SPEC.md#11-backlog) for the policy).

## 1.8.0 - unreleased

**Single-package release: `@pollen-robotics/reachy-mini-host` is
gone.** The host source moved into `@pollen-robotics/reachy-mini-sdk`
under the `./host`, `./host/auto`, `./host/embed`, and `./host/protocol`
subpath exports. App authors should drop the second dep and update
their imports — see the [migration note](../../README.md#migration-from-pollen-roboticsreachy-mini-host)
in the JS workspace README.

The host bundles now import the SDK runtime directly (the SDK
ships in the same npm tarball as `reachy-mini-sdk.js`) and
auto-install it on `window.ReachyMini` at load time. Apps no
longer need a separate `<script type="module">` for the SDK, and
the `useSdk` hook + `connectToHost` no longer race against a
late-arriving global.

### Internal cleanup tied to the merge

- `src/lib/sdk-types.ts` is now a thin re-export from the
  canonical type declarations next to the SDK runtime
  (`js/sdk/reachy-mini-sdk.d.ts`). External consumers who
  `import` only the SDK get the same types automatically; host
  code keeps its non-optional `window.ReachyMini` augmentation
  for the use sites that rely on it.
- `entry/auto.ts` and `entry/embed.ts` synchronously assign
  `window.ReachyMini` when unset and dispatch `reachymini:ready`
  before exposing `window.ReachyMiniHost` / `window.ReachyMiniHostEmbed`.
- `index.ts` (the `./host` npm entry) does the same on import, so
  bundler users of `mountHost` / `connectToHost` get the SDK
  initialised eagerly.

The SDK and host versions stay in lock-step on a single
`pyproject.toml`-driven release.

### SDK changes that ripple through to the host

- **Breaking (SDK)**: `wakeUp()` and `gotoSleep()` now return
  `Promise<void>` (previously `boolean`). The promise resolves on
  the daemon's `{command, completed: true}` response, after the
  trajectory player has fully landed. Apps embedded in this host
  shell that previously relied on the boolean return are unaffected
  in practice (the boolean was rarely consumed strictly); apps that
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

### Host changes

- **Types**: `src/lib/sdk-types.ts` refreshed to cover the full
  SDK 1.8.0 public surface (`autoConnect`, `gotoTarget`,
  `setMotorTorque`, `subscribeLogs`, `requestState`, version /
  hardware-id helpers, `robotState`, `isEmbedded`, jitter buffer
  option, `autoStartFromUrl`). The motion helpers now type as
  `Promise<void>` to match the SDK.

## 0.3.0 - 2026-05-16 (unreleased)

**This release is a full rewrite of the package against
[SPEC.md v1.0](./SPEC.md).** The public surface and the wire
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
