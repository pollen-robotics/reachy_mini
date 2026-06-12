---
title: Reachy Mini Host Harness
emoji: 🧪
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
hf_oauth: true
short_description: Dev harness for the Reachy Mini host shell
---

# Reachy Mini Host Harness

A standalone deployment of the **Reachy Mini host shell**, used to exercise
the host in real conditions - in particular the **Hugging Face OAuth flow**
(real redirect + real Space-injected `OAUTH_CLIENT_ID`), the robot picker,
the welcome / post-OAuth splash, and the leave / sign-out contract.

This is **not** an end-user app. It is the same shell that apps load via
`@pollen-robotics/reachy-mini-sdk/host/auto`, built straight from source
(no CDN, no published npm version in the loop) and served as static files
so it owns the top-level origin and can complete the OAuth round-trip.

## What it validates

- OAuth sign-in redirect and return leg (no button flicker on return).
- "Welcome back" overlay + post-OAuth splash timing.
- Robot picker, connect → wake.
- Leave / End-session contract: app `onLeave` → sleep → torque release →
  `stopSession`, including a graceful sign-out from inside a live session.

## Caveats (different from a real app Space)

- This Space gets its **own** auto-provisioned OAuth app: distinct
  `client_id`, redirect URI and consent screen (owned by this Space).
  The OAuth *logic and UX* are faithful, but the client/redirect identity
  is not 1:1 with any production app Space.
- The robot path uses the default central signaling server, so it matches
  production.
- Final go/no-go for a given app should still be validated inside that
  app's own Space.

## Rebuild & redeploy

From `reachy_mini/ts/host/`:

```bash
vite build --config vite.harness.config.ts   # → dist-harness/
```

Then push `dist-harness/` + this README to the Space.
