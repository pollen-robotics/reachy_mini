/**
 * Single-instance SDK provisioning for the host shell.
 *
 * Invariant: at most one `ReachyMini` instance ever lives on
 * `window` (SPEC §8.1). The host shell only uses the SDK for
 * OAuth (`authenticate()`, `login()`, `logout()`) — it never
 * calls `connect()`, so no SSE is ever opened from the host.
 * The picker fetches robots via the REST `/api/robot-status`
 * endpoint instead (see `useRobots`); this is critical because
 * an SSE registration from the host token interferes with the
 * iframe's subsequent WebRTC handshake.
 *
 * The instance is stored at module scope (NOT in React state)
 * so React Strict Mode's double-mount and any incidental
 * remount can't accidentally create a second SDK.
 */
import { useEffect, useState } from 'react';

import type {
  ReachyMiniInstance,
  ReachyMiniOptions,
} from '../lib/sdk-types';

let singleton: ReachyMiniInstance | null = null;
let lastOptionsKey: string | null = null;

function optionsKey(opts: ReachyMiniOptions): string {
  // Deliberately ignore `appName` here so picking a different
  // app from the picker (legacy) won't tear down the SDK; only
  // signalingUrl / mic / clientId require a new instance.
  return JSON.stringify({
    signalingUrl: opts.signalingUrl ?? null,
    enableMicrophone: Boolean(opts.enableMicrophone),
    clientId: opts.clientId ?? null,
  });
}

/** Get (or lazily create) the singleton SDK instance. */
export function getOrCreateSdk(opts: ReachyMiniOptions): ReachyMiniInstance {
  const key = optionsKey(opts);
  if (singleton && lastOptionsKey === key) return singleton;

  if (singleton) {
    // Different signaling URL / mic policy: tear down and rebuild.
    try {
      singleton.disconnect();
    } catch {
      /* ignore */
    }
  }

  singleton = new window.ReachyMini(opts);
  lastOptionsKey = key;
  return singleton;
}

/** Fully drop the singleton (used on logout / hard reset). */
export function destroySdk(): void {
  if (!singleton) return;
  try {
    singleton.disconnect();
  } catch {
    /* ignore */
  }
  singleton = null;
  lastOptionsKey = null;
}

/**
 * React hook returning the SDK instance + a "ready" boolean that
 * flips to true once `window.ReachyMini` is defined. The hook
 * does NOT call `connect()` / `authenticate()`; that's the
 * responsibility of `useHostPhase`.
 */
export function useSdk(opts: ReachyMiniOptions): {
  sdk: ReachyMiniInstance | null;
  sdkReady: boolean;
} {
  const [sdkReady, setSdkReady] = useState<boolean>(
    typeof window !== 'undefined' && Boolean(window.ReachyMini),
  );
  const [sdk, setSdk] = useState<ReachyMiniInstance | null>(() =>
    typeof window !== 'undefined' && window.ReachyMini
      ? getOrCreateSdk(opts)
      : null,
  );

  useEffect(() => {
    if (sdkReady) return;
    const onReady = (): void => {
      if (!window.ReachyMini) return;
      setSdkReady(true);
      setSdk(getOrCreateSdk(opts));
    };
    window.addEventListener('reachymini:ready', onReady);
    return () => window.removeEventListener('reachymini:ready', onReady);
    // We intentionally only re-evaluate `opts` when its hashed
    // signature changes; the dependency is captured below.
  }, [sdkReady, optionsKey(opts)]);

  // Recreate SDK when relevant options change after first ready.
  useEffect(() => {
    if (!sdkReady) return;
    setSdk(getOrCreateSdk(opts));
  }, [optionsKey(opts), sdkReady]);

  return { sdk, sdkReady };
}
