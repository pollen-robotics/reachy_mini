/**
 * OAuth state tracking for the host shell.
 *
 * The SDK owns the actual OAuth dance (`login()` redirect,
 * `authenticate()` token resolution). The hook merely surfaces
 * "are we signed in?" + names + sign-in / sign-out helpers, and
 * threads the `oauth-pending` flag for the "welcome back"
 * animation across the redirect.
 */
import { useEffect, useState, useCallback } from 'react';

import type { ReachyMiniInstance } from '../lib/sdk-types';
import {
  clearSignedOutFlag,
  consumeOAuthPending,
  hasCachedDevToken,
  isUserSignedOut,
  markOAuthPending,
  markUserSignedOut,
  rehydrateDevToken,
} from '../lib/settings';

export interface OAuthState {
  /** SDK reports an active auth (token + user name resolved). */
  isAuthenticated: boolean;
  /** HF account user name when known. */
  userName: string | null;
  /** Boot started from an OAuth redirect (i.e. `oauth-pending`
   *  flag was set). Reset once `isAuthenticated` flips true. */
  isPostOauthReturn: boolean;
  /** Async wrapper around `sdk.login()`. */
  signIn(): Promise<void>;
  /** Sync wrapper around `sdk.logout()` + mark signed-out. */
  signOut(): void;
}

/**
 * Module-level memo of the OAuth-pending flag. We consume the
 * localStorage flag exactly once per page load (the first time
 * `useOAuth` is invoked) so that:
 *
 *  - the very first render already sees `isPostOauthReturn=true`
 *    when we just returned from a sign-in (no `useEffect` lag
 *    that React 18 would batch away);
 *  - StrictMode dev's "mount → unmount → remount" never
 *    double-consumes the flag (the second mount sees the cached
 *    value, not a fresh `consumeOAuthPending()` call).
 */
let cachedOAuthPending: boolean | null = null;
function readOAuthPendingOnce(): boolean {
  if (cachedOAuthPending === null) {
    cachedOAuthPending = consumeOAuthPending();
  }
  return cachedOAuthPending;
}

export function useOAuth(sdk: ReachyMiniInstance | null): OAuthState {
  const [isAuthenticated, setAuth] = useState<boolean>(() =>
    Boolean(sdk?.isAuthenticated),
  );
  const [userName, setUserName] = useState<string | null>(
    () => sdk?.username ?? null,
  );
  const [isPostOauthReturn, setPostOauth] = useState<boolean>(() =>
    readOAuthPendingOnce(),
  );

  // 2. Try to authenticate from cached tokens once the SDK is
  //    available. Skip if the user explicitly signed out earlier.
  //
  //    We deliberately DO NOT reset `isPostOauthReturn` here, even
  //    once auth resolves. The flag means "this page load was the
  //    return leg of an OAuth redirect" - that's a fact about the
  //    boot, not about the live auth state. Flipping it back to
  //    false on a fast `authenticate()` (~30 ms in prod where the
  //    token is already in sessionStorage) creates a race with the
  //    welcome-back latch in ReachyHostShell: deps change → effect
  //    cleanup fires → fallback timer is cancelled before it has a
  //    chance to mount the overlay. The latch's own one-shot ref
  //    handles "don't fire twice"; the flag just needs to stay true
  //    until the next sign-out / page reload.
  useEffect(() => {
    if (!sdk) return;
    if (isUserSignedOut()) return;
    let alive = true;
    void (async () => {
      try {
        const ok = await sdk.authenticate();
        if (!alive) return;
        setAuth(ok);
        setUserName(sdk.username);
      } catch (err) {
        console.warn('[reachy-mini-sdk/host] authenticate() threw', err);
      }
    })();
    return () => {
      alive = false;
    };
  }, [sdk]);

  // 3. Mirror SDK state changes. The SDK does not emit a
  //    dedicated `authChanged` event - auth resolution always
  //    funnels through `authenticate()` (step 2) - but a
  //    `connected` / `disconnected` cycle is a good moment to
  //    re-read the current auth snapshot in case a refresh-token
  //    swap silently updated `sdk.username`. Cheap, idempotent.
  //    Same reasoning as above re: leaving `isPostOauthReturn`
  //    untouched here.
  useEffect(() => {
    if (!sdk) return;
    const sync = (): void => {
      setAuth(sdk.isAuthenticated);
      setUserName(sdk.username);
    };
    sdk.addEventListener('connected', sync);
    sdk.addEventListener('disconnected', sync);
    return () => {
      sdk.removeEventListener('connected', sync);
      sdk.removeEventListener('disconnected', sync);
    };
  }, [sdk]);

  const signIn = useCallback(async () => {
    if (!sdk) return;
    clearSignedOutFlag();

    // Local dev path: a `devToken` was passed to `mountHost()`
    // earlier. Re-seed the session storage (wiped by the previous
    // `signOut()`) and resolve through `authenticate()`, exactly
    // the way a fresh page reload would do it. This avoids
    // `sdk.login()` throwing `Missing clientId` in environments
    // where no OAuth client ID is configured.
    //
    // The user explicitly clicked "Sign in" so we DO want the
    // welcome-back animation to play - this is functionally the
    // dev equivalent of returning from an HF redirect, even
    // though we're not actually round-tripping. We skip
    // `markOAuthPending()` because the flag is meant to survive
    // a page redirect (which isn't happening here); we set
    // `isPostOauthReturn` directly instead so the latch in
    // ReachyHostShell fires on the very next render.
    if (hasCachedDevToken()) {
      rehydrateDevToken();
      try {
        const ok = await sdk.authenticate();
        setAuth(ok);
        setUserName(sdk.username);
        if (ok) setPostOauth(true);
      } catch (err) {
        console.error('[reachy-mini-sdk/host] dev-token authenticate() threw', err);
        throw err;
      }
      return;
    }

    // Real OAuth path: only here do we mark the flag so the
    // animation plays once on the post-redirect page load.
    markOAuthPending();
    try {
      await sdk.login();
    } catch (err) {
      // login() typically redirects, so a throw here means the
      // redirect was blocked. Clear the pending flag so the
      // next boot doesn't show a confused "welcome back".
      console.error('[reachy-mini-sdk/host] sdk.login() threw', err);
      consumeOAuthPending();
      throw err;
    }
  }, [sdk]);

  const signOut = useCallback(() => {
    if (!sdk) return;
    try {
      sdk.logout();
    } catch (err) {
      console.warn('[reachy-mini-sdk/host] sdk.logout() threw', err);
    }
    markUserSignedOut();
    setAuth(false);
    setUserName(null);
    setPostOauth(false);
  }, [sdk]);

  return {
    isAuthenticated,
    userName,
    isPostOauthReturn,
    signIn,
    signOut,
  };
}
