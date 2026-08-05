/**
 * OAuth state tracking for the host shell.
 *
 * The SDK owns the actual OAuth dance (`login()` redirect,
 * `authenticate()` token resolution). The hook merely surfaces
 * "are we signed in?" + names + sign-in / sign-out helpers, and
 * threads the `oauth-pending` flag for the "welcome back"
 * animation across the redirect.
 *
 * Boot also carries a SILENT sign-in leg: when `authenticate()` finds
 * no token and no guard objects (explicit sign-out, previous attempt
 * this tab, dev token, iframe), the hook redirects once through
 * `login({ prompt: 'none' })`. Users with a live HF session and a
 * prior grant come back signed in without ever seeing SignInView;
 * everyone else bounces straight back to the regular signed-out view.
 */
import { useEffect, useState, useCallback } from 'react';

import type { ReachyMiniInstance } from '../lib/sdk-types';
import {
  clearSignedOutFlag,
  clearSilentAuthAttempted,
  consumeOAuthPending,
  hasCachedDevToken,
  hasSilentAuthAttempted,
  isUserSignedOut,
  markOAuthPending,
  markSilentAuthAttempted,
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
  /** `false` until the boot-time `authenticate()` settles (or we
   *  know synchronously there's nothing to resolve). The shell
   *  uses this to keep a neutral splash up instead of flashing
   *  `SignInView` while the cached token is still resolving. */
  authResolved: boolean;
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

/**
 * How long the boot splash may wait for the SDK bundle before giving
 * up on auth resolution. Every path that resolves `authResolved`
 * requires an `sdk` instance; if the bundle never loads (blocked CDN,
 * stale Space asset, broken build) nothing would ever flip it and the
 * shell would hold its neutral splash forever. Past this grace we
 * declare "not signed in" so the shell lands on SignInView - with its
 * local-dev missing-config hint, the one screen that can explain the
 * situation. A late-arriving SDK still runs `authenticate()` and
 * upgrades the state; the SignInView flash in that race is the
 * accepted cost of never spinning forever.
 */
export const SDK_LOAD_GRACE_MS = 8_000;

/**
 * Should the boot leg auto-redirect into a silent sign-in
 * (`login({ prompt: 'none' })`) after `authenticate()` found no token?
 *
 * Every guard is a "never surprise the user" rule:
 *  - explicit sign-out wins over convenience;
 *  - one attempt per tab (the sessionStorage flag survives the redirect
 *    round trip, so a `login_required` return can't loop);
 *  - dev-token setups never redirect (local dev has no OAuth app);
 *  - never from inside an iframe: with third-party cookies blocked the
 *    silent attempt always comes back `login_required`, and the iframe
 *    would navigate away from its parent's page.
 */
function shouldAttemptSilentSignIn(): boolean {
  if (isUserSignedOut()) return false;
  if (hasSilentAuthAttempted()) return false;
  if (hasCachedDevToken()) return false;
  try {
    if (window.self !== window.top) return false;
  } catch {
    return false; // cross-origin access throw = definitely framed
  }
  return true;
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
  // Whether the boot-time auth resolution has settled. Starts true only when
  // there's nothing async to wait for: the SDK already reports authenticated,
  // or the user explicitly signed out earlier. Otherwise we must wait for
  // `authenticate()` before deciding between SignInView and the picker.
  const [authResolved, setAuthResolved] = useState<boolean>(() =>
    Boolean(sdk?.isAuthenticated) || isUserSignedOut(),
  );

  // 1b. Escape hatch: everything below waits on `sdk`, so a bundle
  //     that never materialises would otherwise pin `authResolved`
  //     false - and the boot splash up - forever. See
  //     `SDK_LOAD_GRACE_MS` for the trade-off.
  useEffect(() => {
    if (sdk) return;
    const t = window.setTimeout(
      () => setAuthResolved(true),
      SDK_LOAD_GRACE_MS,
    );
    return () => window.clearTimeout(t);
  }, [sdk]);

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
    if (isUserSignedOut()) {
      // Nothing to resolve: we already know the user is signed out.
      setAuthResolved(true);
      return;
    }
    let alive = true;
    // True once we've committed to the silent-auth redirect: the page is
    // about to unload, so `authResolved` must stay false to keep the
    // neutral splash up instead of flashing SignInView for a frame.
    let redirecting = false;
    void (async () => {
      try {
        const ok = await sdk.authenticate();
        if (!alive) return;
        if (!ok && shouldAttemptSilentSignIn()) {
          // Silent sign-in leg: a user with a live HF session and a
          // previous grant comes back with a token and never sees the
          // sign-in view; anyone else bounces back with `?error=...`,
          // which `authenticate()` strips on the return leg while the
          // attempt flag routes them to the regular SignInView.
          markSilentAuthAttempted();
          try {
            await sdk.login({ prompt: 'none' });
            redirecting = true;
            return;
          } catch (err) {
            // No client ID (dev setups) or blocked redirect: fall
            // through to the normal signed-out view.
            console.warn('[reachy-mini-sdk/host] silent sign-in failed to start', err);
          }
        }
        setAuth(ok);
        setUserName(sdk.username);
      } catch (err) {
        console.warn('[reachy-mini-sdk/host] authenticate() threw', err);
      } finally {
        // Settled either way - the shell can now pick a definite view.
        if (alive && !redirecting) setAuthResolved(true);
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
    // Explicit click re-arms the one-shot silent attempt for this tab.
    clearSilentAuthAttempted();

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
    authResolved,
    signIn,
    signOut,
  };
}
