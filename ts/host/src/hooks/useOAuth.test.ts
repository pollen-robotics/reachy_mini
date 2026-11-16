/**
 * Boot-resolution tests for `useOAuth`, focused on the one state the
 * whole shell hangs on: `authResolved`.
 *
 * Every path that resolves it needs an `sdk` instance, so the case
 * under test here is the SDK bundle never loading at all (blocked
 * CDN, stale Space asset, broken build). Before the grace-period
 * escape hatch, that state pinned the neutral boot splash up forever
 * with no diagnosable screen behind it.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';

import { SDK_LOAD_GRACE_MS, useOAuth } from './useOAuth';
import { clearSignedOutFlag, markUserSignedOut } from '../lib/settings';

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  clearSignedOutFlag();
  localStorage.clear();
  sessionStorage.clear();
  vi.useRealTimers();
});

describe('useOAuth authResolved with a missing SDK', () => {
  it('holds unresolved while the SDK may still arrive', () => {
    const { result, unmount } = renderHook(() => useOAuth(null));
    expect(result.current.authResolved).toBe(false);

    act(() => {
      vi.advanceTimersByTime(SDK_LOAD_GRACE_MS - 1);
    });
    expect(result.current.authResolved).toBe(false);
    unmount();
  });

  it('gives up after the grace period so the shell can show SignInView', () => {
    const { result, unmount } = renderHook(() => useOAuth(null));

    act(() => {
      vi.advanceTimersByTime(SDK_LOAD_GRACE_MS);
    });
    expect(result.current.authResolved).toBe(true);
    // Not a fake sign-in: we resolve to "not authenticated", which is
    // the state that renders SignInView + the local-dev config hint.
    expect(result.current.isAuthenticated).toBe(false);
    unmount();
  });

  it('resolves synchronously when the user explicitly signed out', () => {
    // Nothing async to wait for in this state: the initializer must
    // settle immediately, SDK or not, so no splash is ever shown.
    markUserSignedOut();
    const { result, unmount } = renderHook(() => useOAuth(null));
    expect(result.current.authResolved).toBe(true);
    unmount();
  });
});
