/**
 * Version-policy tests: semver parsing, the block/notice/null severity
 * matrix, and the cached GitHub "latest release" fetch.
 *
 * The policy functions gate what a user is allowed to run, so every
 * "unknown" input MUST resolve to inaction (fail-open) - that
 * discipline is asserted here, not just documented.
 *
 * The fetch suite re-imports the module per test (`vi.resetModules`)
 * and clears localStorage so every test starts from a cold cache.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  MIN_SUPPORTED_DAEMON_VERSION,
  daemonUpdateSeverity,
  isDaemonOutdated,
  parseSemver,
} from './daemonRelease';

describe('parseSemver', () => {
  it('parses MAJOR.MINOR.PATCH', () => {
    expect(parseSemver('1.8.2')).toEqual({ major: 1, minor: 8, patch: 2 });
  });

  it('tolerates a leading v (GitHub tag form) and whitespace', () => {
    expect(parseSemver('v1.8.2')).toEqual({ major: 1, minor: 8, patch: 2 });
    expect(parseSemver(' 1.8.2 ')).toEqual({ major: 1, minor: 8, patch: 2 });
  });

  it('drops pre-release / build metadata', () => {
    expect(parseSemver('1.8.2-rc1')).toEqual({ major: 1, minor: 8, patch: 2 });
    expect(parseSemver('0.0.0-managed-by-ci')).toEqual({
      major: 0,
      minor: 0,
      patch: 0,
    });
  });

  it('returns null on anything it cannot positively read', () => {
    for (const bad of ['1.8', 'banana', '', null, undefined, 'v..2']) {
      expect(parseSemver(bad)).toBeNull();
    }
  });
});

describe('isDaemonOutdated', () => {
  it('true only when current is strictly behind latest', () => {
    expect(isDaemonOutdated('1.8.0', '1.9.0')).toBe(true);
    expect(isDaemonOutdated('1.9.0', '1.9.0')).toBe(false);
    expect(isDaemonOutdated('1.10.0', '1.9.0')).toBe(false);
  });

  it('fails open on any unparseable side', () => {
    expect(isDaemonOutdated(null, '1.9.0')).toBe(false);
    expect(isDaemonOutdated('1.8.0', null)).toBe(false);
    expect(isDaemonOutdated('banana', '1.9.0')).toBe(false);
    expect(isDaemonOutdated('1.8.0', 'banana')).toBe(false);
  });
});

describe('daemonUpdateSeverity', () => {
  it(`blocks below the ${MIN_SUPPORTED_DAEMON_VERSION} floor`, () => {
    expect(daemonUpdateSeverity('1.8.1', '1.9.0')).toBe('block');
    expect(daemonUpdateSeverity('1.7.9', null)).toBe('block');
  });

  it('notices when merely behind the latest release', () => {
    expect(daemonUpdateSeverity('1.8.2', '1.9.0')).toBe('notice');
  });

  it('stays quiet when current, ahead, or latest unknown', () => {
    expect(daemonUpdateSeverity('1.9.0', '1.9.0')).toBeNull();
    expect(daemonUpdateSeverity('2.0.0', '1.9.0')).toBeNull();
    expect(daemonUpdateSeverity('1.9.0', null)).toBeNull();
  });

  it('fails open on an unknown current version - never block a silent daemon', () => {
    expect(daemonUpdateSeverity(null, '1.9.0')).toBeNull();
    expect(daemonUpdateSeverity('banana', '1.9.0')).toBeNull();
    expect(daemonUpdateSeverity(undefined, '1.9.0')).toBeNull();
  });
});

describe('fetchLatestDaemonVersion', () => {
  type Mod = typeof import('./daemonRelease');

  const CACHE_KEY = 'reachy.daemonLatestRelease.v1';

  /** Cold-import the module so its in-memory memo starts empty. */
  async function freshModule(): Promise<Mod> {
    vi.resetModules();
    return await import('./daemonRelease');
  }

  function mockFetchOnce(response: {
    ok: boolean;
    status?: number;
    json?: unknown;
  }): ReturnType<typeof vi.fn> {
    const fn = vi.fn().mockResolvedValue({
      ok: response.ok,
      status: response.status ?? (response.ok ? 200 : 500),
      json: async () => response.json ?? {},
    });
    vi.stubGlobal('fetch', fn);
    return fn;
  }

  beforeEach(() => {
    localStorage.clear();
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-07-30T12:00:00Z'));
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('resolves the tag, strips the v, and caches it', async () => {
    const mod = await freshModule();
    const fetchFn = mockFetchOnce({ ok: true, json: { tag_name: 'v1.9.0' } });

    expect(await mod.fetchLatestDaemonVersion()).toBe('1.9.0');
    // Second call: served from the localStorage cache, no network.
    expect(await mod.fetchLatestDaemonVersion()).toBe('1.9.0');
    expect(fetchFn).toHaveBeenCalledTimes(1);
    // And the cache survives a page reload (localStorage).
    expect(localStorage.getItem(CACHE_KEY)).toContain('1.9.0');
  });

  it('serves the localStorage cache without any fetch while fresh', async () => {
    localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({ value: '1.8.2', at: Date.now() - 60_000 }),
    );
    const mod = await freshModule();
    const fetchFn = mockFetchOnce({ ok: true, json: { tag_name: 'v9.9.9' } });

    expect(await mod.fetchLatestDaemonVersion()).toBe('1.8.2');
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it('refetches once the TTL (6 h) has expired', async () => {
    localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({ value: '1.8.2', at: Date.now() - 7 * 60 * 60 * 1000 }),
    );
    const mod = await freshModule();
    const fetchFn = mockFetchOnce({ ok: true, json: { tag_name: 'v1.9.0' } });

    expect(await mod.fetchLatestDaemonVersion()).toBe('1.9.0');
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });

  it('keeps serving the stale value on a 403 (rate limit) instead of flapping to null', async () => {
    localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({ value: '1.8.2', at: Date.now() - 7 * 60 * 60 * 1000 }),
    );
    const mod = await freshModule();
    mockFetchOnce({ ok: false, status: 403 });

    expect(await mod.fetchLatestDaemonVersion()).toBe('1.8.2');
  });

  it('returns the stale value on a network error, null with a cold cache', async () => {
    const mod = await freshModule();
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    expect(await mod.fetchLatestDaemonVersion()).toBeNull();

    localStorage.setItem(
      CACHE_KEY,
      JSON.stringify({ value: '1.8.2', at: Date.now() - 7 * 60 * 60 * 1000 }),
    );
    const mod2 = await freshModule();
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    expect(await mod2.fetchLatestDaemonVersion()).toBe('1.8.2');
  });

  it('returns null (and caches the miss) when the release has no tag', async () => {
    const mod = await freshModule();
    const fetchFn = mockFetchOnce({ ok: true, json: {} });

    expect(await mod.fetchLatestDaemonVersion()).toBeNull();
    expect(await mod.fetchLatestDaemonVersion()).toBeNull();
    expect(fetchFn).toHaveBeenCalledTimes(1);
  });
});
