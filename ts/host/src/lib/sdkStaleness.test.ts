/**
 * Tests for the SDK staleness self-check, shaped by the #1316 review:
 * the first cut of this feature warned on any version gap, every
 * reload, against the wrong version source. Each of those regressions
 * is pinned here - if one of these tests starts failing, the feature
 * is drifting back into being a global kill switch for frozen apps.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  dismissSdkWarning,
  fetchLatestSdkVersion,
  isSdkMajorBehind,
  isSdkWarningDismissed,
  maybeWarnSdkStale,
  showSdkStalenessOverlay,
} from './sdkStaleness';

const OVERLAY_ID = 'reachy-sdk-outdated-overlay';

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  document.getElementById(OVERLAY_ID)?.remove();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

/* ─────────────────── Policy: major-only ─────────────────── */

describe('isSdkMajorBehind', () => {
  it('stays silent on minor and patch gaps - working apps must not be nagged', () => {
    expect(isSdkMajorBehind('1.9.0', '1.10.0')).toBe(false);
    expect(isSdkMajorBehind('1.9.0', '1.9.1')).toBe(false);
    expect(isSdkMajorBehind('1.9.0', '1.9.0')).toBe(false);
  });

  it('fires on a strictly newer major', () => {
    expect(isSdkMajorBehind('1.9.0', '2.0.0')).toBe(true);
    expect(isSdkMajorBehind('1.9.0', '3.1.4')).toBe(true);
  });

  it('never fires forward or sideways', () => {
    expect(isSdkMajorBehind('2.0.0', '1.9.9')).toBe(false);
  });

  it('fails open on dev placeholders and unparseable versions', () => {
    // 0.x is what CI-managed builds and local dev report.
    expect(isSdkMajorBehind('0.0.0-managed-by-ci', '2.0.0')).toBe(false);
    expect(isSdkMajorBehind(null, '2.0.0')).toBe(false);
    expect(isSdkMajorBehind('1.9.0', null)).toBe(false);
    expect(isSdkMajorBehind('not-a-version', '2.0.0')).toBe(false);
  });
});

/* ─────────────────── Dismissal: persisted per major ─────────────────── */

describe('persisted dismissal', () => {
  it('one dismissal covers the whole acknowledged major', () => {
    dismissSdkWarning('2.1.0');
    expect(isSdkWarningDismissed('2.1.0')).toBe(true);
    expect(isSdkWarningDismissed('2.5.3')).toBe(true);
  });

  it('a newer major re-arms the warning', () => {
    dismissSdkWarning('2.1.0');
    expect(isSdkWarningDismissed('3.0.0')).toBe(false);
  });

  it('nothing dismissed means nothing suppressed', () => {
    expect(isSdkWarningDismissed('2.0.0')).toBe(false);
  });
});

/* ─────────────────── Latest-version source: npm registry ─────────────────── */

function stubFetchOnce(payload: unknown, ok = true): ReturnType<typeof vi.fn> {
  const fn = vi.fn().mockResolvedValue({
    ok,
    json: async () => payload,
  });
  vi.stubGlobal('fetch', fn);
  return fn;
}

describe('fetchLatestSdkVersion', () => {
  it('reads the npm dist-tag payload and caches it', async () => {
    const fn = stubFetchOnce({ version: '2.3.4' });
    expect(await fetchLatestSdkVersion()).toBe('2.3.4');
    expect(await fetchLatestSdkVersion()).toBe('2.3.4');
    // Second call served from localStorage: one network hit total.
    expect(fn).toHaveBeenCalledTimes(1);
    expect(String(fn.mock.calls[0]?.[0])).toContain('registry.npmjs.org');
  });

  it('refreshes after the TTL', async () => {
    vi.useFakeTimers();
    const fn = stubFetchOnce({ version: '2.3.4' });
    await fetchLatestSdkVersion();
    vi.setSystemTime(Date.now() + 7 * 60 * 60 * 1000); // TTL is 6 h
    await fetchLatestSdkVersion();
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it('serves the stale cache instead of flapping to null on errors', async () => {
    vi.useFakeTimers();
    stubFetchOnce({ version: '2.3.4' });
    await fetchLatestSdkVersion();
    vi.setSystemTime(Date.now() + 7 * 60 * 60 * 1000);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    expect(await fetchLatestSdkVersion()).toBe('2.3.4');
  });

  it('resolves null (never throws) with no cache and no network', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('offline')));
    expect(await fetchLatestSdkVersion()).toBe(null);
  });
});

/* ─────────────────── Overlay + orchestrator ─────────────────── */

describe('overlay', () => {
  it('mounts once, and the dismiss button persists the acknowledgment', () => {
    showSdkStalenessOverlay('1.9.0', '2.0.0');
    showSdkStalenessOverlay('1.9.0', '2.0.0');
    expect(document.querySelectorAll(`#${OVERLAY_ID}`)).toHaveLength(1);

    (document.querySelector(`#${OVERLAY_ID} button`) as HTMLButtonElement).click();
    expect(document.getElementById(OVERLAY_ID)).toBeNull();
    expect(isSdkWarningDismissed('2.0.0')).toBe(true);
  });

  it('re-serialises versions through the semver parser before rendering', () => {
    showSdkStalenessOverlay('1.9.0-rc1+<img src=x>', '2.0.0');
    const body = document.querySelector(`#${OVERLAY_ID} .body`);
    expect(body?.textContent).toContain('v1.9.0');
    expect(document.querySelector(`#${OVERLAY_ID} img`)).toBeNull();
  });

  it('maybeWarnSdkStale stays silent within the same major', async () => {
    stubFetchOnce({ version: '1.42.0' });
    await maybeWarnSdkStale('1.9.0');
    expect(document.getElementById(OVERLAY_ID)).toBeNull();
  });

  it('maybeWarnSdkStale shows on a major gap, then respects the dismissal', async () => {
    stubFetchOnce({ version: '2.0.0' });
    await maybeWarnSdkStale('1.9.0');
    expect(document.getElementById(OVERLAY_ID)).not.toBeNull();

    (document.querySelector(`#${OVERLAY_ID} button`) as HTMLButtonElement).click();
    await maybeWarnSdkStale('1.9.0');
    // Dismissed 2.x: the overlay must NOT come back on reload.
    expect(document.getElementById(OVERLAY_ID)).toBeNull();
  });
});
