/**
 * Daemon version policy for the standalone web shell.
 *
 * Why the shell cares
 * ───────────────────
 * A Space runs against whatever daemon the user's robot happens to have.
 * Neither central nor the host holds a data channel, so the only version
 * signal is the one the embed forwards on `embed:app-state` once its
 * session is up (`daemonVersion`).
 *
 * Why this is softer than the mobile app
 * ──────────────────────────────────────
 * The mobile app blocks the moment the daemon is behind the latest
 * GitHub release. That is defensible for a controlled first-party
 * funnel; it is not for the web, where every published Space would turn
 * a daemon release into a global kill switch for robots that were
 * working fine a minute earlier. So the web policy has two tiers:
 *
 *   block  - the daemon is below `MIN_SUPPORTED_DAEMON_VERSION`, i.e.
 *            too old for the app contract to hold. Blocking is the only
 *            honest answer.
 *   notice - the daemon merely trails the latest release. Dismissable
 *            banner, the app stays usable.
 *
 * Fail-open discipline
 * ────────────────────
 * Every unknown resolves to "no action". An unparseable version, a
 * daemon too old to answer `get_version`, an offline or rate-limited
 * GitHub: none of these may ever cost the user their app. We only act
 * on versions we positively understand.
 */

// NOTE: keep this module React-free. It is imported by the embed
// runtime (`embed/index.ts`), which is bundled without React - the
// `useLatestDaemonVersion` hook lives in `hooks/useLatestDaemonVersion`.

const REPO = 'pollen-robotics/reachy_mini';
const LATEST_URL = `https://api.github.com/repos/${REPO}/releases/latest`;
const CACHE_KEY = 'reachy.daemonLatestRelease.v1';
const TTL_MS = 6 * 60 * 60 * 1000;

/**
 * Floor below which the shell refuses to run an app.
 *
 * Set to the first release that understands the WebRTC `start_update`
 * command (reachy_mini#1208, v1.8.2), which makes it the lowest version
 * we can still repair from here: under it, the gate's own "Update now"
 * button would do nothing and the user has to go through the desktop
 * app anyway.
 */
export const MIN_SUPPORTED_DAEMON_VERSION = '1.8.2';

export interface SemVer {
  major: number;
  minor: number;
  patch: number;
}

/**
 * Parse a `MAJOR.MINOR.PATCH` string into a `SemVer`. Tolerates a
 * leading `v` (GitHub tags) and trailing pre-release / build metadata
 * (`1.8.2-rc1` → `1.8.2`). Returns `null` when the first three numeric
 * components can't be read, so callers can fail-open.
 */
export function parseSemver(value: string | null | undefined): SemVer | null {
  if (typeof value !== 'string') return null;
  const match = value.trim().replace(/^v/i, '').match(/^(\d+)\.(\d+)\.(\d+)/);
  if (!match) return null;
  return {
    major: Number(match[1]),
    minor: Number(match[2]),
    patch: Number(match[3]),
  };
}

/** Negative if `a < b`, positive if `a > b`, 0 if equal. */
export function compareSemver(a: SemVer, b: SemVer): number {
  return a.major - b.major || a.minor - b.minor || a.patch - b.patch;
}

/**
 * True only when BOTH versions parse AND `current` is strictly behind
 * `latest`. Any unparseable input returns `false` (fail-open).
 */
export function isDaemonOutdated(
  current: string | null | undefined,
  latest: string | null | undefined,
): boolean {
  const c = parseSemver(current);
  const l = parseSemver(latest);
  if (!c || !l) return false;
  return compareSemver(c, l) < 0;
}

/**
 * What the shell should do about this daemon.
 *
 *   `'block'`  - below the supported floor: hard gate.
 *   `'notice'` - behind the latest release: dismissable banner.
 *   `null`     - up to date, or not enough information to judge.
 */
export type DaemonUpdateSeverity = 'block' | 'notice';

export function daemonUpdateSeverity(
  current: string | null | undefined,
  latest: string | null | undefined,
): DaemonUpdateSeverity | null {
  const c = parseSemver(current);
  // Unknown version: could be a daemon predating `get_version`, could be
  // a build with an exotic version string. Either way we know nothing,
  // so we do nothing.
  if (!c) return null;

  const floor = parseSemver(MIN_SUPPORTED_DAEMON_VERSION);
  if (floor && compareSemver(c, floor) < 0) return 'block';

  return isDaemonOutdated(current, latest) ? 'notice' : null;
}

/**
 * True when the daemon is new enough to install its own update over the
 * data channel. Unknown / unparseable → false, so the caller points the
 * user at the desktop app rather than offering a dead button.
 */
export function supportsSelfUpdate(current: string | null | undefined): boolean {
  const c = parseSemver(current);
  const min = parseSemver(MIN_SUPPORTED_DAEMON_VERSION);
  return !!c && !!min && compareSemver(c, min) >= 0;
}

interface CacheEntry {
  value: string | null;
  at: number;
}

function readCache(): CacheEntry | null {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<CacheEntry>;
    if (typeof parsed.at !== 'number') return null;
    return {
      value: typeof parsed.value === 'string' ? parsed.value : null,
      at: parsed.at,
    };
  } catch {
    return null;
  }
}

/**
 * Resolve the latest published daemon version (e.g. `"1.8.2"`), or
 * `null` when it can't be determined. Cached in localStorage for
 * `TTL_MS`: the unauthenticated GitHub API allows 60 requests per
 * hour per IP, and a Space reloads far more often than daemons ship.
 * Never throws.
 */
export async function fetchLatestDaemonVersion(): Promise<string | null> {
  const stored = readCache();
  if (stored && Date.now() - stored.at < TTL_MS) return stored.value;

  try {
    const res = await fetch(LATEST_URL, {
      headers: { Accept: 'application/vnd.github+json' },
    });
    if (!res.ok) {
      // Keep serving a stale cached value rather than flapping to null
      // on a transient 403 (rate limit) / 5xx.
      return stored?.value ?? null;
    }
    const json = (await res.json()) as { tag_name?: unknown };
    const tag = typeof json.tag_name === 'string' ? json.tag_name : null;
    const value = tag ? tag.replace(/^v/i, '') : null;
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify({ value, at: Date.now() }));
    } catch {
      /* storage full / unavailable - next call refetches */
    }
    return value;
  } catch {
    return stored?.value ?? null;
  }
}
