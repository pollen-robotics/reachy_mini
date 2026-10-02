/**
 * SDK staleness self-check for the embed runtime.
 *
 * Why this exists
 * ───────────────
 * A published Space bundles its SDK at build time and is never
 * rebuilt: if a bundled SDK generation ever becomes incompatible with
 * the deployed daemons, there is NO channel to tell the users of those
 * frozen apps - except code that ships inside the app itself. This
 * module is that channel. It runs inside the iframe, so every host
 * (web shell, mobile WebView, anything future) is covered with zero
 * parent integration. The one case it structurally cannot cover is a
 * bundle that predates this very check; an independently-updated
 * parent catches that instead (the mobile app, via the absence of
 * `sdkVersion` on app-state).
 *
 * Why it is this conservative (learned in #1316 review)
 * ─────────────────────────────────────────────────────
 * The first cut of this feature was the "global kill switch" the
 * daemon update policy explicitly argues against: it warned on ANY
 * version gap, full-screen, on every reload, comparing the npm bundle
 * version against the DAEMON's GitHub release tag - two version lines
 * that only agree while they happen to be tagged together. This
 * version inverts every one of those choices:
 *
 *   - source of truth: the npm registry entry of the package itself.
 *     The registry is what apps actually install from, so it cannot
 *     drift from the bundle's own version line.
 *   - trigger: MAJOR gap only. A major bump is the only signal that
 *     the app/robot contract may genuinely no longer hold; minor and
 *     patch releases must never nag a working app.
 *   - dismissal: persisted (localStorage). One "I understand" covers
 *     everything up to the dismissed major; the overlay returns only
 *     when a NEWER major ships.
 *   - theming: follows `prefers-color-scheme` instead of hardcoding a
 *     light card into apps that may be dark.
 *
 * Fail-open discipline: unparseable versions, dev placeholders
 * (`0.x`), an unreachable registry - all resolve to "stay silent".
 * Advisory code must never cost the user their app.
 */

// NOTE: keep this module React-free. It is imported by the embed
// runtime (`embed/index.ts`), which is bundled without React.

import { parseSemver } from './daemonRelease';

const NPM_PACKAGE = '@pollen-robotics/reachy-mini-sdk';
const LATEST_URL = `https://registry.npmjs.org/${NPM_PACKAGE}/latest`;
const CACHE_KEY = 'reachy.sdkLatestRelease.v1';
const DISMISS_KEY = 'reachy.sdkStalenessDismissed.v1';
const TTL_MS = 6 * 60 * 60 * 1000;

const OVERLAY_ID = 'reachy-sdk-outdated-overlay';

/* ─────────────────── Latest-version source ─────────────────── */

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

function writeCache(entry: CacheEntry): void {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(entry));
  } catch {
    /* storage full / unavailable: we just fetch again next time */
  }
}

/**
 * Resolve the latest published SDK version from the npm registry's
 * `latest` dist-tag (a ~200-byte JSON, CORS-enabled), or `null` when
 * it can't be determined. Cached in localStorage for `TTL_MS`; a
 * failed refresh serves the stale value rather than flapping to null.
 * Never throws.
 */
export async function fetchLatestSdkVersion(): Promise<string | null> {
  const cached = readCache();
  if (cached && Date.now() - cached.at < TTL_MS) return cached.value;

  try {
    const res = await fetch(LATEST_URL, {
      headers: { Accept: 'application/json' },
    });
    if (!res.ok) return cached?.value ?? null;
    const json = (await res.json()) as { version?: unknown };
    const value = typeof json.version === 'string' ? json.version : null;
    writeCache({ value, at: Date.now() });
    return value;
  } catch {
    return cached?.value ?? null;
  }
}

/* ─────────────────── Policy ─────────────────── */

/**
 * Should the overlay show for this current/latest pair? True only
 * when BOTH versions parse, the current one is a real release (not a
 * `0.x` dev placeholder), and the latest is a strictly NEWER MAJOR.
 * Anything unknown resolves to false.
 */
export function isSdkMajorBehind(
  current: string | null | undefined,
  latest: string | null | undefined,
): boolean {
  const c = parseSemver(current);
  const l = parseSemver(latest);
  if (!c || !l) return false;
  if (c.major === 0) return false; // dev / CI placeholder builds
  return l.major > c.major;
}

/**
 * A dismissal covers every release up to the dismissed latest's
 * major: the overlay returns only when a newer major ships than the
 * one the user already acknowledged.
 */
export function isSdkWarningDismissed(
  latest: string | null | undefined,
): boolean {
  const l = parseSemver(latest);
  if (!l) return false;
  try {
    const dismissed = parseSemver(localStorage.getItem(DISMISS_KEY));
    if (!dismissed) return false;
    return l.major <= dismissed.major;
  } catch {
    return false;
  }
}

export function dismissSdkWarning(latest: string): void {
  try {
    localStorage.setItem(DISMISS_KEY, latest);
  } catch {
    /* private browsing: the dismissal just won't persist */
  }
}

/* ─────────────────── Orchestrator ─────────────────── */

/**
 * Fire-and-forget entry point for the embed: check, and show the
 * overlay when (and only when) the bundled SDK is a full major behind
 * the npm registry and the user hasn't already acknowledged that
 * major. Never throws, never blocks the session.
 */
export async function maybeWarnSdkStale(
  currentVersion: string | null,
): Promise<void> {
  try {
    const latest = await fetchLatestSdkVersion();
    if (!isSdkMajorBehind(currentVersion, latest)) return;
    if (isSdkWarningDismissed(latest)) return;
    showSdkStalenessOverlay(currentVersion as string, latest as string);
  } catch {
    /* purely advisory - never let it interfere with a live session */
  }
}

/* ─────────────────── Overlay (plain DOM) ─────────────────── */

/** Re-serialise a version through the semver parser so nothing
 *  attacker-shaped can ride a version string into `innerHTML`. */
function safeVersion(value: string): string {
  const p = parseSemver(value);
  return p ? `${p.major}.${p.minor}.${p.patch}` : '?';
}

/**
 * Plain-DOM overlay (the embed is framework-agnostic). Colors follow
 * `prefers-color-scheme` - an advisory card must not hardcode a light
 * theme into an app that may be dark. Exported for tests; production
 * code goes through `maybeWarnSdkStale`.
 */
export function showSdkStalenessOverlay(
  current: string,
  latest: string,
): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById(OVERLAY_ID)) return;

  const cur = safeVersion(current);
  const lat = safeVersion(latest);

  document.body.insertAdjacentHTML(
    'beforeend',
    `<div id="${OVERLAY_ID}" role="alertdialog" aria-label="This app may be out of date">
      <style>
        #${OVERLAY_ID} {
          position: fixed; inset: 0; z-index: 2147483000;
          display: flex; align-items: center; justify-content: center;
          padding: 16px; background: rgba(0, 0, 0, 0.55);
          font-family: system-ui, -apple-system, sans-serif;
          --card-bg: #fff; --card-fg: #1a1a1a; --card-muted: #555;
          --btn-bg: #1a1a1a; --btn-fg: #fff;
        }
        @media (prefers-color-scheme: dark) {
          #${OVERLAY_ID} {
            --card-bg: #1e1e1e; --card-fg: #f0f0f0; --card-muted: #aaa;
            --btn-bg: #f0f0f0; --btn-fg: #1a1a1a;
          }
        }
        #${OVERLAY_ID} .card {
          background: var(--card-bg); color: var(--card-fg);
          border-radius: 12px; padding: 24px; max-width: 400px;
          width: 100%; text-align: center;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        }
        #${OVERLAY_ID} .icon { font-size: 32px; line-height: 1; margin-bottom: 12px; }
        #${OVERLAY_ID} .title { font-size: 17px; font-weight: 700; margin-bottom: 8px; }
        #${OVERLAY_ID} .body {
          font-size: 13px; line-height: 1.6;
          color: var(--card-muted); margin-bottom: 16px;
        }
        #${OVERLAY_ID} button {
          width: 100%; padding: 10px 16px; border: none; border-radius: 8px;
          background: var(--btn-bg); color: var(--btn-fg);
          font-size: 13px; font-weight: 600; cursor: pointer;
        }
      </style>
      <div class="card">
        <div class="icon">\u26A0\uFE0F</div>
        <div class="title">This app may be out of date</div>
        <div class="body">It was built with SDK v${cur}, but v${lat} is the latest generation. Some things may not behave as expected with your robot.</div>
        <button type="button">I understand, continue</button>
      </div>
    </div>`,
  );

  const scrim = document.getElementById(OVERLAY_ID);
  const button = scrim?.querySelector('button');
  button?.addEventListener('click', () => {
    dismissSdkWarning(latest);
    scrim?.remove();
  });
  button?.focus();
}
