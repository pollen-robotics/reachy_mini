/**
 * Settings + auth helpers for the host shell.
 *
 * Three concerns live here:
 *  1. OAuth client ID resolution (HF Spaces injection > caller-
 *     supplied > `localStorage`).
 *  2. Dev token seeding: skips the OAuth dance entirely when the
 *     app author passes `devToken` to `mountHost()`.
 *  3. Inter-load flags (`oauth-pending` for the "welcome back"
 *     animation; `signed-out` so we don't auto-relogin after a
 *     user-initiated logout).
 *
 * Storage rules (SPEC §8.2):
 *  - HF token + user name live in `sessionStorage` (tab-scoped,
 *    wiped on close). The Reachy Mini SDK reads them from
 *    `hf_token` / `hf_username` / `hf_token_expires`.
 *  - Persistent flags (`hf_oauth_client_id`, `oauth-pending`,
 *    `signed-out`) live in `localStorage`.
 *  - Tokens have a 15 min TTL by default; the SDK refreshes
 *    on demand if a real OAuth refresh flow is wired.
 */

const STORAGE_KEY_CLIENT_ID = 'reachy_mini_oauth_client_id';
const STORAGE_KEY_OAUTH_PENDING = 'reachy_mini_oauth_pending';
const STORAGE_KEY_SIGNED_OUT = 'reachy_mini_signed_out';

const SESSION_KEY_TOKEN = 'hf_token';
const SESSION_KEY_USER = 'hf_username';
const SESSION_KEY_EXPIRES = 'hf_token_expires';

const TOKEN_TTL_MS = 15 * 60 * 1000;

export type ClientIdSource =
  | 'none'
  | 'caller'
  | 'space'
  | 'localStorage';

export interface DevTokenConfig {
  token: string;
  userName: string;
}

/* ─────────────────── OAuth client ID ─────────────────── */

/** Resolve the OAuth client ID from all known sources. */
export function resolveClientId(callerSupplied?: string): string | undefined {
  if (callerSupplied) return callerSupplied;
  const fromSpace = window.huggingface?.variables?.OAUTH_CLIENT_ID;
  if (fromSpace) return fromSpace;
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY_CLIENT_ID);
    if (stored) return stored;
  } catch {
    /* ignore - private browsing / quota */
  }
  return undefined;
}

/** Same as `resolveClientId` but reports which source won.
 *  Useful for the SignInView's "you're using…" hint. */
export function resolveClientIdSource(
  callerSupplied?: string,
): { clientId: string | undefined; source: ClientIdSource } {
  if (callerSupplied) return { clientId: callerSupplied, source: 'caller' };
  const fromSpace = window.huggingface?.variables?.OAUTH_CLIENT_ID;
  if (fromSpace) return { clientId: fromSpace, source: 'space' };
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY_CLIENT_ID);
    if (stored) return { clientId: stored, source: 'localStorage' };
  } catch {
    /* ignore */
  }
  return { clientId: undefined, source: 'none' };
}

/** Persist an OAuth client ID for future sessions. */
export function storeClientId(clientId: string): void {
  try {
    window.localStorage.setItem(STORAGE_KEY_CLIENT_ID, clientId);
  } catch {
    /* ignore */
  }
}

/* ─────────────────── Dev token ─────────────────── */

/** In-memory copy of the most recent `seedDevToken()` call.
 *  Lets us re-seed after a user-initiated sign-out wipes the
 *  session storage, so a subsequent "Continue with Hugging Face"
 *  click resolves through `authenticate()` instead of throwing
 *  on a missing OAuth client ID. Cleared on full page reload. */
let cachedDevToken: DevTokenConfig | null = null;

/** Pre-seed the SDK's auth cache with a personal access token.
 *  Used in dev to skip the OAuth redirect. */
export function seedDevToken(config: DevTokenConfig): void {
  cachedDevToken = config;
  try {
    window.sessionStorage.setItem(SESSION_KEY_TOKEN, config.token);
    window.sessionStorage.setItem(SESSION_KEY_USER, config.userName);
    window.sessionStorage.setItem(
      SESSION_KEY_EXPIRES,
      new Date(Date.now() + TOKEN_TTL_MS).toISOString(),
    );
  } catch {
    /* ignore */
  }
}

/** `true` if a dev token was passed to `mountHost()` this page
 *  load, regardless of whether the session storage was later
 *  wiped by a sign-out. */
export function hasCachedDevToken(): boolean {
  return cachedDevToken !== null;
}

/** Re-seed the session storage from the cached dev token. No-op
 *  if no dev token was provided to `mountHost()`. Returns `true`
 *  iff a token was re-seeded. */
export function rehydrateDevToken(): boolean {
  if (!cachedDevToken) return false;
  seedDevToken(cachedDevToken);
  return true;
}

/* ─────────────────── OAuth pending flag ─────────────────── */

/** Mark that we're about to start an OAuth redirect. The host
 *  reads this on the next boot to show the "welcome back" view. */
export function markOAuthPending(): void {
  try {
    window.localStorage.setItem(STORAGE_KEY_OAUTH_PENDING, '1');
  } catch {
    /* ignore */
  }
}

/** Read and clear the `oauth-pending` flag in one call. Returns
 *  `true` iff this boot is the result of an OAuth redirect. */
export function consumeOAuthPending(): boolean {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY_OAUTH_PENDING);
    if (value) window.localStorage.removeItem(STORAGE_KEY_OAUTH_PENDING);
    return value === '1';
  } catch {
    return false;
  }
}

/* ─────────────────── Signed-out flag ─────────────────── */

/** Mark the user as explicitly signed out so the SDK doesn't
 *  silently re-authenticate from a leftover session cookie. */
export function markUserSignedOut(): void {
  try {
    window.localStorage.setItem(STORAGE_KEY_SIGNED_OUT, '1');
  } catch {
    /* ignore */
  }
}

/** Clear the signed-out flag (called when the user clicks
 *  "Sign in" after a logout). */
export function clearSignedOutFlag(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY_SIGNED_OUT);
  } catch {
    /* ignore */
  }
}

/** Returns `true` if the user explicitly signed out earlier. */
export function isUserSignedOut(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY_SIGNED_OUT) === '1';
  } catch {
    return false;
  }
}

/* ─────────────────── Token hygiene ─────────────────── */

/** Wipe HF token keys from sessionStorage. Called by the embed
 *  on `host:leaving` before the iframe unmounts (SPEC §8.2). */
export function wipeHfSessionStorage(): void {
  try {
    window.sessionStorage.removeItem(SESSION_KEY_TOKEN);
    window.sessionStorage.removeItem(SESSION_KEY_USER);
    window.sessionStorage.removeItem(SESSION_KEY_EXPIRES);
  } catch {
    /* ignore */
  }
}

/* ─────────────────── URL config helper ─────────────────── */

/** Decode a base64 JSON `?config=` URL parameter into a typed
 *  value. Returns `null` on missing / malformed input. The
 *  caller is responsible for validating the shape. */
export function readUrlConfig<T = unknown>(): T | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = new URLSearchParams(window.location.search).get('config');
    if (!raw) return null;
    const decoded = atob(raw);
    return JSON.parse(decoded) as T;
  } catch {
    return null;
  }
}
