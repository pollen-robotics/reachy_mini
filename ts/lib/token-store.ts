/**
 * Session-scoped persistence of the HF OAuth token.
 *
 * The token lives in `sessionStorage` (tab-scoped, dies with the tab),
 * under keys shared with the host shell and the embed bootstrap. On top
 * of the OAuth expiry, reads enforce a SLIDING IDLE WINDOW: a token
 * whose last successful read is older than `TOKEN_MAX_IDLE_MS` is
 * treated as absent and wiped. That property replaced the host's old
 * "wipe on pagehide" hygiene, which fired on every reload / navigation
 * (pagehide can't tell a refresh from a tab close) and forced a
 * re-login on every F5. The idle window keeps the one attack the wipe
 * actually covered - a tab resurrected days later via session restore
 * gets its sessionStorage back - without punishing reloads.
 *
 * `consumeOAuthErrorParams` supports the silent sign-in flow
 * (`login({ prompt: 'none' })`): a failed silent authorization comes
 * back as `?error=login_required...` query params, which must be
 * stripped from the URL before the app router or a copy-pasted link
 * leaks them around.
 */

const KEY_TOKEN = 'hf_token';
const KEY_USER = 'hf_username';
const KEY_EXPIRES = 'hf_token_expires';
const KEY_LAST_SEEN = 'hf_token_last_seen';

/** Idle cutoff: a cached token unused for longer than this is dropped. */
export const TOKEN_MAX_IDLE_MS = 24 * 60 * 60 * 1000;

export interface StoredToken {
    token: string;
    username: string;
    /** ISO date string of the OAuth expiry. */
    expires: string;
}

/** Persist a token bundle and stamp its last-seen time. */
export function writeStoredToken(bundle: StoredToken): void {
    try {
        sessionStorage.setItem(KEY_TOKEN, bundle.token);
        sessionStorage.setItem(KEY_USER, bundle.username);
        sessionStorage.setItem(KEY_EXPIRES, bundle.expires);
        sessionStorage.setItem(KEY_LAST_SEEN, new Date().toISOString());
    } catch {
        /* ignore - private browsing / quota */
    }
}

/**
 * Read the cached token if it is still usable, refreshing its
 * last-seen stamp (sliding window). Returns `null` - and wipes the
 * store - when the token is expired or idle for too long.
 *
 * A missing last-seen stamp (bundle written by an older SDK, or seeded
 * by the embed's fragment-credentials path) is treated as fresh and
 * stamped now, so a rollout never logs anyone out.
 */
export function readUsableToken(now: number = Date.now()): StoredToken | null {
    try {
        const token = sessionStorage.getItem(KEY_TOKEN);
        const username = sessionStorage.getItem(KEY_USER);
        const expires = sessionStorage.getItem(KEY_EXPIRES);
        if (!token || !expires) return null;

        if (new Date(expires).getTime() <= now) {
            clearStoredToken();
            return null;
        }

        const lastSeen = sessionStorage.getItem(KEY_LAST_SEEN);
        if (lastSeen !== null) {
            const lastSeenMs = new Date(lastSeen).getTime();
            if (!Number.isFinite(lastSeenMs) || now - lastSeenMs > TOKEN_MAX_IDLE_MS) {
                clearStoredToken();
                return null;
            }
        }
        sessionStorage.setItem(KEY_LAST_SEEN, new Date(now).toISOString());
        return { token, username: username ?? '', expires };
    } catch {
        return null;
    }
}

/** Remove every token key from the store. */
export function clearStoredToken(): void {
    try {
        sessionStorage.removeItem(KEY_TOKEN);
        sessionStorage.removeItem(KEY_USER);
        sessionStorage.removeItem(KEY_EXPIRES);
        sessionStorage.removeItem(KEY_LAST_SEEN);
    } catch {
        /* ignore */
    }
}

/**
 * OAuth error codes a `prompt=none` silent authorization can bounce
 * back with. Gating on this set keeps `consumeOAuthErrorParams` from
 * swallowing an app's own unrelated `?error=` param (and rewriting
 * history for it) on a page load that has nothing to do with OAuth.
 */
const OAUTH_ERROR_CODES = new Set([
    'login_required',
    'consent_required',
    'interaction_required',
    'access_denied',
]);

/**
 * Detect an OAuth error return (`?error=...` from a `prompt=none`
 * silent authorization), strip the OAuth params from the URL, and
 * return the error code. Returns `null` when the URL carries no OAuth
 * error. Other query params are preserved.
 */
export function consumeOAuthErrorParams(): string | null {
    try {
        const url = new URL(window.location.href);
        const error = url.searchParams.get('error');
        if (error === null || !OAUTH_ERROR_CODES.has(error)) return null;
        url.searchParams.delete('error');
        url.searchParams.delete('error_description');
        url.searchParams.delete('state');
        window.history.replaceState(null, '', url.toString());
        return error;
    } catch {
        return null;
    }
}
