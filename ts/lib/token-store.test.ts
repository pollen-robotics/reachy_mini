/**
 * Token-store tests: the sliding idle window that replaced the host's
 * pagehide wipe, and the OAuth-error URL scrubbing used by the silent
 * sign-in flow.
 *
 * The property that must hold: a token can NEVER be served past its
 * OAuth expiry, nor after sitting unused longer than the idle window -
 * but a plain reload (seconds of idle) always keeps the session.
 */
import { afterEach, describe, expect, it } from 'vitest';

import {
    TOKEN_MAX_IDLE_MS,
    clearStoredToken,
    consumeOAuthErrorParams,
    readUsableToken,
    writeStoredToken,
} from './token-store.js';

const NOW = 1_700_000_000_000;

function seed(opts?: { expiresInMs?: number; lastSeenAgoMs?: number | null }): void {
    const expiresInMs = opts?.expiresInMs ?? 60 * 60 * 1000;
    writeStoredToken({
        token: 'hf_test_token',
        username: 'thibaud',
        expires: new Date(NOW + expiresInMs).toISOString(),
    });
    if (opts?.lastSeenAgoMs === null) {
        sessionStorage.removeItem('hf_token_last_seen');
    } else if (opts?.lastSeenAgoMs !== undefined) {
        sessionStorage.setItem(
            'hf_token_last_seen',
            new Date(NOW - opts.lastSeenAgoMs).toISOString(),
        );
    }
}

afterEach(() => {
    clearStoredToken();
    window.history.replaceState(null, '', '/');
});

describe('readUsableToken', () => {
    it('round-trips a fresh token and refreshes its last-seen stamp', () => {
        seed({ lastSeenAgoMs: 60_000 });
        const read = readUsableToken(NOW);
        expect(read).toEqual({
            token: 'hf_test_token',
            username: 'thibaud',
            expires: new Date(NOW + 60 * 60 * 1000).toISOString(),
        });
        expect(sessionStorage.getItem('hf_token_last_seen')).toBe(
            new Date(NOW).toISOString(),
        );
    });

    it('returns null when nothing is stored', () => {
        expect(readUsableToken(NOW)).toBeNull();
    });

    it('drops an expired token and wipes the store', () => {
        seed({ expiresInMs: -1 });
        expect(readUsableToken(NOW)).toBeNull();
        expect(sessionStorage.getItem('hf_token')).toBeNull();
    });

    it('drops a token idle for longer than the window (session restore)', () => {
        seed({ lastSeenAgoMs: TOKEN_MAX_IDLE_MS + 1 });
        expect(readUsableToken(NOW)).toBeNull();
        expect(sessionStorage.getItem('hf_token')).toBeNull();
    });

    it('keeps a token idle for less than the window (plain reload)', () => {
        seed({ lastSeenAgoMs: TOKEN_MAX_IDLE_MS - 1 });
        expect(readUsableToken(NOW)).not.toBeNull();
    });

    it('accepts and stamps a legacy bundle without a last-seen key', () => {
        seed({ lastSeenAgoMs: null });
        expect(readUsableToken(NOW)).not.toBeNull();
        expect(sessionStorage.getItem('hf_token_last_seen')).toBe(
            new Date(NOW).toISOString(),
        );
    });

    it('drops a token whose last-seen stamp is unparsable', () => {
        seed();
        sessionStorage.setItem('hf_token_last_seen', 'not-a-date');
        expect(readUsableToken(NOW)).toBeNull();
    });
});

describe('consumeOAuthErrorParams', () => {
    it('returns null and leaves the URL alone without an error param', () => {
        window.history.replaceState(null, '', '/?config=abc');
        expect(consumeOAuthErrorParams()).toBeNull();
        expect(window.location.search).toBe('?config=abc');
    });

    it('consumes a silent-auth decline and scrubs only the OAuth params', () => {
        window.history.replaceState(
            null,
            '',
            '/?config=abc&error=login_required&error_description=User+must+be+logged+in&state=xyz',
        );
        expect(consumeOAuthErrorParams()).toBe('login_required');
        expect(window.location.search).toBe('?config=abc');
    });
});
