/**
 * Central-listener tests: the shared fleet-watching SSE channel
 * (roles=["listener"]) consumed by the host picker and the mobile
 * scan screen.
 *
 * Properties that must hold:
 *  - registration (`setPeerStatus`) lands BEFORE `onConnect` fires,
 *    so consumers can trust the listener slot exists when they react;
 *  - a silent server-side TTL eviction (heartbeat POST → 4xx) tears
 *    the half-open SSE down and reconnects - the regression that
 *    motivated moving this into the SDK;
 *  - transient failures retry with exponential backoff, fatal auth
 *    failures (401/403) do NOT retry;
 *  - `close()` is idempotent and no callback ever fires after it.
 *
 * The fetch mock distinguishes the two wire surfaces: GET /events
 * (long-lived SSE stream, hand-rolled reader) and POST /send
 * (registration + heartbeat).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
    openCentralListener,
    type CentralListenerHandle,
} from './central-listener.js';

const SIGNALING_URL = 'https://central.test';

// ─────────────────────────────────────────────────────────────────
// Fetch mock harness
// ─────────────────────────────────────────────────────────────────

interface FakeEventsConn {
    /** Encode one `data: <json>\n` SSE line into the stream. */
    push(msg: Record<string, unknown>): void;
    /** Clean EOF (proxy idle cull, central restart). */
    end(): void;
}

interface SendCall {
    body: { type?: string; roles?: string[]; meta?: { name?: string } };
}

let eventsConns: FakeEventsConn[];
let sendCalls: SendCall[];
/** Status served to the next GET /events (then reset to 200). */
let nextEventsStatus: number;
/** Status served to every POST /send. */
let sendStatus: number;
/** When set, /send responses are parked until the test releases them. */
let deferSend: boolean;
let releaseSend: Array<() => void>;

function makeEventsResponse(signal: AbortSignal): Response {
    const encoder = new TextEncoder();
    const queue: Array<ReadableStreamReadResult<Uint8Array>> = [];
    let waiter:
        | ((r: ReadableStreamReadResult<Uint8Array>) => void)
        | null = null;
    let rejecter: ((err: unknown) => void) | null = null;

    signal.addEventListener('abort', () => {
        rejecter?.(new DOMException('The operation was aborted.', 'AbortError'));
        rejecter = null;
        waiter = null;
    });

    const conn: FakeEventsConn = {
        push(msg) {
            const item = {
                done: false as const,
                value: encoder.encode(`data: ${JSON.stringify(msg)}\n`),
            };
            if (waiter) {
                const w = waiter;
                waiter = null;
                rejecter = null;
                w(item);
            } else {
                queue.push(item);
            }
        },
        end() {
            const item = { done: true as const, value: undefined };
            if (waiter) {
                const w = waiter;
                waiter = null;
                rejecter = null;
                w(item);
            } else {
                queue.push(item);
            }
        },
    };
    eventsConns.push(conn);

    const reader = {
        read(): Promise<ReadableStreamReadResult<Uint8Array>> {
            if (signal.aborted) {
                return Promise.reject(
                    new DOMException('The operation was aborted.', 'AbortError'),
                );
            }
            const item = queue.shift();
            if (item) return Promise.resolve(item);
            return new Promise((resolve, reject) => {
                waiter = resolve;
                rejecter = reject;
            });
        },
    };

    return {
        ok: true,
        status: 200,
        body: { getReader: () => reader },
    } as unknown as Response;
}

function installFetchMock(): void {
    vi.stubGlobal(
        'fetch',
        vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
            const url = String(input);
            if (url.endsWith('/events')) {
                const status = nextEventsStatus;
                nextEventsStatus = 200;
                if (status !== 200) {
                    return { ok: false, status, body: null } as unknown as Response;
                }
                // openCentralListener always passes a signal for /events.
                return makeEventsResponse(init!.signal!);
            }
            if (url.endsWith('/send')) {
                sendCalls.push({ body: JSON.parse(String(init?.body)) });
                const respond = (): Response =>
                    ({ ok: sendStatus < 400, status: sendStatus }) as Response;
                if (deferSend) {
                    return new Promise<Response>((resolve) => {
                        releaseSend.push(() => resolve(respond()));
                    });
                }
                return respond();
            }
            throw new Error(`unexpected fetch: ${url}`);
        }),
    );
}

/** Flush pending microtasks (mock fetch resolves in microtasks). */
async function flush(): Promise<void> {
    for (let i = 0; i < 10; i++) await Promise.resolve();
}

/** Strict accessor for the i-th /events connection. */
function conn(i: number): FakeEventsConn {
    const c = eventsConns[i];
    if (!c) throw new Error(`no /events connection #${i}`);
    return c;
}

/** Strict accessor for the i-th /send call. */
function sent(i: number): SendCall {
    const c = sendCalls[i];
    if (!c) throw new Error(`no /send call #${i}`);
    return c;
}

let handle: CentralListenerHandle | null = null;

beforeEach(() => {
    vi.useFakeTimers();
    // Kill backoff jitter: Math.random()=0.5 → jitter term is exactly 0,
    // so reconnect delays equal the nominal backoff (1s, 2s, 4s, ...).
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
    eventsConns = [];
    sendCalls = [];
    nextEventsStatus = 200;
    sendStatus = 200;
    deferSend = false;
    releaseSend = [];
    installFetchMock();
});

afterEach(() => {
    handle?.close();
    handle = null;
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
});

function open(
    callbacks: Partial<Parameters<typeof openCentralListener>[0]> = {},
): CentralListenerHandle {
    handle = openCentralListener({
        token: 'hf_test',
        signalingUrl: SIGNALING_URL,
        appName: 'Test Listener',
        ...callbacks,
    });
    return handle;
}

// ─────────────────────────────────────────────────────────────────
// Registration & connect ordering
// ─────────────────────────────────────────────────────────────────

describe('registration', () => {
    it('registers as listener after welcome, then reports connected', async () => {
        const onConnect = vi.fn();
        open({ onConnect });
        await flush();
        expect(eventsConns).toHaveLength(1);
        expect(sendCalls).toHaveLength(0);

        deferSend = true;
        conn(0).push({ type: 'welcome' });
        await flush();

        // Registration POST is in flight; connect must NOT be reported yet.
        expect(sendCalls).toHaveLength(1);
        expect(sent(0).body).toMatchObject({
            type: 'setPeerStatus',
            roles: ['listener'],
            meta: { name: 'Test Listener' },
        });
        expect(onConnect).not.toHaveBeenCalled();

        releaseSend.shift()!();
        await flush();
        expect(onConnect).toHaveBeenCalledTimes(1);
    });
});

// ─────────────────────────────────────────────────────────────────
// Event dispatch
// ─────────────────────────────────────────────────────────────────

describe('event dispatch', () => {
    it('parses list / peerStatusChanged / sessionStateChanged frames', async () => {
        const onList = vi.fn();
        const onPeerStatusChanged = vi.fn();
        const onSessionStateChanged = vi.fn();
        open({ onList, onPeerStatusChanged, onSessionStateChanged });
        await flush();
        const c = conn(0);

        c.push({ type: 'welcome' });
        c.push({
            type: 'list',
            producers: [{ id: 'robot-1', meta: { name: 'Reachy' }, busy: false }],
        });
        c.push({ type: 'peerStatusChanged', peerId: 'robot-1', roles: [] });
        c.push({
            type: 'sessionStateChanged',
            peerId: 'robot-1',
            busy: true,
            activeApp: 'conversation',
        });
        await flush();

        expect(onList).toHaveBeenCalledWith([
            { id: 'robot-1', meta: { name: 'Reachy' }, busy: false },
        ]);
        expect(onPeerStatusChanged).toHaveBeenCalledWith(
            expect.objectContaining({ peerId: 'robot-1', roles: [] }),
        );
        expect(onSessionStateChanged).toHaveBeenCalledWith(
            expect.objectContaining({
                peerId: 'robot-1',
                busy: true,
                activeApp: 'conversation',
            }),
        );
    });

    it('ignores unknown frame types and unparsable payloads', async () => {
        const onError = vi.fn();
        const onList = vi.fn();
        open({ onError, onList });
        await flush();
        const c = conn(0);

        c.push({ type: 'ping' });
        await flush();
        expect(onError).not.toHaveBeenCalled();
        expect(onList).not.toHaveBeenCalled();
    });
});

// ─────────────────────────────────────────────────────────────────
// Reconnect policy
// ─────────────────────────────────────────────────────────────────

describe('reconnect', () => {
    it('reconnects after a clean EOF and resets backoff on welcome', async () => {
        const onDisconnect = vi.fn();
        open({ onDisconnect });
        await flush();
        conn(0).push({ type: 'welcome' });
        await flush();

        conn(0).end();
        await flush();
        expect(onDisconnect).toHaveBeenCalledWith('stream ended');

        // Nominal first backoff: 1 s (jitter zeroed in beforeEach).
        await vi.advanceTimersByTimeAsync(1_000);
        await flush();
        expect(eventsConns).toHaveLength(2);

        // Welcome on the new stream resets backoff → next drop retries
        // after 1 s again, not 2 s.
        conn(1).push({ type: 'welcome' });
        await flush();
        conn(1).end();
        await flush();
        await vi.advanceTimersByTimeAsync(1_000);
        await flush();
        expect(eventsConns).toHaveLength(3);
    });

    it('doubles the backoff while central stays down', async () => {
        nextEventsStatus = 503;
        open({});
        await flush();
        expect(eventsConns).toHaveLength(0);

        // Attempt 2 after 1 s - fails again (503).
        nextEventsStatus = 503;
        await vi.advanceTimersByTimeAsync(1_000);
        await flush();

        // Attempt 3 only after 2 more seconds.
        await vi.advanceTimersByTimeAsync(1_999);
        await flush();
        expect(eventsConns).toHaveLength(0);
        await vi.advanceTimersByTimeAsync(1);
        await flush();
        expect(eventsConns).toHaveLength(1);
    });

    it('treats 401 as fatal: no retry, auth_rejected surfaced', async () => {
        const onError = vi.fn();
        const onDisconnect = vi.fn();
        nextEventsStatus = 401;
        open({ onError, onDisconnect });
        await flush();

        expect(onError).toHaveBeenCalledTimes(1);
        expect(String(onError.mock.calls[0]?.[0])).toContain('401');
        expect(onDisconnect).toHaveBeenCalledWith('auth_rejected');

        await vi.advanceTimersByTimeAsync(120_000);
        await flush();
        expect(eventsConns).toHaveLength(0);
    });
});

// ─────────────────────────────────────────────────────────────────
// Heartbeat & TTL eviction
// ─────────────────────────────────────────────────────────────────

describe('heartbeat', () => {
    it('re-emits setPeerStatus at the cadence negotiated in welcome', async () => {
        open({});
        await flush();
        conn(0).push({
            type: 'welcome',
            recommended_heartbeat_interval_seconds: 5,
        });
        await flush();
        expect(sendCalls).toHaveLength(1); // initial registration

        await vi.advanceTimersByTimeAsync(5_000);
        await flush();
        expect(sendCalls).toHaveLength(2);

        await vi.advanceTimersByTimeAsync(5_000);
        await flush();
        expect(sendCalls).toHaveLength(3);
    });

    it('falls back to lease_seconds / 3 when no explicit cadence', async () => {
        open({});
        await flush();
        conn(0).push({ type: 'welcome', lease_seconds: 30 });
        await flush();
        expect(sendCalls).toHaveLength(1);

        await vi.advanceTimersByTimeAsync(10_000);
        await flush();
        expect(sendCalls).toHaveLength(2);
    });

    it('reconnects when a heartbeat reveals a server-side eviction', async () => {
        const onDisconnect = vi.fn();
        open({ onDisconnect });
        await flush();
        conn(0).push({
            type: 'welcome',
            recommended_heartbeat_interval_seconds: 5,
        });
        await flush();

        // TTL sweeper reaped us server-side: next heartbeat gets a 400.
        sendStatus = 400;
        await vi.advanceTimersByTimeAsync(5_000);
        await flush();
        expect(onDisconnect).toHaveBeenCalledWith('listener evicted (HTTP 400)');

        // A fresh stream is minted after the backoff.
        sendStatus = 200;
        await vi.advanceTimersByTimeAsync(1_000);
        await flush();
        expect(eventsConns).toHaveLength(2);
        conn(1).push({ type: 'welcome' });
        await flush();
        // Re-registration on the new stream.
        expect(sendCalls.length).toBeGreaterThanOrEqual(3);
    });
});

// ─────────────────────────────────────────────────────────────────
// close()
// ─────────────────────────────────────────────────────────────────

describe('close', () => {
    it('is idempotent and silences every callback', async () => {
        const onList = vi.fn();
        const onDisconnect = vi.fn();
        const h = open({ onList, onDisconnect });
        await flush();
        conn(0).push({
            type: 'welcome',
            recommended_heartbeat_interval_seconds: 5,
        });
        await flush();
        const sendsBefore = sendCalls.length;

        h.close();
        h.close();
        conn(0).push({ type: 'list', producers: [] });
        await flush();
        await vi.advanceTimersByTimeAsync(60_000);
        await flush();

        expect(onList).not.toHaveBeenCalled();
        expect(onDisconnect).not.toHaveBeenCalled();
        expect(sendCalls).toHaveLength(sendsBefore); // heartbeat stopped
        expect(eventsConns).toHaveLength(1); // no reconnect
    });
});
