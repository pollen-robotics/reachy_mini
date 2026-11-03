/**
 * Every call to central is bounded.
 *
 * A path that stalls without failing (lossy Wi-Fi, a half-dead connection)
 * used to hang whatever awaited it: a re-dial waiting on connect(), a
 * session start waiting for an offer that would never come, a stop waiting
 * on its endSession. The properties that must hold:
 *   - a /send that never answers resolves null after CENTRAL_REQUEST_TIMEOUT_MS;
 *   - connect() rejects when /events never answers, or answers but never
 *     sends `welcome`, and leaves the instance disconnected;
 *   - a stream that is welcomed in time is not aborted afterwards;
 *   - a startSession that central did not take fails the dial at once.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';

const encoder = new TextEncoder();

interface Internals {
    _token: string | null;
    _state: 'disconnected' | 'connected' | 'streaming';
    _sseAbortController: AbortController | null;
    _sendToServer(msg: Record<string, unknown>): Promise<unknown>;
}

/** A fetch that never settles on its own, only when aborted. */
function hangingFetch(): void {
    vi.stubGlobal(
        'fetch',
        vi.fn(
            (_url: string, init?: RequestInit) =>
                new Promise<Response>((_resolve, reject) => {
                    init?.signal?.addEventListener('abort', () =>
                        reject(new DOMException('aborted', 'AbortError')),
                    );
                }),
        ),
    );
}

/** A /events stream that answers 200, then sends `firstLine` if given. */
function eventsStream(firstLine?: string): void {
    vi.stubGlobal(
        'fetch',
        vi.fn(async (_url: string, init?: RequestInit) => {
            const body = new ReadableStream<Uint8Array>({
                start(c) {
                    if (firstLine) c.enqueue(encoder.encode(firstLine));
                    init?.signal?.addEventListener('abort', () =>
                        c.error(new DOMException('aborted', 'AbortError')),
                    );
                },
            });
            return new Response(body, { status: 200 });
        }),
    );
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
});

describe('central calls are bounded', () => {
    it('a /send that never answers resolves null', async () => {
        hangingFetch();
        const r = new ReachyMini();
        const s = r as unknown as Internals;
        s._token = 'hf_test';
        let result: unknown = 'pending';
        void s._sendToServer({ type: 'endSession' }).then((v) => { result = v; });

        await vi.advanceTimersByTimeAsync(9_000);
        expect(result).toBe('pending');
        await vi.advanceTimersByTimeAsync(2_000);
        expect(result).toBeNull();
    });

    it('connect() rejects when /events never answers', async () => {
        hangingFetch();
        const r = new ReachyMini();
        const outcome = r.connect('hf_test').then(() => 'resolved', (e: Error) => e.message);

        await vi.advanceTimersByTimeAsync(16_000);

        expect(await outcome).toMatch(/no answer from central/);
        expect(r.state).toBe('disconnected');
        expect((r as unknown as Internals)._sseAbortController).toBeNull();
    });

    it('connect() rejects when central answers but never welcomes', async () => {
        eventsStream();
        const r = new ReachyMini();
        (r as unknown as Internals)._sendToServer = vi.fn().mockResolvedValue(null);
        const outcome = r.connect('hf_test').then(() => 'resolved', (e: Error) => e.message);

        await vi.advanceTimersByTimeAsync(16_000);

        expect(await outcome).toMatch(/no welcome from central/);
        expect(r.state).toBe('disconnected');
    });

    it('keeps a stream welcomed in time', async () => {
        eventsStream('data: {"type":"welcome","peerId":"me"}\n\n');
        const r = new ReachyMini();
        (r as unknown as Internals)._sendToServer = vi.fn().mockResolvedValue(null);
        const disconnected = vi.fn();
        r.addEventListener('disconnected', disconnected);

        await r.connect('hf_test');
        await vi.advanceTimersByTimeAsync(20_000);

        expect(r.state).toBe('connected');
        expect(disconnected).not.toHaveBeenCalled();
        r.disconnect();
    });
});

describe('a startSession central did not take', () => {
    it('fails the dial at once instead of waiting for an offer', async () => {
        vi.stubGlobal(
            'RTCPeerConnection',
            class {
                iceConnectionState = 'new';
                close(): void { /* nothing to release */ }
            },
        );
        const r = new ReachyMini();
        const s = r as unknown as Internals;
        s._token = 'hf_test';
        s._state = 'connected';
        s._sendToServer = vi.fn().mockResolvedValue(null);

        const outcome = r.startSession('robot-1').then(() => 'resolved', (e: Error) => e.message);
        await vi.advanceTimersByTimeAsync(0);

        expect(await outcome).toMatch(/Could not reach central/);
    });
});
