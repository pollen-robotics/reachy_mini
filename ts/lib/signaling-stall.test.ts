/**
 * A signaling stream that stalls without closing must not wedge the SDK.
 *
 * Central pings an idle stream every 30 s. When a phone locks its screen or
 * drops off the network, the stream can go silent while the SDK's POSTs keep
 * working: every startSession then reaches the robot but its offer never
 * comes back. Past SIGNALING_STALL_MS of silence the stream is aborted and
 * the regular "SSE closed" path takes over.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';

const encoder = new TextEncoder();

/** A /events stream that sends `welcome`, then whatever the test pushes. */
function stalledEvents(): { push: (line: string) => void } {
    let controller!: ReadableStreamDefaultController<Uint8Array>;
    vi.stubGlobal(
        'fetch',
        vi.fn(async (_url: string, init?: RequestInit) => {
            const body = new ReadableStream<Uint8Array>({
                start(c) {
                    controller = c;
                    c.enqueue(encoder.encode('data: {"type":"welcome","peerId":"me"}\n\n'));
                    init?.signal?.addEventListener('abort', () =>
                        c.error(new DOMException('aborted', 'AbortError')),
                    );
                },
            });
            return new Response(body, { status: 200 });
        }),
    );
    return { push: (line) => controller.enqueue(encoder.encode(line)) };
}

function connectedInstance(): { r: ReachyMini; disconnected: ReturnType<typeof vi.fn> } {
    const r = new ReachyMini();
    (r as unknown as { _sendToServer: () => Promise<null> })._sendToServer = vi
        .fn()
        .mockResolvedValue(null);
    const disconnected = vi.fn();
    r.addEventListener('disconnected', disconnected);
    return { r, disconnected };
}

describe('signaling stream stall', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
    });

    it('drops a stream that went silent so the next dial reconnects', async () => {
        stalledEvents();
        const { r, disconnected } = connectedInstance();
        await r.connect('token');

        await vi.advanceTimersByTimeAsync(80_000);

        expect(disconnected).toHaveBeenCalledOnce();
        expect(r.state).toBe('disconnected');
    });

    it('keeps a stream that central keeps pinging', async () => {
        const events = stalledEvents();
        const { r, disconnected } = connectedInstance();
        await r.connect('token');

        for (let i = 0; i < 6; i++) {
            await vi.advanceTimersByTimeAsync(30_000);
            events.push(': ping\n\n');
        }

        expect(disconnected).not.toHaveBeenCalled();
        expect(r.state).toBe('connected');
        r.disconnect();
    });
});
