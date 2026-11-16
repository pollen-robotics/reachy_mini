/**
 * Session liveness follows the peer-to-peer link, not central's signaling.
 *
 * The properties that must hold:
 *   - a reason-less endSession from central (its own eviction after our
 *     signaling timed out) is advisory while the peer link is live;
 *   - an endSession with a reason, or one arriving on an idle link, still
 *     ends the session;
 *   - losing signaling while streaming keeps the session (state stays
 *     `streaming`) and reconnects signaling in the background.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';
import type { SessionSupervisor } from './session-supervisor.js';

interface Internals {
    _state: 'disconnected' | 'connected' | 'streaming';
    _sessionId: string | null;
    _dcOpen: boolean;
    _dc: { readyState: string; close: () => void; onclose?: () => void } | null;
    _pc: { iceConnectionState: string; close: () => void } | null;
    _token: string | null;
    _signalingDown: boolean;
    _supervisor: SessionSupervisor;
    _sendToServer(msg: Record<string, unknown>): Promise<unknown>;
    _handleEndSession(msg: Record<string, unknown>): void;
    _openSignaling(): Promise<void>;
}

function liveSession(): { r: ReachyMini; s: Internals } {
    const r = new ReachyMini();
    const s = r as unknown as Internals;
    s._state = 'streaming';
    s._sessionId = 'central-1';
    s._dcOpen = true;
    s._dc = { readyState: 'open', close: vi.fn() };
    s._pc = { iceConnectionState: 'connected', close: vi.fn() };
    s._token = 'hf_test';
    s._sendToServer = vi.fn().mockResolvedValue(null);
    s._supervisor.stampDcInbound();
    return { r, s };
}

function events(r: ReachyMini, name: string): unknown[] {
    const seen: unknown[] = [];
    r.addEventListener(name, (e) => seen.push((e as CustomEvent).detail));
    return seen;
}

beforeEach(() => {
    vi.useFakeTimers();
});

afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
});

describe('central endSession on a streaming session', () => {
    it('is advisory without a reason while the peer link is live', () => {
        const { r, s } = liveSession();
        const stopped = events(r, 'sessionStopped');
        const advisory = events(r, 'signalingSessionEnded');

        s._handleEndSession({ type: 'endSession', sessionId: 'central-1' });

        expect(stopped).toEqual([]);
        expect(advisory).toEqual([{ reason: null }]);
        expect(s._state).toBe('streaming');
        expect(s._sessionId).toBeNull(); // stale on central now
    });

    it('ends the session when it carries a reason', () => {
        const { r, s } = liveSession();
        const stopped = events(r, 'sessionStopped');

        s._handleEndSession({ type: 'endSession', sessionId: 'central-1', reason: 'local_app_started' });

        // Exactly one event, with the robot's reason (no extra `user` one).
        expect(stopped).toEqual([{ reason: 'local_app_started', message: 'Disconnected: a local Python app started on the robot' }]);
        expect(s._state).toBe('connected');
    });

    it('ends the session when the peer link has gone quiet', () => {
        const { r, s } = liveSession();
        const stopped = events(r, 'sessionStopped');
        vi.advanceTimersByTime(10_000); // nothing heard from the robot

        s._handleEndSession({ type: 'endSession', sessionId: 'central-1' });

        expect(stopped[0]).toMatchObject({ reason: 'remote_end' });
    });
});

describe('signaling loss while streaming', () => {
    function sseResponse(lines: string[]): Response {
        const body = new ReadableStream<Uint8Array>({
            start(controller) {
                for (const l of lines) controller.enqueue(new TextEncoder().encode(l));
                controller.close();
            },
        });
        return new Response(body, { status: 200 });
    }

    it('keeps the session and reconnects signaling in the background', async () => {
        const { r, s } = liveSession();
        const lost = events(r, 'signalingLost');
        const restored = events(r, 'signalingRestored');
        const disconnected = events(r, 'disconnected');
        const fetchMock = vi.fn().mockResolvedValue(
            sseResponse(['data: {"type":"welcome","peerId":"me"}\n']),
        );
        vi.stubGlobal('fetch', fetchMock);

        // A stream that welcomes us and then ends: the drop happens mid-session.
        await s._openSignaling();
        await vi.advanceTimersByTimeAsync(0);

        expect(restored).toEqual([{ peerId: 'me' }]);
        expect(lost).toEqual([{ reason: 'SSE closed' }]);
        expect(disconnected).toEqual([]);
        expect(s._state).toBe('streaming');
        expect(s._signalingDown).toBe(true);

        // The background retry dials central again after a short backoff.
        await vi.advanceTimersByTimeAsync(1_000);
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });
});
