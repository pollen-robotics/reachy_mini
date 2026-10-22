/**
 * Races between dials, signaling opens and late replies.
 *
 * Found in review of the session-id and time-limit changes:
 *   - a startSession reply that lands after its dial was abandoned must not
 *     become the next dial's id (it would hide that dial's own rejection or
 *     its robot's end); it is retired and the robot told to end it;
 *   - two signaling opens at once share one stream instead of running two;
 *   - a stall timer that fires late because the page was frozen gives the
 *     buffered bytes a moment before aborting;
 *   - robot `peer` messages for a retired session are dropped;
 *   - the robot list fetch is bounded;
 *   - with a stale id kept after central's advisory end, an endSession for
 *     another id is still ignored.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';
import type { SessionSupervisor } from './session-supervisor.js';

const encoder = new TextEncoder();
const WELCOME = 'data: {"type":"welcome","peerId":"me"}\n\n';

interface Internals {
    _state: 'disconnected' | 'connected' | 'streaming';
    _selectedRobotId: string | null;
    _sessionId: string | null;
    _sessionReject: ((e: Error) => void) | null;
    _pc: unknown;
    _dc: unknown;
    _dcOpen: boolean;
    _token: string | null;
    _robots: unknown[];
    _retiredSessionIds: Set<string>;
    _supervisor: SessionSupervisor;
    _sendToServer(m: Record<string, unknown>): Promise<unknown>;
    _handleEndSession(m: Record<string, unknown>): void;
    _handlePeerMessage(m: Record<string, unknown>): Promise<void>;
    _fetchOwnedRobots(): Promise<unknown[]>;
}

function streamResponse(init: RequestInit | undefined, first?: string): Response {
    const body = new ReadableStream<Uint8Array>({
        start(c) {
            if (first) c.enqueue(encoder.encode(first));
            init?.signal?.addEventListener('abort', () =>
                c.error(new DOMException('aborted', 'AbortError')),
            );
        },
    });
    return new Response(body, { status: 200 });
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
});

describe('late startSession replies', () => {
    function redialing(): {
        r: ReachyMini;
        s: Internals;
        replies: Array<(v: unknown) => void>;
        sent: () => Array<Record<string, unknown>>;
    } {
        vi.stubGlobal(
            'RTCPeerConnection',
            class {
                iceConnectionState = 'new';
                close(): void { /* nothing to release */ }
            },
        );
        const r = new ReachyMini();
        const s = r as unknown as Internals;
        s._token = 'hf';
        s._state = 'streaming';
        s._selectedRobotId = 'robot-1';
        s._sessionId = 'sess-0';
        s._pc = { close: vi.fn(), iceConnectionState: 'connected' };
        const replies: Array<(v: unknown) => void> = [];
        const send = vi.fn((m: Record<string, unknown>) =>
            m.type === 'startSession'
                ? new Promise((res) => replies.push(res))
                : Promise.resolve(null),
        );
        s._sendToServer = send;
        const sent = () => send.mock.calls.map((c) => c[0] as Record<string, unknown>);
        return { r, s, replies, sent };
    }

    it("an abandoned dial's late reply does not hide the next dial's rejection", async () => {
        const { r, s, replies, sent } = redialing();
        s._supervisor.maybeBeginRedial('test');
        await vi.advanceTimersByTimeAsync(0);
        // Attempt 1 is rejected over SSE before its HTTP reply.
        s._handleEndSession({ type: 'endSession', sessionId: 'X1', reason: 'robot_busy_local' });
        await vi.advanceTimersByTimeAsync(2_100);
        const starts = () => sent().filter((m) => m.type === 'startSession').length;
        expect(starts()).toBe(2);

        replies[0]!({ type: 'sessionStarted', sessionId: 'X1' });
        await vi.advanceTimersByTimeAsync(0);
        expect(s._sessionId).toBeNull();
        expect(s._retiredSessionIds.has('X1')).toBe(true);
        // The robot is told to end the session it may have started for it.
        expect(sent()).toContainEqual({ type: 'endSession', sessionId: 'X1', reason: 'client_stop' });

        // Attempt 2's own rejection still applies: the loop moves on.
        s._handleEndSession({ type: 'endSession', sessionId: 'Y2', reason: 'robot_busy_local' });
        await vi.advanceTimersByTimeAsync(4_100);
        expect(starts()).toBe(3);
        r.disconnect();
    });

    it('a central refusal fails the dial at once', async () => {
        const { r, s, replies } = redialing();
        s._state = 'connected';
        s._pc = null;
        const outcome = r.startSession('robot-1').then(() => 'resolved', (e: Error) => e.message);
        await vi.advanceTimersByTimeAsync(0);
        replies[0]!({ type: 'error', message: 'Producer not found' });
        await vi.advanceTimersByTimeAsync(0);
        expect(await outcome).toMatch(/Central refused the session: Producer not found/);
        r.disconnect();
    });
});

describe('signaling opens', () => {
    it('a second connect() joins the open in flight: one stream, both resolve', async () => {
        let release!: () => void;
        const fetchMock = vi.fn(
            (_u: string, init?: RequestInit) =>
                new Promise<Response>((res) => {
                    release = () => res(streamResponse(init, WELCOME));
                }),
        );
        vi.stubGlobal('fetch', fetchMock);
        const r = new ReachyMini();
        (r as unknown as Internals)._sendToServer = vi.fn().mockResolvedValue(null);

        const a = r.connect('hf');
        const b = r.connect('hf');
        release();
        await Promise.all([a, b]);

        expect(fetchMock).toHaveBeenCalledTimes(1);
        expect(r.state).toBe('connected');
        r.disconnect();
    });

    it('connect() right after disconnect() opens a fresh stream', async () => {
        const fetchMock = vi.fn(async (_u: string, init?: RequestInit) =>
            streamResponse(init, WELCOME));
        vi.stubGlobal('fetch', fetchMock);
        const r = new ReachyMini();
        (r as unknown as Internals)._sendToServer = vi.fn().mockResolvedValue(null);

        const first = r.connect('hf').catch(() => 'aborted');
        r.disconnect(); // aborts the open in flight
        await r.connect('hf');
        await first;

        expect(fetchMock).toHaveBeenCalledTimes(2);
        expect(r.state).toBe('connected');
        r.disconnect();
    });

    it('a stall timer fired late by a frozen page re-judges after a grace', async () => {
        let push!: (line: string) => void;
        vi.stubGlobal(
            'fetch',
            vi.fn(async (_u: string, init?: RequestInit) => {
                const body = new ReadableStream<Uint8Array>({
                    start(c) {
                        push = (line) => c.enqueue(encoder.encode(line));
                        c.enqueue(encoder.encode(WELCOME));
                        init?.signal?.addEventListener('abort', () =>
                            c.error(new DOMException('aborted', 'AbortError')),
                        );
                    },
                });
                return new Response(body, { status: 200 });
            }),
        );
        const r = new ReachyMini();
        (r as unknown as Internals)._sendToServer = vi.fn().mockResolvedValue(null);
        const disconnected = vi.fn();
        r.addEventListener('disconnected', disconnected);
        await r.connect('hf');

        // The page freezes for 3 minutes: wall clock jumps, the timer fires late.
        vi.setSystemTime(Date.now() + 180_000);
        await vi.advanceTimersByTimeAsync(75_000);
        expect(disconnected).not.toHaveBeenCalled();
        // The bytes central buffered meanwhile are read within the grace.
        push(': ping\n\n');
        await vi.advanceTimersByTimeAsync(6_000);
        expect(disconnected).not.toHaveBeenCalled();
        expect(r.state).toBe('connected');
        r.disconnect();
    });
});

describe('retired sessions', () => {
    it("drops a robot offer for a retired session", async () => {
        // Real enough for the handler to reach setRemoteDescription.
        vi.stubGlobal('RTCSessionDescription', class {
            constructor(init: RTCSessionDescriptionInit) { Object.assign(this, init); }
        });
        const r = new ReachyMini();
        const s = r as unknown as Internals;
        const setRemoteDescription = vi.fn();
        s._pc = { setRemoteDescription, close: vi.fn() };
        s._retiredSessionIds.add('old');
        await s._handlePeerMessage({ type: 'peer', sessionId: 'old', sdp: { type: 'answer', sdp: 'v=0' } });
        expect(setRemoteDescription).not.toHaveBeenCalled();
        // Control: the same message for a live session does reach the pc.
        await s._handlePeerMessage({ type: 'peer', sessionId: 'live', sdp: { type: 'answer', sdp: 'v=0' } });
        expect(setRemoteDescription).toHaveBeenCalledOnce();
    });

    it('with a stale id kept, an endSession for another id is ignored', () => {
        const r = new ReachyMini();
        const s = r as unknown as Internals;
        s._token = 'hf';
        s._state = 'streaming';
        s._sessionId = 'central-1';
        s._dcOpen = true;
        s._dc = { readyState: 'open', close: vi.fn() };
        s._pc = { iceConnectionState: 'connected', close: vi.fn() };
        s._sendToServer = vi.fn().mockResolvedValue(null);
        s._supervisor.stampDcInbound();
        // Central's advisory end: the id becomes stale, the session lives on.
        s._handleEndSession({ type: 'endSession', sessionId: 'central-1' });
        expect(s._sessionId).toBeNull();
        const stopped = vi.fn();
        r.addEventListener('sessionStopped', stopped);

        s._handleEndSession({ type: 'endSession', sessionId: 'someone-else', reason: 'robot_stopping' });

        expect(stopped).not.toHaveBeenCalled();
        expect(s._state).toBe('streaming');
    });
});

describe('robot list fetch', () => {
    it('falls back to the SSE list when /api/robot-status never answers', async () => {
        vi.stubGlobal(
            'fetch',
            vi.fn(
                (_u: string, init?: RequestInit) =>
                    new Promise<Response>((_res, rej) =>
                        init?.signal?.addEventListener('abort', () =>
                            rej(new DOMException('aborted', 'AbortError')),
                        ),
                    ),
            ),
        );
        const r = new ReachyMini();
        const s = r as unknown as Internals;
        s._token = 'hf';
        s._robots = [{ id: 'robot-1', meta: { name: 'Reachy' } }];
        let result: unknown = 'pending';
        void s._fetchOwnedRobots().then((v) => { result = v; });
        await vi.advanceTimersByTimeAsync(11_000);
        expect(result).toEqual([
            expect.objectContaining({ id: 'robot-1', name: 'Reachy' }),
        ]);
    });
});
