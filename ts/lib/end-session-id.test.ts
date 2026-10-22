/**
 * An endSession only ends the session it names.
 *
 * Central routes endSession by session id to the consumer peer, and a
 * re-dial reuses that peer: a late endSession for the old session (the
 * robot superseding its ghost after our client_redial got lost) must not
 * end the new one. The properties that must hold:
 *   - an endSession for a session we already closed is ignored;
 *   - an endSession for another id than the current one is ignored;
 *   - with no id known yet (startSession in flight) it still applies, so
 *     rejections keep working;
 *   - after central dropped our session while the link lived on, the
 *     robot can still end it under its old id.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';
import type { SessionSupervisor } from './session-supervisor.js';

interface Internals {
    _state: 'disconnected' | 'connected' | 'streaming';
    _sessionId: string | null;
    _dcOpen: boolean;
    _dc: { readyState: string; close: () => void } | null;
    _pc: { iceConnectionState: string; close: () => void } | null;
    _token: string | null;
    _supervisor: SessionSupervisor;
    _sessionReject: ((err: Error) => void) | null;
    _sendToServer(msg: Record<string, unknown>): Promise<unknown>;
    _handleEndSession(msg: Record<string, unknown>): void;
    _teardownForRedial(): void;
}

function streamOn(s: Internals, sessionId: string): void {
    s._state = 'streaming';
    s._sessionId = sessionId;
    s._dcOpen = true;
    s._dc = { readyState: 'open', close: vi.fn() };
    s._pc = { iceConnectionState: 'connected', close: vi.fn() };
    s._supervisor.stampDcInbound();
}

function liveSession(sessionId = 'central-1'): { r: ReachyMini; s: Internals } {
    const r = new ReachyMini();
    const s = r as unknown as Internals;
    s._token = 'hf_test';
    s._sendToServer = vi.fn().mockResolvedValue(null);
    streamOn(s, sessionId);
    return { r, s };
}

function stoppedEvents(r: ReachyMini): unknown[] {
    const seen: unknown[] = [];
    r.addEventListener('sessionStopped', (e) => seen.push((e as CustomEvent).detail));
    return seen;
}

beforeEach(() => {
    vi.useFakeTimers();
});

afterEach(() => {
    vi.useRealTimers();
});

describe('endSession is matched to the current session', () => {
    it("ignores a late end for a re-dial's old session", () => {
        const { r, s } = liveSession('central-1');
        s._teardownForRedial(); // retires central-1
        streamOn(s, 'central-2');
        const stopped = stoppedEvents(r);

        s._handleEndSession({ type: 'endSession', sessionId: 'central-1', reason: 'superseded' });

        expect(stopped).toEqual([]);
        expect(s._state).toBe('streaming');
        expect(s._sessionId).toBe('central-2');
    });

    it('ignores an end for an id that is not ours', () => {
        const { r, s } = liveSession('central-1');
        const stopped = stoppedEvents(r);

        s._handleEndSession({ type: 'endSession', sessionId: 'someone-else', reason: 'robot_stopping' });

        expect(stopped).toEqual([]);
        expect(s._state).toBe('streaming');
    });

    it('still ends the current session', () => {
        const { r, s } = liveSession('central-1');
        const stopped = stoppedEvents(r);

        s._handleEndSession({ type: 'endSession', sessionId: 'central-1', reason: 'robot_stopping' });

        expect(stopped).toEqual([{ reason: 'robot_stopping', message: null }]);
        expect(s._state).toBe('connected');
    });

    it('applies while startSession is in flight, so rejections still land', () => {
        const r = new ReachyMini();
        const s = r as unknown as Internals;
        s._state = 'connected';
        s._sessionId = null; // the startSession reply has not come back yet
        const reject = vi.fn();
        s._sessionReject = reject;

        s._handleEndSession({ type: 'endSession', sessionId: 'central-9', reason: 'robot_busy_local' });

        expect(reject).toHaveBeenCalledOnce();
    });

    it("a rejected start leaves no id behind to hide the next start's rejection", () => {
        const r = new ReachyMini();
        const s = r as unknown as Internals;
        s._state = 'connected';
        // First dial: central gave it id X, then the robot refused it.
        s._sessionId = 'central-X';
        const firstReject = vi.fn();
        s._sessionReject = firstReject;
        s._handleEndSession({ type: 'endSession', sessionId: 'central-X', reason: 'robot_busy_local' });
        expect(firstReject).toHaveBeenCalledOnce();
        expect(s._sessionId).toBeNull();

        // Second dial: its rejection (id Y) lands before the startSession
        // reply has set the id. It must still reject this dial.
        const secondReject = vi.fn();
        s._sessionReject = secondReject;
        s._handleEndSession({ type: 'endSession', sessionId: 'central-Y', reason: 'robot_busy_local' });
        expect(secondReject).toHaveBeenCalledOnce();

        // And a late repeat for X is ignored.
        const thirdReject = vi.fn();
        s._sessionReject = thirdReject;
        s._handleEndSession({ type: 'endSession', sessionId: 'central-X', reason: 'robot_busy_local' });
        expect(thirdReject).not.toHaveBeenCalled();
    });

    it('lets the robot end a session central already dropped', () => {
        const { r, s } = liveSession('central-1');
        const stopped = stoppedEvents(r);

        // Central's own eviction: advisory, the link is live.
        s._handleEndSession({ type: 'endSession', sessionId: 'central-1' });
        expect(s._state).toBe('streaming');
        expect(s._sessionId).toBeNull();

        s._handleEndSession({ type: 'endSession', sessionId: 'central-1', reason: 'robot_stopping' });

        expect(stopped).toEqual([{ reason: 'robot_stopping', message: null }]);
    });
});
