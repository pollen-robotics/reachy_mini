/**
 * connectLocal() - the SDK running ON the robot (a JS app's robot module),
 * over the daemon's loopback app link instead of a WebRTC DataChannel.
 *
 * The properties that must hold:
 *   - commands go out on the link exactly as on the DataChannel, and the
 *     pose stream is subscribed right away;
 *   - replies settle the usual waiters, pose frames feed `robotState` and
 *     stale frames (lower `seq`) are dropped;
 *   - the link closing settles pending waiters and ends the session.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ReachyMini } from './reachy-mini.js';

class FakeWebSocket {
    static OPEN = 1;
    static last: FakeWebSocket | null = null;
    readyState = 0;
    bufferedAmount = 0;
    sent: Array<Record<string, unknown>> = [];
    onopen: (() => void) | null = null;
    onmessage: ((ev: { data: string }) => void) | null = null;
    onerror: (() => void) | null = null;
    onclose: ((ev: { code: number; reason: string }) => void) | null = null;

    constructor(public url: string) {
        FakeWebSocket.last = this;
        queueMicrotask(() => {
            this.readyState = FakeWebSocket.OPEN;
            this.onopen?.();
        });
    }
    send(data: string) {
        this.sent.push(JSON.parse(data) as Record<string, unknown>);
    }
    close() {
        this.readyState = 3;
        this.onclose?.({ code: 1000, reason: 'closed by test' });
    }
    deliver(msg: Record<string, unknown>) {
        this.onmessage?.({ data: JSON.stringify(msg) });
    }
}

beforeEach(() => {
    vi.stubGlobal('WebSocket', FakeWebSocket);
});

afterEach(() => {
    vi.unstubAllGlobals();
});

async function linked(): Promise<{ r: ReachyMini; ws: FakeWebSocket }> {
    const r = new ReachyMini();
    await r.connectLocal('ws://127.0.0.1:1234/?token=t');
    return { r, ws: FakeWebSocket.last! };
}

describe('connectLocal', () => {
    it('streams from the start and sends commands on the link', async () => {
        const { r, ws } = await linked();
        expect(ws.url).toBe('ws://127.0.0.1:1234/?token=t');
        expect(r.state).toBe('streaming');
        expect(ws.sent[0]).toEqual({ type: 'subscribe_pose' });

        expect(r.setAntennasDeg(10, -10)).toBe(true);
        expect(ws.sent[1]?.type).toBe('set_full_target');
    });

    it('settles replies and applies pose frames, dropping stale ones', async () => {
        const { r, ws } = await linked();
        const version = r.getVersion();
        ws.deliver({ version: '9.9.9' });
        await expect(version).resolves.toBe('9.9.9');

        const state = (antennas: number[], seq: number) => ({
            seq,
            state: {
                antennas,
                body_yaw: 0,
                motor_mode: 'enabled',
                is_recording: false,
                is_move_running: false,
                face_target: {},
            },
        });
        ws.deliver(state([0.1, 0.2], 5));
        expect(r.robotState.antennas).toEqual([0.1, 0.2]);
        ws.deliver(state([0.9, 0.9], 4));
        expect(r.robotState.antennas).toEqual([0.1, 0.2]);
    });

    it('ends the session and fails waiters when the link closes', async () => {
        const { r, ws } = await linked();
        const stopped = vi.fn();
        r.addEventListener('sessionStopped', stopped);
        const pending = r.rpcCall('app.$snapshot');
        ws.close();
        await expect(pending).rejects.toThrow(/link closed/);
        expect(stopped).toHaveBeenCalledOnce();
        expect(r.state).toBe('disconnected');
        expect(r.setAntennasDeg(0, 0)).toBe(false);
    });
});
