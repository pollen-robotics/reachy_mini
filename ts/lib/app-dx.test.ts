/**
 * Small DX contracts that fresh app authors (human or agent) kept tripping on:
 *   - `playMove` frames captured from `state` carry a flat `number[16]` head,
 *     but the daemon's recorded-move parser needs a nested 4×4;
 *   - `matrixToRpy` must accept that same flat head;
 *   - app audio reaches the robot speaker through a public method, not `_pc`;
 *   - `safelyReturnToPose` stops head tracking first, since tracking at
 *     weight 1 overrides the return-to-rest goto.
 */
import { describe, expect, it, vi } from 'vitest';

import { safelyReturnToPose } from '../animation/safe-return.js';
import { matrixToRpy, rpyToMatrix } from './math.js';
import { ReachyMini } from './reachy-mini.js';
import type { ReachyMiniInstance } from './types.js';
import { normalizeMoveFrames } from './upload-helpers.js';

describe('normalizeMoveFrames', () => {
    it('reshapes a flat 16-float head into a nested 4×4, row-major', () => {
        const flat = Array.from({ length: 16 }, (_, i) => i);
        const [frame] = normalizeMoveFrames([{ head: flat, antennas: [0.1, -0.1], body_yaw: 0 }]);
        expect(frame).toEqual({
            head: [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            antennas: [0.1, -0.1],
            body_yaw: 0,
        });
    });

    it('passes nested heads and head-less frames through untouched', () => {
        const nested = { head: rpyToMatrix(0, 10, 0) };
        const antennasOnly = { antennas: [0, 0] };
        const out = normalizeMoveFrames([nested, antennasOnly]);
        expect(out[0]).toBe(nested);
        expect(out[1]).toBe(antennasOnly);
    });
});

describe('matrixToRpy', () => {
    it('gives the same angles for the nested and the flat form', () => {
        const nested = rpyToMatrix(5, -10, 20);
        const fromNested = matrixToRpy(nested);
        const fromFlat = matrixToRpy(nested.flat());
        expect(fromFlat.roll).toBeCloseTo(fromNested.roll);
        expect(fromFlat.pitch).toBeCloseTo(fromNested.pitch);
        expect(fromFlat.yaw).toBeCloseTo(fromNested.yaw);
        expect(fromNested.yaw).toBeCloseTo(20);
    });
});

/** Private surface the audio tests poke at. */
interface AudioInternals {
    _pc: { getTransceivers: () => unknown[] } | null;
    _micStream: { getAudioTracks: () => MediaStreamTrack[] } | null;
}

function fakeTrack(kind: 'audio' | 'video', id: string = kind): MediaStreamTrack {
    return { kind, id } as unknown as MediaStreamTrack;
}

function withAudioSender(r: ReachyMini) {
    const replaceTrack = vi.fn(async () => { /* accepted */ });
    const internals = r as unknown as AudioInternals;
    const placeholder = fakeTrack('audio', 'placeholder');
    internals._micStream = { getAudioTracks: () => [placeholder] };
    internals._pc = {
        getTransceivers: () => [
            { receiver: { track: { kind: 'video' } }, sender: { replaceTrack: vi.fn() } },
            { receiver: { track: { kind: 'audio' } }, sender: { replaceTrack } },
        ],
    };
    return { replaceTrack, placeholder };
}

describe('setOutgoingAudioTrack', () => {
    it('puts the app track on the audio sender', async () => {
        const r = new ReachyMini();
        const { replaceTrack } = withAudioSender(r);
        const tts = fakeTrack('audio', 'tts');
        await expect(r.setOutgoingAudioTrack(tts)).resolves.toBe(true);
        expect(replaceTrack).toHaveBeenCalledWith(tts);
    });

    it('null goes back to the silent placeholder', async () => {
        const r = new ReachyMini();
        const { replaceTrack, placeholder } = withAudioSender(r);
        await r.setOutgoingAudioTrack(null);
        expect(replaceTrack).toHaveBeenLastCalledWith(placeholder);
    });

    it('resolves false before a session exists, without throwing', async () => {
        const r = new ReachyMini();
        await expect(r.setOutgoingAudioTrack(fakeTrack('audio'))).resolves.toBe(false);
    });

    it('rejects non-audio tracks', async () => {
        const r = new ReachyMini();
        await expect(r.setOutgoingAudioTrack(fakeTrack('video'))).rejects.toThrow(TypeError);
    });
});

describe('safelyReturnToPose', () => {
    it('stops head tracking before re-enabling torque and sending the goto', () => {
        const calls: string[] = [];
        const reachy = {
            robotState: {},
            stopHeadTracking: () => { calls.push('stopHeadTracking'); return true; },
            setMotorMode: () => { calls.push('setMotorMode'); return true; },
            gotoTarget: () => { calls.push('gotoTarget'); return true; },
        } as unknown as ReachyMiniInstance;
        safelyReturnToPose(reachy);
        expect(calls).toEqual(['stopHeadTracking', 'setMotorMode', 'gotoTarget']);
    });

    it('still returns to pose when the tracking command throws', () => {
        const goto = vi.fn(() => true);
        const reachy = {
            robotState: {},
            stopHeadTracking: () => { throw new Error('channel closed'); },
            setMotorMode: () => true,
            gotoTarget: goto,
        } as unknown as ReachyMiniInstance;
        safelyReturnToPose(reachy);
        expect(goto).toHaveBeenCalledOnce();
    });
});
