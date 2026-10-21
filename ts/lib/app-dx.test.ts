/**
 * Small DX contracts that fresh app authors (human or agent) kept tripping on:
 *   - `matrixToRpy` must accept the flat `number[16]` head that `state` carries;
 *   - app audio reaches the robot speaker through a public method, not `_pc`;
 *   - `safelyReturnToPose` stops head tracking first, since tracking at
 *     weight 1 overrides the return-to-rest goto.
 */
import { describe, expect, it, vi } from 'vitest';

import { safelyReturnToPose } from '../animation/safe-return.js';
import { matrixToRpy, rpyToMatrix } from './math.js';
import { ReachyMini } from './reachy-mini.js';
import type { ReachyMiniInstance } from './types.js';

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
    _micSupported: boolean;
    _outgoingAudioTrack: MediaStreamTrack | null;
}

function fakeTrack(kind: 'audio' | 'video', id: string = kind): MediaStreamTrack {
    return { kind, id } as unknown as MediaStreamTrack;
}

function withAudioSender(r: ReachyMini) {
    const replaceTrack = vi.fn(async () => { /* accepted */ });
    const internals = r as unknown as AudioInternals;
    const placeholder = fakeTrack('audio', 'placeholder');
    internals._micStream = { getAudioTracks: () => [placeholder] };
    internals._micSupported = true;
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

    it('resolves false when the robot did not offer two-way audio', async () => {
        const r = new ReachyMini();
        const { replaceTrack } = withAudioSender(r);
        (r as unknown as AudioInternals)._micSupported = false;
        await expect(r.setOutgoingAudioTrack(fakeTrack('audio'))).resolves.toBe(false);
        expect(replaceTrack).not.toHaveBeenCalled();
    });

    it('is forgotten by an explicit stopSession', async () => {
        const r = new ReachyMini();
        await r.setOutgoingAudioTrack(fakeTrack('audio'));
        await r.stopSession();
        expect((r as unknown as AudioInternals)._outgoingAudioTrack).toBeNull();
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
