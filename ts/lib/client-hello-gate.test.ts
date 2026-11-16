/**
 * client_hello is only sent to daemons that advertise it.
 *
 * Older daemons answer an unknown command with `{error}`, which the SDK
 * re-emits to the app as a robot error: sending the hello there would make
 * every JS app see an error on each session. The daemon's `get_version`
 * reply carries `client_hello: true` when it knows the command.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { resetPublicSpaceIdCache } from './client-hello.js';
import { ReachyMini } from './reachy-mini.js';

interface Internals {
    _dc: { readyState: string; send: (msg: string) => void } | null;
    _handleRobotMessage(data: Record<string, unknown>): void;
}

function makeInstance(options = {}): { internals: Internals; sent: Array<Record<string, unknown>> } {
    const internals = new ReachyMini(options) as unknown as Internals;
    const sent: Array<Record<string, unknown>> = [];
    internals._dc = {
        readyState: 'open',
        send: (msg: string) => { sent.push(JSON.parse(msg) as Record<string, unknown>); },
    };
    return { internals, sent };
}

const hellos = (sent: Array<Record<string, unknown>>) => sent.filter((m) => m.type === 'client_hello');

beforeEach(() => {
    resetPublicSpaceIdCache();
    (globalThis as { huggingface?: unknown }).huggingface = {
        variables: { SPACE_ID: 'pollen-robotics/emotions' },
    };
    vi.stubGlobal('fetch', vi.fn(async () => Response.json({ private: false })));
});

afterEach(() => {
    delete (globalThis as { huggingface?: unknown }).huggingface;
    vi.unstubAllGlobals();
});

describe('client_hello gate', () => {
    it('stays silent with a daemon that does not advertise it', async () => {
        const { internals, sent } = makeInstance();
        internals._handleRobotMessage({ version: '1.13.0' });
        await new Promise((resolve) => setTimeout(resolve, 0));
        expect(hellos(sent)).toEqual([]);
    });

    it('sends one hello with the public app id when advertised', async () => {
        const { internals, sent } = makeInstance({ client: 'mobile/0.11.4 (android)' });
        internals._handleRobotMessage({ version: '1.14.0', client_hello: true });
        internals._handleRobotMessage({ version: '1.14.0', client_hello: true });
        await vi.waitFor(() => expect(hellos(sent)).toHaveLength(1));
        expect(hellos(sent)[0]).toEqual({
            type: 'client_hello',
            client: 'mobile/0.11.4 (android)',
            app: 'pollen-robotics/emotions',
        });
    });

    it('drops the hello if the channel changed before the Space check returned', async () => {
        const { internals, sent } = makeInstance();
        internals._handleRobotMessage({ version: '1.14.0', client_hello: true });
        internals._dc = { readyState: 'open', send: () => {} };
        await new Promise((resolve) => setTimeout(resolve, 0));
        expect(hellos(sent)).toEqual([]);
    });
});
