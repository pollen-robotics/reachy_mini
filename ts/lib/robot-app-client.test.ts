/**
 * startRobotApp() / attachRobotApp() - the client half of a robot module.
 *
 * The properties that must hold:
 *   - the bundle is uploaded only when the robot doesn't have that SHA-256,
 *     in chunks small enough for a DataChannel message;
 *   - a module already running under the same name is reused on request,
 *     otherwise the running app is stopped first;
 *   - calls map to `app.<method>`, events/logs/state to the module's
 *     notifications, and state never goes back in time (`rev`).
 */
import { describe, expect, it } from 'vitest';

import { defineRobotApp } from '../robot/index.js';
import { attachRobotApp, startRobotApp } from './robot-app-client.js';
import type { ReachyMiniInstance } from './types.js';

type Params = Record<string, unknown>;

function fakeRobot(opts: { present?: boolean; running?: string | null; status?: string } = {}) {
    const calls: Array<[string, Params]> = [];
    const listeners = new Map<string, Set<(p: Params) => void>>();
    const reachy = {
        rpcCall: async (method: string, params: Params = {}) => {
            calls.push([method, params]);
            switch (method) {
                case 'apps.status':
                    return opts.running
                        ? { state: opts.status ?? 'running', info: { name: opts.running } }
                        : { state: 'idle', info: null };
                case 'apps.has_bundle':
                    return { present: opts.present ?? false };
                case 'apps.put_bundle':
                    return { complete: true };
                case 'app.$snapshot':
                    return { app: 'demo', protocol: 1, methods: ['wave'], state: { n: 1 }, rev: 3 };
                case 'app.wave':
                    return { waved: params.times };
                default:
                    return {};
            }
        },
        onNotification: (method: string, cb: (p: Params) => void) => {
            const set = listeners.get(method) ?? new Set();
            set.add(cb);
            listeners.set(method, set);
            return () => set.delete(cb);
        },
    } as unknown as ReachyMiniInstance;
    const notify = (method: string, params: Params) => {
        for (const cb of listeners.get(method) ?? []) cb(params);
    };
    return { reachy, calls, notify };
}

const methods = (calls: Array<[string, Params]>) => calls.map(([m]) => m);

describe('startRobotApp', () => {
    it('uploads in chunks, then starts, then reads the state', async () => {
        const { reachy, calls } = fakeRobot();
        const code = 'x'.repeat(100 * 1024);
        const app = await startRobotApp(reachy, { name: 'demo', code, allowHosts: ['api.x.com'] });

        const puts = calls.filter(([m]) => m === 'apps.put_bundle');
        expect(puts).toHaveLength(3);
        expect(puts.map(([, p]) => (p.data as string).length).reduce((a, b) => a + b)).toBe(code.length);
        expect(puts.every(([, p]) => p.sha256 === app.sha256 && p.count === 3)).toBe(true);
        expect(app.sha256).toMatch(/^[0-9a-f]{64}$/);

        const start = calls.find(([m]) => m === 'apps.start_bundle')![1];
        expect(start).toEqual({ name: 'demo', sha256: app.sha256, allow_hosts: ['api.x.com'] });
        expect(app.state).toEqual({ n: 1 });
    });

    it('skips the upload when the robot has the bundle', async () => {
        const { reachy, calls } = fakeRobot({ present: true });
        await startRobotApp(reachy, { name: 'demo', code: 'export default {}' });
        expect(methods(calls)).not.toContain('apps.put_bundle');
        expect(methods(calls)).toContain('apps.start_bundle');
    });

    it('reattaches to a running module instead of restarting it', async () => {
        const { reachy, calls } = fakeRobot({ running: 'demo' });
        await startRobotApp(reachy, { name: 'demo', code: 'export default {}', reuseRunning: true });
        expect(methods(calls)).toEqual(['apps.status', 'app.$snapshot']);
    });

    it('clears a crashed app before starting', async () => {
        const { reachy, calls } = fakeRobot({ present: true, status: 'error', running: 'x' });
        await startRobotApp(reachy, { name: 'demo', code: 'export default {}' });
        const m = methods(calls);
        expect(m.indexOf('apps.stop')).toBeLessThan(m.indexOf('apps.start_bundle'));
    });

    it('stops another running app before starting', async () => {
        const { reachy, calls } = fakeRobot({ running: 'other', present: true });
        await startRobotApp(reachy, { name: 'demo', code: 'export default {}' });
        const m = methods(calls);
        expect(m.indexOf('apps.stop')).toBeLessThan(m.indexOf('apps.start_bundle'));
    });
});

describe('attachRobotApp', () => {
    it('maps calls, events, logs and ordered state', async () => {
        const { reachy, calls, notify } = fakeRobot();
        const app = await attachRobotApp(reachy, 'demo');

        await expect(app.call('wave', { times: 2 })).resolves.toEqual({ waved: 2 });
        expect(calls.at(-1)).toEqual(['app.wave', { times: 2 }]);

        const events: unknown[] = [];
        const off = app.on('pressed', (d) => events.push(d));
        notify('app.$event', { name: 'pressed', data: { side: 'left' } });
        notify('app.$event', { name: 'other', data: 1 });
        off();
        notify('app.$event', { name: 'pressed', data: { side: 'right' } });
        expect(events).toEqual([{ side: 'left' }]);

        const logs: string[] = [];
        app.onLog((l) => logs.push(l));
        notify('app.$log', { line: 'hello', t: 0 });
        expect(logs).toEqual(['hello']);

        const states: unknown[] = [];
        app.onState((s) => states.push(s));
        notify('app.$state', { state: { n: 5 }, rev: 5 });
        notify('app.$state', { state: { n: 4 }, rev: 4 });
        expect(states).toEqual([{ n: 5 }]);
        expect(app.state).toEqual({ n: 5 });
    });
});

describe('types', () => {
    it('flow from the robot module definition to the client calls', async () => {
        const robotModule = defineRobotApp({
            state: { n: 0 },
            methods: {
                wave: ({ times }: { times: number }) => ({ waved: times }),
            },
        });
        const { reachy } = fakeRobot();
        const app = await attachRobotApp<typeof robotModule>(reachy, 'demo');
        const result: { waved: number } = await app.call('wave', { times: 2 });
        expect(result).toEqual({ waved: 2 });
        const n: number = app.state.n;
        expect(n).toBe(1);
        // @ts-expect-error - unknown method
        void app.call('nope', {});
        // @ts-expect-error - wrong params
        void app.call('wave', { times: 'two' });
    });
});
