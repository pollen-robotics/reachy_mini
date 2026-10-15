/**
 * Client side of a JS app's robot module.
 *
 * `startRobotApp()` uploads the module (only if the robot doesn't already
 * have that exact bundle: they are stored by SHA-256), starts it, and returns
 * a handle to call its methods, read its state and listen to its events. All
 * of it rides the session's DataChannel as JSON-RPC (`apps.*` to the daemon,
 * `app.*` relayed to the module).
 */

import type { ReachyMiniInstance } from './types.js';

/** Data-channel-friendly upload chunk (characters of JS source). */
const BUNDLE_CHUNK_CHARS = 48 * 1024;

export interface StartRobotAppOptions {
    /** App name shown by the robot (`apps.status`). */
    name: string;
    /** The bundled robot module source... */
    code?: string;
    /** ...or where to fetch it from (e.g. `new URL('./robot.js', location.href)`). */
    url?: string | URL;
    /** Internet hosts the module may reach (`api.example.com`, `host:443`). */
    allowHosts?: string[];
    /** Reattach if the same bundle is already running instead of restarting it. */
    reuseRunning?: boolean;
}

export interface RobotAppSnapshot<S> {
    app: string;
    protocol: number;
    methods: string[];
    state: S;
    rev: number;
}

// `any` so an untyped handle accepts any params; a module's own types (via
// `typeof` its default export) narrow them.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MethodMap = Record<string, (params: any, ctx: any) => unknown>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ParamsOf<F> = F extends (params: infer P, ctx: any) => unknown ? P : Record<string, unknown>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ResultOf<F> = F extends (...args: any[]) => infer R ? Awaited<R> : unknown;

/** Handle on a running robot module. `D` is `typeof` its default export. */
export interface RobotAppHandle<
    D extends { state?: object; methods?: MethodMap } = { state?: Record<string, unknown>; methods?: MethodMap },
> {
    readonly name: string;
    readonly sha256: string;
    /** Last state received from the robot. */
    readonly state: NonNullable<D['state']>;
    /** Call a robot method; resolves with what it returned. */
    call<K extends keyof NonNullable<D['methods']> & string>(
        method: K,
        params?: ParamsOf<NonNullable<D['methods']>[K]>,
        opts?: { timeoutMs?: number },
    ): Promise<ResultOf<NonNullable<D['methods']>[K]>>;
    /** Listen to `ctx.emit(event, data)`. Returns an unsubscribe function. */
    on(event: string, cb: (data: unknown) => void): () => void;
    /** Listen to state changes. Returns an unsubscribe function. */
    onState(cb: (state: NonNullable<D['state']>) => void): () => void;
    /** Listen to `ctx.log(...)` lines. Returns an unsubscribe function. */
    onLog(cb: (line: string) => void): () => void;
    /** Re-read the state (after a reconnect, say). */
    refresh(): Promise<RobotAppSnapshot<NonNullable<D['state']>>>;
    /** Stop the module on the robot. */
    stop(): Promise<void>;
    /** Drop listeners without stopping the module. */
    detach(): void;
}

interface AppsStatus {
    state: string;
    info: { name: string } | null;
}

/** Upload (if needed) and start a robot module, then attach to it. */
export async function startRobotApp<
    D extends { state?: object; methods?: MethodMap } = { state?: Record<string, unknown>; methods?: MethodMap },
>(reachy: ReachyMiniInstance, opts: StartRobotAppOptions): Promise<RobotAppHandle<D>> {
    const code = opts.code ?? (await fetchText(opts.url));
    const sha256 = await sha256Hex(code);

    const status = await reachy.rpcCall<AppsStatus>('apps.status');
    const running = status.state === 'running' && status.info?.name === opts.name;
    if (!(running && opts.reuseRunning)) {
        const { present } = await reachy.rpcCall<{ present: boolean }>('apps.has_bundle', { sha256 });
        if (!present) {
            const count = Math.max(1, Math.ceil(code.length / BUNDLE_CHUNK_CHARS));
            for (let index = 0; index < count; index++) {
                const data = code.slice(index * BUNDLE_CHUNK_CHARS, (index + 1) * BUNDLE_CHUNK_CHARS);
                await reachy.rpcCall('apps.put_bundle', { sha256, index, count, data });
            }
        }
        // Any app holding the slot, including one that crashed ('error'),
        // must be cleared first.
        if (status.state !== 'idle') await reachy.rpcCall('apps.stop', {}, { timeoutMs: 30000 });
        await reachy.rpcCall(
            'apps.start_bundle',
            { name: opts.name, sha256, allow_hosts: opts.allowHosts ?? [] },
            { timeoutMs: 30000 },
        );
    }
    return attachRobotApp<D>(reachy, opts.name, sha256);
}

/** Attach to an already running robot module (e.g. after reconnecting). */
export async function attachRobotApp<
    D extends { state?: object; methods?: MethodMap } = { state?: Record<string, unknown>; methods?: MethodMap },
>(reachy: ReachyMiniInstance, name: string, sha256 = ''): Promise<RobotAppHandle<D>> {
    type S = NonNullable<D['state']>;
    let state = {} as S;
    let rev = -1;
    const stateCbs = new Set<(s: S) => void>();
    const offs: Array<() => void> = [];

    const applyState = (next: S, nextRev: number) => {
        if (nextRev <= rev) return;
        rev = nextRev;
        state = next;
        for (const cb of stateCbs) cb(state);
    };
    offs.push(
        reachy.onNotification('app.$state', (p) => applyState(p.state as S, p.rev as number)),
    );

    const refresh = async () => {
        const snap = await reachy.rpcCall<RobotAppSnapshot<S>>('app.$snapshot');
        applyState(snap.state, snap.rev);
        return snap;
    };
    await refresh();

    return {
        name,
        sha256,
        get state() {
            return state;
        },
        call: (method, params, callOpts) =>
            reachy.rpcCall(`app.${method}`, (params ?? {}) as Record<string, unknown>, callOpts ?? {}),
        on: (event, cb) =>
            reachy.onNotification('app.$event', (p) => {
                if (p.name === event) cb(p.data);
            }),
        onState: (cb) => {
            stateCbs.add(cb);
            return () => stateCbs.delete(cb);
        },
        onLog: (cb) => reachy.onNotification('app.$log', (p) => cb(String(p.line))),
        refresh,
        stop: async () => {
            await reachy.rpcCall('apps.stop', {}, { timeoutMs: 30000 });
        },
        detach: () => {
            for (const off of offs) off();
            stateCbs.clear();
        },
    } as RobotAppHandle<D>;
}

async function fetchText(url: string | URL | undefined): Promise<string> {
    if (!url) throw new Error('startRobotApp: pass `code` or `url`');
    const res = await fetch(url);
    if (!res.ok) throw new Error(`cannot fetch robot module ${String(url)}: HTTP ${res.status}`);
    return res.text();
}

async function sha256Hex(text: string): Promise<string> {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
    return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}
