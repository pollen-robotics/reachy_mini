/**
 * Embed runtime orchestration tests (`connectToHost` boot pipeline +
 * graceful leave).
 *
 * This module is the embed side of the lifecycle contract
 * (docs/source/SDK/lifecycle-contract.md): it owns the ORDER of the
 * boot steps, the "wake is awaited before the app is revealed"
 * guarantee, and the sleep → disable → daemon-echo → `embed:left`
 * leave sequence the host relies on to unmount without racing the
 * daemon's idle-reset. None of that ordering was pinned by a test
 * before this file - a reorder (e.g. acking `embed:left` before the
 * motors-off echo) would have shipped silently.
 *
 * Harness notes
 * ─────────────
 *  - The module keeps its idempotency latch (`bootPromise`) and the
 *    version latches at module level, so every test re-imports a fresh
 *    copy via `vi.resetModules()` + dynamic `import()`.
 *  - jsdom's `window.parent === window`, which would make
 *    `awaitHostInit` take its no-iframe shortcut. We install a fake
 *    parent object instead: it both restores the real iframe code path
 *    and captures every outbound `postMessage` for assertions.
 *  - `window.ReachyMini` is a fake SDK (EventTarget + the methods the
 *    boot touches) with per-test gates on `ensureAwake` / `gotoSleep` /
 *    `getVersion`, so tests can hold the pipeline at a precise step.
 *  - A single `timeline` array records SDK calls and outbound protocol
 *    messages in arrival order - the assertions are mostly "A happened
 *    before B", which is exactly what this module exists to guarantee.
 *  - Fake timers throughout: the boot and leave paths are budget-driven
 *    (2 s host:init, 2.5 s version, 6.5 s sleep hard cap, 1 s echo
 *    wait), and the degraded-daemon tests advance through those budgets
 *    deterministically.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  PROTOCOL_SOURCE,
  PROTOCOL_VERSION,
  encodeCredsToHash,
  type CredsBundle,
} from '../lib/protocol';
import type { ConnectedHandle } from './index';

/** Central REST is the only network dependency of the boot pipeline
 *  (peer-id re-resolution). Mocked so tests control the robot list. */
const { fetchRobotsFromCentralMock } = vi.hoisted(() => ({
  fetchRobotsFromCentralMock: vi.fn(),
}));
vi.mock('../lib/centralRest', () => ({
  fetchRobotsFromCentral: fetchRobotsFromCentralMock,
}));

/* ─────────────────── Timeline + deferred helpers ─────────────────── */

let timeline: string[] = [];
const mark = (tag: string): void => {
  timeline.push(tag);
};
const indexOf = (tag: string): number => timeline.indexOf(tag);
const countOf = (tag: string): number =>
  timeline.filter((t) => t === tag).length;

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (err: unknown) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (err: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Drain the microtask queue so the boot pipeline can chain through
 *  its (non-timer) awaits. Plain `Promise.resolve()` hops are enough:
 *  everything timer-based is advanced explicitly per test. */
async function flush(rounds = 25): Promise<void> {
  for (let i = 0; i < rounds; i++) await Promise.resolve();
}

/* ─────────────────── Fake SDK ─────────────────── */

class FakeReachyMini extends EventTarget {
  static instances: FakeReachyMini[] = [];
  static get last(): FakeReachyMini {
    const sdk = FakeReachyMini.instances.at(-1);
    if (!sdk) throw new Error('no FakeReachyMini constructed yet');
    return sdk;
  }

  robotState: { motor_mode: string } = { motor_mode: 'enabled' };
  requestStateCount = 0;

  /** When set, `ensureAwake` / `gotoSleep` / `getVersion` block on the
   *  gate instead of resolving immediately - lets a test freeze the
   *  pipeline at that exact step. */
  wakeGate: Deferred<boolean> | null = null;
  sleepGate: Deferred<void> | null = null;
  versionGate: Deferred<string> | null = null;

  constructor(_options: unknown) {
    super();
    FakeReachyMini.instances.push(this);
    mark('sdk:new');
  }

  authenticate = async (): Promise<boolean> => {
    mark('sdk:authenticate');
    return true;
  };

  connect = async (): Promise<void> => {
    mark('sdk:connect');
  };

  startSession = async (peerId: string): Promise<void> => {
    mark(`sdk:startSession:${peerId}`);
  };

  getVersion = (): Promise<string> => {
    mark('sdk:getVersion');
    return this.versionGate ? this.versionGate.promise : Promise.resolve('9.9.9');
  };

  ensureAwake = (): Promise<boolean> => {
    mark('sdk:ensureAwake');
    return this.wakeGate ? this.wakeGate.promise : Promise.resolve(true);
  };

  isAwake = (): boolean => this.robotState.motor_mode === 'enabled';

  gotoSleep = (_opts?: { timeoutMs?: number }): Promise<void> => {
    mark('sdk:gotoSleep');
    return this.sleepGate ? this.sleepGate.promise : Promise.resolve();
  };

  setMotorMode = (mode: string): boolean => {
    mark(`sdk:setMotorMode:${mode}`);
    return true;
  };

  requestState = (): void => {
    this.requestStateCount += 1;
  };

  stopSession = (): void => {
    mark('sdk:stopSession');
  };
}

/* ─────────────────── Fake parent (outbound capture) ─────────────────── */

type PostedMsg = { type?: string } & Record<string, unknown>;

let posted: PostedMsg[] = [];

function installFakeParent(): void {
  const fakeParent = {
    postMessage: (msg: unknown): void => {
      const m = msg as PostedMsg;
      posted.push(m);
      // Keep the timeline readable: debug frames are high-volume and
      // ordering-irrelevant here.
      if (m.type === 'embed:debug') return;
      if (m.type === 'embed:app-state') {
        const step = (m as { connectingStep?: string | null }).connectingStep;
        mark(`post:app-state:${(m as { phase?: string }).phase}${step ? `:${step}` : ''}`);
        return;
      }
      mark(`post:${m.type}`);
    },
  };
  Object.defineProperty(window, 'parent', {
    configurable: true,
    get: () => fakeParent,
  });
}

/* ─────────────────── Boot fixtures ─────────────────── */

const CREDS: CredsBundle = {
  hfToken: 'hf_test_token',
  userName: 'tfrere',
  robotPeerId: 'peer-handed',
  robotHardwareId: null,
  signalingUrl: 'wss://central.example/sig',
  theme: 'light',
  config: null,
  hostName: 'Creds Host',
  appName: 'Test App',
};

function hostInitMsg(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    source: PROTOCOL_SOURCE,
    type: 'host:init',
    version: PROTOCOL_VERSION,
    theme: 'light',
    config: null,
    appName: 'Test App',
    hostName: 'Init Host',
    userName: 'tfrere',
    robotPeerId: 'peer-live',
    robotHardwareId: null,
    ...overrides,
  };
}

function deliverToEmbed(data: Record<string, unknown>): void {
  window.dispatchEvent(
    new MessageEvent('message', { data, origin: 'https://host.example' }),
  );
}

function hostLeavingMsg(): Record<string, unknown> {
  return {
    source: PROTOCOL_SOURCE,
    type: 'host:leaving',
    version: PROTOCOL_VERSION,
    reason: 'user-action',
  };
}

type EmbedModule = typeof import('./index');

/**
 * Kick a boot and drive it to the point where `host:init` is due.
 * `configureSdk` runs after the fake SDK exists but BEFORE the boot
 * proceeds past `host:init` - the hook for arming gates.
 */
async function startBoot(opts: {
  configureSdk?: (sdk: FakeReachyMini) => void;
  hostInit?: boolean;
  hostInitOverrides?: Record<string, unknown>;
} = {}): Promise<{
  embed: EmbedModule;
  bootP: Promise<ConnectedHandle<unknown>>;
  sdk: FakeReachyMini;
}> {
  window.location.hash = `#${encodeCredsToHash(CREDS)}`;
  const embed = (await import('./index')) as EmbedModule;
  const bootP = embed.connectToHost();
  // Steps 1-5 (creds, SDK construction, embed:ready) complete on the
  // microtask queue; the boot is now parked in `awaitHostInit`.
  await flush();
  const sdk = FakeReachyMini.last;
  opts.configureSdk?.(sdk);
  if (opts.hostInit === false) {
    // Let the 2 s host:init soft deadline expire → creds fallback.
    await vi.advanceTimersByTimeAsync(2_000);
  } else {
    deliverToEmbed(hostInitMsg(opts.hostInitOverrides));
  }
  await flush();
  return { embed, bootP, sdk };
}

/** Full happy-path boot, for the leave tests. */
async function bootToLive(): Promise<{
  handle: ConnectedHandle<unknown>;
  sdk: FakeReachyMini;
}> {
  const { bootP, sdk } = await startBoot();
  const handle = await bootP;
  return { handle, sdk };
}

const appStates = (): string[] =>
  timeline.filter((t) => t.startsWith('post:app-state:'));

/* ─────────────────── Setup / teardown ─────────────────── */

/**
 * `vi.resetModules()` hands each test a fresh embed module, but jsdom's
 * `window` is shared across the file - and the PREVIOUS test's bridge
 * left its `message` / `pagehide` listeners armed on it. A dispatched
 * `host:leaving` would then also wake the stale bridges, whose fake
 * SDKs happily write into the shared timeline. Track every listener the
 * code under test registers and strip them between tests.
 */
let trackedListeners: Array<{
  type: string;
  listener: EventListenerOrEventListenerObject;
}> = [];
const realAddEventListener = window.addEventListener.bind(window);

beforeEach(() => {
  vi.useFakeTimers();
  vi.resetModules();
  timeline = [];
  posted = [];
  FakeReachyMini.instances = [];
  trackedListeners = [];
  sessionStorage.clear();
  installFakeParent();
  vi.spyOn(window, 'addEventListener').mockImplementation(
    (type: string, listener: EventListenerOrEventListenerObject, options?: unknown) => {
      trackedListeners.push({ type, listener });
      realAddEventListener(
        type,
        listener,
        options as AddEventListenerOptions | boolean | undefined,
      );
    },
  );
  (window as unknown as { ReachyMini: unknown }).ReachyMini = FakeReachyMini;
  // Default central answer: reachable but empty. Only the peer-id
  // re-resolution tests care; everything else never calls it (their
  // creds carry no `robotHardwareId`).
  fetchRobotsFromCentralMock.mockReset();
  fetchRobotsFromCentralMock.mockResolvedValue({ ok: true, robots: [] });
  // The boot fires a fire-and-forget npm-registry staleness check;
  // it must never hit the network from a unit test.
  vi.stubGlobal('fetch', vi.fn(async () => {
    throw new Error('network disabled in tests');
  }));
});

afterEach(() => {
  for (const { type, listener } of trackedListeners) {
    window.removeEventListener(type, listener);
  }
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/* ─────────────────── Boot pipeline ─────────────────── */

describe('connectToHost boot pipeline', () => {
  it('runs the contract order and only announces live after the wake settles', async () => {
    const wakeGate = deferred<boolean>();
    const { bootP } = await startBoot({
      configureSdk: (sdk) => {
        sdk.wakeGate = wakeGate;
      },
    });

    // Parked on the wake: every earlier step already happened, in order.
    expect(appStates()).toEqual([
      'post:app-state:connecting:link',
      'post:app-state:connecting:session',
      'post:app-state:connecting:wake',
    ]);
    expect(indexOf('sdk:authenticate')).toBeLessThan(indexOf('sdk:connect'));
    expect(indexOf('sdk:connect')).toBeLessThan(
      indexOf('sdk:startSession:peer-live'),
    );
    // Version read sits between session-up and wake, so a gating host
    // can decide while the connecting splash is still on screen.
    expect(indexOf('sdk:getVersion')).toBeGreaterThan(
      indexOf('sdk:startSession:peer-live'),
    );
    expect(indexOf('sdk:getVersion')).toBeLessThan(indexOf('sdk:ensureAwake'));
    // The wake is awaited, not fire-and-forget: no `live` yet.
    expect(timeline).not.toContain('post:app-state:live');

    wakeGate.resolve(true);
    await flush();
    const handle = await bootP;

    expect(timeline).toContain('post:app-state:live');
    expect(handle.reachy).toBe(FakeReachyMini.last);
    // `embed:ready` is the very first protocol frame out the door.
    expect(posted[0]?.type).toBe('embed:ready');
    // The daemon version resolved during bring-up rides the live frame.
    const liveFrame = posted.find(
      (m) => m.type === 'embed:app-state' && m.phase === 'live',
    );
    expect(liveFrame?.daemonVersion).toBe('9.9.9');
    // Boot consumed the creds hash and wiped it from the URL.
    expect(window.location.hash).toBe('');
  });

  it('is idempotent across double invocation (Strict Mode): one SDK, one embed:ready', async () => {
    const { embed, bootP } = await startBoot();
    const second = embed.connectToHost();
    const [h1, h2] = await Promise.all([bootP, second]);

    expect(h1).toBe(h2);
    expect(FakeReachyMini.instances).toHaveLength(1);
    expect(countOf('post:embed:ready')).toBe(1);
  });

  it('falls back to hash creds when host:init never arrives', async () => {
    const { bootP } = await startBoot({ hostInit: false });
    const handle = await bootP;

    // Creds-derived live state (host:init would have said "Init Host")
    // and the handed-in peer id, since no re-resolution source exists.
    expect(handle.hostName).toBe('Creds Host');
    expect(timeline).toContain('sdk:startSession:peer-handed');
    expect(timeline).toContain('post:app-state:live');
  });

  it('a daemon that never answers get_version costs its 2.5 s budget, not the boot', async () => {
    const { bootP } = await startBoot({
      configureSdk: (sdk) => {
        sdk.versionGate = deferred<string>();
      },
    });

    // Blocked on the version read: the wake must not have started yet
    // (the read runs before it) and `live` is out of the question.
    expect(timeline).not.toContain('sdk:ensureAwake');
    expect(timeline).not.toContain('post:app-state:live');

    await vi.advanceTimersByTimeAsync(2_500);
    await flush();
    await bootP;

    expect(timeline).toContain('post:app-state:live');
    const liveFrame = posted.find(
      (m) => m.type === 'embed:app-state' && m.phase === 'live',
    );
    expect(liveFrame?.daemonVersion).toBeNull();
  });
});

/* ─────────────────── Peer-id re-resolution ─────────────────── */

/**
 * The handed-in `robotPeerId` is a snapshot the host captured at picker
 * time; central rotates it on every relay reconnect, so after a Space
 * cold-start it is frequently dead. When the host also hands a stable
 * `robotHardwareId`, the embed re-resolves the CURRENT peer id from
 * central right before dialing. The contract under test is twofold:
 * the happy path actually swaps the id, and every failure mode is
 * FAIL-OPEN (dial the handed-in id rather than not dialing at all - a
 * refactor that lets a central hiccup throw would break every boot).
 */
describe('peer-id re-resolution before startSession', () => {
  const withHardwareId = { robotHardwareId: 'hw-1' };

  it('dials the fresh peer id when central knows the hardware id', async () => {
    fetchRobotsFromCentralMock.mockResolvedValue({
      ok: true,
      robots: [
        { id: 'peer-other', hardwareId: 'hw-2' },
        { id: 'peer-fresh', hardwareId: 'hw-1' },
      ],
    });
    const { bootP } = await startBoot({ hostInitOverrides: withHardwareId });
    await bootP;

    expect(timeline).toContain('sdk:startSession:peer-fresh');
    expect(timeline).not.toContain('sdk:startSession:peer-live');
    expect(fetchRobotsFromCentralMock).toHaveBeenCalledWith({
      signalingUrl: CREDS.signalingUrl,
      hfToken: CREDS.hfToken,
    });
  });

  it('falls back to the handed-in id when the hardware id is unknown to central', async () => {
    fetchRobotsFromCentralMock.mockResolvedValue({
      ok: true,
      robots: [{ id: 'peer-other', hardwareId: 'hw-2' }],
    });
    const { bootP } = await startBoot({ hostInitOverrides: withHardwareId });
    await bootP;

    expect(timeline).toContain('sdk:startSession:peer-live');
  });

  it('falls back to the handed-in id when central answers not-ok', async () => {
    fetchRobotsFromCentralMock.mockResolvedValue({
      ok: false,
      robots: [],
      reason: 'HTTP 503',
    });
    const { bootP } = await startBoot({ hostInitOverrides: withHardwareId });
    await bootP;

    expect(timeline).toContain('sdk:startSession:peer-live');
  });

  it('a central that throws must not break the boot (fail-open)', async () => {
    fetchRobotsFromCentralMock.mockRejectedValue(new Error('network down'));
    const { bootP } = await startBoot({ hostInitOverrides: withHardwareId });
    await bootP;

    expect(timeline).toContain('sdk:startSession:peer-live');
    expect(timeline).toContain('post:app-state:live');
  });

  it('skips the central roundtrip entirely when no hardware id was handed in', async () => {
    const { bootP } = await startBoot(); // default: robotHardwareId null
    await bootP;

    expect(fetchRobotsFromCentralMock).not.toHaveBeenCalled();
    expect(timeline).toContain('sdk:startSession:peer-live');
  });
});

/* ─────────────────── Graceful leave ─────────────────── */

describe('graceful leave (host:leaving)', () => {
  it('acks embed:left only after sleep, disable, and the daemon echoing motors off', async () => {
    const { handle, sdk } = await bootToLive();
    const onLeave = vi.fn();
    handle.onLeave(onLeave);

    const sleepGate = deferred<void>();
    sdk.sleepGate = sleepGate;
    deliverToEmbed(hostLeavingMsg());
    await flush();

    // App cleanup fired immediately; the sleep is in flight and the
    // disable must NOT preempt it (disabling mid-trajectory drops the
    // robot wherever it is).
    expect(onLeave).toHaveBeenCalledTimes(1);
    expect(timeline).toContain('sdk:gotoSleep');
    expect(timeline).not.toContain('sdk:setMotorMode:disabled');

    sleepGate.resolve();
    await flush();

    // Sleep done → disable sent. But `setMotorMode` is fire-and-forget:
    // the ack must wait for the daemon to echo the new mode, otherwise
    // the host unmounts the iframe while the command may still sit in
    // the SCTP send queue.
    expect(timeline).toContain('sdk:setMotorMode:disabled');
    expect(timeline).not.toContain('post:embed:left');
    // The echo waiter actively nudges state polls rather than trusting
    // the SDK's own (possibly stood-down) poller.
    expect(sdk.requestStateCount).toBeGreaterThan(0);

    sdk.robotState.motor_mode = 'disabled';
    sdk.dispatchEvent(new Event('state'));
    await flush();

    expect(timeline).toContain('post:embed:left');
    expect(indexOf('sdk:gotoSleep')).toBeLessThan(
      indexOf('sdk:setMotorMode:disabled'),
    );
    expect(indexOf('sdk:setMotorMode:disabled')).toBeLessThan(
      indexOf('post:embed:left'),
    );
  });

  it('a wedged goto_sleep cannot stall the leave: hard cap, then disable, then bounded echo wait', async () => {
    const { sdk } = await bootToLive();
    sdk.sleepGate = deferred<void>(); // never resolves

    deliverToEmbed(hostLeavingMsg());
    await flush();
    expect(timeline).toContain('sdk:gotoSleep');
    expect(timeline).not.toContain('sdk:setMotorMode:disabled');

    // Hard cap (6.5 s) fires → we fall through to the disable anyway.
    await vi.advanceTimersByTimeAsync(6_500);
    expect(timeline).toContain('sdk:setMotorMode:disabled');
    expect(timeline).not.toContain('post:embed:left');

    // The daemon never echoes motors-off either: the 1 s confirm budget
    // expires and the leave still acks - it must never hang, the
    // daemon's idle-reset is the backstop for the robot itself.
    await vi.advanceTimersByTimeAsync(1_000);
    await flush();
    expect(timeline).toContain('post:embed:left');
  });

  it('is idempotent: a duplicate host:leaving replays neither sleep nor cleanup', async () => {
    const { handle, sdk } = await bootToLive();
    const onLeave = vi.fn();
    handle.onLeave(onLeave);

    deliverToEmbed(hostLeavingMsg());
    deliverToEmbed(hostLeavingMsg());
    sdk.robotState.motor_mode = 'disabled';
    sdk.dispatchEvent(new Event('state'));
    await flush();

    expect(countOf('sdk:gotoSleep')).toBe(1);
    expect(countOf('post:embed:left')).toBe(1);
    expect(onLeave).toHaveBeenCalledTimes(1);
  });

  it('pagehide runs app cleanup + stopSession but NO sleep trajectory (idle-reset owns that)', async () => {
    const { handle } = await bootToLive();
    const onLeave = vi.fn();
    handle.onLeave(onLeave);

    window.dispatchEvent(new Event('pagehide'));
    await flush();

    // A dying tab has no time to play a 2 s trajectory: release the
    // session and let the daemon's idle-reset park the robot.
    expect(onLeave).toHaveBeenCalledTimes(1);
    expect(timeline).toContain('sdk:stopSession');
    expect(timeline).not.toContain('sdk:gotoSleep');
    expect(timeline).not.toContain('post:embed:left');
  });
});
