/**
 * Embedded-app client.
 *
 * Vanilla TypeScript helper that lives in the iframe side of the
 * host / app split. Consumed by `src/embed.{ts,tsx}` in each app
 * (or via the CDN entry
 * `@pollen-robotics/reachy-mini-sdk/host/embed` script tag).
 *
 *   import { connectToHost } from '@pollen-robotics/reachy-mini-sdk/host/embed';
 *
 *   const handle = await connectToHost<MyAppConfig>();
 *   handle.onLeave(() => { /* clean up before unmount *\/ });
 *   handle.reachy.setHeadRpyDeg(0, 10, 0);
 *
 * Boot sequence (canonical reference: APP_CREATION_GUIDE
 * §13.4 handoff sequence + §13.6 protocol v1):
 *  1. Read `#creds=<base64>` synchronously and wipe the hash
 *     with `history.replaceState`.
 *  2. Wait for `window.ReachyMini` (8 s timeout).
 *  3. Instantiate the SDK, seed the HF token into
 *     `sessionStorage`.
 *  4. Send `embed:ready` to the parent.
 *  5. Wait for `host:init` (2 s soft timeout; on hit we proceed
 *     from the hash creds alone via `liveStateFromCreds`, which
 *     carries the same fields as `liveStateFromInit`).
 *  6. `connect()` → `startSession()` → wake (awaiting the full wake-up
 *     trajectory, not fire-and-forget), emitting `embed:app-state` at each
 *     step.
 *  7. Resolve `connectToHost()` with the live SDK handle.
 *
 * Strict Mode safety (APP_CREATION_GUIDE §13.5.4): the function is idempotent
 * across multiple awaits via a module-level promise. Calling
 * `connectToHost()` twice returns the same in-flight promise;
 * a single SDK instance is created, a single `embed:ready` is
 * posted.
 */
import type {
  ReachyMiniInstance,
  ReachyMiniOptions,
} from '../lib/sdk-types';
import {
  PROTOCOL_SOURCE,
  PROTOCOL_VERSION,
  decodeCredsFromHash,
  isProtocolMessage,
} from '../lib/protocol';
import { fetchRobotsFromCentral } from '../lib/centralRest';
import {
  fetchLatestDaemonVersion,
  isDaemonOutdated,
  parseSemver,
} from '../lib/daemonRelease';
import type {
  AppConnectingStep,
  AppPhase,
  ConfigPayload,
  CredsBundle,
  EmbedToHostMsg,
  EmbedUpdateProgressMsg,
  HostInitMsg,
  HostToEmbedMsg,
  ThemeMode,
} from '../lib/protocol';

const SDK_READY_TIMEOUT_MS = 8000;
// Soft deadline for the parent's `host:init` reply. The embed can
// boot from `#creds=` alone (see `liveStateFromCreds`), so this is
// purely the upper bound on "wait a touch in case `host:init` is in
// flight". Was 8000 ms when the cross-origin filter was broken (every
// message got dropped, every Space sat the full 8 s before falling
// back). With the origin filter fixed the host:init typically lands
// in <100 ms; 2 s is comfortable defensive slack.
const HOST_INIT_TIMEOUT_MS = 2000;
const TOKEN_TTL_MS = 15 * 60 * 1000;
// Upper bound on how long we wait for the wake-up trajectory to finish
// before revealing the app anyway. Matches the SDK's own `wakeUp` default
// (and the ConnectingView "slow hint" at ~6 s), so a slow or older daemon
// degrades into "land the user" rather than trapping them on the splash.
const WAKE_TRAJECTORY_TIMEOUT_MS = 8000;
// Budget for the `get_version` round-trip we run between session-up and
// wake. Short on purpose: the answer only feeds the host's update gate,
// so an unresponsive or too-old daemon must cost the boot a blink, not a
// stall. Mirrors the mobile app's own bring-up version read.
const DAEMON_VERSION_TIMEOUT_MS = 2500;
// Sleep-on-leave budget. The host layer owns wake/sleep, so on an explicit
// graceful leave the embed plays goto_sleep itself (immediate) then forces
// motors disabled - mirroring the mobile app (`sleepAndDisableRobot`). We wait
// for it to finish BEFORE letting the host unmount, so the motors are already
// `Disabled` when the app-slot lock frees: the daemon's idle-reset then skips
// (no double trajectory / go_sleep sound), with no daemon-side change needed.
// SDK-level timeout on the goto_sleep move + a slightly larger JS hard cap so a
// wedged daemon can't stall the leave forever.
const LEAVE_SLEEP_TIMEOUT_MS = 6000;
const LEAVE_SLEEP_HARD_TIMEOUT_MS = 6500;
// How long we wait for the daemon to echo the motors-off state back before
// acking the leave. `setMotorMode` is fire-and-forget: it queues the command
// on the data channel and returns, so acking on the next line lets the host
// unmount this iframe - tearing the peer connection down - while the command
// may still be sitting in the SCTP send queue. Losing it is precisely what
// the sequence below relies on NOT happening, since the daemon would then
// still see `Enabled` when the app-slot lock frees and replay its own
// goto_sleep. Small enough that the whole leave stays under the host's own
// `LEAVING_ACK_CAP_MS` even stacked on the sleep hard cap above.
//
// Sized against the slowest confirmation path rather than the typical one.
// With a pose stream running the state lands within a frame (~33 ms), and
// with no stream a poll reply takes about one RTT. But an app that releases
// its pose subscription from `onLeave` leaves the SDK's `POSE_STREAM_FRESH_MS`
// window (750 ms) still armed, and poll replies are ignored for its remainder,
// so the confirmation can legitimately take a beat under a second.
const LEAVE_DISABLE_CONFIRM_TIMEOUT_MS = 1000;

/**
 * Land the user in a *settled* robot.
 *
 * Unlike `sdk.ensureAwake()` - which kicks `wakeUp()` fire-and-forget and
 * resolves the instant the command is *sent* - this awaits the wake-up
 * trajectory to actual completion (the daemon's `completed: true` ack), so
 * the host only reveals the app once the robot has finished its wake
 * animation instead of mid-move.
 *
 * Ordering:
 *   1. If the motor mode is still unknown (fresh session), ask for a state
 *      snapshot and wait briefly for it so `isAwake()` is meaningful.
 *   2. Already awake -> return immediately (no trajectory to play).
 *   3. Asleep -> await the full `wakeUp()` trajectory. A timeout or error is
 *      swallowed and we reveal the app anyway (degraded but reachable),
 *      never trapping the user on the connecting splash.
 */
async function awaitFullyAwake(sdk: ReachyMiniInstance): Promise<void> {
  if (sdk.robotState.motor_mode === undefined) {
    await new Promise<void>((resolve) => {
      const done = (): void => {
        sdk.removeEventListener('state', done);
        clearTimeout(timer);
        resolve();
      };
      const timer = setTimeout(done, 1000);
      sdk.addEventListener('state', done);
      sdk.requestState();
    });
  }
  if (sdk.isAwake()) return;
  try {
    await sdk.wakeUp({ timeoutMs: WAKE_TRAJECTORY_TIMEOUT_MS });
  } catch {
    /* degraded: reveal the app anyway rather than trap the user on splash */
  }
}

/**
 * Wait until the daemon echoes back `motor_mode: 'disabled'`.
 *
 * Used to close a graceful leave: a state frame carrying the new mode is
 * proof that the fire-and-forget `set_motor_mode` actually reached the
 * daemon, rather than having been dropped with the peer connection when the
 * host unmounted us (see `LEAVE_DISABLE_CONFIRM_TIMEOUT_MS`).
 *
 * We nudge `requestState()` on a short interval rather than trusting the
 * SDK's own 500 ms poll, which stands down while a pose stream is feeding
 * state and would otherwise leave us waiting on the next pushed frame.
 * A timeout resolves anyway: the leave must never hang on this, and the
 * daemon's idle-reset remains the backstop.
 */
async function awaitMotorsDisabled(sdk: ReachyMiniInstance): Promise<void> {
  if (sdk.robotState.motor_mode === 'disabled') return;
  await new Promise<void>((resolve) => {
    let settled = false;
    let pollId = 0;
    let timerId = 0;
    const finish = (): void => {
      if (settled) return;
      settled = true;
      sdk.removeEventListener('state', onState);
      window.clearInterval(pollId);
      window.clearTimeout(timerId);
      resolve();
    };
    // Hoisted so `finish` can unregister it.
    function onState(): void {
      if (sdk.robotState.motor_mode === 'disabled') finish();
    }
    sdk.addEventListener('state', onState);
    timerId = window.setTimeout(finish, LEAVE_DISABLE_CONFIRM_TIMEOUT_MS);
    pollId = window.setInterval(() => sdk.requestState(), 120);
    sdk.requestState();
  });
}

/**
 * Stable surface for the robot's WebRTC media. All accessors are
 * synchronous and safe to read at any point after the handle is
 * returned by `connectToHost()`. References are cached: calling
 * `media.robotStream` repeatedly returns the same `MediaStream`
 * instance (good for React effect deps), and the stream auto-clears
 * on `sessionStopped`.
 */
export interface RobotMedia {
  /**
   * Bind the robot's video element. Internally:
   *   1. Calls the SDK's own `attachVideo()` (which keeps the
   *      element's `muted` flag in sync with `audioMuted`, kicks
   *      off the latency monitor, and resets `srcObject` on
   *      `sessionStopped`).
   *   2. Replays the cached `robotStream` so a late-mounting
   *      `<video>` element catches up immediately.
   *
   * Returns a cleanup function that detaches the SDK listeners.
   */
  attachVideo(el: HTMLVideoElement): () => void;
  /**
   * Robot's outbound MediaStream (video + audio in a single
   * stream, as the daemon emits it). `null` until the WebRTC
   * tracks have arrived; cleared on `sessionStopped`.
   *
   * Hand it to `<video>.srcObject`, an `AudioContext`'s
   * `createMediaStreamSource`, or any other consumer.
   */
  readonly robotStream: MediaStream | null;
  /**
   * Local microphone MediaStream. Mirrors `reachy._micStream` so
   * apps don't reach into underscore-prefixed SDK internals.
   * `null` when `enableMicrophone` was `false` at construction
   * time, when the daemon refused bidirectional audio, or when
   * the session has stopped.
   */
  readonly micStream: MediaStream | null;
}

/** Resolved state at the moment `connectToHost()` returns. */
export interface ConnectedHandle<TConfig = unknown> {
  /** Live SDK instance: connected, session started, robot awake. */
  reachy: ReachyMiniInstance;
  /** Current theme; updated via `onThemeChange`. */
  theme: ThemeMode;
  /** Initial config (from URL `?config=` or mobile handoff).
   *  Updates pushed via `onConfigChange`. */
  config: TConfig | null;
  /** App display name as passed by the host. */
  appName: string;
  /** Host display name (e.g. "Reachy Mini"). */
  hostName: string;
  /** HF user name when known (from `host:init`). */
  userName: string | null;

  /**
   * Stable accessors for the WebRTC media streams negotiated
   * during `startSession()`.
   *
   * Why apps must use these (not `reachy.attachVideo` /
   * `reachy._pc` / `reachy._micStream`):
   *   `connectToHost()` fully completes the WebRTC handshake
   *   before the embedded app's React tree mounts. The SDK's
   *   one-shot `videoTrack` event and the underlying `pc.ontrack`
   *   event have therefore ALREADY fired by the time a
   *   freshly-mounted component subscribes - any listener
   *   registered after `connectToHost()` resolves will sit silent
   *   until the next `startSession()`, which embeds never
   *   trigger. This API replays the streams from a synchronous
   *   snapshot of the peer-connection's receivers, so
   *   late-mounting consumers see the camera + audio immediately.
   *
   * For the data channel, mute toggles, motor commands and state
   * updates there is no race: the bridge only resolves once ICE
   * AND the data channel are connected, and state events stream
   * continuously at 50 Hz from the daemon. Apps can keep using
   * `reachy.setHeadRpyDeg(…)`, `reachy.setMicMuted(…)`,
   * `reachy.addEventListener('state', …)` directly.
   */
  readonly media: RobotMedia;

  /** Register a teardown callback. Fires on `host:leaving`
   *  (one-shot) or `pagehide`. Return a promise to keep the host
   *  waiting (bounded by the host's `timeoutMs`). Returns an
   *  unsubscribe function. */
  onLeave(cb: () => void | Promise<void>): () => void;
  /** Register a theme-change handler. */
  onThemeChange(cb: (theme: ThemeMode) => void): () => void;
  /** Register a config-change handler. */
  onConfigChange(cb: (config: TConfig | null) => void): () => void;

  /** Push an app-level state update upstream so the host can
   *  drive its ConnectingView overlay. */
  setAppState(state: {
    phase: AppPhase;
    connectingStep?: AppConnectingStep | null;
    message?: string | null;
  }): void;
  /** Ask the host to start the leave sequence. */
  requestLeave(): void;
  /** Report an error. `fatal: true` switches the host to ErrorView. */
  reportError(
    message: string,
    opts?: { fatal?: boolean; detail?: unknown },
  ): void;
}

export interface ConnectToHostOptions {
  /** Forwarded to the SDK constructor. `appName`, `signalingUrl`,
   *  `clientId` are auto-set from the creds bundle. */
  sdkOptions?: Partial<ReachyMiniOptions>;
  /** Origin of the host's window. Defaults to
   *  `window.location.origin` (same-origin iframe). */
  expectedOrigin?: string;
}

/* ─────────────────── Module-level idempotency ─────────────────── */

let bootPromise: Promise<ConnectedHandle<unknown>> | null = null;

/**
 * Target origin used by every outgoing `postMessage`. In Mode A
 * (host shell same-origin as the embed) this equals
 * `window.location.origin`. In Mode B (mobile WebView at a
 * different origin like `tauri.localhost`) we infer the parent's
 * origin from `document.referrer` at boot and fall back to `'*'`
 * if even that is empty.
 *
 * Same value drives the INBOUND filter (`expectedOrigin` in
 * `bootOnce`): we accept `host:init` and other host-to-embed
 * messages from this origin only. Previously the inbound filter
 * defaulted to `window.location.origin`, which silently dropped
 * every cross-origin message and stalled Mode B boots for the
 * full `HOST_INIT_TIMEOUT_MS`.
 *
 * Outgoing messages carry no secrets (the HF token lives in the
 * URL hash, never in postMessage payloads), so `'*'` is safe as a
 * last-resort target for diagnostics + lifecycle pings.
 */
let parentTargetOrigin: string = '*';

function detectParentOrigin(): string {
  try {
    if (typeof document !== 'undefined' && document.referrer) {
      return new URL(document.referrer).origin;
    }
  } catch {
    /* ignore malformed referrer */
  }
  return '*';
}

/** Boot the embedded app. Idempotent: calling twice returns the
 *  same in-flight promise. */
export async function connectToHost<TConfig = unknown>(
  options: ConnectToHostOptions = {},
): Promise<ConnectedHandle<TConfig>> {
  if (!bootPromise) {
    bootPromise = bootOnce(options) as Promise<ConnectedHandle<unknown>>;
  }
  return (await bootPromise) as ConnectedHandle<TConfig>;
}

/* ─────────────────── Boot pipeline ─────────────────── */

async function bootOnce(
  options: ConnectToHostOptions,
): Promise<ConnectedHandle<unknown>> {
  // Detect the parent's origin once: drives both the outbound
  // `postMessage` target AND the inbound message filter. Previously
  // the inbound filter defaulted to `window.location.origin` (the
  // EMBED's own origin), which is never what we want for cross-
  // origin Mode B (mobile shell at `tauri.localhost`, embed at
  // `*.hf.space`): every incoming `host:init` got dropped, the embed
  // sat the full `HOST_INIT_TIMEOUT_MS` and fell back to creds.
  // `detectParentOrigin()` returns the actual parent origin (from
  // `document.referrer`) or `'*'` if the referrer is empty; either
  // way it matches what `event.origin` carries on incoming messages.
  parentTargetOrigin = detectParentOrigin();
  const expectedOrigin = options.expectedOrigin ?? parentTargetOrigin;

  // 1. Parse creds from the URL hash and wipe it synchronously.
  const creds = decodeCredsFromHash(window.location.hash);
  wipeUrlHash();

  if (!creds) {
    throw new Error(
      '[reachy-mini-sdk/host/embed] no creds bundle found in URL hash. ' +
        'Was the embed mounted directly without ?embedded=1#creds=...?',
    );
  }

  // 2. Wait for the SDK script to finish loading.
  const sdkReady = await waitForSdkReady(SDK_READY_TIMEOUT_MS);
  if (!sdkReady) {
    throw new Error(
      '[reachy-mini-sdk/host/embed] window.ReachyMini did not become ' +
        `available within ${SDK_READY_TIMEOUT_MS}ms - check the SDK CDN tag.`,
    );
  }

  // 3. Seed the HF token before SDK construction so authenticate()
  //    resolves without a redirect. Lenient on the user-name key:
  //    the canonical schema is `userName` (camelCase) but earlier
  //    mobile builds wrote `username` (lowercase). Accept both so
  //    a stale shell in the wild keeps working; the `CredsBundle`
  //    interface stays strict on the writing side as the single
  //    source of truth.
  const credsUserName =
    creds.userName ??
    ((creds as unknown as { username?: string | null }).username ?? null);
  if (creds.hfToken && credsUserName) {
    seedSessionToken(creds.hfToken, credsUserName);
  }

  // 4. Build the SDK with the bundled signaling URL + appName.
  const sdk: ReachyMiniInstance = new window.ReachyMini({
    appName: creds.appName,
    signalingUrl: creds.signalingUrl,
    ...options.sdkOptions,
  });
  // Not OUR build version: the embed rides whatever `window.ReachyMini`
  // the app loaded, and that bundle is what talks to the robot. Latched
  // module-level so every app-state re-advertises it (same lifecycle as
  // `daemonVersion` below). SDKs that predate the field leave it null.
  appSdkVersion = typeof sdk.sdkVersion === 'string' ? sdk.sdkVersion : null;

  // 5. Build the bridge (subscriber registry) + post ready.
  const bridge = createBridge(expectedOrigin, sdk);
  postToHost({
    source: PROTOCOL_SOURCE,
    type: 'embed:ready',
    version: PROTOCOL_VERSION,
  });
  bridge.start();

  // 6. Wait for host:init. Both Mode A (same-origin host shell) and
  //    Mode B (cross-origin mobile shell at e.g. `tauri.localhost`)
  //    send this message; the cross-origin path was previously
  //    broken by a `event.origin !== window.location.origin` filter
  //    that silently dropped parent messages. Origin handling is
  //    now driven by `expectedOrigin` (computed above from
  //    `detectParentOrigin()`). On timeout we fall back to
  //    `liveStateFromCreds`, which carries the same fields as
  //    `liveStateFromInit` (verified in `liveStateFrom*` below).
  const live = await bridge.awaitHostInit(HOST_INIT_TIMEOUT_MS, creds);

  // 7. Sequence: connect → startSession → wake (await full trajectory).
  pushAppState('connecting', 'link');
  postDebug('boot:link:start', { robotPeerId: live.robotPeerId });
  await sdk.authenticate();
  postDebug('boot:authenticate:ok', { state: (sdk as { state?: string }).state });
  await sdk.connect();
  postDebug('boot:connect:ok', {
    state: (sdk as { state?: string }).state,
    robots: ((sdk as { robots?: unknown[] }).robots ?? []).length,
  });

  // Re-resolve the CURRENT peer id from central by the robot's stable
  // hardware id before dialing. The `robotPeerId` we were handed is a
  // snapshot the host captured earlier (picker selection / mobile
  // handoff); the central peer id rotates on every relay reconnect, so
  // after a Space cold-start it is frequently dead. Matching on
  // `hardware_id` self-heals against that rotation. A `null` hardware
  // id (older daemon) or any central hiccup falls back to the handed-in
  // peer id unchanged.
  const targetPeerId = await resolveLivePeerId(live, creds);
  pushAppState('connecting', 'session');
  postDebug('boot:session:start', {
    robotPeerId: targetPeerId,
    handedPeerId: live.robotPeerId,
    reresolved: targetPeerId !== live.robotPeerId,
  });
  installSdkProbe(sdk);
  try {
    await sdk.startSession(targetPeerId);
    postDebug('boot:session:ok');
  } catch (err) {
    postDebug('boot:session:error', {
      message: (err as Error)?.message ?? String(err),
    });
    throw err;
  }

  // Read the daemon version now that the data channel is up, and before
  // the wake, so a host that gates on it (the web shell's update gate)
  // can decide while the user is still on the connecting splash rather
  // than after the app has painted. Fail-open: stays `null` on timeout.
  daemonVersion = await readDaemonVersion(sdk);
  postDebug('boot:daemon-version', { daemonVersion });

  // Await the wake-up trajectory to completion (not fire-and-forget) so the
  // app is only revealed once the robot has finished its wake animation.
  pushAppState('connecting', 'wake');
  postDebug('boot:wake:start');
  await awaitFullyAwake(sdk);
  postDebug('boot:wake:ok', { awake: sdk.isAwake() });

  // 8. We're live. Wire pagehide cleanup so the SDK releases the
  //    robot if the browser kills the tab.
  bridge.attachPageHide(sdk);
  pushAppState('live', null);

  // Surface the SDK's automatic session re-dial to the host: the shell
  // already renders a full-screen ConnectingView whenever we report the
  // `connecting` phase, so a reconnect reads as "Reconnecting…" instead
  // of a dead frozen app. No-op against an older SDK bundle that doesn't
  // emit these events.
  installReconnectBridge(sdk);

  // Self-serve staleness check: this runs INSIDE the app's iframe, so
  // it covers every consumer (web shell, mobile app, anything else)
  // without the parent lifting a finger. Fire-and-forget: the GitHub
  // lookup must never delay or break going live.
  void maybeWarnSdkOutdated();

  // 9. Start sampling our own WebRTC RTT and reporting it upstream so
  //    a host shell that handed its session off to us (mobile app)
  //    can show a true live latency instead of a frozen value.
  startLiveLinkMonitor(sdk);

  return bridge.buildHandle<unknown>(sdk, live);
}

/* ─────────────────── Bridge state ─────────────────── */

interface LiveState {
  theme: ThemeMode;
  config: ConfigPayload;
  appName: string;
  hostName: string;
  userName: string | null;
  robotPeerId: string;
  /** Stable hardware id used to re-resolve `robotPeerId` at dial time.
   *  `null` when the host didn't provide one (older daemon). */
  robotHardwareId: string | null;
}

function liveStateFromCreds(creds: CredsBundle): LiveState {
  return {
    theme: creds.theme,
    config: creds.config,
    appName: creds.appName,
    hostName: creds.hostName,
    userName: creds.userName ?? null,
    robotPeerId: creds.robotPeerId,
    robotHardwareId: creds.robotHardwareId ?? null,
  };
}

function liveStateFromInit(msg: HostInitMsg): LiveState {
  return {
    theme: msg.theme,
    config: msg.config,
    appName: msg.appName,
    hostName: msg.hostName,
    userName: msg.userName ?? null,
    robotPeerId: msg.robotPeerId,
    robotHardwareId: msg.robotHardwareId ?? null,
  };
}

/**
 * Re-resolve the live central peer id for the target robot from its
 * stable hardware id, right before `startSession()`.
 *
 * The handed-in `live.robotPeerId` is a snapshot: central rotates a
 * robot's peer id on every relay reconnect, so by the time the iframe
 * has cold-started and reached this point the id can already be dead.
 * We fetch the current robot list from central and match on
 * `hardware_id` (stable per physical robot) to recover the fresh id.
 *
 * Fail-open: no hardware id (older daemon), no token, central
 * unreachable, or no match all return the handed-in peer id unchanged,
 * so this can only ever improve on the previous behaviour.
 */
async function resolveLivePeerId(
  live: LiveState,
  creds: CredsBundle,
): Promise<string> {
  const hardwareId = live.robotHardwareId;
  const hfToken = creds.hfToken;
  if (!hardwareId || !hfToken) return live.robotPeerId;
  try {
    const res = await fetchRobotsFromCentral({
      signalingUrl: creds.signalingUrl,
      hfToken,
    });
    if (!res.ok) return live.robotPeerId;
    const match = res.robots.find((r) => r.hardwareId === hardwareId);
    return match?.id ?? live.robotPeerId;
  } catch {
    return live.robotPeerId;
  }
}

function createBridge(expectedOrigin: string, sdk: ReachyMiniInstance) {
  type LeaveCb = () => void | Promise<void>;
  type ThemeCb = (t: ThemeMode) => void;
  type ConfigCb = (c: unknown) => void;

  const leaveListeners = new Set<LeaveCb>();
  const themeListeners = new Set<ThemeCb>();
  const configListeners = new Set<ConfigCb>();

  let current: LiveState | null = null;
  let leaveTriggered = false;
  let graceLeaveStarted = false;

  // Listener installed lazily so `embed:ready` is the only
  // outgoing event before the host has time to respond.
  let started = false;
  let onMessage: ((event: MessageEvent) => void) | null = null;

  function dispatchMessage(msg: HostToEmbedMsg): void {
    switch (msg.type) {
      case 'host:init': {
        current = liveStateFromInit(msg);
        // Re-notify subscribers in case the init arrives after
        // they registered (shouldn't happen with the current
        // boot order but cheap defensive code).
        themeListeners.forEach((cb) => cb(current!.theme));
        configListeners.forEach((cb) => cb(current!.config));
        break;
      }
      case 'host:theme-changed': {
        if (current) current.theme = msg.theme;
        themeListeners.forEach((cb) => cb(msg.theme));
        break;
      }
      case 'host:config-changed': {
        if (current) current.config = msg.config;
        configListeners.forEach((cb) => cb(msg.config));
        break;
      }
      case 'host:leaving': {
        void runGracefulLeave();
        break;
      }
      case 'host:start-update': {
        startDaemonUpdateForHost(sdk, msg.preRelease === true);
        break;
      }
      case 'host:cancel-update': {
        // The host's stall timer gave up on the job we started: put
        // the update-mode plumbing back (sessionStopped translator,
        // auto-reconnect). No-op when nothing is in flight.
        disarmActiveUpdate?.();
        break;
      }
    }
  }

  function runLeaveOnce(): void {
    if (leaveTriggered) return;
    leaveTriggered = true;
    // App-level cleanup. Fire and forget: the host budgets its own
    // teardown deadline, it doesn't wait on individual `onLeave` cbs.
    leaveListeners.forEach((cb) => {
      try {
        void cb();
      } catch (err) {
        console.warn('[reachy-mini-sdk/host/embed] onLeave threw', err);
      }
    });
  }

  /**
   * Graceful `host:leaving` teardown. The host layer owns the wake/sleep
   * contract, so on an explicit leave the SDK - not the app - puts the robot
   * to sleep, mirroring the mobile app's `sleepAndDisableRobot`:
   *
   *   1. Run the app's `onLeave` callbacks (their own cleanup).
   *   2. Dispatch `gotoSleep` immediately so the robot starts sleeping the
   *      instant the user leaves (no waiting on the daemon idle-reset debounce),
   *      bounded by a hard cap so a wedged daemon can't stall us.
   *   3. Force motors `Disabled` - deterministic off-switch, and it lands while
   *      the WebRTC session is still up, so the app-slot lock frees only AFTER
   *      the robot is already asleep. The daemon's idle-reset then sees
   *      `Disabled` and skips, so there's no second goto_sleep (no double
   *      trajectory / go_sleep sound) - all without any daemon-side change.
   *   4. Wait for the daemon to confirm the new motor mode, so we don't ack a
   *      command that never left the send queue (see `awaitMotorsDisabled`).
   *   5. Post `embed:left` so the host can unmount right away instead of waiting
   *      out its safety cap.
   *
   * Only wired to the explicit `host:leaving` path - `pagehide` (tab kill) has
   * no time to play a trajectory, so the daemon idle-reset covers that instead.
   */
  async function runGracefulLeave(): Promise<void> {
    if (graceLeaveStarted) return;
    graceLeaveStarted = true;

    runLeaveOnce();

    try {
      await Promise.race([
        sdk.gotoSleep({ timeoutMs: LEAVE_SLEEP_TIMEOUT_MS }),
        new Promise<void>((resolve) =>
          window.setTimeout(resolve, LEAVE_SLEEP_HARD_TIMEOUT_MS),
        ),
      ]);
    } catch {
      /* wedged/older daemon: fall through to the explicit disable */
    }
    try {
      sdk.setMotorMode('disabled');
      // Only ack once the daemon has echoed the mode back: the host unmounts
      // this iframe on the ack, which would kill the command in flight.
      await awaitMotorsDisabled(sdk);
    } catch {
      /* channel may already be closing - best effort */
    }

    postToHost({
      source: PROTOCOL_SOURCE,
      type: 'embed:left',
      version: PROTOCOL_VERSION,
    });
  }

  return {
    start(): void {
      if (started) return;
      started = true;
      onMessage = (event: MessageEvent) => {
        // `'*'` means we couldn't detect the parent's origin (empty
        // `document.referrer`); fall back to payload-only validation
        // via `isProtocolMessage`. The protocol carries no secrets,
        // so a spoofed message can only corrupt our own life-state -
        // bounded blast radius.
        if (expectedOrigin !== '*' && event.origin !== expectedOrigin) return;
        if (!isProtocolMessage(event.data)) return;
        dispatchMessage(event.data as HostToEmbedMsg);
      };
      window.addEventListener('message', onMessage);
    },

    async awaitHostInit(
      timeoutMs: number,
      fallbackCreds: CredsBundle,
    ): Promise<LiveState> {
      // No-iframe path (rare: direct page load for testing): the
      // parent IS this window, so no one will ever reply - resolve
      // synchronously from the hash creds. Both real Mode A
      // (same-origin shell + iframe) and Mode B (cross-origin
      // shell + iframe) send `host:init` and follow the listener
      // path below.
      const isInIframe = window.parent !== window;
      if (!isInIframe) {
        current = liveStateFromCreds(fallbackCreds);
        return current;
      }

      // If host:init already arrived (race), use it.
      if (current) return current;

      return new Promise((resolve) => {
        const initListener = (event: MessageEvent): void => {
          // Same wildcard tolerance as the main bridge listener -
          // accept any origin when we couldn't detect the parent's
          // (empty referrer). `isProtocolMessage` is the real
          // payload safety net.
          if (expectedOrigin !== '*' && event.origin !== expectedOrigin) return;
          if (!isProtocolMessage(event.data)) return;
          const data = event.data as HostToEmbedMsg;
          if (data.type !== 'host:init') return;
          window.removeEventListener('message', initListener);
          window.clearTimeout(timer);
          current = liveStateFromInit(data);
          resolve(current);
        };
        const timer = window.setTimeout(() => {
          window.removeEventListener('message', initListener);
          // Timeout: fall back to creds. Useful when the parent
          // never sends init (older host versions, manual
          // testing).
          if (!current) current = liveStateFromCreds(fallbackCreds);
          resolve(current);
        }, timeoutMs);
        window.addEventListener('message', initListener);
      });
    },

    attachPageHide(sdk: ReachyMiniInstance): void {
      const onPageHide = (): void => {
        runLeaveOnce();
        try {
          void sdk.stopSession();
        } catch {
          /* ignore - tab is going away anyway */
        }
      };
      window.addEventListener('pagehide', onPageHide, { once: true });
    },

    buildHandle<TConfig>(
      sdk: ReachyMiniInstance,
      live: LiveState,
    ): ConnectedHandle<TConfig> {
      current = live;
      const media = createRobotMedia(sdk);
      return {
        reachy: sdk,
        media,
        get theme(): ThemeMode {
          return current!.theme;
        },
        get config(): TConfig | null {
          return current!.config as TConfig | null;
        },
        get appName(): string {
          return current!.appName;
        },
        get hostName(): string {
          return current!.hostName;
        },
        get userName(): string | null {
          return current!.userName;
        },
        onLeave(cb) {
          leaveListeners.add(cb);
          return () => leaveListeners.delete(cb);
        },
        onThemeChange(cb) {
          themeListeners.add(cb);
          return () => themeListeners.delete(cb);
        },
        onConfigChange(cb) {
          const wrapped = (c: unknown) => cb(c as TConfig | null);
          configListeners.add(wrapped);
          return () => configListeners.delete(wrapped);
        },
        setAppState(state) {
          pushAppState(
            state.phase,
            state.connectingStep ?? null,
            state.message ?? null,
          );
        },
        requestLeave() {
          postToHost({
            source: PROTOCOL_SOURCE,
            type: 'embed:request-leave',
            version: PROTOCOL_VERSION,
          });
        },
        reportError(message, opts) {
          postToHost({
            source: PROTOCOL_SOURCE,
            type: 'embed:error',
            version: PROTOCOL_VERSION,
            message,
            fatal: opts?.fatal === true,
            detail: opts?.detail,
          });
        },
      };
    },
  };
}

/* ─────────────────── Helpers ─────────────────── */

function wipeUrlHash(): void {
  // Best-effort: replaceState fails on `file://` and a few exotic
  // schemes. We don't want to throw in the embed for that.
  try {
    const cleanUrl =
      window.location.pathname + window.location.search;
    history.replaceState(history.state, document.title, cleanUrl);
  } catch {
    /* ignore */
  }
}

function seedSessionToken(token: string, userName: string): void {
  try {
    sessionStorage.setItem('hf_token', token);
    sessionStorage.setItem('hf_username', userName);
    sessionStorage.setItem(
      'hf_token_expires',
      new Date(Date.now() + TOKEN_TTL_MS).toISOString(),
    );
  } catch {
    /* ignore - private browsing / quota */
  }
}

function waitForSdkReady(timeoutMs: number): Promise<boolean> {
  return new Promise((resolve) => {
    if (typeof window === 'undefined') {
      resolve(false);
      return;
    }
    if (window.ReachyMini) {
      resolve(true);
      return;
    }
    let settled = false;
    const onReady = (): void => {
      if (settled) return;
      settled = true;
      window.removeEventListener('reachymini:ready', onReady);
      window.clearTimeout(timer);
      resolve(Boolean(window.ReachyMini));
    };
    const timer = window.setTimeout(() => {
      if (settled) return;
      settled = true;
      window.removeEventListener('reachymini:ready', onReady);
      resolve(false);
    }, timeoutMs);
    window.addEventListener('reachymini:ready', onReady);
  });
}

function postToHost(msg: EmbedToHostMsg): void {
  if (typeof window === 'undefined') return;
  // Mode B (mobile WebView) embeds us in an iframe at a DIFFERENT
  // origin than the parent shell. Sending to
  // `window.location.origin` then makes the browser drop the
  // message and warn "Recipient has origin <X>". `parentTargetOrigin`
  // is set once at boot to the parent's referrer-derived origin,
  // falling back to `'*'` (safe - payloads carry no secrets).
  try {
    window.parent.postMessage(msg, parentTargetOrigin);
  } catch (err) {
    console.warn('[reachy-mini-sdk/host/embed] postMessage to host failed', err);
  }
}

/**
 * Run the daemon self-update on the host's behalf and relay its
 * progress back over the bridge.
 *
 * The host asks for this because only we hold a data channel. Daemon
 * events are forwarded verbatim; on top of them we synthesise the one
 * the daemon can't send - `rebooting`, when the restart takes the
 * session down mid-install, which is what a success looks like from
 * here.
 */
/**
 * Disarm handle for the in-flight `host:start-update` job, armed by
 * `startDaemonUpdateForHost` and consumed by `host:cancel-update`.
 * Null whenever no update is in flight (a cancel is then a no-op).
 * The host sends the cancel when ITS stall timer gives up: only the
 * host runs one, so without this hook a host-side timeout would leave
 * the sessionStopped translator armed and auto-reconnect off for the
 * rest of the session.
 */
let disarmActiveUpdate: (() => void) | null = null;

function startDaemonUpdateForHost(
  sdk: ReachyMiniInstance,
  preRelease: boolean,
): void {
  const postProgress = (msg: Omit<EmbedUpdateProgressMsg, 'source' | 'type' | 'version'>): void => {
    postToHost({
      source: PROTOCOL_SOURCE,
      type: 'embed:update-progress',
      version: PROTOCOL_VERSION,
      ...msg,
    });
  };

  // Stand the SDK's auto re-dial down for the duration of the update:
  // a successful install ends with `systemctl restart`, and that
  // teardown must surface as `sessionStopped` IMMEDIATELY (it's the
  // "install done, rebooting" signal below) — not get absorbed by
  // ~22 s of reconnect attempts against a robot that is rebooting.
  // Feature-detected so an older SDK bundle is a no-op.
  const setAutoReconnect = (enabled: boolean): void => {
    const fn = (sdk as unknown as { setAutoReconnect?: (e: boolean) => void })
      .setAutoReconnect;
    if (typeof fn === 'function') {
      try { fn.call(sdk, enabled); } catch { /* ignore */ }
    }
  };
  setAutoReconnect(false);

  // A successful install ends with `systemctl restart`, which kills the
  // session before the daemon can report anything: the transport
  // dropping IS the completion signal. Translate it into an explicit
  // `rebooting` so the host stops waiting on a channel that will never
  // speak again. The listener must NOT outlive a failed update: a
  // leftover copy would fire on the user's next normal end-session and
  // flip the (already failed) gate into a bogus reboot wait.
  const onSessionStopped = (): void => {
    disarmActiveUpdate = null;
    sdk.removeEventListener('sessionStopped', onSessionStopped);
    postProgress({ status: 'rebooting' });
  };
  const abandonUpdate = (): void => {
    disarmActiveUpdate = null;
    sdk.removeEventListener('sessionStopped', onSessionStopped);
    setAutoReconnect(true);
  };

  let sent: boolean;
  try {
    sent = sdk.startDaemonUpdate({
      preRelease,
      onProgress: (event) => {
        // Terminal failure: the daemon is alive and NOT rebooting, so
        // the session outlives the update. Restore normal resilience
        // and drop the reboot translator.
        if (event.status === 'failed') abandonUpdate();
        postProgress({
          status: event.status,
          line: event.line ?? null,
          error: event.error ?? null,
        });
      },
    });
  } catch (err) {
    abandonUpdate();
    postProgress({ status: 'failed', error: (err as Error)?.message ?? String(err) });
    return;
  }
  // Channel closed: the daemon never saw the request, so nothing else
  // will ever report on it. Say so now rather than leaving the host to
  // time out on a job that was never started.
  if (!sent) {
    abandonUpdate();
    postProgress({ status: 'failed', error: 'Data channel not open' });
    return;
  }

  try {
    sdk.addEventListener('sessionStopped', onSessionStopped);
  } catch {
    /* older SDK without the event: the host falls back to its stall timer */
  }
  disarmActiveUpdate = abandonUpdate;
}

/**
 * Daemon version of the robot we're talking to, resolved once the
 * session is up (see `readDaemonVersion`). Module-level so every
 * subsequent `pushAppState` re-advertises it: the host has no data
 * channel of its own, so this is its only way to learn what software
 * the robot runs. `null` until resolved, or forever against a daemon
 * that predates `get_version`.
 */
let daemonVersion: string | null = null;

/**
 * Version of the SDK bundle the app loaded (`instance.sdkVersion`),
 * captured at construction in `bootOnce`. Drives our own staleness
 * check (`maybeWarnSdkOutdated`) and is advertised on every app-state
 * for parents that update independently of the app (see `sdkVersion`
 * in protocol.ts). `null` against an SDK old enough not to carry the
 * field.
 */
let appSdkVersion: string | null = null;

/**
 * Ask the daemon its version, bounded so a silent or too-old daemon
 * can't stall the boot. Fail-open: any timeout / rejection resolves
 * `null`, which every consumer reads as "unknown, carry on".
 */
async function readDaemonVersion(sdk: ReachyMiniInstance): Promise<string | null> {
  try {
    const version = await Promise.race([
      sdk.getVersion(),
      new Promise<null>((resolve) =>
        setTimeout(() => resolve(null), DAEMON_VERSION_TIMEOUT_MS),
      ),
    ]);
    return typeof version === 'string' && version.length > 0 ? version : null;
  } catch {
    return null;
  }
}

function pushAppState(
  phase: AppPhase,
  connectingStep: AppConnectingStep | null,
  message: string | null = null,
  rttMs: number | null = null,
): void {
  postToHost({
    source: PROTOCOL_SOURCE,
    type: 'embed:app-state',
    version: PROTOCOL_VERSION,
    phase,
    connectingStep,
    message,
    rttMs,
    daemonVersion,
    sdkVersion: appSdkVersion,
  });
}

/* ─────────────────── SDK staleness self-check ─────────────────── */

/**
 * Warn the user, from inside the iframe, when the SDK bundle this app
 * shipped trails the latest release.
 *
 * Why here and not in the parent: the embed is the only code that runs
 * identically under every host (web shell, mobile app WebView), so a
 * check living here needs zero integration on the parent's side. The
 * one case it structurally CANNOT cover is an app whose SDK predates
 * this very code - a frozen bundle can't warn about itself. Only a
 * parent that updates independently of the app could catch that (the
 * mobile app, via the absence of `sdkVersion` on app-state); the web
 * shell can't, being part of the same frozen bundle.
 *
 * Silent when: the version is a dev placeholder (`0.0.0-*`) or
 * unparseable, the latest release can't be fetched, or we're simply up
 * to date. Never blocks - one click dismisses it for the session.
 */
async function maybeWarnSdkOutdated(): Promise<void> {
  try {
    const parsed = parseSemver(appSdkVersion);
    if (!parsed || parsed.major === 0) return;
    const latest = await fetchLatestDaemonVersion();
    if (!isDaemonOutdated(appSdkVersion, latest)) return;
    showSdkOutdatedOverlay(appSdkVersion as string, latest as string);
  } catch {
    /* purely advisory - never let it interfere with a live session */
  }
}

/** Plain-DOM overlay (the embed is framework-agnostic): dark scrim,
 *  centred card, one dismiss button. Styling is self-contained so it
 *  renders the same over any app theme. */
function showSdkOutdatedOverlay(current: string, latest: string): void {
  if (typeof document === 'undefined') return;
  if (document.getElementById('reachy-sdk-outdated-overlay')) return;

  const scrim = document.createElement('div');
  scrim.id = 'reachy-sdk-outdated-overlay';
  scrim.setAttribute('role', 'alertdialog');
  scrim.setAttribute('aria-label', 'This app may be out of date');
  scrim.style.cssText = [
    'position:fixed',
    'inset:0',
    'z-index:2147483000',
    'display:flex',
    'align-items:center',
    'justify-content:center',
    'padding:16px',
    'background:rgba(0,0,0,0.55)',
    'font-family:system-ui,-apple-system,sans-serif',
  ].join(';');

  const card = document.createElement('div');
  card.style.cssText = [
    'background:#fff',
    'color:#1a1a1a',
    'border-radius:12px',
    'padding:24px',
    'max-width:400px',
    'width:100%',
    'text-align:center',
    'box-shadow:0 8px 32px rgba(0,0,0,0.35)',
  ].join(';');

  const icon = document.createElement('div');
  icon.textContent = '\u26A0\uFE0F';
  icon.style.cssText = 'font-size:32px;line-height:1;margin-bottom:12px';

  const title = document.createElement('div');
  title.textContent = 'This app may be out of date';
  title.style.cssText = 'font-size:17px;font-weight:700;margin-bottom:8px';

  const body = document.createElement('div');
  body.textContent =
    `It was built with SDK v${current}, but v${latest} is the latest. ` +
    'Some things may not behave as expected with your robot.';
  body.style.cssText =
    'font-size:13px;line-height:1.6;color:#555;margin-bottom:16px';

  const button = document.createElement('button');
  button.textContent = 'I understand, continue';
  button.style.cssText = [
    'width:100%',
    'padding:10px 16px',
    'border:none',
    'border-radius:8px',
    'background:#1a1a1a',
    'color:#fff',
    'font-size:13px',
    'font-weight:600',
    'cursor:pointer',
  ].join(';');
  button.addEventListener('click', () => scrim.remove());

  card.append(icon, title, body, button);
  scrim.appendChild(card);
  document.body.appendChild(scrim);
  button.focus();
}

/* ─────────────────── Live link latency monitor ─────────────────── */

/** Poll cadence for the live RTT sampler. Matches the host-side
 *  `TransportMonitor` so the number updates at the same rhythm the
 *  user is used to on the session screen. */
const LINK_MONITOR_INTERVAL_MS = 1500;
/** Rolling-min window: ~6 ticks ≈ 9 s of history, so a single Wi-Fi
 *  jitter spike doesn't bounce the displayed latency. Mirrors the
 *  host's `RTT_WINDOW_SIZE`. */
const RTT_WINDOW_SIZE = 6;

/**
 * Sample the selected ICE candidate pair's `currentRoundTripTime`
 * from a peer connection and return it in milliseconds, or `null`
 * when no nominated pair exposes it yet (or the platform omits it,
 * e.g. iOS WKWebView). Pure read of `getStats()` - same field the
 * host's `TransportMonitor` reads.
 */
async function sampleRttMs(pc: RTCPeerConnection): Promise<number | null> {
  try {
    const stats = await pc.getStats();
    let rtt: number | null = null;
    stats.forEach((report) => {
      const r = report as {
        type: string;
        selected?: boolean;
        nominated?: boolean;
        state?: string;
        currentRoundTripTime?: number;
      };
      if (r.type !== 'candidate-pair') return;
      const isSelected =
        r.selected === true || (r.nominated === true && r.state === 'succeeded');
      if (!isSelected) return;
      if (typeof r.currentRoundTripTime === 'number') {
        rtt = r.currentRoundTripTime * 1000;
      }
    });
    return rtt;
  } catch {
    return null;
  }
}

/**
 * Once the app is live, periodically sample the embed's OWN WebRTC
 * RTT and re-emit `embed:app-state` (still `phase: 'live'`) carrying
 * the rolling-min latency. This is what lets a host shell that has
 * released its session to us (the mobile app) display a TRUE,
 * up-to-date link latency: the host no longer holds a connection, so
 * the iframe is the only place the candidate pair still exists.
 *
 * Self-stops on `sessionStopped` and `pagehide`. No-op (never emits)
 * when the platform doesn't expose RTT, so the host keeps hiding the
 * latency pill rather than showing a frozen value.
 */
function startLiveLinkMonitor(sdk: ReachyMiniInstance): void {
  if (typeof window === 'undefined') return;
  // Re-read on every tick (not captured once): the SDK's auto-reconnect
  // replaces the RTCPeerConnection mid-session, and a sampler pinned to
  // the old pc would freeze the RTT forever.
  const livePc = (): RTCPeerConnection | null => livePeerConnection(sdk);
  if (!livePc()) return;

  const windowMs: number[] = [];
  let stopped = false;

  const tick = async (): Promise<void> => {
    if (stopped) return;
    const pc = livePc();
    if (!pc || pc.connectionState === 'closed') return;
    const sample = await sampleRttMs(pc);
    if (sample === null) return;
    windowMs.push(sample);
    if (windowMs.length > RTT_WINDOW_SIZE) windowMs.shift();
    pushAppState('live', null, null, Math.min(...windowMs));
  };

  const interval = window.setInterval(() => void tick(), LINK_MONITOR_INTERVAL_MS);
  // Kick a first sample shortly after going live (the pair is already
  // nominated by the time `connectToHost()` resolves).
  window.setTimeout(() => void tick(), 600);

  const stop = (): void => {
    if (stopped) return;
    stopped = true;
    window.clearInterval(interval);
  };
  try {
    sdk.addEventListener('sessionStopped', stop);
  } catch {
    /* ignore - sampler will keep running until pagehide */
  }
  window.addEventListener('pagehide', stop, { once: true });
}

/**
 * Live RTCPeerConnection of the SDK bundle the app shipped. Prefers the
 * public `peerConnection` getter; falls back to the private `_pc` field
 * for bundles that predate it (the embed rides whatever SDK the app
 * bundled, so both shapes are in the wild). `null` between sessions.
 */
function livePeerConnection(sdk: ReachyMiniInstance): RTCPeerConnection | null {
  const s = sdk as unknown as {
    peerConnection?: RTCPeerConnection | null;
    _pc?: RTCPeerConnection | null;
  };
  return s.peerConnection ?? s._pc ?? null;
}

/**
 * Dev-only diagnostic channel. Forwards a tag + payload to the host
 * so the parent's console (visible to devtools and the Cursor MCP
 * browser) shows the embed's boot progression. The host's
 * `ReachyHostShell` listens for `embed:debug` and `console.info`s
 * the payload.
 */
function postDebug(tag: string, payload: Record<string, unknown> = {}): void {
  if (typeof window === 'undefined') return;
  try {
    window.parent.postMessage(
      {
        source: PROTOCOL_SOURCE,
        type: 'embed:debug',
        version: PROTOCOL_VERSION,
        tag,
        payload,
      },
      parentTargetOrigin,
    );
  } catch {
    /* ignore */
  }
  try {
    let asJson = '';
    try {
      asJson = JSON.stringify(payload);
    } catch {
      asJson = '<unserializable>';
    }
    console.info(`[embed-debug] ${tag} ${asJson}`);
  } catch {
    /* ignore */
  }
}

/**
 * Build the `RobotMedia` surface for a freshly-resolved SDK
 * handle.
 *
 * Background
 * ──────────
 * `connectToHost()` only resolves once the SDK's `startSession()`
 * has completed - which means ICE is connected, the data channel
 * is open, AND `pc.ontrack` has already fired for every remote
 * track in the SDP answer. By the time the embedded React tree
 * mounts and a `<video>` element calls `attachVideo()`, the SDK's
 * one-shot `videoTrack` event has therefore already happened and
 * a freshly-registered listener will sit silent.
 *
 * Implementation
 * ──────────────
 * We avoid mutating the SDK (no monkey-patching of
 * `reachy.attachVideo`) and instead build a thin parallel API
 * around it:
 *   - The `robotStream` is captured by reading
 *     `sdk._pc.getReceivers()` lazily on first access. By then
 *     ICE+DC are connected so every receiver has a live track.
 *     The result is cached as a single stable `MediaStream`
 *     instance - good for React `useEffect` deps.
 *   - `attachVideo()` calls the SDK's own `attachVideo()` first
 *     (so we keep its mute-sync, latency monitor and
 *     `sessionStopped` cleanup) and then immediately replays the
 *     cached stream into the element. Late-mounting consumers
 *     therefore see the camera within one paint instead of
 *     waiting forever.
 *   - On `sessionStopped` we drop the cached stream so subsequent
 *     reads return `null`. (`connectToHost()` is one-shot per
 *     page load, so we do not currently rebuild the cache after
 *     a stop / restart - apps tear down on session end.)
 *
 * `micStream` is just a delegating getter to `reachy._micStream`:
 * the SDK exposes it synchronously and there is no race - it is
 * acquired during `startSession()` before `connectToHost()`
 * resolves. We surface it on the handle so apps don't poke into
 * underscore-prefixed SDK internals.
 */
function createRobotMedia(sdk: ReachyMiniInstance): RobotMedia {
  let cached: MediaStream | null = null;

  const sdkInternals = sdk as unknown as {
    _micStream: MediaStream | null;
  };

  const buildFromReceivers = (): MediaStream | null => {
    const pc = livePeerConnection(sdk);
    if (!pc) return null;
    const tracks = pc
      .getReceivers()
      .map((rcv) => rcv.track)
      .filter(
        (t): t is MediaStreamTrack =>
          t !== null && t.kind !== '' && t.readyState === 'live',
      );
    if (tracks.length === 0) return null;
    return new MediaStream(tracks);
  };

  const ensureCached = (): MediaStream | null => {
    if (cached) return cached;
    cached = buildFromReceivers();
    if (cached) {
      postDebug('media:cache:init', {
        videoTracks: cached.getVideoTracks().length,
        audioTracks: cached.getAudioTracks().length,
      });
    }
    return cached;
  };

  // Drop the cache as soon as the daemon tears down - keeps
  // `media.robotStream` honest if anything reads it after
  // `sessionStopped`. Same on `sessionReconnecting`: the SDK's
  // auto re-dial replaces the RTCPeerConnection, so the cached
  // tracks are dead and must be rebuilt from the new receivers.
  const onSessionStopped = (): void => {
    if (cached) {
      cached = null;
      postDebug('media:cache:clear');
    }
  };
  sdk.addEventListener('sessionStopped', onSessionStopped);
  sdk.addEventListener('sessionReconnecting', onSessionStopped);

  return {
    attachVideo(el: HTMLVideoElement): () => void {
      const detach = sdk.attachVideo(el);
      const stream = ensureCached();
      if (stream && el.srcObject !== stream) {
        try {
          el.srcObject = stream;
          // Best-effort autoplay. Browsers gate this on a user
          // gesture; the host-side picker tap typically satisfies
          // it. Swallow the rejection so a Safari pre-gesture
          // mount never crashes the boot path.
          void el.play().catch(() => {
            /* ignore */
          });
          postDebug('media:attach:replay', {
            videoTracks: stream.getVideoTracks().length,
            audioTracks: stream.getAudioTracks().length,
          });
        } catch (err) {
          postDebug('media:attach:replay:error', {
            message: (err as Error)?.message ?? String(err),
          });
        }
      }
      return detach;
    },
    get robotStream(): MediaStream | null {
      return ensureCached();
    },
    get micStream(): MediaStream | null {
      return sdkInternals._micStream;
    },
  };
}

/**
 * Mirror the SDK's automatic session re-dial into host app-states.
 *
 * The SDK (>= the version carrying `autoReconnect`) re-dials the robot
 * by itself when an established session's transport dies. While it
 * retries, the app iframe is functionally frozen — motion/RPC calls
 * fail — so the honest UX is the same connecting splash the boot flow
 * uses. On success the SDK re-fires `streaming` (video re-attaches on
 * its own) and we flip back to `live`; on a terminal
 * `sessionStopped { reason: 'reconnect_failed' }` we report a fatal
 * error so the shell offers reload / back-to-picker.
 *
 * Registered with plain string event names so an app that shipped an
 * OLDER SDK bundle (no such events) is a silent no-op.
 */
function installReconnectBridge(sdk: ReachyMiniInstance): void {
  const on = (
    name: string,
    cb: (e: { detail?: Record<string, unknown> }) => void,
  ): void => {
    try {
      (sdk as unknown as {
        addEventListener: (n: string, c: (e: unknown) => void) => void;
      }).addEventListener(name, cb as (e: unknown) => void);
    } catch {
      /* older SDK bundle */
    }
  };

  on('sessionReconnecting', (e) => {
    const attempt = Number(e.detail?.attempt ?? 1);
    const max = Number(e.detail?.maxAttempts ?? 1);
    pushAppState(
      'connecting',
      'session',
      attempt > 1
        ? `Reconnecting to the robot (attempt ${attempt}/${max})`
        : 'Connection lost — reconnecting to the robot',
    );
  });

  on('sessionReconnected', () => {
    pushAppState('live', null);
  });

  on('sessionStopped', (e) => {
    if (e.detail?.reason !== 'reconnect_failed') return;
    postToHost({
      source: PROTOCOL_SOURCE,
      type: 'embed:error',
      version: PROTOCOL_VERSION,
      message: 'Lost the connection to the robot and could not reconnect.',
      fatal: true,
      detail: typeof e.detail?.message === 'string' ? e.detail.message : undefined,
    });
  });
}

/**
 * One-shot SDK probe used while we hunt the "stuck at session" bug.
 * Subscribes to every internal event the SDK is known to emit and
 * forwards them to the host via `embed:debug`. No-op in production
 * once the bug is fixed.
 */
function installSdkProbe(sdk: ReachyMiniInstance): void {
  const events = [
    'connected',
    'disconnected',
    'streaming',
    'sessionStopped',
    'sessionRejected',
    'sessionReconnecting',
    'sessionReconnected',
    'robotsChanged',
    'error',
    'state',
    'log',
    'message',
  ];
  for (const ev of events) {
    try {
      (sdk as unknown as {
        addEventListener: (n: string, cb: (e: unknown) => void) => void;
      }).addEventListener(ev, (e: unknown) => {
        let detail: Record<string, unknown> = {};
        const evObj = e as { detail?: unknown };
        if (evObj && typeof evObj === 'object' && 'detail' in evObj) {
          try {
            detail = JSON.parse(JSON.stringify(evObj.detail ?? null));
          } catch {
            detail = { _unserializable: true };
          }
        }
        postDebug(`sdk:${ev}`, detail);
      });
    } catch {
      /* ignore */
    }
  }
  // Wrap _handleSignalingMessage so we see every payload central
  // delivers via SSE (peer offers, ICE candidates, sessionRejected,
  // etc.). If we never see a `peer` message of kind `sdp/offer`
  // here, central is dropping the offer or routing it to a stale
  // peer.
  try {
    const sdkAny = sdk as unknown as {
      _handleSignalingMessage?: (msg: unknown) => unknown;
    };
    const orig = sdkAny._handleSignalingMessage;
    if (typeof orig === 'function') {
      const sendOrig = (sdkAny as Record<string, unknown>)._sendToServer as
        | ((this: unknown, payload: unknown) => Promise<unknown>)
        | undefined;
      if (typeof sendOrig === 'function') {
        (sdkAny as Record<string, unknown>)._sendToServer =
          async function patchedSend(this: unknown, payload: unknown) {
            const p = payload as Record<string, unknown>;
            const dbg: Record<string, unknown> = { type: p?.type ?? '?' };
            if (p && 'peerId' in p) dbg.peerId = String(p.peerId);
            if (p && 'sessionId' in p) dbg.sessionId = String(p.sessionId);
            if (p && 'sdp' in p) {
              const sdp = p.sdp as { type?: string; sdp?: string } | undefined;
              dbg.sdpType = sdp?.type ?? '?';
              dbg.sdpLen = sdp?.sdp?.length ?? 0;
            }
            if (p && 'ice' in p) {
              const ice = p.ice as { candidate?: string } | undefined;
              dbg.iceCand =
                (ice?.candidate ?? '').slice(0, 60) || '<end-of-candidates>';
            }
            postDebug('sdk:send', dbg);
            try {
              const res = await sendOrig.call(this, payload);
              const rj = res as Record<string, unknown> | undefined;
              postDebug('sdk:send:res', {
                inFor: dbg.type,
                resType: rj?.type ?? null,
                keys: rj ? Object.keys(rj) : [],
              });
              return res;
            } catch (err) {
              postDebug('sdk:send:err', {
                inFor: dbg.type,
                msg: (err as Error)?.message ?? String(err),
              });
              throw err;
            }
          };
      }
      sdkAny._handleSignalingMessage = function patched(msg: unknown) {
        const m = msg as Record<string, unknown>;
        const payload: Record<string, unknown> = { type: m?.type ?? '?' };
        if ('sessionId' in m) payload.sessionId = String(m.sessionId);
        if ('peerId' in m) payload.peerId = String(m.peerId);
        if ('sdp' in m) {
          const sdp = m.sdp as { type?: string; sdp?: string } | undefined;
          payload.sdpType = sdp?.type ?? '?';
          payload.sdpLen = sdp?.sdp?.length ?? 0;
        }
        if ('ice' in m) {
          const ice = m.ice as { candidate?: string } | undefined;
          payload.iceCand =
            (ice?.candidate ?? '').slice(0, 60) || '<end-of-candidates>';
        }
        if ('reason' in m) payload.reason = String(m.reason);
        postDebug('sdk:sse', payload);
        return orig.call(this, msg);
      };
    }
  } catch {
    /* ignore */
  }
  const probeStart = Date.now();
  const interval = window.setInterval(() => {
    const sdkAny = sdk as unknown as {
      _pc?: RTCPeerConnection;
      _dc?: RTCDataChannel;
      _sessionId?: string;
      _peerId?: string;
      _state?: string;
      _sseAbortController?: { signal?: { aborted?: boolean } };
    };
    const pc = sdkAny._pc;
    const dc = sdkAny._dc;
    postDebug('sdk:probe', {
      elapsedMs: Date.now() - probeStart,
      myPeerId: sdkAny._peerId ?? null,
      state: sdkAny._state ?? null,
      sseAborted: sdkAny._sseAbortController?.signal?.aborted ?? null,
      pcState: pc?.connectionState ?? null,
      iceState: pc?.iceConnectionState ?? null,
      iceGather: pc?.iceGatheringState ?? null,
      signalingState: pc?.signalingState ?? null,
      dcState: dc?.readyState ?? null,
      sessionId: sdkAny._sessionId ?? null,
    });
    if (Date.now() - probeStart > 30_000) window.clearInterval(interval);
  }, 1500);
}
