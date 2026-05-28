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
 * Boot sequence (canonical reference: SPEC.md §6.4):
 *  1. Read `#creds=<base64>` synchronously and wipe the hash
 *     with `history.replaceState`.
 *  2. Wait for `window.ReachyMini` (8 s timeout).
 *  3. Instantiate the SDK, seed the HF token into
 *     `sessionStorage`.
 *  4. Send `embed:ready` to the parent.
 *  5. Wait for `host:init` (2 s soft timeout; on hit we proceed
 *     from the hash creds alone via `liveStateFromCreds`, which
 *     carries the same fields as `liveStateFromInit`).
 *  6. `connect()` → `startSession()` → `ensureAwake()`, emitting
 *     `embed:app-state` at each step.
 *  7. Resolve `connectToHost()` with the live SDK handle.
 *
 * Strict Mode safety (SPEC §8.4): the function is idempotent
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
import type {
  AppConnectingStep,
  AppPhase,
  ConfigPayload,
  CredsBundle,
  EmbedToHostMsg,
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

  // 5. Build the bridge (subscriber registry) + post ready.
  const bridge = createBridge(expectedOrigin);
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

  // 7. Sequence: connect → startSession → ensureAwake.
  pushAppState('connecting', 'link');
  postDebug('boot:link:start', { robotPeerId: live.robotPeerId });
  await sdk.authenticate();
  postDebug('boot:authenticate:ok', { state: (sdk as { state?: string }).state });
  await sdk.connect();
  postDebug('boot:connect:ok', {
    state: (sdk as { state?: string }).state,
    robots: ((sdk as { robots?: unknown[] }).robots ?? []).length,
  });

  pushAppState('connecting', 'session');
  postDebug('boot:session:start', { robotPeerId: live.robotPeerId });
  installSdkProbe(sdk);
  try {
    await sdk.startSession(live.robotPeerId);
    postDebug('boot:session:ok');
  } catch (err) {
    postDebug('boot:session:error', {
      message: (err as Error)?.message ?? String(err),
    });
    throw err;
  }

  pushAppState('connecting', 'wake');
  postDebug('boot:wake:start');
  await sdk.ensureAwake();
  postDebug('boot:wake:ok');

  // 8. We're live. Wire pagehide cleanup so the SDK releases the
  //    robot if the browser kills the tab.
  bridge.attachPageHide(sdk);
  pushAppState('live', null);

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
}

function liveStateFromCreds(creds: CredsBundle): LiveState {
  return {
    theme: creds.theme,
    config: creds.config,
    appName: creds.appName,
    hostName: creds.hostName,
    userName: creds.userName ?? null,
    robotPeerId: creds.robotPeerId,
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
  };
}

function createBridge(expectedOrigin: string) {
  type LeaveCb = () => void | Promise<void>;
  type ThemeCb = (t: ThemeMode) => void;
  type ConfigCb = (c: unknown) => void;

  const leaveListeners = new Set<LeaveCb>();
  const themeListeners = new Set<ThemeCb>();
  const configListeners = new Set<ConfigCb>();

  let current: LiveState | null = null;
  let leaveTriggered = false;

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
        runLeaveOnce();
        break;
      }
    }
  }

  function runLeaveOnce(): void {
    if (leaveTriggered) return;
    leaveTriggered = true;
    // Fire and forget; the host doesn't wait for an ack, it just
    // unmounts the iframe after `timeoutMs`.
    leaveListeners.forEach((cb) => {
      try {
        void cb();
      } catch (err) {
        console.warn('[reachy-mini-sdk/host/embed] onLeave threw', err);
      }
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

function pushAppState(
  phase: AppPhase,
  connectingStep: AppConnectingStep | null,
  message: string | null = null,
): void {
  postToHost({
    source: PROTOCOL_SOURCE,
    type: 'embed:app-state',
    version: PROTOCOL_VERSION,
    phase,
    connectingStep,
    message,
  });
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
    _pc: RTCPeerConnection | null;
    _micStream: MediaStream | null;
  };

  const buildFromReceivers = (): MediaStream | null => {
    const pc = sdkInternals._pc;
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
  // `sessionStopped`.
  const onSessionStopped = (): void => {
    if (cached) {
      cached = null;
      postDebug('media:cache:clear');
    }
  };
  sdk.addEventListener('sessionStopped', onSessionStopped);

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
