/**
 * postMessage protocol v1 between the host shell (parent window,
 * exposed via `@pollen-robotics/reachy-mini-sdk/host`) and an
 * embedded Reachy Mini app (iframe).
 *
 * Canonical reference: APP_CREATION_GUIDE §13.6 (Protocol v1).
 *
 * Contract identity
 * ─────────────────
 * - Every message carries `version: 1`. Bumping that integer is
 *   the ONLY way to introduce a breaking change to the wire
 *   protocol. Additive changes (new optional fields, new typed
 *   messages) ship without a version bump.
 * - Every message carries `source: 'reachy-mini'`. Lets receivers
 *   distinguish our envelopes from unrelated `postMessage` traffic
 *   (DevTools, MUI portals, browser extensions, ...).
 * - Both sides validate `event.origin` before trusting the payload,
 *   but against different references: the web shell requires strict
 *   same-origin (`window.location.origin` - Mode A, the Space embeds
 *   itself), while the embed accepts its PARENT's origin resolved via
 *   `document.referrer` (Mode B, the mobile shell at e.g.
 *   `tauri.localhost` is a different origin by construction).
 *
 * Message families
 * ────────────────
 * 1. Lifecycle: boot / ready / leaving. Drive the visible state
 *    of the host shell.
 * 2. State: `embed:app-state` lets the host render accurate
 *    connection / wake-up overlays.
 * 3. Config & theme: opaque `config` payload + theme push.
 * 4. Error: `embed:error` for tear-down on app failure.
 *
 * Intentionally NOT in v1 (see APP_CREATION_GUIDE §13.6):
 * - No `host:custom` / `embed:custom` free-form channel.
 * - No `embed:request-config-update` (apps don't push config).
 * - No heartbeat / ping-pong.
 */

/** Protocol version. Bump on breaking changes. */
export const PROTOCOL_VERSION = 1;

/** Source tag attached to every envelope. */
export const PROTOCOL_SOURCE = 'reachy-mini' as const;

/** Theme mode the host applies to the embedded app. */
export type ThemeMode = 'dark' | 'light';

/**
 * Connection lifecycle, as observed by the embedded app. Drives
 * the host's `ConnectingView` stepper and visibility of the
 * iframe.
 *
 *   boot       : app loaded, no SDK action yet
 *   connecting : ReachyMini.connect() / startSession() in flight
 *   live       : session up, motors awake, app interactive
 *   leaving    : `host:leaving` received, app tearing down
 *   error      : non-recoverable failure
 */
export type AppPhase = 'boot' | 'connecting' | 'live' | 'leaving' | 'error';

/**
 * Fine-grained step inside `connecting`. Maps to the 3 dots in
 * `StepsProgressIndicator`:
 *   - `link`    : waiting for `host:init` / `connect()` in flight
 *   - `session` : `startSession()` in flight
 *   - `wake`    : `ensureAwake()` in flight
 *
 * Apps that don't differentiate can omit this field; the host
 * treats it as `link`.
 */
export type AppConnectingStep = 'link' | 'session' | 'wake';

/** Opaque app-specific payload routed through the host. */
export type ConfigPayload = unknown;

/** Reason supplied with `host:leaving` for logging. */
export type LeavingReason =
  | 'user-action'
  | 'session-stopped'
  | 'error'
  | 'pagehide';

/* ─────────────────── HOST → EMBED ─────────────────── */

/**
 * First message sent by the host once the iframe has shouted
 * `embed:ready`. Carries the credentials + initial state the app
 * needs to bring a session up.
 *
 * Same-origin iframe: the SDK CDN script is imported with a
 * relative URL from `index.html`. We still pass `signalingUrl`
 * here so the host can swap centrals (staging / self-hosted)
 * without a rebuild of the app.
 */
export interface HostInitMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'host:init';
  version: 1;
  theme: ThemeMode;
  signalingUrl: string;
  /** HF OAuth bearer token. Optional for apps that don't need to
   *  authenticate (rare). */
  hfToken?: string;
  /** HF account user name, when known. */
  userName?: string | null;
  /** Robot ID selected by the host's picker. Unstable: the central
   *  peer id rotates on every relay reconnect, so the embed treats
   *  this as a starting point and re-resolves the live id from
   *  `robotHardwareId` right before `startSession()`. */
  robotPeerId: string;
  /** Stable hardware id of the selected robot (central
   *  `meta.hardware_id`), when the daemon exposes one. Lets the embed
   *  re-resolve the CURRENT `robotPeerId` from central just before
   *  dialing, so a rotated peer id (long iframe cold-start, relay
   *  reconnect) doesn't strand the app on a dead producer. Omitted for
   *  daemons too old to advertise it - the embed then uses
   *  `robotPeerId` as-is. */
  robotHardwareId?: string | null;
  /** Optional opaque payload from `?config=<base64>` or from the
   *  mobile-app handoff. App is responsible for parsing /
   *  validating. */
  config: ConfigPayload;
  /** Host display name (e.g. "Reachy Mini") - useful for the
   *  embed if it wants to surface "Connected via …" copy. */
  hostName: string;
  /** Embedded app's display name (passed by the dispatcher). */
  appName: string;
}

/** Theme changed live (user toggled OS palette, host UI switched). */
export interface HostThemeChangedMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'host:theme-changed';
  version: 1;
  theme: ThemeMode;
}

/** Config updated live without an iframe reload. */
export interface HostConfigChangedMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'host:config-changed';
  version: 1;
  config: ConfigPayload;
}

/**
 * Host is asking the app to wind down cleanly. The app SHOULD:
 *   1. Stop emitting motion commands.
 *   2. Disconnect any non-SDK resources (timers, listeners).
 *   3. Resolve every registered `onLeave` callback before the
 *      `timeoutMs` deadline; otherwise the host force-unmounts
 *      the iframe.
 *
 * The host does NOT expect an explicit `leave-ack`; resolving
 * `onLeave` callbacks within the deadline is sufficient.
 */
export interface HostLeavingMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'host:leaving';
  version: 1;
  reason: LeavingReason;
  /** Soft deadline in ms. After this the host unmounts the iframe
   *  regardless. */
  timeoutMs: number;
}

/**
 * Ask the embed to trigger the daemon's PyPI self-update.
 *
 * Only the embed holds a data channel, so the host cannot send
 * `start_update` itself - it delegates here and watches the
 * `embed:update-progress` replies. A successful update ends with a
 * `systemctl restart` on the robot, which kills the session: the embed
 * will go silent and the host is expected to remount the iframe once
 * the robot is back on central.
 *
 * Additive message (no version bump): an embed too old to know this
 * type ignores it, so the host must not block on a reply.
 */
export interface HostStartUpdateMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'host:start-update';
  version: 1;
  /** Install the latest pre-release instead of the latest stable. */
  preRelease?: boolean;
}

/**
 * The host gave up on the update it started (its own stall timer
 * fired while the session was still alive). This does NOT abort the
 * daemon-side job - nothing can, the install owns the robot - it
 * disarms the embed's update-mode plumbing: the sessionStopped →
 * `rebooting` translator and the auto-reconnect stand-down. Without
 * it, the user's next NORMAL end-session would replay a stale
 * `rebooting` frame into a host that already declared the update
 * failed.
 *
 * Additive message (no version bump): an embed too old to know this
 * type ignores it, and its own translator dies with the iframe.
 */
export interface HostCancelUpdateMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'host:cancel-update';
  version: 1;
}

export type HostToEmbedMsg =
  | HostInitMsg
  | HostThemeChangedMsg
  | HostConfigChangedMsg
  | HostLeavingMsg
  | HostStartUpdateMsg
  | HostCancelUpdateMsg;

/* ─────────────────── EMBED → HOST ─────────────────── */

/**
 * Emitted by the app as early as possible (synchronous tick of
 * the embed entry, before the SDK is touched). Tells the host
 * the iframe is alive and ready to receive `host:init`. The host
 * MUST NOT send `host:init` before seeing this.
 */
export interface EmbedReadyMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'embed:ready';
  version: 1;
}

/**
 * App-level connection state. The host renders its
 * `ConnectingView` overlay over the (still-mounted) iframe until
 * it sees `phase === 'live'`. Apps SHOULD emit this on every
 * transition; the host caches the last value and won't re-render
 * unless something changes.
 */
export interface EmbedAppStateMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'embed:app-state';
  version: 1;
  phase: AppPhase;
  /** Sub-step inside `connecting`. Ignored for other phases. */
  connectingStep?: AppConnectingStep | null;
  /** Optional human-readable hint shown in the overlay caption. */
  message?: string | null;
  /**
   * Rolling-min round-trip time (ms) on the embed's own WebRTC
   * candidate pair, sampled from `sdk._pc.getStats()`. Additive
   * field (no version bump): emitted periodically once `phase ===
   * 'live'` so a host shell that has handed its session off to this
   * iframe (mobile app) can still surface a TRUE link latency -
   * the host itself no longer holds a connection to measure.
   *
   * `null` (or omitted) when the platform doesn't expose RTT (iOS
   * WKWebView) or no pair is nominated yet. Hosts that don't care
   * (the standalone web shell, which measures its own link) simply
   * ignore it.
   */
  rttMs?: number | null;
  /**
   * Daemon version reported by the robot (`get_version` over the data
   * channel), resolved once the session is up. Additive field (no
   * version bump).
   *
   * The host has no data channel of its own and central exposes no
   * version, so this is the ONLY way a shell can tell whether the robot
   * it just handed the app to is running current software. The web
   * shell uses it to gate on a minimum supported version; hosts that
   * don't care ignore it.
   *
   * `null` (or omitted) until resolved, or when the daemon predates
   * `get_version`. Receivers MUST treat "unknown" as "fine" - never
   * block a robot just because it stayed silent.
   */
  daemonVersion?: string | null;
  /**
   * Build version of the SDK bundle the app loaded
   * (`ReachyMini.version`, generated at publish time). Additive field
   * (no version bump).
   *
   * The embed rides whatever `window.ReachyMini` the app shipped, so
   * this tells the shell how stale the app's robot stack is - e.g. to
   * warn that behaviour may be off against a newer daemon. `null` (or
   * omitted) when the SDK predates the field - which, unlike
   * `daemonVersion`, is a meaningful signal: every SDK from the release
   * that introduced this field reports it, so silence through a live
   * session means "built before that". Receivers may surface a notice
   * on it, but MUST never block on it.
   */
  sdkVersion?: string | null;
}

/**
 * Progress of a `host:start-update` job, relayed from the SDK's
 * `update_progress` stream one message per log line.
 *
 *   in_progress : a log line of the install.
 *   rebooting   : the WebRTC session died while the update was in
 *                 flight. Synthesised by the embed, NOT by the daemon -
 *                 a successful install restarts the daemon before it
 *                 can report anything, so the transport dropping is the
 *                 success signal. Nothing more will arrive on this
 *                 channel: the host should tear the iframe down and
 *                 wait for the robot to reappear on central.
 *   done        : the daemon explicitly reported completion. Rare (see
 *                 above), but forwarded when it happens.
 *   failed      : the daemon declined the job (not wireless, no update
 *                 available, one already running) or the install raised
 *                 before the restart.
 */
export interface EmbedUpdateProgressMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'embed:update-progress';
  version: 1;
  status: 'in_progress' | 'rebooting' | 'done' | 'failed';
  /** Log line for `in_progress`. */
  line?: string | null;
  /** Reason for `failed`. */
  error?: string | null;
}

/** App requests to leave (user clicked an in-app exit, error,
 *  ...). The host runs the same tear-down as a top-bar
 *  "End session". */
export interface EmbedRequestLeaveMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'embed:request-leave';
  version: 1;
}

/**
 * The embed finished its `host:leaving` tear-down: app `onLeave`
 * callbacks ran AND the SDK's host-owned sleep sequence completed
 * (goto_sleep trajectory + motors disabled). Lets the host unmount as
 * soon as the robot is actually asleep instead of after a fixed
 * worst-case timeout - and, because the WebRTC session is still up when
 * this fires, the motors are already `Disabled` before the app-slot lock
 * frees, so the daemon's idle-reset sees "already asleep" and does NOT
 * replay a second goto_sleep (no double trajectory / go_sleep sound).
 * The host still force-unmounts after its own cap if this never arrives.
 */
export interface EmbedLeftMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'embed:left';
  version: 1;
}

/** App-level error report. `fatal: true` switches the host to
 *  ErrorView; `fatal: false` is logged and may surface a toast. */
export interface EmbedErrorMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'embed:error';
  version: 1;
  message: string;
  fatal: boolean;
  detail?: unknown;
}

export type EmbedToHostMsg =
  | EmbedReadyMsg
  | EmbedAppStateMsg
  | EmbedRequestLeaveMsg
  | EmbedLeftMsg
  | EmbedErrorMsg
  | EmbedUpdateProgressMsg;

/* ─────────────────── CREDS BUNDLE ─────────────────── */

/**
 * Serialised credentials passed from the host (Mode A) or the
 * mobile app (Mode B) to the embed via the URL hash fragment.
 *
 * Hash-only (never URL search): browsers don't send the hash to
 * any server, so the HF token never appears in access logs,
 * referer headers, or HF Spaces frontend logs.
 *
 * The embed wipes the hash with `history.replaceState` on its
 * first synchronous tick, before any `await`, then proceeds with
 * the rest of the boot using the in-memory bundle.
 */
export interface CredsBundle {
  hfToken?: string | null;
  userName?: string | null;
  robotPeerId: string;
  /** Stable hardware id of the selected robot (central
   *  `meta.hardware_id`), when the daemon exposes one. Lets the embed
   *  re-resolve the CURRENT `robotPeerId` from central just before
   *  dialing, so a rotated peer id (long iframe cold-start, relay
   *  reconnect) doesn't strand the app on a dead producer. Omitted for
   *  daemons too old to advertise it - the embed then uses
   *  `robotPeerId` as-is. */
  robotHardwareId?: string | null;
  signalingUrl: string;
  theme: ThemeMode;
  config: ConfigPayload;
  hostName: string;
  appName: string;
}

/* ─────────────────── HELPERS ─────────────────── */

/**
 * Cheap discriminator: does this `unknown` look like a v1
 * envelope from our protocol? Receivers call this before
 * narrowing on `type`. Conservative on the version: an unknown
 * version is treated as "not our protocol" so the receiver
 * silently ignores it (forward-compat for a future v2 peer).
 */
export function isProtocolMessage(
  value: unknown,
): value is { source: typeof PROTOCOL_SOURCE; type: string; version: number } {
  if (!value || typeof value !== 'object') return false;
  const record = value as Record<string, unknown>;
  return (
    record.source === PROTOCOL_SOURCE &&
    typeof record.type === 'string' &&
    record.version === PROTOCOL_VERSION
  );
}

/**
 * Encode a creds bundle to the URL hash fragment.
 * URL-safe base64 wrapper around JSON to avoid percent-encoding
 * noise.
 */
export function encodeCredsToHash(bundle: CredsBundle): string {
  const json = JSON.stringify(bundle);
  const b64 = encodeBase64Utf8(json);
  return `creds=${encodeURIComponent(b64)}`;
}

/**
 * Decode the creds bundle from a URL hash fragment. Returns
 * `null` if no `creds=` segment is present or if the payload is
 * malformed (caller handles the error by rendering ErrorView).
 */
export function decodeCredsFromHash(hash: string | null): CredsBundle | null {
  if (!hash) return null;
  const raw = hash.startsWith('#') ? hash.slice(1) : hash;
  for (const segment of raw.split('&')) {
    if (!segment.startsWith('creds=')) continue;
    try {
      const b64 = decodeURIComponent(segment.slice('creds='.length));
      const json = decodeBase64Utf8(b64);
      return JSON.parse(json) as CredsBundle;
    } catch {
      return null;
    }
  }
  return null;
}

/* UTF-8 safe base64 helpers (btoa / atob choke on non-Latin
 * characters in `config` payloads). Works in browsers and Node
 * (for unit tests). */

function encodeBase64Utf8(input: string): string {
  if (typeof window !== 'undefined' && typeof window.btoa === 'function') {
    const bytes = new TextEncoder().encode(input);
    let bin = '';
    for (const b of bytes) bin += String.fromCharCode(b);
    return window.btoa(bin);
  }
  // Node fallback (tests / SSR).
  return Buffer.from(input, 'utf8').toString('base64');
}

function decodeBase64Utf8(input: string): string {
  if (typeof window !== 'undefined' && typeof window.atob === 'function') {
    const bin = window.atob(input);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder().decode(bytes);
  }
  return Buffer.from(input, 'base64').toString('utf8');
}
