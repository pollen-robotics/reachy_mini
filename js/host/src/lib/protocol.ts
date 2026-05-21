/**
 * postMessage protocol v1 between the host shell (parent window,
 * exposed via `@pollen-robotics/reachy-mini-sdk/host`) and an
 * embedded Reachy Mini app (iframe).
 *
 * Canonical reference: SPEC.md §6.
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
 * - Both sides validate `event.origin` against the expected origin
 *   before trusting the payload. In our deployment (same-origin
 *   iframe within an HF Space) the expected origin is
 *   `event.origin === window.location.origin`.
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
 * Intentionally NOT in v1 (see SPEC §6 and §11):
 * - No `host:custom` / `embed:custom` free-form channel.
 * - No `embed:request-config-update` (apps don't push config).
 * - No heartbeat / ping-pong (backlog).
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
  /** Robot ID selected by the host's picker. */
  robotPeerId: string;
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

export type HostToEmbedMsg =
  | HostInitMsg
  | HostThemeChangedMsg
  | HostConfigChangedMsg
  | HostLeavingMsg;

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
}

/** App requests to leave (user clicked an in-app exit, error,
 *  ...). The host runs the same tear-down as a top-bar
 *  "End session". */
export interface EmbedRequestLeaveMsg {
  source: typeof PROTOCOL_SOURCE;
  type: 'embed:request-leave';
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
  | EmbedErrorMsg;

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
    } catch (err) {
      console.warn('[reachy-mini-sdk/host] failed to decode creds hash', err);
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
