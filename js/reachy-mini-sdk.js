/**
 * reachy-mini.js — Browser SDK for controlling a Reachy Mini robot over WebRTC.
 * https://github.com/pollen-robotics/reachy-mini
 *
 * QUICK START
 * ───────────
 *   import { ReachyMini } from "./reachy-mini-sdk.js";
 *   const robot = new ReachyMini();
 *
 *   // 1. Auth (HuggingFace OAuth — required for the signaling server)
 *   if (!await robot.authenticate()) { robot.login(); return; }
 *
 *   // 2. Connect to signaling server (SSE)
 *   await robot.connect();
 *
 *   // 3. Pick a robot once the list arrives
 *   robot.addEventListener("robotsChanged", (e) => {
 *       const robots = e.detail.robots;  // [{ id, meta: { name } }, ...]
 *   });
 *
 *   // 4. Start a WebRTC session (resolves when video + data channel ready)
 *   const detach = robot.attachVideo(document.querySelector("video"));
 *   await robot.startSession(robotId);
 *
 *   // 5. Send commands — degree-friendly helpers, all built on setTarget()
 *   robot.setHeadRpyDeg(0, 10, -5);   // roll, pitch, yaw in degrees
 *   robot.setAntennasDeg(30, -30);    // right, left in degrees
 *   robot.setBodyYawDeg(15);          // body yaw in degrees
 *
 *   // …or compose an atomic update in raw wire units (full SE(3); no XYZ loss):
 *   robot.setTarget({
 *       head: rpyToMatrix(0, 10, -5).flat(),  // number[16] flat row-major 4×4
 *       antennas: [degToRad(30), degToRad(-30)],
 *       body_yaw: degToRad(15),
 *   });
 *   robot.playSound("wake_up.wav");   // filename on robot
 *   const ver = await robot.getVersion(); // e.g. "1.5.1"
 *
 *   // …or play a recorded move (motion + optional audio) on the daemon's
 *   //  local clock — smooth on wireless robots, single-clock A/V sync.
 *   await robot.playMove(motion, { audioBlob, audioLeadMs: -100 });
 *   //  robot.cancelMove() to stop early.
 *   //
 *   //  For record-time flows that need the same audio pipeline at both
 *   //  capture and replay:
 *   //    const audioId = await robot.uploadAudio(blob);
 *   //    await robot.playUploadedAudio(audioId);   // sync anchor
 *   //    // …start your motion capture now…
 *   //    robot.cancelAudio();                       // when capture stops
 *
 *   // 6. Receive live state (emitted every ~500 ms while streaming; call
 *   //    robot.requestState() yourself for higher rates — see its JSDoc).
 *   //    State payload is the daemon's raw wire shape — use the math
 *   //    utilities exported from this module for degree conversions.
 *   robot.addEventListener("state", (e) => {
 *       const { head, antennas, body_yaw, motor_mode, is_move_running } = e.detail;
 *       // head:            number[16]   — flat row-major 4×4 (full SE(3))
 *       // antennas:        [rightRad, leftRad]
 *       // body_yaw:        number       — radians
 *       // motor_mode:      "enabled" | "disabled" | "gravity_compensation"
 *       // is_move_running: boolean
 *       // For human-friendly head RPY:
 *       //   const rpy = matrixToRpy(head);   // { roll, pitch, yaw } in degrees
 *   });
 *
 *   // 7. Audio controls
 *   robot.setAudioMuted(false);   // unmute robot speaker (muted by default)
 *   robot.setMicMuted(false);     // unmute your mic → robot speaker (if supported)
 *   // XVF3800 audio-board tuning (works on both Lite and Wireless — the
 *   // daemon talks to the audio board over USB on whichever host it runs):
 *   await robot.applyAudioConfig([{ name: "AUDIO_MGR_MIC_GAIN", values: [1.0] }]);
 *   const v = await robot.readAudioParameter("AUDIO_MGR_MIC_GAIN"); // [1.0]
 *
 *   // 8. Cleanup
 *   detach();                      // remove video binding
 *   await robot.stopSession();     // back to 'connected'
 *   robot.disconnect();            // back to 'disconnected' (keeps auth)
 *   robot.logout();                // clear HF credentials too
 *
 *
 * STATE MACHINE
 * ─────────────
 *   'disconnected' ──connect()──▸ 'connected' ──startSession()──▸ 'streaming'
 *        ▴ disconnect()                ▴ stopSession()
 *        └─────────────────────────────┘
 *
 *
 * CONSTRUCTOR OPTIONS
 * ───────────────────
 *   new ReachyMini({
 *     signalingUrl:              string,   // default: "https://pollen-robotics-reachy-mini-central.hf.space"
 *     enableMicrophone:          boolean,  // default: true  — acquire mic for bidirectional audio
 *     videoJitterBufferTargetMs: number,   // default: 0     — receiver-side jitter buffer hint, ms
 *                                          //                  0 = "render ASAP" (teleop). Spec range [0, 4000].
 *                                          //                  Raise (100–400) on flaky links to trade latency for resilience.
 *     autoStartFromUrl:          boolean,  // default: false — when true AND the URL carries a `robot_peer_id` hint,
 *                                          //                  auto-call `startSession(preselectedRobotId)` after
 *                                          //                  `connect()` resolves and that robot appears online.
 *                                          //                  One-shot per page load; suits iframe-embedded apps
 *                                          //                  that want zero-tap entry from the host shell.
 *   })
 *
 *
 * READ-ONLY PROPERTIES
 * ────────────────────
 *   .state            "disconnected" | "connected" | "streaming"
 *   .robots           Array<{ id: string, meta: { name: string } }>
 *   .robotState       Mirror of the latest "state" event detail —
 *                     { head: number[16], antennas: [rightRad, leftRad],
 *                       body_yaw, motor_mode, is_move_running }
 *                     (fields only present once the daemon sends them;
 *                      see EVENTS below)
 *   .username         string | null     — HF username after authenticate()
 *   .isAuthenticated  boolean           — true if a valid HF token is available
 *   .micSupported     boolean           — true if robot offers bidirectional audio
 *   .micMuted         boolean           — your microphone mute state
 *   .audioMuted       boolean           — robot speaker mute state (local)
 *   .preselectedRobotId string | null   — peer id from `?robot_peer_id=` /
 *                                          `#robot_peer_id=`; null if absent.
 *                                          Use it to skip your robot picker
 *                                          when a host iframe (e.g. the
 *                                          Reachy Mini mobile shell) embeds
 *                                          this app.
 *   .isEmbedded       boolean           — true iff `preselectedRobotId !==
 *                                          null`. Branch your UX on this:
 *                                          when true, hide the robot picker
 *                                          and your sign-in screen (the
 *                                          host has already handled both).
 *
 *
 * EVENTS  (EventTarget — use addEventListener)
 * ──────────────────────────────────────────────
 *   "connected"       { peerId: string }
 *   "disconnected"    { reason: string }
 *   "robotsChanged"   { robots: Array<{ id, meta }> }
 *   "streaming"       { sessionId: string, robotId: string }
 *   "sessionStopped"  { reason: string }
 *   "state"           { head: number[16],                    // flat row-major 4×4, when daemon sends head_pose
 *                       antennas: [rightRad, leftRad],       // when daemon sends antennas
 *                       body_yaw: number,                    // radians, when daemon sends body_yaw
 *                       motor_mode: string,                  // when daemon sends motor_mode
 *                       is_move_running: boolean }           // when daemon sends is_move_running
 *   "videoTrack"      { track: MediaStreamTrack, stream: MediaStream }
 *   "micSupported"    { supported: boolean }
 *   "error"           { source: "signaling"|"webrtc"|"robot", error: Error|string }
 *
 *
 * EXPORTS
 * ───────
 *   export default ReachyMini;
 *   export { ReachyMini, rpyToMatrix, matrixToRpy, degToRad, radToDeg };
 */

import {
    oauthHandleRedirectIfPresent,
    oauthLoginUrl,
} from "@huggingface/hub";

// ─── Math utilities ──────────────────────────────────────────────────────────

/** @param {number} deg @returns {number} */
export function degToRad(deg) { return deg * Math.PI / 180; }

/** @param {number} rad @returns {number} */
export function radToDeg(rad) { return rad * 180 / Math.PI; }

/**
 * Roll/pitch/yaw (degrees) → 4×4 rotation matrix (ZYX convention).
 * This is the wire format for the robot's `set_target` command.
 * @param {number} rollDeg  @param {number} pitchDeg  @param {number} yawDeg
 * @returns {number[][]} 4×4 matrix
 */
export function rpyToMatrix(rollDeg, pitchDeg, yawDeg) {
    const r = degToRad(rollDeg), p = degToRad(pitchDeg), y = degToRad(yawDeg);
    const cy = Math.cos(y), sy = Math.sin(y);
    const cp = Math.cos(p), sp = Math.sin(p);
    const cr = Math.cos(r), sr = Math.sin(r);
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, 0],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, 0],
        [-sp, cp * sr, cp * cr, 0],
        [0, 0, 0, 1],
    ];
}

/**
 * Rotation matrix (3×3 or 4×4) → { roll, pitch, yaw } in degrees.
 * @param {number[][]} m  @returns {{ roll: number, pitch: number, yaw: number }}
 */
export function matrixToRpy(m) {
    return {
        roll: radToDeg(Math.atan2(m[2][1], m[2][2])),
        pitch: radToDeg(Math.asin(-m[2][0])),
        yaw: radToDeg(Math.atan2(m[1][0], m[0][0])),
    };
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/** Clamp a volume to [0, 100] and round to integer — mirrors the server-side
 *  Field(..., ge=0, le=100) validator so calling setVolume(150) doesn't 400. */
function clampVolume(v) {
    const n = Math.round(Number(v) || 0);
    return Math.max(0, Math.min(100, n));
}

/**
 * Pick up HuggingFace credentials passed via the URL fragment and move them
 * into `sessionStorage`, where `authenticate()` looks them up.
 *
 * This is the bridge that lets a host page (e.g. the Reachy Mini mobile
 * app, or the vibe-coder preview iframe) embed a Space hosting a SDK
 * consumer despite `X-Frame-Options: SAMEORIGIN` on `huggingface.co/login`:
 * the host already holds a valid token (through its own OAuth flow) and
 * appends it to the iframe URL as
 *
 *     #hf_token=<jwt>&hf_username=<handle>&hf_token_expires=<iso>
 *
 * Fragments are NOT sent over HTTP, so the credentials never leak to
 * the HF Space backend or to intermediate proxies.
 *
 * Why all three keys: `authenticate()`'s cache check requires the token,
 * the username AND a future expiry to ALL be present in `sessionStorage`,
 * otherwise it returns `false` and the app falls through to a full OAuth
 * round-trip — which can't complete inside an iframe.
 *
 * Called once from the top of `authenticate()` so SDK consumers don't
 * need any boilerplate of their own. We clear the fragment right after
 * reading it so a page reload does not keep the credentials visible in
 * the address bar.
 *
 * No-op when:
 *   - there is no `window` (SSR / Worker contexts),
 *   - the URL has no fragment,
 *   - the fragment carries no `hf_token` (other apps may use the
 *     fragment for theme / route / etc.; we leave those alone).
 */
function consumeFragmentCredentials() {
    if (typeof window === 'undefined' || !window.location.hash) return;
    const raw = window.location.hash.startsWith('#')
        ? window.location.hash.slice(1)
        : window.location.hash;
    let params;
    try { params = new URLSearchParams(raw); } catch (_e) { return; }
    const token = params.get('hf_token');
    if (!token) return;
    // `hf_username` is required by the cache check. Hosts that haven't
    // resolved the user's HF handle yet may pass a literal "user"
    // placeholder; the SDK only uses the value for display and never
    // round-trips it server-side, so the placeholder is harmless.
    const username = params.get('hf_username') || 'user';
    // `hf_token_expires` is a far-future ISO date for personal access
    // tokens (no real expiry). Hosts typically synthesise ~1 year out;
    // we accept whatever was sent and fall back to "1 year from now"
    // if the parameter is missing or unparseable, so a partial fragment
    // still gets the user logged in.
    const expiresParam = params.get('hf_token_expires');
    const expires =
        expiresParam && !Number.isNaN(new Date(expiresParam).getTime())
            ? expiresParam
            : new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString();
    try {
        sessionStorage.setItem('hf_token', token);
        sessionStorage.setItem('hf_username', username);
        sessionStorage.setItem('hf_token_expires', expires);
    } catch (err) {
        console.warn('[reachy-mini] could not persist pre-seeded HF credentials:', err);
    }
    // Strip the auth keys from the address bar but keep any other hash
    // params the app or SDK might care about (theme, embedded, …).
    params.delete('hf_token');
    params.delete('hf_username');
    params.delete('hf_token_expires');
    const remaining = params.toString();
    const cleanUrl =
        window.location.pathname +
        window.location.search +
        (remaining ? '#' + remaining : '');
    try { window.history.replaceState(null, '', cleanUrl); } catch (_e) {}
}

/**
 * Pick up a preselected robot peer id from the URL.
 *
 * Looked up in this order:
 *   1. URL fragment   (`#robot_peer_id=<peerId>`)
 *   2. URL query      (`?robot_peer_id=<peerId>`)
 *
 * Both spellings are accepted because:
 *   - the Reachy Mini mobile shell sends it in the query today,
 *   - the vibe-coder preview / future hosts may prefer the fragment for
 *     symmetry with `consumeFragmentCredentials`,
 *   - the value is NOT a secret (peer ids are public on the central
 *     signaling server's robot listing) so query is fine.
 *
 * Returns `null` when no peer id is found in either location, when there
 * is no `window` (SSR / Worker context), or on parse error. Unlike
 * credentials, we do NOT strip the param from the URL: the value is
 * harmless to keep visible and removing it would break tools that read
 * the URL for context.
 *
 * @returns {string|null}
 */
function readPreselectedRobotIdFromUrl() {
    if (typeof window === 'undefined') return null;
    // 1. Fragment (`#robot_peer_id=…`).
    if (window.location.hash) {
        const raw = window.location.hash.startsWith('#')
            ? window.location.hash.slice(1)
            : window.location.hash;
        try {
            const params = new URLSearchParams(raw);
            const fromHash = params.get('robot_peer_id');
            if (fromHash) return fromHash;
        } catch (_e) { /* malformed fragment — fall through */ }
    }
    // 2. Query (`?robot_peer_id=…`).
    if (window.location.search) {
        try {
            const params = new URLSearchParams(window.location.search);
            const fromQuery = params.get('robot_peer_id');
            if (fromQuery) return fromQuery;
        } catch (_e) { /* malformed query — fall through */ }
    }
    return null;
}

/** Check if the audio m= section of an SDP has a=sendrecv (bidirectional audio). */
function sdpHasAudioSendRecv(sdp) {
    const lines = sdp.split('\r\n');
    let inAudio = false;
    for (const line of lines) {
        if (line.startsWith('m=audio')) inAudio = true;
        else if (line.startsWith('m=')) inAudio = false;
        if (inAudio && line === 'a=sendrecv') return true;
    }
    return false;
}

// ─── Daemon-side upload helpers ──────────────────────────────────────────────
// Wire-level constants and pure helpers used by playMove / uploadAudio /
// playUploadedAudio. Private to this module — apps call the public methods.

// Conservative per-message size for the data channel. 16 KB is the cross-
// browser safe ceiling; we slice payloads at 12 KB and let the JSON envelope
// add ~80 bytes.
const UPLOAD_CHUNK_SIZE = 12 * 1024;
// Backpressure thresholds: pause sending if `bufferedAmount` climbs over the
// high watermark; resume once it drains below the low watermark. WebRTC's
// SCTP can buffer plenty, but spiking it to tens of megabytes degrades every
// other channel on the same peer connection.
const UPLOAD_BUFFERED_HIGH_WATER = 1 * 1024 * 1024;
const UPLOAD_BUFFERED_LOW_WATER = 512 * 1024;

function hasCompressionStream() {
    return typeof CompressionStream !== "undefined";
}

/** Cheap unique upload id; collision odds within a session are negligible. */
function makeUploadId() {
    return "u" + Math.random().toString(36).slice(2, 11)
        + Date.now().toString(36);
}

/** Base64-encode a Uint8Array, chunking to avoid call-stack overflow on multi-MB blobs. */
function bytesToBase64(bytes) {
    let str = "";
    const STEP = 0x8000;
    for (let i = 0; i < bytes.length; i += STEP) {
        str += String.fromCharCode.apply(null, bytes.subarray(i, i + STEP));
    }
    return btoa(str);
}

/** Base64(gzip(utf8(s))) via the browser CompressionStream API. */
async function gzipBase64(jsonStr) {
    const enc = new TextEncoder().encode(jsonStr);
    const compressed = await new Response(
        new Blob([enc]).stream().pipeThrough(new CompressionStream("gzip"))
    ).arrayBuffer();
    return bytesToBase64(new Uint8Array(compressed));
}

// ─── ReachyMini class ────────────────────────────────────────────────────────

export class ReachyMini extends EventTarget {

    /** @param {{ signalingUrl?: string, enableMicrophone?: boolean, clientId?: string, appName?: string, videoJitterBufferTargetMs?: number, autoStartFromUrl?: boolean }} [options] */
    constructor(options = {}) {
        super();
        this._signalingUrl = options.signalingUrl || 'https://pollen-robotics-reachy-mini-central.hf.space';
        this._enableMicrophone = options.enableMicrophone !== false;
        this._clientId = options.clientId || null;
        this._appName = options.appName || 'unknown';
        // Hint to the receiver's WebRTC jitter buffer (ms). 0 = "render ASAP",
        // appropriate for teleop. Spec range [0, 4000]. Browsers that don't
        // implement RTCRtpReceiver.jitterBufferTarget fall back to default
        // buffering (~150-200 ms).
        this._videoJitterBufferTargetMs = options.videoJitterBufferTargetMs ?? 0;
        // When true AND the URL carried a `robot_peer_id` hint at
        // construction (so `preselectedRobotId !== null`), the SDK
        // auto-calls `startSession(preselectedRobotId)` as soon as
        // that robot appears in the central's robot list after the
        // app's own `connect()` resolves. Lets host-iframe-embedded
        // consumers (mobile shell, vibe-coder preview) skip their
        // robot picker AND skip the manual `startSession` call —
        // they just `await robot.connect()` and receive a `streaming`
        // event when the SDK has dialed in. One-shot: a manual
        // `stopSession()` followed by another `startSession()` is
        // not auto-replayed. Default `false` keeps the standalone
        // Space behavior unchanged.
        this._autoStartFromUrl = options.autoStartFromUrl === true;
        this._autoStartAttempted = false;

        this._state = 'disconnected';                 // 'disconnected' | 'connected' | 'streaming'
        this._robots = [];                             // latest robot list from signaling
        this._robotState = {};                         // populated from daemon state events (wire shape)

        // Preselected robot peer id read from the URL at construction
        // time. When a host iframe (typically the Reachy Mini mobile
        // shell) embeds an SDK consumer, it appends the peer id of the
        // robot it's already connected to via
        // `?robot_peer_id=…` (or `#robot_peer_id=…`). Apps can read
        // `robot.preselectedRobotId` and call `startSession(id)`
        // directly to skip their robot picker. Captured ONCE at
        // construction so subsequent URL changes (history navigation,
        // hash mutations from `consumeFragmentCredentials`) don't move
        // the target out from under the consumer.
        this._preselectedRobotId = readPreselectedRobotIdFromUrl();

        // Auth
        this._token = null;
        this._username = null;
        this._tokenExpires = null;

        // Signaling
        this._peerId = null;
        this._sseAbortController = null;

        // WebRTC
        this._pc = null;           // RTCPeerConnection
        this._dc = null;           // RTCDataChannel (robot commands)
        this._sessionId = null;
        this._selectedRobotId = null;

        // Audio
        this._micStream = null;    // MediaStream from getUserMedia
        this._micMuted = true;
        this._audioMuted = true;
        this._micSupported = false; // set after SDP negotiation

        // Timers
        this._latencyMonitorId = null;
        this._stateRefreshInterval = null;

        // getVersion() / getHardwareId() promise plumbing
        this._versionResolve = null;
        this._hardwareIdResolve = null;

        // Volume getter/setter promise plumbing (get_volume / set_volume).
        // Speaker and microphone are tracked separately so two in-flight
        // requests can't collide on the same slot.
        this._volumeResolve = null;
        this._micVolumeResolve = null;

        // applyAudioConfig() / readAudioParameter() promise plumbing.
        // Each has its own slot so the two can be in-flight concurrently.
        this._applyAudioConfigResolve = null;
        this._readAudioParameterResolve = null;

        // subscribeLogs(): a Set of {onLine, onError} subscribers. The
        // first add sends `subscribe_logs`; removing the last sends
        // `unsubscribe_logs`. We keep a single daemon-side stream and
        // fan out to local subscribers in `_handleRobotMessage`.
        this._logSubscribers = new Set();

        // Pending one-shot broadcast waiters used by playMove /
        // playUploadedAudio. Each entry is a { predicate, resolve, timer }
        // installed by `_waitForBroadcast`. Dispatched in `_handleRobotMessage`
        // when the daemon broadcasts a matching {type, upload_id, ...}
        // event. LIFO order so the most recently registered waiter wins
        // on duplicates (rare but possible during reconnect).
        this._broadcastWaiters = [];

        // upload_id of the in-flight playMove / playUploadedAudio so
        // the parameter-less `cancelMove()` / `cancelAudio()` calls
        // can target the right run. The daemon now requires upload_id
        // on cancels (otherwise back-to-back plays would cross-cancel).
        // Apps that hold the id themselves can pass it explicitly
        // instead — see the cancelMove / cancelAudio JSDoc.
        this._activeMoveUploadId = null;
        this._activeAudioUploadId = null;

        // startSession() promise plumbing
        this._sessionResolve = null;
        this._sessionReject = null;
        this._iceConnected = false;
        this._dcOpen = false;

        // Motion-completion plumbing for wakeUp() / gotoSleep().
        //
        // The daemon's data-channel handler dispatches `wake_up` and
        // `goto_sleep` as async tasks and replies with
        // `{status: "ok", command, completed: true}` when the trajectory
        // ACTUALLY finishes (or with `{error, command}` on failure). We
        // surface that as a Promise so callers can `await robot.gotoSleep()`
        // and chain `setMotorMode('disabled')` without racing the
        // trajectory player.
        //
        // Queues, not single slots: the data-channel protocol has no
        // request IDs, but FIFO ordering is guaranteed by the daemon's
        // serialised dispatcher, so the N-th response matches the N-th
        // request. A queue makes back-to-back calls (e.g. teardown firing
        // on top of an in-flight wake_up) safe; a single slot would
        // silently drop the earlier awaiter.
        this._pendingMotionCompletions = {
            wake_up: [],
            goto_sleep: [],
        };

        // Set by attachVideo()
        this._videoElement = null;
    }

    // ─── Read-only properties ────────────────────────────────────────────

    /** @returns {"disconnected"|"connected"|"streaming"} */
    get state() { return this._state; }

    /** @returns {Array<{id: string, meta: {name: string}}>} */
    get robots() { return this._robots; }

    /**
     * Latest robot state (same shape as the "state" event detail).
     * Mirrors the daemon's wire format — fields appear only once the
     * daemon has sent the corresponding source field. Use the exported
     * math utilities (``matrixToRpy``, ``radToDeg``) for human units.
     * @returns {{
     *   head?: number[],
     *   antennas?: number[],
     *   body_yaw?: number,
     *   motor_mode?: "enabled"|"disabled"|"gravity_compensation",
     *   is_move_running?: boolean,
     * }}
     */
    get robotState() { return this._robotState; }

    /** @returns {string|null} HuggingFace username, set after authenticate(). */
    get username() { return this._username; }

    /** @returns {boolean} True if a valid HF token is available. */
    get isAuthenticated() { return !!this._token; }

    /** @returns {boolean} True if the robot's SDP offered bidirectional audio. */
    get micSupported() { return this._micSupported; }

    /** @returns {boolean} */
    get micMuted() { return this._micMuted; }

    /** @returns {boolean} */
    get audioMuted() { return this._audioMuted; }

    /**
     * Peer id of the robot the embedding host wants this session to
     * target, captured from the URL at construction time. Apps that
     * want to support iframe-embedding without forcing the user to
     * re-pick a robot read this and pass it straight to
     * `startSession()` once `connect()` resolves:
     *
     *     await robot.connect();
     *     await robot.startSession(robot.preselectedRobotId ?? pickedId);
     *
     * Returns `null` when the URL carries no `robot_peer_id` (typical
     * standalone Space load). The value is also exposed on the
     * "robotsChanged" payload via the `meta` sidecar for
     * convenience, but the most direct read is right here.
     *
     * @returns {string|null}
     */
    get preselectedRobotId() { return this._preselectedRobotId; }

    /**
     * Convenience flag for apps that want to branch their UX on
     * "am I embedded in a host shell?". True iff the URL carried a
     * `robot_peer_id` hint at construction time (which only happens
     * when a host iframe — mobile shell, vibe-coder preview, etc. —
     * is the parent). Apps typically use it to skip their robot
     * picker and their sign-in screen, since both are duplicated
     * work the host has already done.
     *
     * @returns {boolean}
     */
    get isEmbedded() { return this._preselectedRobotId !== null; }

    /**
     * Internal: try to honour the `autoStartFromUrl` constructor
     * option. Called from the signaling-message handler after every
     * `robotsChanged` emit, so a robot that comes online after the
     * SDK is already `connected` still triggers the auto-start.
     * No-op unless `autoStartFromUrl` is set, the URL carries a
     * preselect, the SDK is `connected`, the preselected robot is
     * in the latest list, and we haven't already attempted in this
     * page load. Errors are swallowed to a `console.warn` — the
     * normal `startSession` rejection / `sessionRejected` event
     * still fires for app-level handling.
     *
     * Defers the actual `startSession()` call by one macrotask
     * (`setTimeout(..., 0)`) so it runs OUTSIDE the
     * `_handleSignalingMessage` callstack that just processed the
     * `'list'` message. Reproduced on Android WebView: firing
     * `startSession` synchronously inside the SSE handler races the
     * daemon's setup, leading to a connected-but-no-keyframe state
     * where the receiver eternally NACKs and the iframe shows a
     * black <video>. The macrotask-deferral is the minimum nudge
     * that consistently resolves the race in our reproduction; if
     * it ever proves insufficient on slower hardware, bump to
     * a small explicit delay (e.g. 250 ms).
     */
    _maybeAutoStart() {
        if (!this._autoStartFromUrl) return;
        if (this._autoStartAttempted) return;
        if (!this._preselectedRobotId) return;
        if (this._state !== 'connected') return;
        const match = this._robots.find((r) => r.id === this._preselectedRobotId);
        if (!match) return;
        this._autoStartAttempted = true;
        const peerId = this._preselectedRobotId;
        setTimeout(() => {
            // Re-check state in case a manual stopSession / disconnect
            // landed between the schedule and the fire.
            if (this._state !== 'connected') return;
            this.startSession(peerId).catch((err) => {
                console.warn('[reachy-mini] autoStartFromUrl: startSession rejected:', err);
            });
        }, 0);
    }

    // ─── Auth ────────────────────────────────────────────────────────────

    /**
     * Check for a valid HuggingFace token.
     *
     * Resolution order:
     *   1. URL fragment hand-off (`#hf_token=…&hf_username=…&hf_token_expires=…`).
     *      A host iframe — typically the Reachy Mini mobile app or a
     *      vibe-coder preview — can pass credentials through the URL
     *      fragment to bypass HF's `X-Frame-Options: SAMEORIGIN` block
     *      on `huggingface.co/login`. Seeded into `sessionStorage` and
     *      then stripped from the address bar so a page reload does not
     *      keep the credentials visible.
     *   2. OAuth redirect callback (standalone Space, first sign-in).
     *   3. `sessionStorage` cache (subsequent loads in any context).
     *
     * @returns {Promise<boolean>} true → token ready, false → call login()
     */
    async authenticate() {
        try {
            // 1. Iframe hand-off. No-op when the URL has no fragment or
            //    the fragment carries no `hf_token`, so this is free on
            //    standalone Space loads.
            consumeFragmentCredentials();

            // 2. OAuth redirect callback.
            const result = await oauthHandleRedirectIfPresent();
            if (result) {
                this._username = result.userInfo.preferred_username || result.userInfo.name;
                this._token = result.accessToken;
                this._tokenExpires = result.accessTokenExpiresAt;
                sessionStorage.setItem('hf_token', this._token);
                sessionStorage.setItem('hf_username', this._username);
                sessionStorage.setItem('hf_token_expires', this._tokenExpires);
                return true;
            }

            // 3. sessionStorage cache. Both paths above also write here,
            //    so this is the canonical lookup for any subsequent call.
            const t = sessionStorage.getItem('hf_token');
            const u = sessionStorage.getItem('hf_username');
            const e = sessionStorage.getItem('hf_token_expires');
            if (t && u && e && new Date(e) > new Date()) {
                this._token = t;
                this._username = u;
                this._tokenExpires = e;
                return true;
            }
            return false;
        } catch (e) {
            console.error('Auth error:', e);
            return false;
        }
    }

    /** Redirect the browser to the HuggingFace OAuth login page. */
    async login() {
        const opts = {};
        if (this._clientId) opts.clientId = this._clientId;
        window.location.href = await oauthLoginUrl(opts);
    }

    /** Clear stored HF credentials and disconnect everything. */
    logout() {
        sessionStorage.removeItem('hf_token');
        sessionStorage.removeItem('hf_username');
        sessionStorage.removeItem('hf_token_expires');
        this._username = null;
        this._tokenExpires = null;
        this.disconnect();
    }

    // ─── Lifecycle ───────────────────────────────────────────────────────

    /**
     * Open SSE signaling connection.  Resolves once the server sends `welcome`.
     * Emits "robotsChanged" as robots come and go.
     * @param {string} [token] — HF access token.  Omit to use the one from authenticate().
     * @returns {Promise<void>}
     */
    async connect(token) {
        if (this._state !== 'disconnected') throw new Error('Already connected');
        if (token) this._token = token;
        if (!this._token) throw new Error('No token — call authenticate() first or pass a token');
        this._sseAbortController = new AbortController();

        let res;
        try {
            // Token goes in the Authorization header, not the URL —
            // keeps it out of DevTools Network tab, browser history,
            // Referer, and any server/proxy access log. We use fetch
            // + manual stream reader (below) rather than EventSource
            // specifically to allow custom headers.
            res = await fetch(
                `${this._signalingUrl}/events`,
                {
                    signal: this._sseAbortController.signal,
                    headers: { 'Authorization': `Bearer ${this._token}` },
                },
            );
        } catch (e) {
            this._sseAbortController = null;
            throw e;
        }
        if (!res.ok) {
            this._sseAbortController = null;
            throw new Error(`HTTP ${res.status}`);
        }

        return new Promise((resolve, reject) => {
            let welcomed = false;
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            const readLoop = async () => {
                try {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\n');
                        buffer = lines.pop();
                        for (const line of lines) {
                            if (!line.startsWith('data:')) continue;
                            try {
                                const msg = JSON.parse(line.slice(5).trim());
                                if (!welcomed && msg.type === 'welcome') {
                                    welcomed = true;
                                    this._peerId = msg.peerId;
                                    this._state = 'connected';
                                    await this._sendToServer({
                                        type: 'setPeerStatus',
                                        roles: ['listener'],
                                        meta: { name: this._appName },
                                    });
                                    this._emit('connected', { peerId: msg.peerId });
                                    resolve();
                                }
                                this._handleSignalingMessage(msg);
                            } catch (_) { /* malformed JSON — skip */ }
                        }
                    }
                } catch (e) {
                    if (e.name !== 'AbortError') {
                        this._emit('error', { source: 'signaling', error: e });
                    }
                    if (!welcomed) { reject(e); return; }
                }
                // SSE stream ended (server closed or network drop)
                if (this._state !== 'disconnected') {
                    this._state = 'disconnected';
                    this._emit('disconnected', { reason: 'SSE closed' });
                }
                if (!welcomed) reject(new Error('Connection closed before welcome'));
            };

            readLoop();
        });
    }

    /**
     * One-shot bring-up: auth → SSE connect → robot selection → session →
     * wake up. The all-in-one entry point that captures the common
     * "embed *or* standalone, just get me streaming" flow so each
     * consumer does not have to re-implement it.
     *
     * What it does, in order:
     *   1. **Auth.** If `this._token` is not set, calls `authenticate()`
     *      (which honours the iframe URL-fragment hand-off, the OAuth
     *      redirect callback, and the `sessionStorage` cache). Throws if
     *      none yield a token — the consumer should call `login()` and
     *      retry after the redirect. Pass an explicit `token` to skip
     *      `authenticate()` entirely.
     *   2. **Connect.** If `state === 'disconnected'`, opens the SSE
     *      signaling channel.
     *   3. **Pick a robot.**
     *      - **Embed mode** (`this.isEmbedded`): uses
     *        `this._preselectedRobotId` from the URL. No picker callback
     *        invoked; we briefly wait for that robot to appear in the
     *        SSE list, then proceed.
     *      - **Standalone**: GETs `/api/robot-status` for the owner's
     *        robots with busy state, dedupes by `install_id`, sorts by
     *        freshness. If `autoPickIfSingle` and exactly one free, picks
     *        it. Else calls the consumer-supplied `pickRobot(robots)`
     *        callback. Throws if neither yields an id.
     *   4. **Start session.** Awaits `startSession(robotId)` (ICE + DC).
     *   5. **Wake up.** Awaits `ensureAwake()` so sliders don't silently
     *      no-op against a torque-off robot.
     *
     * @param {{
     *   token?: string,                           // skip authenticate(); use this raw HF token
     *   pickRobot?: (robots: Array<{
     *     id: string,
     *     name: string|null,
     *     busy: boolean,
     *     activeApp: string|null,
     *     meta: object,
     *     lastSeenAgeSeconds: number|null,
     *   }>) => Promise<string|null>,              // called only in standalone, multi-robot case
     *   autoPickIfSingle?: boolean,               // default true — skip the callback when 1 free robot
     *   filterBusy?: boolean,                     // default true — hide busy robots from the picker
     *   wakeOnConnect?: boolean,                  // default true — call ensureAwake() after startSession
     * }} [options]
     * @returns {Promise<{
     *   robotId: string,
     *   robotName: string|null,
     *   isEmbedded: boolean,
     *   alreadyStreaming?: boolean,
     * }>}
     */
    async autoConnect(options = {}) {
        const {
            token = null,
            pickRobot = null,
            autoPickIfSingle = true,
            filterBusy = true,
            wakeOnConnect = true,
        } = options;

        // Idempotent fast-path: caller invoked autoConnect() on an
        // already-streaming session (e.g. on a route change inside an
        // SPA). Return the current selection rather than tearing down.
        if (this._state === 'streaming') {
            const cur = this._robots?.find((r) => r.id === this._selectedRobotId);
            return {
                robotId: this._selectedRobotId,
                robotName: cur?.meta?.name ?? null,
                isEmbedded: this.isEmbedded,
                alreadyStreaming: true,
            };
        }

        // autoConnect takes over the bring-up — disable the SDK's
        // own `autoStartFromUrl` so the two paths don't race and
        // both call `startSession()` against the same preselected
        // robot. The race used to manifest as central rejecting the
        // second attempt with "Robot is busy: <appName>" — the
        // appName being our own first attempt. Restored on the way
        // out so a later `stopSession()` followed by a fresh
        // listener attach still benefits from auto-start.
        const _prevAutoStartFromUrl = this._autoStartFromUrl;
        this._autoStartFromUrl = false;

        try {
        // 1. Auth.
        if (token) {
            this._token = token;
        } else if (!this._token) {
            const ok = await this.authenticate();
            if (!ok) {
                // login() does a full page redirect; we don't trigger
                // it here so the consumer can decide (a desktop tray
                // wants different recovery than a standalone Space).
                throw new Error('Not authenticated — call login() or pass a token');
            }
        }

        // 2. SSE connect.
        if (this._state === 'disconnected') {
            await this.connect();
        }

        // 3. Resolve the target robot.
        let robotId;
        let robotName = null;
        if (this.isEmbedded) {
            robotId = this._preselectedRobotId;
            // Wait briefly for the preselected robot to surface in the
            // SSE list. Best-effort: if it never shows we still try
            // startSession() — central may know about a robot the SSE
            // list pushes only a moment later.
            try {
                await this._waitForRobotInList(robotId, 5000);
            } catch (_) { /* fall through */ }
            const found = this._robots?.find((r) => r.id === robotId);
            robotName = found?.meta?.name ?? null;
        } else {
            const robots = await this._fetchOwnedRobots({ filterBusy });
            if (robots.length === 0) {
                throw new Error('No reachable robots');
            }
            if (autoPickIfSingle && robots.length === 1 && !robots[0].busy) {
                robotId = robots[0].id;
                robotName = robots[0].name;
            } else if (pickRobot) {
                const picked = await pickRobot(robots);
                if (!picked) throw new Error('Robot selection cancelled');
                robotId = picked;
                robotName = robots.find((r) => r.id === picked)?.name ?? null;
            } else {
                throw new Error(
                    'Multiple robots available — pass a pickRobot callback to autoConnect()',
                );
            }
        }

        // 4. Session.
        await this.startSession(robotId);

        // 5. Wake.
        if (wakeOnConnect && typeof this.ensureAwake === 'function') {
            try { await this.ensureAwake(); }
            catch (e) { console.warn('[reachy-mini] autoConnect: ensureAwake failed:', e); }
        }

        return { robotId, robotName, isEmbedded: this.isEmbedded };
        } finally {
            this._autoStartFromUrl = _prevAutoStartFromUrl;
        }
    }

    /**
     * Fetch the caller's robots with busy state, deduped + sorted.
     * One-shot snapshot — no live subscription. Falls back to the SSE
     * `_robots` cache if `/api/robot-status` is unavailable (older
     * central deployments don't expose it).
     *
     * Dedup: same physical robot can appear twice transiently after a
     * daemon reinstall (new peerId, same install_id). Last-writer-wins
     * on `install_id`, then `hardware_id`, then `peerId` (= no dedup).
     *
     * @returns {Promise<Array<{
     *   id: string,
     *   name: string|null,
     *   busy: boolean,
     *   activeApp: string|null,
     *   meta: object,
     *   lastSeenAgeSeconds: number|null,
     * }>>}
     */
    async _fetchOwnedRobots({ filterBusy = true } = {}) {
        try {
            const res = await fetch(`${this._signalingUrl}/api/robot-status`, {
                headers: { 'Authorization': `Bearer ${this._token}` },
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const json = await res.json();
            const seen = new Map();  // dedup key → projected robot
            for (const r of (json.robots || [])) {
                if (filterBusy && r.busy) continue;
                const key = r.meta?.install_id ?? r.meta?.hardware_id ?? r.peerId;
                seen.set(key, {
                    id: r.peerId,
                    name: r.robotName ?? r.meta?.name ?? null,
                    busy: !!r.busy,
                    activeApp: r.activeApp ?? null,
                    meta: r.meta ?? {},
                    lastSeenAgeSeconds: r.last_seen_age_seconds ?? null,
                });
            }
            return Array.from(seen.values()).sort(
                (a, b) => (a.lastSeenAgeSeconds ?? Infinity) - (b.lastSeenAgeSeconds ?? Infinity),
            );
        } catch (e) {
            console.warn('[reachy-mini] /api/robot-status unavailable, using SSE list:', e);
            return (this._robots || []).map((r) => ({
                id: r.id,
                name: r.meta?.name ?? null,
                busy: false,             // unknown — SSE list does not carry busy state
                activeApp: null,
                meta: r.meta ?? {},
                lastSeenAgeSeconds: null,
            }));
        }
    }

    /**
     * Resolve once `robotId` appears in `_robots`, or reject after
     * `timeoutMs`. Used by `autoConnect()`'s embed branch so the preselected
     * robot has a chance to surface from the first SSE `list` push before
     * `startSession()` is fired.
     */
    _waitForRobotInList(robotId, timeoutMs) {
        if (this._robots?.find((r) => r.id === robotId)) return Promise.resolve();
        return new Promise((resolve, reject) => {
            const onChange = () => {
                if (this._robots?.find((r) => r.id === robotId)) {
                    this.removeEventListener('robotsChanged', onChange);
                    clearTimeout(timeoutId);
                    resolve();
                }
            };
            const timeoutId = setTimeout(() => {
                this.removeEventListener('robotsChanged', onChange);
                reject(new Error(`Timeout waiting for robot ${robotId} in list`));
            }, timeoutMs);
            this.addEventListener('robotsChanged', onChange);
        });
    }

    /**
     * Start a WebRTC session with the given robot.
     * Acquires the microphone (if enabled), negotiates SDP, and waits for
     * both ICE connection and data channel to be ready before resolving.
     * Emits "videoTrack" when the robot's camera stream arrives.
     * Emits "micSupported" once SDP negotiation reveals whether the robot
     * accepts bidirectional audio.
     * @param {string} robotId — one of the ids from the robots list
     * @returns {Promise<void>}
     */
    async startSession(robotId) {
        if (this._state !== 'connected') throw new Error('Not connected');
        this._selectedRobotId = robotId;
        this._iceConnected = false;
        this._dcOpen = false;
        this._micSupported = false;
        // Buffer for ICE candidates that arrive before the SDP
        // exchange completes (see _handlePeerMessage). Reset on every
        // fresh session so stale candidates from a previous attempt
        // don't get applied to a new RTCPeerConnection.
        this._pendingRemoteIce = [];

        // Acquire mic eagerly so the browser permission prompt appears now,
        // but tracks stay disabled (muted) until the user explicitly unmutes.
        if (this._enableMicrophone) {
            try {
                this._micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this._micStream.getAudioTracks().forEach(t => { t.enabled = false; });
                this._micMuted = true;
            } catch (e) {
                console.warn('Microphone not available:', e);
                // Fall back to a silent placeholder track. We MUST add an
                // audio track before createAnswer or the answer SDP comes
                // back as recvonly for audio - which negotiates the audio
                // SENDER side off the wire entirely. A host that wants to
                // later inject a different audio source (e.g. a synthesised
                // AI voice via replaceTrack on the sender) needs a live
                // sendrecv slot, even if the initial track is silent.
                try {
                    const ctx = new (window.AudioContext || window.webkitAudioContext)();
                    const dst = ctx.createMediaStreamDestination();
                    // A muted oscillator keeps the track "alive" without
                    // emitting any audible signal.
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    gain.gain.value = 0;
                    osc.connect(gain).connect(dst);
                    osc.start();
                    this._micStream = dst.stream;
                    this._micStream.getAudioTracks().forEach(t => { t.enabled = false; });
                    this._micMuted = true;
                    this._silentMicFallback = { ctx, osc };
                } catch (fallbackErr) {
                    console.warn('Silent mic fallback failed:', fallbackErr);
                    this._micStream = null;
                }
            }
        }

        this._pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
        });

        return new Promise((resolve, reject) => {
            this._sessionResolve = resolve;
            this._sessionReject = reject;

            this._pc.ontrack = (e) => {
                if (e.track.kind === 'video') {
                    // Tell the receiver's jitter buffer to minimise its hold
                    // time. Both properties target the same internal buffer;
                    // browsers ignore whichever they don't implement.
                    const ms = this._videoJitterBufferTargetMs;
                    try { e.receiver.jitterBufferTarget = ms; } catch (_) {}
                    try { e.receiver.playoutDelayHint = ms / 1000; } catch (_) {}
                    this._emit('videoTrack', { track: e.track, stream: e.streams[0] });
                }
            };

            this._pc.onicecandidate = async (e) => {
                if (e.candidate && this._sessionId) {
                    await this._sendToServer({
                        type: 'peer',
                        sessionId: this._sessionId,
                        ice: {
                            candidate: e.candidate.candidate,
                            sdpMLineIndex: e.candidate.sdpMLineIndex,
                            sdpMid: e.candidate.sdpMid,
                        },
                    });
                }
            };

            this._pc.oniceconnectionstatechange = () => {
                const s = this._pc?.iceConnectionState;
                if (!s) return;
                if (s === 'connected' || s === 'completed') {
                    this._iceConnected = true;
                    this._checkSessionReady();
                } else if (s === 'failed') {
                    const err = new Error('ICE connection failed');
                    if (this._sessionReject) {
                        this._sessionReject(err);
                        this._sessionResolve = null;
                        this._sessionReject = null;
                    }
                    this._emit('error', { source: 'webrtc', error: err });
                } else if (s === 'disconnected') {
                    this._emit('error', { source: 'webrtc', error: new Error('ICE disconnected') });
                }
            };

            this._pc.ondatachannel = (e) => {
                this._dc = e.channel;
                this._dc.onopen = () => {
                    this._dcOpen = true;
                    this._checkSessionReady();
                };
                this._dc.onmessage = (ev) => this._handleRobotMessage(JSON.parse(ev.data));
            };

            this._sendToServer({ type: 'startSession', peerId: robotId }).then((r) => {
                if (r?.type === 'sessionRejected') {
                    this._failSessionRejected(r);
                    return;
                }
                if (r?.sessionId) this._sessionId = r.sessionId;
            });
        });
    }

    /**
     * Internal: handle a sessionRejected response from central.
     * Releases resources allocated by startSession() (RTCPeerConnection,
     * microphone stream) and rejects the pending startSession() promise
     * with an Error carrying `.reason` and `.activeApp`.
     *
     * Called from both the POST-response path (primary) and the SSE
     * handler (defensive, in case the server changes).
     */
    _failSessionRejected(msg) {
        const err = new Error(
            msg.reason === 'robot_busy'
                ? `Robot is busy: "${msg.activeApp || 'another app'}" is already connected`
                : `Session rejected: ${msg.reason || 'unknown reason'}`
        );
        err.reason = msg.reason;
        err.activeApp = msg.activeApp;

        // Release resources allocated optimistically in startSession()
        // before we knew the server would refuse.
        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
        this._iceConnected = false;
        this._dcOpen = false;
        this._micMuted = true;
        this._micSupported = false;

        this._emit('sessionRejected', { reason: msg.reason, activeApp: msg.activeApp });

        if (this._sessionReject) {
            const reject = this._sessionReject;
            this._sessionResolve = null;
            this._sessionReject = null;
            reject(err);
        }
    }

    /**
     * End the WebRTC session.  Returns to "connected" state so you can
     * startSession() again with the same or a different robot.
     * @returns {Promise<void>}
     */
    async stopSession() {
        if (this._versionResolve) { this._versionResolve(null); this._versionResolve = null; }
        if (this._hardwareIdResolve) { this._hardwareIdResolve(null); this._hardwareIdResolve = null; }
        if (this._volumeResolve) { this._volumeResolve(null); this._volumeResolve = null; }
        if (this._micVolumeResolve) { this._micVolumeResolve(null); this._micVolumeResolve = null; }
        if (this._applyAudioConfigResolve) { this._applyAudioConfigResolve(false); this._applyAudioConfigResolve = null; }
        if (this._readAudioParameterResolve) { this._readAudioParameterResolve(null); this._readAudioParameterResolve = null; }
        // Drop any active log subscribers — the daemon-side subprocess
        // is torn down on peer-disconnect, so resubscribing across a
        // reconnect requires a fresh subscribeLogs() call from the
        // consumer.
        this._logSubscribers.clear();
        // Drain any in-flight wakeUp() / gotoSleep() awaiters before
        // the data channel is killed below, so callers don't sit on a
        // promise that can never resolve.
        this._rejectPendingMotionCompletions(new Error('Session stopped'));
        if (this._sessionReject) {
            this._sessionReject(new Error('Session stopped'));
            this._sessionResolve = null;
            this._sessionReject = null;
        }

        if (this._stateRefreshInterval) { clearInterval(this._stateRefreshInterval); this._stateRefreshInterval = null; }
        if (this._latencyMonitorId) { clearInterval(this._latencyMonitorId); this._latencyMonitorId = null; }

        if (this._sessionId) {
            await this._sendToServer({ type: 'endSession', sessionId: this._sessionId });
        }

        if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
        this._micMuted = true;
        this._micSupported = false;

        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._dc) { this._dc.close(); this._dc = null; }

        this._sessionId = null;
        this._iceConnected = false;
        this._dcOpen = false;

        const wasStreaming = this._state === 'streaming';
        if (wasStreaming) {
            this._state = 'connected';
            this._emit('sessionStopped', { reason: 'user' });
        }
    }

    /**
     * Full teardown — abort SSE, close WebRTC.
     * Auth state is preserved (call logout() to also clear credentials).
     */
    disconnect() {
        if (this._sseAbortController) { this._sseAbortController.abort(); this._sseAbortController = null; }

        if (this._versionResolve) { this._versionResolve(null); this._versionResolve = null; }
        if (this._hardwareIdResolve) { this._hardwareIdResolve(null); this._hardwareIdResolve = null; }
        if (this._volumeResolve) { this._volumeResolve(null); this._volumeResolve = null; }
        if (this._micVolumeResolve) { this._micVolumeResolve(null); this._micVolumeResolve = null; }
        if (this._applyAudioConfigResolve) { this._applyAudioConfigResolve(false); this._applyAudioConfigResolve = null; }
        if (this._readAudioParameterResolve) { this._readAudioParameterResolve(null); this._readAudioParameterResolve = null; }
        this._logSubscribers.clear();
        // Same rationale as in stopSession(): drain pending motion
        // awaiters before tearing down the data channel.
        this._rejectPendingMotionCompletions(new Error('Disconnected'));
        if (this._sessionReject) {
            this._sessionReject(new Error('Disconnected'));
            this._sessionResolve = null;
            this._sessionReject = null;
        }

        if (this._stateRefreshInterval) { clearInterval(this._stateRefreshInterval); this._stateRefreshInterval = null; }
        if (this._latencyMonitorId) { clearInterval(this._latencyMonitorId); this._latencyMonitorId = null; }

        if (this._sessionId && this._token) {
            this._sendToServer({ type: 'endSession', sessionId: this._sessionId }); // fire-and-forget
        }

        if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
        if (this._pc) { this._pc.close(); this._pc = null; }
        if (this._dc) { this._dc.close(); this._dc = null; }

        this._sessionId = null;
        this._micMuted = true;
        this._micSupported = false;
        this._iceConnected = false;
        this._dcOpen = false;
        this._peerId = null;
        this._robots = [];
        this._state = 'disconnected';
        this._emit('disconnected', { reason: 'user' });
    }

    // ─── Commands ────────────────────────────────────────────────────────
    // All return false if the data channel is not open, true if sent.

    /**
     * Send a target pose to the robot. Wire-shape, raw units only —
     * single source of truth for motion commands. Every field is
     * optional; omitted fields leave the daemon's previous target
     * unchanged, so partial updates compose naturally.
     *
     * For human units (degrees), use the ``setHeadRpyDeg`` /
     * ``setAntennasDeg`` / ``setBodyYawDeg`` thin wrappers below.
     *
     * @param {object} [target]
     * @param {number[]} [target.head] 16-element flat row-major 4×4
     *   matrix (full SE(3); preserves translation, no XYZ loss).
     * @param {number[]} [target.antennas] ``[rightRad, leftRad]``.
     * @param {number} [target.body_yaw] Body yaw in radians.
     * @returns {boolean} false if the data channel is not open.
     * @throws {TypeError} if any provided field has the wrong shape or
     *   contains a non-finite value (NaN, Infinity). Validation runs at
     *   the JS boundary so caller mistakes surface with a stack trace
     *   pointing to the call site, not as a confusing daemon-side error.
     */
    setTarget({ head, antennas, body_yaw } = {}) {
        const cmd = { type: "set_full_target" };
        if (head !== undefined) {
            if (!Array.isArray(head) || head.length !== 16
                || !head.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'setTarget: head must be a 16-element flat row-major 4×4 matrix '
                    + `of finite numbers; got ${Array.isArray(head) ? `Array(${head.length})` : typeof head}`
                );
            }
            cmd.head = head;
        }
        if (antennas !== undefined) {
            if (!Array.isArray(antennas) || antennas.length !== 2
                || !antennas.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'setTarget: antennas must be [rightRad, leftRad] (2 finite numbers); '
                    + `got ${Array.isArray(antennas) ? `Array(${antennas.length})` : typeof antennas}`
                );
            }
            cmd.antennas = antennas;
        }
        if (body_yaw !== undefined) {
            if (!Number.isFinite(body_yaw)) {
                throw new TypeError(
                    `setTarget: body_yaw must be a finite number (radians); got ${body_yaw}`
                );
            }
            cmd.body_yaw = body_yaw;
        }
        return this._sendCommand(cmd);
    }

    /**
     * Smooth daemon-side interpolation to a target pose over
     * ``duration`` seconds. Mirrors ``setTarget``'s wire shape (head
     * is a 16-element flat row-major 4×4, antennas are
     * ``[rightRad, leftRad]``, body_yaw is radians) and adds a
     * required ``duration`` field. The daemon dispatches the command
     * to its lerp planner instead of jumping to the target.
     *
     * Use this for one-shot smooth approaches to an arbitrary pose
     * (e.g. soft-return-to-base after recording, or pre-positioning
     * before a streamed playback). For continuous streamed motion,
     * use ``setTarget`` and lerp client-side.
     *
     * @param {{head?: number[], antennas?: number[], body_yaw?: number, duration: number}} args
     * @returns {boolean} false if the data channel is not open.
     * @throws {TypeError} if any provided field has the wrong shape
     *   or contains a non-finite value (NaN, Infinity), or if
     *   ``duration`` is missing or non-positive.
     */
    gotoTarget({ head, antennas, body_yaw, duration } = {}) {
        const cmd = { type: "goto_target" };
        if (head !== undefined) {
            if (!Array.isArray(head) || head.length !== 16
                || !head.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'gotoTarget: head must be a 16-element flat row-major 4×4 matrix '
                    + `of finite numbers; got ${Array.isArray(head) ? `Array(${head.length})` : typeof head}`
                );
            }
            cmd.head = head;
        }
        if (antennas !== undefined) {
            if (!Array.isArray(antennas) || antennas.length !== 2
                || !antennas.every((n) => Number.isFinite(n))) {
                throw new TypeError(
                    'gotoTarget: antennas must be [rightRad, leftRad] (2 finite numbers); '
                    + `got ${Array.isArray(antennas) ? `Array(${antennas.length})` : typeof antennas}`
                );
            }
            cmd.antennas = antennas;
        }
        if (body_yaw !== undefined) {
            if (!Number.isFinite(body_yaw)) {
                throw new TypeError(
                    `gotoTarget: body_yaw must be a finite number (radians); got ${body_yaw}`
                );
            }
            cmd.body_yaw = body_yaw;
        }
        if (!Number.isFinite(duration) || duration <= 0) {
            throw new TypeError(
                `gotoTarget: duration must be a positive finite number (seconds); got ${duration}`
            );
        }
        cmd.duration = duration;
        return this._sendCommand(cmd);
    }

    /**
     * Set head orientation from roll/pitch/yaw in degrees.
     * Convenience wrapper over ``setTarget``.
     * @param {number} rollDeg @param {number} pitchDeg @param {number} yawDeg
     * @returns {boolean}
     */
    setHeadRpyDeg(rollDeg, pitchDeg, yawDeg) {
        return this.setTarget({ head: rpyToMatrix(rollDeg, pitchDeg, yawDeg).flat() });
    }

    /**
     * Set antenna positions from degrees.
     * Convenience wrapper over ``setTarget``.
     * @param {number} rightDeg @param {number} leftDeg
     * @returns {boolean}
     */
    setAntennasDeg(rightDeg, leftDeg) {
        return this.setTarget({ antennas: [degToRad(rightDeg), degToRad(leftDeg)] });
    }

    /**
     * Set body yaw from degrees.
     * Convenience wrapper over ``setTarget``.
     * @param {number} yawDeg
     * @returns {boolean}
     */
    setBodyYawDeg(yawDeg) {
        return this.setTarget({ body_yaw: degToRad(yawDeg) });
    }

    /**
     * Play a sound file on the robot.
     * @param {string} file — filename available on the robot (e.g. "wake_up.wav")
     * @returns {boolean}
     */
    playSound(file) {
        return this._sendCommand({ type: "play_sound", file });
    }

    /**
     * Set the motor control mode.
     *
     * @param {"enabled"|"disabled"|"gravity_compensation"} mode
     *   - "enabled"              torque on, position-controlled.
     *   - "disabled"             torque off; the robot is backdrivable
     *                            and will not hold any pose.
     *   - "gravity_compensation" torque on in current-control mode;
     *                            motors actively cancel gravity so the
     *                            robot is easy to move by hand.
     * @returns {boolean} false if the data channel is not open.
     */
    setMotorMode(mode) {
        return this._sendCommand({ type: "set_motor_mode", mode });
    }

    /**
     * Toggle torque on/off, optionally per-motor.
     *
     * @param {boolean} on
     * @param {string[]} [ids]  motor names (e.g. ["left_antenna"]). When
     *   omitted, applies globally — equivalent to setMotorMode("enabled"
     *   | "disabled").
     * @returns {boolean} false if the data channel is not open.
     */
    setMotorTorque(on, ids = null) {
        return this._sendCommand({ type: "set_torque", on, ids });
    }

    /**
     * Play the wake-up animation (full head/antennas trajectory on the
     * robot, ~1-3 s depending on the starting head pose) and resolve
     * when the daemon reports the trajectory player has actually
     * finished.
     *
     * This helper sends a ``set_motor_mode: "enabled"`` command *before*
     * the ``wake_up`` command so the animation actually moves the motors.
     * The robot's ``wake_up`` handler does not touch motor mode itself;
     * if torque is off when the trajectory runs, the commanded positions
     * are silently ignored and the robot stays limp. Both commands
     * travel over the same data channel so ordering at the backend is
     * preserved.
     *
     * The returned promise resolves on the daemon's
     * ``{command: "wake_up", completed: true}`` response (sent after
     * the trajectory player is fully done, not just when the command
     * is enqueued). Lets a UI overlay (e.g. the host's "Wake-up" step)
     * stay up for exactly the right duration, and lets callers chain
     * setup that depends on the head being in the awake pose without
     * racing the trajectory.
     *
     * Semantics match the REST endpoint ``POST /api/move/play/wake_up``
     * plus the LAN convention of enabling motors before playing motion
     * trajectories.
     *
     * @param {object} [options]
     * @param {number} [options.timeoutMs=8000] hard upper bound; the
     *   promise rejects with a TimeoutError-shaped Error if the daemon
     *   stops responding (e.g. data channel went down mid-animation
     *   without firing close events).
     * @returns {Promise<void>}
     */
    wakeUp({ timeoutMs = 8000 } = {}) {
        this._sendCommand({ type: "set_motor_mode", mode: "enabled" });
        return this._sendCommandAwaitCompletion("wake_up", timeoutMs);
    }

    /**
     * Play the goto-sleep animation and resolve when the daemon reports
     * the trajectory player has finished. See ``wakeUp`` for the
     * completion-signal rationale.
     *
     * Does NOT touch motor mode: the daemon's ``goto_sleep`` handler
     * manages the transition out of torque on its own (motors must stay
     * powered during the trajectory to move into the sleep pose, then
     * are typically disabled by the daemon once the pose is reached).
     *
     * The awaitable form lets callers chain ``setMotorMode('disabled')``
     * AFTER the trajectory lands instead of racing it, which previously
     * caused the head to drop mid-animation when consumers tore down
     * too eagerly.
     *
     * Semantics match ``POST /api/move/play/goto_sleep`` and the
     * ``"goto_sleep"`` WebRTC command.
     *
     * @param {object} [options]
     * @param {number} [options.timeoutMs=8000]
     * @returns {Promise<void>}
     */
    gotoSleep({ timeoutMs = 8000 } = {}) {
        return this._sendCommandAwaitCompletion("goto_sleep", timeoutMs);
    }

    /**
     * Internal: send a motion command and resolve when the daemon's
     * matching ``{command, completed: true}`` response lands.
     *
     * Pushes one entry onto ``_pendingMotionCompletions[command]``; the
     * data-channel reader (``_handleRobotMessage``) shifts the oldest
     * entry off the queue when a response arrives, which preserves the
     * FIFO matching that the daemon's serialised dispatcher relies on.
     *
     * Rejects immediately if the data channel is not open; the underlying
     * ``_sendCommand`` returns false in that case and we never enqueue an
     * awaiter that the daemon could never reach.
     *
     * @param {"wake_up"|"goto_sleep"} command
     * @param {number} timeoutMs
     * @returns {Promise<void>}
     */
    _sendCommandAwaitCompletion(command, timeoutMs) {
        if (!this._sendCommand({ type: command })) {
            return Promise.reject(new Error(`${command}: data channel not open`));
        }
        return new Promise((resolve, reject) => {
            const entry = {
                resolve,
                reject,
                timer: setTimeout(() => {
                    const queue = this._pendingMotionCompletions[command];
                    const idx = queue.indexOf(entry);
                    if (idx !== -1) queue.splice(idx, 1);
                    reject(new Error(`${command} timed out after ${timeoutMs}ms`));
                }, timeoutMs),
            };
            this._pendingMotionCompletions[command].push(entry);
        });
    }

    /**
     * Internal: drain every pending motion-completion resolver with the
     * given error. Called by ``stopSession()`` and ``disconnect()`` so a
     * teardown that interrupts an in-flight ``gotoSleep`` does not leave
     * the caller awaiting forever.
     */
    _rejectPendingMotionCompletions(error) {
        for (const command of Object.keys(this._pendingMotionCompletions)) {
            const queue = this._pendingMotionCompletions[command];
            while (queue.length) {
                const entry = queue.shift();
                clearTimeout(entry.timer);
                entry.reject(error);
            }
        }
    }

    /**
     * Whether the robot's motors are currently powered (the "awake" state).
     *
     * Reads ``motor_mode`` from the last state event. Both ``"enabled"``
     * and ``"gravity_compensation"`` count as awake: in gravity-comp the
     * motors are actively holding the arm against gravity, so the robot
     * is *not* limp and playing wake_up on top would fight the user.
     * Only ``"disabled"`` (true sleep) is considered not-awake.
     *
     * Returns ``false`` before the first state event arrives (typical
     * right after ``startSession()``). Use ``ensureAwake()`` if you want
     * to wait for the first state before deciding.
     *
     * @returns {boolean}
     */
    isAwake() {
        const mode = this._robotState?.motor_mode;
        return mode === "enabled" || mode === "gravity_compensation";
    }

    /**
     * Wake the robot up if it is currently asleep, otherwise no-op.
     *
     * Intended as the first line of any app after ``startSession()``
     * resolves — robots are often left in the sleep pose (torque off,
     * head resting on the base) and commanded positions are silently
     * ignored in that state.
     *
     * If no state event has arrived yet, waits up to ``timeoutMs`` for
     * one before deciding. If still no state, falls back to sending
     * ``wakeUp()`` (safe: the daemon's wake_up handler is idempotent
     * at the motion level — it moves to the awake pose from wherever
     * the head currently is).
     *
     * @param {number} [timeoutMs=1000] how long to wait for the first
     *   state event before falling through to wakeUp().
     * @returns {Promise<boolean>} true if the robot is awake afterwards.
     */
    async ensureAwake(timeoutMs = 1000) {
        if (this._robotState?.motor_mode === undefined) {
            await new Promise((resolve) => {
                const done = () => {
                    this.removeEventListener('state', done);
                    clearTimeout(timer);
                    resolve();
                };
                const timer = setTimeout(done, timeoutMs);
                this.addEventListener('state', done);
                this.requestState();
            });
        }
        if (this.isAwake()) return true;
        // wakeUp() now returns a Promise. Fire-and-forget here — we
        // intentionally do not await the trajectory completion, the
        // caller of ensureAwake() decides whether to block on the
        // animation. Catch the rejection so a teardown that interrupts
        // the wake doesn't surface an unhandledrejection event from
        // this internal helper.
        this.wakeUp().catch(() => { /* swallow: caller may have torn down */ });
        return true;
    }

    /**
     * Request the daemon version.
     * Resolves with the version string (or null if unavailable).
     * @returns {Promise<string|null>}
     */
    getVersion() {
        return new Promise((resolve, reject) => {
            if (!this._dc || this._dc.readyState !== 'open') {
                reject(new Error('Data channel not open'));
                return;
            }
            if (this._versionResolve) {
                this._versionResolve(null);
            }
            this._versionResolve = resolve;
            this._sendCommand({ type: "get_version" });
        });
    }

    /**
     * Request the robot's unique hardware ID — the Pollen audio device's
     * USB serial. Same value across Lite and Wireless variants, stable
     * across reboots and OS reinstalls. Useful for fleet management,
     * per-robot calibration cache keys, or identifying which physical
     * robot a session is bound to.
     * Resolves with the hardware ID string (or null if no robot is
     * attached, e.g. the daemon is running on a developer machine).
     * @returns {Promise<string|null>}
     */
    getHardwareId() {
        return new Promise((resolve, reject) => {
            if (!this._dc || this._dc.readyState !== 'open') {
                reject(new Error('Data channel not open'));
                return;
            }
            if (this._hardwareIdResolve) {
                this._hardwareIdResolve(null);
            }
            this._hardwareIdResolve = resolve;
            this._sendCommand({ type: "get_hardware_id" });
        });
    }

    /**
     * Query the current speaker volume (0-100).
     * Resolves with null if volume control is unavailable (platform unsupported
     * or audio stack down).
     * @returns {Promise<number|null>}
     */
    getVolume() {
        return this._volumeRoundtrip({ type: "get_volume" }, "_volumeResolve");
    }

    /**
     * Set the speaker volume (0-100). Persists for the next connection
     * (same semantics as the REST /api/volume/set endpoint).
     * Resolves with the applied volume, or null on failure.
     * @param {number} volume 0-100
     * @returns {Promise<number|null>}
     */
    setVolume(volume) {
        return this._volumeRoundtrip(
            { type: "set_volume", volume: clampVolume(volume) },
            "_volumeResolve",
        );
    }

    /**
     * Query the current microphone input volume (0-100).
     * @returns {Promise<number|null>}
     */
    getMicrophoneVolume() {
        return this._volumeRoundtrip(
            { type: "get_microphone_volume" },
            "_micVolumeResolve",
        );
    }

    /**
     * Set the microphone input volume (0-100). Persists across sessions.
     * @param {number} volume 0-100
     * @returns {Promise<number|null>}
     */
    setMicrophoneVolume(volume) {
        return this._volumeRoundtrip(
            { type: "set_microphone_volume", volume: clampVolume(volume) },
            "_micVolumeResolve",
        );
    }

    /**
     * Apply a batch of XVF3800 audio-board parameters on the robot.
     * Mirrors the on-robot `AudioBase.apply_audio_config()` SDK call.
     *
     * @param {Array<{name: string, values: number[]}>} config Parameter
     *        names and values to write (see `audio_control_utils.PARAMETERS`).
     * @param {{verify?: boolean}} [opts] When verify is true (default),
     *        each parameter is read back after writing.
     * @returns {Promise<boolean>} True iff every parameter was written
     *        and (when verify=true) read back successfully. Resolves false
     *        if the audio board is unavailable.
     */
    applyAudioConfig(config, { verify = true } = {}) {
        return this._volumeRoundtrip(
            { type: "apply_audio_config", config, verify },
            "_applyAudioConfigResolve",
        );
    }

    /**
     * Read a single XVF3800 parameter by name.
     * Mirrors the on-robot `ReSpeaker.read_values()` SDK call.
     *
     * @param {string} name Parameter name (see `PARAMETERS` catalog).
     * @returns {Promise<number[]|null>} Decoded numeric values, or null
     *        if the parameter is unknown / unreadable / the audio board
     *        is unavailable.
     */
    readAudioParameter(name) {
        return this._volumeRoundtrip(
            { type: "read_audio_parameter", name },
            "_readAudioParameterResolve",
        );
    }

    /**
     * Internal: send a volume command and await the single-slot response.
     * The slot name selects which resolver (speaker vs mic) owns the
     * pending request so the two can be in-flight concurrently without
     * collision.
     */
    _volumeRoundtrip(command, slot) {
        return new Promise((resolve, reject) => {
            if (!this._dc || this._dc.readyState !== 'open') {
                reject(new Error('Data channel not open'));
                return;
            }
            // If a previous request on the same slot is still pending,
            // resolve it to null so its caller doesn't hang forever.
            if (this[slot]) this[slot](null);
            this[slot] = resolve;
            this._sendCommand(command);
        });
    }

    /**
     * Send an arbitrary JSON command over the data channel.
     * @param {object} data  @returns {boolean}
     */
    sendRaw(data) {
        return this._sendCommand(data);
    }

    /**
     * Subscribe to the daemon's `journalctl -u reachy-mini-daemon`
     * stream over the WebRTC data channel.
     *
     * One daemon-side subprocess is shared across all local subscribers:
     * the first call sends `subscribe_logs`, removing the last subscriber
     * sends `unsubscribe_logs`. Calling the returned `unsubscribe()`
     * twice is a no-op.
     *
     * @param {{
     *   onLine: (entry: { timestamp: string, line: string }) => void,
     *   onError?: (error: string) => void,
     * }} options
     * @returns {() => void} unsubscribe
     */
    subscribeLogs({ onLine, onError } = {}) {
        if (typeof onLine !== 'function') {
            throw new TypeError('subscribeLogs: onLine callback is required');
        }
        const sub = { onLine, onError };
        const wasEmpty = this._logSubscribers.size === 0;
        this._logSubscribers.add(sub);
        if (wasEmpty) this._sendCommand({ type: 'subscribe_logs' });

        let detached = false;
        return () => {
            if (detached) return;
            detached = true;
            this._logSubscribers.delete(sub);
            if (this._logSubscribers.size === 0) {
                this._sendCommand({ type: 'unsubscribe_logs' });
            }
        };
    }

    /**
     * Request a state snapshot.  The response arrives as a "state" event.
     * Called automatically every 500 ms while streaming.
     *
     * Safe to call at a higher rate if you need faster telemetry: e.g.
     * ``setInterval(() => robot.requestState(), 20)`` for ~50 Hz, or drive
     * it from a ``requestAnimationFrame`` loop for display-rate updates.
     * On LAN the daemon can sustain ~90-100 Hz round-trips over the
     * datachannel; over the internet expect the WebRTC path's RTT to
     * dominate. The built-in 500 ms poll keeps running in parallel — it
     * is harmless, as state responses are idempotent.
     *
     * @returns {boolean}
     */
    requestState() {
        return this._sendCommand({ type: "get_state" });
    }

    // ─── Audio ───────────────────────────────────────────────────────────

    /**
     * Mute/unmute the robot's audio playback (speaker) locally.
     * Audio is muted by default — browsers require a user gesture to unmute.
     * @param {boolean} muted
     */
    setAudioMuted(muted) {
        this._audioMuted = muted;
        if (this._videoElement) this._videoElement.muted = muted;
    }

    /**
     * Mute/unmute your microphone.  Only works if micSupported is true.
     * Mic is muted by default even after acquisition.
     * @param {boolean} muted
     */
    setMicMuted(muted) {
        this._micMuted = muted;
        if (this._micStream) {
            this._micStream.getAudioTracks().forEach(t => { t.enabled = !muted; });
        }
    }

    // ─── Video helper ────────────────────────────────────────────────────

    /**
     * Bind a `<video>` element to this robot's stream.
     * Call before startSession().  Sets srcObject when the video track arrives,
     * applies audio mute state, and runs a latency monitor that snaps to the
     * live edge if the buffer grows > 0.5 s.
     *
     * @param {HTMLVideoElement} videoElement
     * @returns {() => void} cleanup function — call to detach video and stop monitoring
     */
    attachVideo(videoElement) {
        this._videoElement = videoElement;
        videoElement.muted = this._audioMuted;

        const onVideoTrack = (e) => {
            videoElement.srcObject = e.detail.stream;
            videoElement.playsInline = true;
            if ('requestVideoFrameCallback' in videoElement) {
                this._startLatencyMonitor(videoElement);
            }
        };

        const onSessionStopped = () => { videoElement.srcObject = null; };

        this.addEventListener('videoTrack', onVideoTrack);
        this.addEventListener('sessionStopped', onSessionStopped);

        return () => {
            this.removeEventListener('videoTrack', onVideoTrack);
            this.removeEventListener('sessionStopped', onSessionStopped);
            if (this._latencyMonitorId) { clearInterval(this._latencyMonitorId); this._latencyMonitorId = null; }
            videoElement.srcObject = null;
            this._videoElement = null;
        };
    }

    // ─── Daemon-side recorded-move playback ──────────────────────────────
    //
    // These methods talk to the daemon's `feature/daemon-side-move-upload`
    // protocol: motion (and optional audio) are uploaded once over the
    // data channel, then the daemon's play_move loop runs at the requested
    // frequency server-side — no per-frame WebRTC round-trip, smooth on
    // wireless robots. Audio, when present, plays on the same daemon-side
    // GStreamer pipeline so motion and sound share a single clock.
    //
    // For record-time flows that need the SAME audio pipeline at capture
    // and replay (so pipeline latency cancels), use uploadAudio +
    // playUploadedAudio to play audio standalone with a sync anchor.

    /**
     * Upload a recorded move (and optionally its audio) and play it on
     * the daemon's local clock. Resolves when playback ends with the
     * daemon's final broadcast — `{finished: true}`, `{cancelled: true}`,
     * or `{error: string}`.
     *
     * `audioLeadMs` shifts audio relative to motion:
     *  - Positive: audio fires N ms BEFORE motion (compensates motor pickup).
     *  - Negative: motion fires N ms BEFORE audio (compensates pipeline warmup).
     *  - Default `-100` is the empirically-measured system-wide constant
     *    (combined motor + GStreamer playbin warmup); tune per setup only
     *    if you've measured a different value.
     *
     * @param {{ time: number[], set_target_data: object[] }} motion
     * @param {object} [opts]
     * @param {Blob}    [opts.audioBlob]            - canonical 16 kHz mono PCM WAV
     * @param {number}  [opts.audioLeadMs=-100]
     * @param {string}  [opts.description="move"]
     * @param {"gzip+base64"|"json"} [opts.encoding="gzip+base64"]
     * @param {number}  [opts.playFrequency=100]
     * @param {number}  [opts.initialGotoDuration=0]
     * @param {number}  [opts.startTimeoutMs=8000]
     * @param {(p: { phase: string, sent?: number, total?: number,
     *               bytes?: number, encoding?: string,
     *               duration_s?: number }) => void} [opts.onProgress]
     * @param {(s: { duration_s: number, has_audio: boolean }) => void} [opts.onStarted]
     * @returns {Promise<{ finished?: boolean, cancelled?: boolean,
     *                     error?: string, has_audio?: boolean }>}
     */
    async playMove(motion, {
        audioBlob = null,
        audioLeadMs = -100,
        description = "move",
        encoding = "gzip+base64",
        playFrequency = 100,
        initialGotoDuration = 0,
        startTimeoutMs = 8000,
        onProgress = () => {},
        onStarted = () => {},
    } = {}) {
        if (!this._dc || this._dc.readyState !== "open") {
            throw new Error("data channel not open");
        }
        if (!motion?.time?.length || !motion?.set_target_data?.length) {
            throw new Error("playMove: motion must have time + set_target_data");
        }
        const uploadId = makeUploadId();
        // Publish the id so `cancelMove()` without args targets this
        // run. Cleared in finally to avoid stale cancels biting the
        // next run.
        this._activeMoveUploadId = uploadId;

        // Encode the move payload. gzip+base64 typically compresses
        // recorded-move JSON ~3× thanks to repeated float patterns;
        // falls back to plain JSON if CompressionStream is missing.
        const moveDict = {
            description,
            time: motion.time,
            set_target_data: motion.set_target_data,
        };
        const jsonStr = JSON.stringify(moveDict);
        let payload;
        let effectiveEncoding;
        if (encoding === "gzip+base64" && hasCompressionStream()) {
            payload = await gzipBase64(jsonStr);
            effectiveEncoding = "gzip+base64";
        } else {
            payload = jsonStr;
            effectiveEncoding = "json";
        }
        const totalChunks = Math.ceil(payload.length / UPLOAD_CHUNK_SIZE) || 1;

        onProgress({
            phase: "starting",
            sent: 0,
            total: totalChunks,
            bytes: payload.length,
            encoding: effectiveEncoding,
        });

        // 1. Open the move slot.
        this._sendCommand({
            type: "upload_move_start",
            upload_id: uploadId,
            total_chunks: totalChunks,
            description,
            encoding: effectiveEncoding,
        });
        // 2. Pipeline motion chunks. No per-chunk acks; pace on
        //    bufferedAmount so a long song doesn't blow up the channel.
        for (let i = 0; i < totalChunks; i++) {
            if (this._dc.bufferedAmount > UPLOAD_BUFFERED_HIGH_WATER) {
                await this._awaitDataChannelDrain();
            }
            const start = i * UPLOAD_CHUNK_SIZE;
            this._sendCommand({
                type: "upload_move_chunk",
                upload_id: uploadId,
                chunk_index: i,
                chunk: payload.slice(start, start + UPLOAD_CHUNK_SIZE),
            });
            onProgress({ phase: "upload", sent: i + 1, total: totalChunks });
        }
        // 3. Close the slot. Daemon parses synchronously.
        this._sendCommand({ type: "upload_move_finish", upload_id: uploadId });
        onProgress({ phase: "uploaded", sent: totalChunks, total: totalChunks });

        // 3b. Optional audio: pipelined under the SAME upload_id so the
        //     daemon pairs it with the move at play time. Raw WAV bytes
        //     are base64-encoded (no gzip; PCM compresses poorly).
        if (audioBlob) {
            const rawBytes = new Uint8Array(await audioBlob.arrayBuffer());
            const audioB64 = bytesToBase64(rawBytes);
            const audioTotal = Math.ceil(audioB64.length / UPLOAD_CHUNK_SIZE) || 1;
            onProgress({
                phase: "audio-starting",
                sent: 0,
                total: audioTotal,
                bytes: audioB64.length,
            });
            this._sendCommand({
                type: "upload_audio_start",
                upload_id: uploadId,
                total_chunks: audioTotal,
                encoding: "wav-base64",
                description,
            });
            for (let i = 0; i < audioTotal; i++) {
                if (this._dc.bufferedAmount > UPLOAD_BUFFERED_HIGH_WATER) {
                    await this._awaitDataChannelDrain();
                }
                const start = i * UPLOAD_CHUNK_SIZE;
                this._sendCommand({
                    type: "upload_audio_chunk",
                    upload_id: uploadId,
                    chunk_index: i,
                    chunk: audioB64.slice(start, start + UPLOAD_CHUNK_SIZE),
                });
                onProgress({ phase: "audio-upload", sent: i + 1, total: audioTotal });
            }
            this._sendCommand({ type: "upload_audio_finish", upload_id: uploadId });
            onProgress({ phase: "audio-uploaded", sent: audioTotal, total: audioTotal });
        }

        // 4. Trigger playback; await the daemon's "started" broadcast.
        this._sendCommand({
            type: "play_uploaded_move",
            upload_id: uploadId,
            play_frequency: playFrequency,
            initial_goto_duration: initialGotoDuration,
            audio_lead_ms: audioLeadMs,
        });
        let startedAck;
        try {
            startedAck = await this._waitForBroadcast(
                (m) =>
                    m?.type === "play_uploaded_move"
                    && m?.upload_id === uploadId
                    && (m.started === true || typeof m.error === "string"),
                { timeoutMs: startTimeoutMs, debugLabel: "play_uploaded_move started" },
            );
        } catch (e) {
            throw new Error(
                "Daemon did not respond to play_uploaded_move "
                + "(requires the reachy_mini daemon with feature/daemon-side-move-upload). "
                + `Underlying: ${e.message}`,
            );
        }
        if (typeof startedAck.error === "string") {
            throw new Error(`play_uploaded_move: ${startedAck.error}`);
        }
        try {
            onStarted({
                duration_s: startedAck.duration_s,
                has_audio: startedAck.has_audio === true,
            });
        } catch (e) {
            // onStarted is user code — never let it abort playback.
            console.warn("playMove.onStarted threw:", e);
        }
        onProgress({ phase: "playing", duration_s: startedAck.duration_s });

        // 5. Wait for the final broadcast.
        const final = await this._waitForBroadcast(
            (m) =>
                m?.type === "play_uploaded_move"
                && m?.upload_id === uploadId
                && (m.finished === true
                    || m.cancelled === true
                    || typeof m.error === "string"),
            {
                timeoutMs: (startedAck.duration_s + 30) * 1000,
                debugLabel: "play_uploaded_move final",
            },
        );
        // Release the "current move id" pointer so the next no-arg
        // cancelMove() doesn't target an already-ended run. Guarded
        // against the unlikely case that a concurrent playMove has
        // already overwritten it.
        if (this._activeMoveUploadId === uploadId) {
            this._activeMoveUploadId = null;
        }
        return final;
    }

    /**
     * Cancel an in-flight `playMove`. Fire-and-forget; the daemon
     * broadcasts the cancelled event which `playMove` resolves with.
     *
     * Pass `uploadId` explicitly to target a specific run; defaults to
     * the most recent in-flight `playMove`. The daemon now scopes
     * cancels by upload_id so two back-to-back plays can't cross-cancel
     * each other — a cancel with no live target is a no-op.
     *
     * @param {string} [uploadId] - optional; defaults to the active playMove id
     * @returns {boolean} false if the data channel isn't open or no run to target
     */
    cancelMove(uploadId = null) {
        const id = uploadId ?? this._activeMoveUploadId;
        if (!id) return false;
        return this._sendCommand({ type: "cancel_move", upload_id: id });
    }

    /**
     * Upload audio to the daemon as a standalone slot (no motion attached).
     * Used by recording flows that want the SAME audio pipeline at record
     * time and play time — pipeline latency cancels, so a single per-system
     * `audioLeadMs` is enough for sync.
     *
     * @param {Blob} audioBlob - canonical 16 kHz mono PCM WAV
     * @param {object} [opts]
     * @param {string} [opts.description="audio"]
     * @param {(p: { phase: string, sent?: number, total?: number,
     *               bytes?: number }) => void} [opts.onProgress]
     * @returns {Promise<string>} uploadId — pair with playUploadedAudio
     */
    async uploadAudio(audioBlob, { description = "audio", onProgress = () => {} } = {}) {
        if (!this._dc || this._dc.readyState !== "open") {
            throw new Error("data channel not open");
        }
        if (!(audioBlob instanceof Blob)) {
            throw new TypeError("uploadAudio: expected a Blob");
        }
        const uploadId = makeUploadId();
        const rawBytes = new Uint8Array(await audioBlob.arrayBuffer());
        const audioB64 = bytesToBase64(rawBytes);
        const total = Math.ceil(audioB64.length / UPLOAD_CHUNK_SIZE) || 1;
        onProgress({ phase: "audio-starting", sent: 0, total, bytes: audioB64.length });
        this._sendCommand({
            type: "upload_audio_start",
            upload_id: uploadId,
            total_chunks: total,
            encoding: "wav-base64",
            description,
        });
        for (let i = 0; i < total; i++) {
            if (this._dc.bufferedAmount > UPLOAD_BUFFERED_HIGH_WATER) {
                await this._awaitDataChannelDrain();
            }
            const start = i * UPLOAD_CHUNK_SIZE;
            this._sendCommand({
                type: "upload_audio_chunk",
                upload_id: uploadId,
                chunk_index: i,
                chunk: audioB64.slice(start, start + UPLOAD_CHUNK_SIZE),
            });
            onProgress({ phase: "audio-upload", sent: i + 1, total });
        }
        this._sendCommand({ type: "upload_audio_finish", upload_id: uploadId });
        onProgress({ phase: "audio-uploaded", sent: total, total });
        return uploadId;
    }

    /**
     * Trigger daemon-side playback of a previously-uploaded audio.
     * Resolves when the daemon broadcasts the "started" event — this is
     * the sync anchor callers use as t=0 for related capture.
     *
     * The daemon does NOT emit a finished event for standalone audio;
     * callers know the duration from the WAV header and send
     * `cancelAudio()` when they're done (e.g. recording stopped).
     *
     * @param {string} uploadId
     * @param {object} [opts]
     * @param {number} [opts.timeoutMs=8000]
     * @returns {Promise<{ started: true }>}
     */
    async playUploadedAudio(uploadId, { timeoutMs = 8000 } = {}) {
        if (!this._dc || this._dc.readyState !== "open") {
            throw new Error("data channel not open");
        }
        const waiter = this._waitForBroadcast(
            (m) =>
                m?.type === "play_uploaded_audio"
                && m?.upload_id === uploadId
                && (m.started === true || typeof m.error === "string"),
            { timeoutMs, debugLabel: "play_uploaded_audio started" },
        );
        this._sendCommand({ type: "play_uploaded_audio", upload_id: uploadId });
        const ack = await waiter;
        if (typeof ack.error === "string") throw new Error(ack.error);
        // Publish the id so a no-arg `cancelAudio()` targets this run.
        // The daemon has no "audio ended" event, so the id stays set
        // until either cancelAudio() is called or playUploadedAudio()
        // is called again with a different id.
        this._activeAudioUploadId = uploadId;
        return ack;
    }

    /**
     * Cancel an in-flight `playUploadedAudio`. Fire-and-forget.
     *
     * Pass `uploadId` explicitly to target a specific run; defaults
     * to the most recent `playUploadedAudio`. The daemon scopes
     * cancels by upload_id so a stale cancel won't kill the audio
     * attached to a concurrently-running `playMove`.
     *
     * @param {string} [uploadId] - optional; defaults to the active playUploadedAudio id
     * @returns {boolean} false if the data channel isn't open or no run to target
     */
    cancelAudio(uploadId = null) {
        const id = uploadId ?? this._activeAudioUploadId;
        if (!id) return false;
        if (this._activeAudioUploadId === id) {
            this._activeAudioUploadId = null;
        }
        return this._sendCommand({ type: "cancel_audio", upload_id: id });
    }

    // ─── Private ─────────────────────────────────────────────────────────

    _emit(name, detail) {
        this.dispatchEvent(new CustomEvent(name, { detail }));
    }

    /**
     * Register a one-shot waiter for a daemon broadcast event. Resolves
     * with the matching payload, rejects on `timeoutMs`. Used internally
     * by `playMove` / `playUploadedAudio`.
     */
    _waitForBroadcast(predicate, { timeoutMs = 5000, debugLabel = "" } = {}) {
        return new Promise((resolve, reject) => {
            const slot = { predicate, resolve };
            slot.timer = setTimeout(() => {
                const i = this._broadcastWaiters.indexOf(slot);
                if (i !== -1) this._broadcastWaiters.splice(i, 1);
                reject(new Error(`broadcast timeout (${timeoutMs} ms): ${debugLabel}`));
            }, timeoutMs);
            this._broadcastWaiters.push(slot);
        });
    }

    /**
     * Wait until `_dc.bufferedAmount` drops below the low watermark. Polls
     * at ~30 ms; browsers don't expose a uniform event for SCTP data channels
     * (`onbufferedamountlow` is patchy across engines).
     */
    async _awaitDataChannelDrain() {
        while (this._dc && this._dc.bufferedAmount > UPLOAD_BUFFERED_LOW_WATER) {
            await new Promise((r) => setTimeout(r, 30));
            if (!this._dc || this._dc.readyState !== "open") {
                throw new Error("data channel closed mid-upload");
            }
        }
    }

    async _sendToServer(message) {
        // Mirrors connect()'s guard — a missing token between connect and
        // send (e.g. logout mid-session) would otherwise produce
        // "Authorization: Bearer undefined", which central correctly 401s
        // but silently returns null to the caller, hiding the real cause.
        if (!this._token) throw new Error('No token — authenticate() first');
        try {
            // Token in Authorization header, not URL — same reasoning as
            // connect()'s SSE fetch: never put secrets in URLs.
            const res = await fetch(`${this._signalingUrl}/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this._token}`,
                },
                body: JSON.stringify(message),
            });
            if (!res.ok) {
                // Surface 4xx/5xx with the rejected message type. The
                // browser already logs "Failed to load resource: <status>"
                // but never says which call produced it, which makes
                // tardy `peer`/`endSession`/`setPeerStatus` races
                // (typical after a session has been torn down) hard to
                // diagnose. Returning null preserves the historical
                // contract for callers that only care about the success
                // path.
                let body = '';
                try { body = await res.text(); } catch { /* ignore */ }
                console.warn(
                    `[reachy-mini] /send rejected (${res.status}) for type=${message?.type}; body=${body || '<empty>'}`,
                );
                return null;
            }
            return await res.json();
        } catch (e) {
            console.error('Send error:', e);
            return null;
        }
    }

    _sendCommand(cmd) {
        if (!this._dc || this._dc.readyState !== 'open') return false;
        this._dc.send(JSON.stringify(cmd));
        return true;
    }

    /** Resolves the startSession() promise once both ICE and datachannel are ready. */
    _checkSessionReady() {
        if (this._iceConnected && this._dcOpen && this._sessionResolve) {
            this._state = 'streaming';
            this.requestState();
            this._stateRefreshInterval = setInterval(() => this.requestState(), 500);
            this._emit('streaming', { sessionId: this._sessionId, robotId: this._selectedRobotId });
            this._sessionResolve();
            this._sessionResolve = null;
            this._sessionReject = null;
        }
    }

    async _handleSignalingMessage(msg) {
        switch (msg.type) {
            case 'welcome':
                break; // handled in connect()
            case 'list':
                this._robots = msg.producers || [];
                this._emit('robotsChanged', { robots: this._robots });
                this._maybeAutoStart();
                break;
            case 'peerStatusChanged': {
                const list = await this._sendToServer({ type: 'list' });
                if (list?.producers) {
                    this._robots = list.producers;
                    this._emit('robotsChanged', { robots: this._robots });
                    this._maybeAutoStart();
                }
                break;
            }
            case 'sessionStarted':
                this._sessionId = msg.sessionId;
                break;
            case 'sessionRejected':
                // Defensive: the current central server returns sessionRejected
                // as the direct POST response (handled at the startSession() call
                // site). This branch is a safety net in case the server ever
                // pushes the rejection over SSE instead.
                this._failSessionRejected(msg);
                break;
            case 'endSession':
                this._handleEndSession(msg);
                break;
            case 'peer':
                this._handlePeerMessage(msg);
                break;
        }
    }

    /**
     * Internal: handle an endSession pushed from central.
     *
     * Two user-visible scenarios converge here:
     *
     * 1. Pending startSession — central accepted our request but the
     *    robot-side relay then refused (e.g. a local Python app holds
     *    the daemon's robot lock). Without this handler, the client's
     *    startSession() promise would hang forever because ICE/datachannel
     *    never come up.
     * 2. Streaming was evicted — a local Python app started on the robot
     *    while we were connected, so the relay tore down our session.
     *
     * The ``reason`` is forwarded verbatim from the relay through central:
     * - "robot_busy_local_app": daemon lock held by a local Python app
     *   (refused before any media negotiation).
     * - "local_app_started": a local Python app started mid-session and
     *   evicted us.
     * - "robot_busy_local": relay's safety-net for stale/concurrent
     *   sessions (should be rare).
     */
    _handleEndSession(msg) {
        const reason = msg.reason;
        const friendly = reason === 'robot_busy_local_app'
            ? 'Robot is busy: a local Python app is running'
            : reason === 'local_app_started'
                ? 'Disconnected: a local Python app started on the robot'
                : reason === 'robot_busy_local'
                    ? 'Robot is busy: another session is already active'
                    : null;

        // Case 1: a startSession() is still pending — reject its promise
        // with the same error shape as sessionRejected so app code can
        // treat both paths identically.
        if (this._sessionReject) {
            const err = new Error(
                friendly || `Session ended before it could start: ${reason || 'unknown reason'}`
            );
            err.reason = reason;
            this._emit('sessionRejected', { reason, activeApp: null });
            // Release resources allocated optimistically by startSession().
            if (this._pc) { this._pc.close(); this._pc = null; }
            if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
            this._iceConnected = false;
            this._dcOpen = false;
            this._micMuted = true;
            this._micSupported = false;
            const reject = this._sessionReject;
            this._sessionResolve = null;
            this._sessionReject = null;
            reject(err);
            return;
        }

        // Case 2: we were streaming and got kicked. Leave cleanup of _pc,
        // _dc, mic and timers to stopSession() so the event listener path
        // matches the user-initiated stop path exactly.
        if (this._state === 'streaming') {
            this._emit('sessionStopped', {
                reason: reason || 'remote_end',
                message: friendly,
            });
            // Fire-and-forget: we just react to what the server already
            // decided. stopSession() sends its own endSession back but
            // central has already dropped the session, so the echo is
            // harmless.
            this.stopSession().catch(() => { });
        }
    }

    async _handlePeerMessage(msg) {
        if (!this._pc) return;
        try {
            if (msg.sdp) {
                const sdp = msg.sdp;
                if (sdp.type === 'offer') {
                    const supportsMic = sdpHasAudioSendRecv(sdp.sdp);
                    this._micSupported = supportsMic;
                    this._emit('micSupported', { supported: supportsMic });

                    // Mic track must be added BEFORE setRemoteDescription so the
                    // generated answer naturally includes sendrecv for audio.
                    if (supportsMic && this._micStream) {
                        for (const track of this._micStream.getAudioTracks()) {
                            this._pc.addTrack(track, this._micStream);
                        }
                    }

                    await this._pc.setRemoteDescription(new RTCSessionDescription(sdp));
                    const answer = await this._pc.createAnswer();
                    await this._pc.setLocalDescription(answer);
                    await this._sendToServer({
                        type: 'peer',
                        sessionId: this._sessionId,
                        sdp: { type: 'answer', sdp: answer.sdp },
                    });
                } else {
                    await this._pc.setRemoteDescription(new RTCSessionDescription(sdp));
                }
                // Replay any ICE candidates that arrived before the
                // SDP exchange completed (see the buffering branch
                // below for context).
                const pending = this._pendingRemoteIce;
                if (pending && pending.length) {
                    this._pendingRemoteIce = [];
                    for (const ice of pending) {
                        try {
                            await this._pc.addIceCandidate(new RTCIceCandidate(ice));
                        } catch (err) {
                            console.warn('[reachy-mini] buffered ICE candidate rejected:', err);
                        }
                    }
                }
            }
            if (msg.ice) {
                // Safari (and the iOS WKWebView Tauri ships on) rejects
                // empty candidate strings with `OperationError: Expect
                // line: candidate:<candidate-str>`. The signaling
                // server uses an empty string as the end-of-candidates
                // marker (legal per the WebRTC spec but optional).
                // Chrome / Firefox swallow it silently; we mirror that
                // here so the iOS WebView stops surfacing the noise as
                // a robot-side WebRTC error event.
                if (!msg.ice.candidate) return;
                if (this._pc.remoteDescription) {
                    await this._pc.addIceCandidate(new RTCIceCandidate(msg.ice));
                } else {
                    // The signaling transport (SSE through central) is
                    // not strictly ordered across the offer / ICE
                    // streams when the SDK runs inside a cross-origin
                    // iframe or in Safari / iOS WKWebView: the first
                    // ICE candidates can land before the offer SDP
                    // does. Calling addIceCandidate before
                    // setRemoteDescription throws
                    // `InvalidStateError: The remote description was
                    // null` and the candidate is silently lost,
                    // sometimes wedging ICE altogether. Buffer here
                    // and replay above as soon as the offer has been
                    // applied.
                    if (!this._pendingRemoteIce) this._pendingRemoteIce = [];
                    this._pendingRemoteIce.push(msg.ice);
                }
            }
        } catch (e) {
            console.error('WebRTC error:', e);
            this._emit('error', { source: 'webrtc', error: e });
        }
    }

    /** Parse robot messages and dispatch. */
    _handleRobotMessage(data) {
        if ('version' in data && this._versionResolve) {
            this._versionResolve(data.version);
            this._versionResolve = null;
            return;
        }
        if ('hardware_id' in data && this._hardwareIdResolve) {
            this._hardwareIdResolve(data.hardware_id);
            this._hardwareIdResolve = null;
            return;
        }
        // Volume responses. Backend tags each response with `command` so we
        // know which pending resolver (speaker vs mic) to fulfil. If a
        // response arrives with no matching pending request (e.g. stale
        // after reconnect), just ignore it — the data channel protocol
        // has no multiplexing, so unmatched replies are expected.
        if (data.command === 'get_volume' || data.command === 'set_volume') {
            if (this._volumeResolve) {
                this._volumeResolve(data.status === 'error' ? null : data.volume);
                this._volumeResolve = null;
            }
            return;
        }
        if (data.command === 'get_microphone_volume' || data.command === 'set_microphone_volume') {
            if (this._micVolumeResolve) {
                this._micVolumeResolve(data.status === 'error' ? null : data.volume);
                this._micVolumeResolve = null;
            }
            return;
        }
        if (data.command === 'apply_audio_config') {
            if (this._applyAudioConfigResolve) {
                this._applyAudioConfigResolve(data.error ? false : !!data.applied);
                this._applyAudioConfigResolve = null;
            }
            return;
        }
        if (data.command === 'read_audio_parameter') {
            if (this._readAudioParameterResolve) {
                this._readAudioParameterResolve(data.error ? null : (data.values ?? null));
                this._readAudioParameterResolve = null;
            }
            return;
        }
        // Motion completion responses. The daemon emits
        // `{status: "ok", command: "wake_up"|"goto_sleep", completed: true}`
        // after the trajectory player is fully done, or `{error, command}`
        // on failure. Route them to the FIFO queue of pending awaiters
        // populated by `_sendCommandAwaitCompletion`; the N-th response
        // matches the N-th request thanks to the daemon's serialised
        // dispatcher.
        if (
            (data.command === 'wake_up' || data.command === 'goto_sleep') &&
            this._pendingMotionCompletions &&
            this._pendingMotionCompletions[data.command]
        ) {
            const queue = this._pendingMotionCompletions[data.command];
            if (data.completed === true && queue.length > 0) {
                const entry = queue.shift();
                clearTimeout(entry.timer);
                entry.resolve();
                return;
            }
            if (data.error && queue.length > 0) {
                const entry = queue.shift();
                clearTimeout(entry.timer);
                entry.reject(new Error(`${data.command}: ${data.error}`));
                return;
            }
        }
        if (data.type === 'log_line') {
            for (const sub of this._logSubscribers) {
                try {
                    sub.onLine({ timestamp: data.timestamp, line: data.line });
                } catch (e) {
                    console.error('subscribeLogs onLine threw:', e);
                }
            }
            return;
        }
        if (data.type === 'log_stream_error') {
            for (const sub of this._logSubscribers) {
                if (typeof sub.onError === 'function') {
                    try { sub.onError(data.error); }
                    catch (e) { console.error('subscribeLogs onError threw:', e); }
                }
            }
            return;
        }
        if (data.state) {
            const s = data.state;
            // Wire-shape pass-through. The daemon ships the head pose as a
            // nested 4×4 (numpy tolist()); we flatten to 16 numbers so
            // consumers can hand it straight to WebGL / Three.js / trajectory
            // logs. Everything else is forwarded as-is.
            if (s.head_pose) this._robotState.head = s.head_pose.flat();
            if (s.antennas) this._robotState.antennas = [s.antennas[0], s.antennas[1]];
            if (typeof s.body_yaw === 'number') this._robotState.body_yaw = s.body_yaw;
            if (s.motor_mode) this._robotState.motor_mode = s.motor_mode;
            if (typeof s.is_move_running === 'boolean') this._robotState.is_move_running = s.is_move_running;
            this._emit('state', { ...this._robotState });
        }
        if (data.error) {
            this._emit('error', { source: 'robot', error: data.error });
        }
        // Daemon-side upload/play broadcasts: dispatch to any waiter
        // whose predicate matches. Iterating in reverse keeps the
        // newest registration first (FIFO across same-predicate
        // duplicates would yield the wrong upload on a stale resend).
        if (this._broadcastWaiters.length > 0) {
            for (let i = this._broadcastWaiters.length - 1; i >= 0; i--) {
                const slot = this._broadcastWaiters[i];
                if (slot.predicate(data)) {
                    this._broadcastWaiters.splice(i, 1);
                    clearTimeout(slot.timer);
                    slot.resolve(data);
                    return;
                }
            }
        }
    }

    /** Snap video playback to live edge if buffered lag exceeds 0.5 s. */
    _startLatencyMonitor(video) {
        if (this._latencyMonitorId) clearInterval(this._latencyMonitorId);
        this._latencyMonitorId = setInterval(() => {
            if (!video.srcObject || video.paused) return;
            const buf = video.buffered;
            if (buf.length > 0) {
                const end = buf.end(buf.length - 1);
                const lag = end - video.currentTime;
                if (lag > 0.5) {
                    console.log(`Latency correction: was ${lag.toFixed(2)}s behind`);
                    video.currentTime = end - 0.1;
                }
            }
        }, 2000);
    }
}

export default ReachyMini;
