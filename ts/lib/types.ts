/**
 * Public type surface for the ReachyMini SDK.
 *
 * These types are exported from the package barrel
 * (`reachy-mini-sdk.ts`) and re-exported by the host shell via
 * `host/src/lib/sdk-types.ts`.
 */

/** Robot summary returned by the central via `robots`/`robotsChanged`. */
export interface RobotInfo {
    id: string;
    meta?: { name?: string };
    /**
     * `true` when the central reports an active session held by
     * another consumer. Older centrals omit it; default `false`.
     */
    busy?: boolean;
    /**
     * Friendly name of the app currently holding the session, when
     * the busy consumer advertised one. Pure UX hint, never trust
     * for security decisions. `null` / undefined when the robot is
     * free.
     */
    activeApp?: string | null;
    /**
     * Network transport reported by the daemon (`wifi` / `usb` /
     * arbitrary string). Surfaced as a chip on the discovery card.
     * Older centrals or non-conformant daemons may omit it.
     */
    transport?: string | null;
    /**
     * Hardware id reported by the daemon (e.g. serial number).
     * When present, the picker shows it in place of the (longer,
     * less human-friendly) peer id. Optional.
     */
    hardwareId?: string | null;
}

/** Latest robot telemetry mirrored on `robot.robotState` (wire shape). */
export interface RobotState {
    /** Flat row-major 4×4 head pose (16 numbers). */
    head?: number[];
    /** [rightRad, leftRad]. */
    antennas?: number[];
    /** Radians. */
    body_yaw?: number;
    motor_mode?: 'enabled' | 'disabled' | 'gravity_compensation';
    is_move_running?: boolean;
}

/** SDK constructor options. */
export interface ReachyMiniOptions {
    signalingUrl?: string;
    /**
     * @deprecated The SDK no longer acquires the user's microphone. A silent
     * placeholder audio sender is always set up so apps can `replaceTrack`
     * their own audio (TTS, files, or — for teleop — the user's mic) onto
     * it. This option is parsed for backward compatibility but has no
     * effect.
     */
    enableMicrophone?: boolean;
    clientId?: string;
    appName?: string;
    /**
     * Hint to the receiver's WebRTC jitter buffer (ms). 0 = "render
     * ASAP", appropriate for teleop. Spec range [0, 4000]. Browsers
     * that don't implement `RTCRtpReceiver.jitterBufferTarget` fall
     * back to default buffering (~150-200 ms).
     */
    videoJitterBufferTargetMs?: number;
    /**
     * When true AND the URL carries a `robot_peer_id` hint, the SDK
     * auto-calls `startSession(preselectedRobotId)` once that robot
     * appears in the central's list after `connect()` resolves. One-shot
     * per page load; the host iframe relies on this to skip the picker.
     */
    autoStartFromUrl?: boolean;
}

/** Options accepted by `autoConnect()`. */
export interface AutoConnectOptions {
    /** Skip `authenticate()`; use this raw HF token. */
    token?: string;
    /**
     * Called in the standalone, multi-robot case to let the consumer
     * pick a robot from the live list. Return the robot id, or `null`
     * to cancel.
     */
    pickRobot?: (robots: AutoConnectRobotChoice[]) => Promise<string | null>;
    /** Skip `pickRobot` when exactly one robot is free. Default `true`. */
    autoPickIfSingle?: boolean;
    /** Hide busy robots from the picker. Default `true`. */
    filterBusy?: boolean;
    /** Call `ensureAwake()` after `startSession()`. Default `true`. */
    wakeOnConnect?: boolean;
}

/** Robot row passed to `pickRobot` callback (richer than `RobotInfo`). */
export interface AutoConnectRobotChoice {
    id: string;
    name: string | null;
    busy: boolean;
    activeApp: string | null;
    meta: Record<string, unknown>;
    lastSeenAgeSeconds: number | null;
}

/** Resolution payload of `autoConnect()`. */
export interface AutoConnectResult {
    robotId: string;
    robotName: string | null;
    isEmbedded: boolean;
    /** Set when `autoConnect()` short-circuited on an already-streaming session. */
    alreadyStreaming?: boolean;
}

/** Options for the awaitable motion helpers (`wakeUp`, `gotoSleep`). */
export interface MotionAwaitOptions {
    /**
     * Hard upper bound for the trajectory completion response from the
     * daemon. Defaults to 8000 ms (typical wake_up is ~1-3 s, but
     * deeply offset poses can take longer). The returned promise
     * rejects with a `${command} timed out after ${timeoutMs}ms` error
     * if the daemon goes silent.
     */
    timeoutMs?: number;
}

/** `subscribeLogs()` argument. */
export interface SubscribeLogsOptions {
    onLine: (entry: { timestamp: string; line: string }) => void;
    onError?: (error: string) => void;
}

// ─── XVF3800 audio-board config (Wireless only) ──────────────────────────────

/** Single XVF3800 parameter write. */
export interface AudioConfigEntry {
    name: string;
    values: number[];
}

export interface ApplyAudioConfigOptions {
    /** Read each parameter back after writing. Default `true`. */
    verify?: boolean;
}

// ─── Move / audio upload ─────────────────────────────────────────────────────

export interface MoveData {
    time: number[];
    set_target_data: object[];
}

export interface PlayMoveProgress {
    phase: string;
    sent?: number;
    total?: number;
    bytes?: number;
    encoding?: string;
    duration_s?: number;
}

export interface PlayMoveStartedInfo {
    duration_s: number;
    has_audio: boolean;
}

export interface PlayMoveOptions {
    audioBlob?: Blob | null;
    audioLeadMs?: number;
    description?: string;
    encoding?: 'gzip+base64' | 'json';
    playFrequency?: number;
    initialGotoDuration?: number;
    startTimeoutMs?: number;
    onProgress?: (p: PlayMoveProgress) => void;
    onStarted?: (s: PlayMoveStartedInfo) => void;
}

export interface PlayMoveResult {
    finished?: boolean;
    cancelled?: boolean;
    error?: string;
    has_audio?: boolean;
    /** Daemon may attach extra fields; tolerate them. */
    [key: string]: unknown;
}

export interface UploadAudioProgress {
    phase: string;
    sent?: number;
    total?: number;
    bytes?: number;
}

export interface UploadAudioOptions {
    description?: string;
    onProgress?: (p: UploadAudioProgress) => void;
}

export interface PlayUploadedAudioOptions {
    timeoutMs?: number;
}

// ─── Event detail shapes ─────────────────────────────────────────────────────

export interface ConnectedEventDetail { peerId: string; }
export interface DisconnectedEventDetail { reason: string; }
export interface RobotsChangedEventDetail { robots: RobotInfo[]; }
export interface StreamingEventDetail { sessionId: string; robotId: string; }
export interface SessionStoppedEventDetail { reason: string; message?: string | null; }
export interface SessionRejectedEventDetail {
    reason: string | undefined;
    activeApp: string | null | undefined;
}
export type StateEventDetail = RobotState;
export interface VideoTrackEventDetail { track: MediaStreamTrack; stream: MediaStream; }
export interface MicSupportedEventDetail { supported: boolean; }
export interface ErrorEventDetail {
    source: 'signaling' | 'webrtc' | 'robot';
    error: Error | string;
}

/** Map of event names to their detail shapes. */
export interface ReachyMiniEventMap {
    connected: CustomEvent<ConnectedEventDetail>;
    disconnected: CustomEvent<DisconnectedEventDetail>;
    robotsChanged: CustomEvent<RobotsChangedEventDetail>;
    streaming: CustomEvent<StreamingEventDetail>;
    sessionStopped: CustomEvent<SessionStoppedEventDetail>;
    sessionRejected: CustomEvent<SessionRejectedEventDetail>;
    state: CustomEvent<StateEventDetail>;
    videoTrack: CustomEvent<VideoTrackEventDetail>;
    micSupported: CustomEvent<MicSupportedEventDetail>;
    error: CustomEvent<ErrorEventDetail>;
}

/** Public surface of a ReachyMini SDK instance. */
export interface ReachyMiniInstance extends EventTarget {
    readonly state: 'disconnected' | 'connected' | 'streaming';
    readonly robots: RobotInfo[];
    /**
     * Mirror of the latest "state" event detail. Fields appear only once
     * the daemon has sent the corresponding source field.
     */
    readonly robotState: RobotState;
    readonly username: string | null;
    readonly isAuthenticated: boolean;
    readonly micSupported: boolean;
    readonly micMuted: boolean;
    readonly audioMuted: boolean;
    /** Set by the SDK from `?robot_peer_id=` / `#robot_peer_id=`. */
    readonly preselectedRobotId: string | null;
    /** `true` iff `preselectedRobotId !== null`. UX branching helper. */
    readonly isEmbedded: boolean;

    /** Underlying RTCPeerConnection. Apps can read it to inspect
     *  audio / video transceivers. */
    _pc: RTCPeerConnection | null;
    /** Silent placeholder MediaStream the SDK feeds the WebRTC audio
     *  sender so robot-speaker output can negotiate sendrecv. Apps inject
     *  their own audio (TTS, files, the user's mic for teleop) by calling
     *  `replaceTrack()` on the audio sender from `_pc.getSenders()`. */
    _micStream: MediaStream | null;

    authenticate(): Promise<boolean>;
    login(): Promise<void>;
    logout(): void;

    connect(token?: string): Promise<void>;
    /**
     * One-shot bring-up: auth → SSE connect → robot selection → session →
     * wake up. The all-in-one entry point that captures the common
     * "embed *or* standalone, just get me streaming" flow.
     */
    autoConnect(options?: AutoConnectOptions): Promise<AutoConnectResult>;
    disconnect(): void;

    startSession(robotId: string): Promise<void>;
    stopSession(): Promise<void>;

    attachVideo(el: HTMLVideoElement): () => void;

    setHeadRpyDeg(roll: number, pitch: number, yaw: number): boolean;
    setAntennasDeg(right: number, left: number): boolean;
    setBodyYawDeg(yawDeg: number): boolean;
    /**
     * Streaming variant for the trajectory player: sets the full
     * target frame at 50-100 Hz. Unspecified joints keep their
     * previous target. Returns `false` if the data channel is not
     * open.
     */
    setTarget(args: {
        head?: number[];
        antennas?: number[];
        body_yaw?: number;
    }): boolean;
    /**
     * Smooth daemon-side interpolation to a target pose over `duration`
     * seconds. Mirrors `setTarget`'s wire shape and adds a required
     * `duration` field. Throws `TypeError` on invalid input.
     */
    gotoTarget(args: {
        head?: number[];
        antennas?: number[];
        body_yaw?: number;
        duration: number;
    }): boolean;
    /** Send an arbitrary JSON message on the data channel. */
    sendRaw(data: unknown): boolean;
    /** Play a sound file on the robot's speakers (basename). */
    playSound(file: string): boolean;

    setAudioMuted(muted: boolean): void;
    setMicMuted(muted: boolean): void;

    getVolume(): Promise<number | null>;
    setVolume(volume: number): Promise<number | null>;
    getMicrophoneVolume(): Promise<number | null>;
    setMicrophoneVolume(volume: number): Promise<number | null>;

    /**
     * Apply a batch of XVF3800 audio-board parameters on the robot.
     * Mirrors the on-robot `AudioBase.apply_audio_config()` SDK call.
     * Resolves `true` when every parameter was written (and, when
     * `verify` is `true`, read back successfully); `false` when the
     * audio board is unavailable.
     */
    applyAudioConfig(
        config: AudioConfigEntry[],
        opts?: ApplyAudioConfigOptions,
    ): Promise<boolean>;
    /**
     * Read a single XVF3800 parameter by name. Mirrors the on-robot
     * `ReSpeaker.read_values()` SDK call. Resolves the decoded
     * numeric values, or `null` when the parameter is unknown /
     * unreadable / the audio board is unavailable.
     */
    readAudioParameter(name: string): Promise<number[] | null>;

    /** Daemon version string, or `null` when unavailable. */
    getVersion(): Promise<string | null>;
    /** Hardware ID (USB serial), or `null` on developer machines. */
    getHardwareId(): Promise<string | null>;
    /** Force a `state` event right now (background poll runs at 500 ms). */
    requestState(): boolean;
    /** Subscribe to daemon `journalctl` logs over the data channel. */
    subscribeLogs(options: SubscribeLogsOptions): () => void;

    /**
     * Motor torque mode: `enabled` (position control), `disabled`
     * (limp), `gravity_compensation` (float-by-hand). Returns
     * `false` if the data channel is not open.
     */
    setMotorMode(
        mode: 'enabled' | 'disabled' | 'gravity_compensation',
    ): boolean;
    /**
     * Per-motor torque toggle. When `ids` is omitted, applies globally
     * (equivalent to `setMotorMode("enabled" | "disabled")`).
     */
    setMotorTorque(on: boolean, ids?: string[] | null): boolean;
    /**
     * Play the wake-up trajectory (enables motors first). Resolves on
     * the daemon's `{command: "wake_up", completed: true}` response,
     * rejects on `timeoutMs` elapsed or on session teardown.
     */
    wakeUp(options?: MotionAwaitOptions): Promise<void>;
    /**
     * Play the goto-sleep trajectory. Resolves when the daemon reports
     * completion. Does NOT touch motor mode; caller manages it.
     */
    gotoSleep(options?: MotionAwaitOptions): Promise<void>;
    /** Read the awake state from the cached `motor_mode`. */
    isAwake(): boolean;
    /**
     * Idempotent wakeUp: noop when already awake. Resolves with
     * the post-call awake state. Does NOT await the trajectory.
     */
    ensureAwake(timeoutMs?: number): Promise<boolean>;

    /** Upload a recorded move (and optional audio) and play it on the daemon's local clock. */
    playMove(motion: MoveData, opts?: PlayMoveOptions): Promise<PlayMoveResult>;
    /** Cancel an in-flight `playMove`. */
    cancelMove(uploadId?: string | null): boolean;
    /** Upload audio as a standalone slot (no motion attached). */
    uploadAudio(audioBlob: Blob, opts?: UploadAudioOptions): Promise<string>;
    /** Trigger daemon-side playback of a previously-uploaded audio. */
    playUploadedAudio(uploadId: string, opts?: PlayUploadedAudioOptions): Promise<{ started: true }>;
    /** Cancel an in-flight `playUploadedAudio`. */
    cancelAudio(uploadId?: string | null): boolean;
}

export type ReachyMiniConstructor = new (
    options?: ReachyMiniOptions,
) => ReachyMiniInstance;

/**
 * Internal "session reject" error shape. Surfaced by `startSession()`
 * rejections; consumers can downcast a caught `Error` to read the
 * structured fields.
 */
export interface SessionRejectError extends Error {
    reason?: string | null;
    activeApp?: string | null;
}
