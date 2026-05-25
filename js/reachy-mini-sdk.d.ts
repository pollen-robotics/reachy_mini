/**
 * Type declarations for the ReachyMini browser SDK. These are the
 * canonical types shipped with the npm package — the host shell
 * (`./host` subpath) re-exports them via `host/src/lib/sdk-types.ts`
 * so React components, hooks, and the embed client all see exactly
 * the same surface.
 *
 * The runtime lives in `reachy-mini-sdk.js`; this `.d.ts` only
 * declares the public types. We hand-maintain it (rather than
 * generating from the JSDoc) so consumers get rich types without an
 * extra build step on our side.
 *
 * Legacy CDN consumers can still expose the constructor on
 * `window.ReachyMini` themselves; the global declaration at the
 * bottom keeps that shape typed.
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
}

export type ReachyMiniConstructor = new (
  options?: ReachyMiniOptions,
) => ReachyMiniInstance;

/** SDK constructor. */
export const ReachyMini: ReachyMiniConstructor;

/** Degrees → radians. */
export function degToRad(deg: number): number;
/** Radians → degrees. */
export function radToDeg(rad: number): number;
/** Roll/pitch/yaw (degrees) → 4×4 rotation matrix (ZYX convention). */
export function rpyToMatrix(
  rollDeg: number,
  pitchDeg: number,
  yawDeg: number,
): number[][];
/** Rotation matrix (3×3 or 4×4) → `{ roll, pitch, yaw }` in degrees. */
export function matrixToRpy(m: number[][]): {
  roll: number;
  pitch: number;
  yaw: number;
};

export default ReachyMini;
