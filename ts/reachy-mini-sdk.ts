/**
 * reachy-mini-sdk — Browser SDK for controlling a Reachy Mini robot over WebRTC.
 * https://github.com/pollen-robotics/reachy-mini
 *
 * QUICK START
 * ───────────
 *   import { ReachyMini } from "@pollen-robotics/reachy-mini-sdk";
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
 *       head: rpyToMatrix(0, 10, -5).flat(),
 *       antennas: [degToRad(30), degToRad(-30)],
 *       body_yaw: degToRad(15),
 *   });
 *
 *   // 6. Receive live state (emitted every ~500 ms while streaming).
 *   robot.addEventListener("state", (e) => {
 *       const { head, antennas, body_yaw, motor_mode, is_move_running } = e.detail;
 *   });
 *
 *   // 7. Audio controls
 *   robot.setAudioMuted(false);
 *   robot.setMicMuted(false);
 *
 *   // XVF3800 audio-board tuning (the daemon owns the USB board; works
 *   // on both Lite and Wireless robots):
 *   await robot.applyAudioConfig([{ name: "AUDIO_MGR_MIC_GAIN", values: [1.0] }]);
 *   const v = await robot.readAudioParameter("AUDIO_MGR_MIC_GAIN"); // [1.0]
 *
 *   // 8. Cleanup
 *   detach();
 *   await robot.stopSession();
 *   robot.disconnect();
 *   robot.logout();
 *
 *
 * EXPORTS
 * ───────
 *   export default ReachyMini;
 *   export { ReachyMini, rpyToMatrix, matrixToRpy, degToRad, radToDeg };
 *   plus the public type surface (RobotInfo, RobotState, ReachyMiniOptions, …).
 */

export { ReachyMini } from './lib/reachy-mini';
export { degToRad, radToDeg, rpyToMatrix, matrixToRpy } from './lib/math';
export type {
    RobotInfo,
    RobotState,
    ReachyMiniOptions,
    ReachyMiniInstance,
    ReachyMiniConstructor,
    AutoConnectOptions,
    AutoConnectRobotChoice,
    AutoConnectResult,
    MotionAwaitOptions,
    SubscribeLogsOptions,
    AudioConfigEntry,
    ApplyAudioConfigOptions,
    MoveData,
    PlayMoveOptions,
    PlayMoveProgress,
    PlayMoveResult,
    PlayMoveStartedInfo,
    PlayUploadedAudioOptions,
    UploadAudioOptions,
    UploadAudioProgress,
    SessionRejectError,
    ReachyMiniEventMap,
    ConnectedEventDetail,
    DisconnectedEventDetail,
    RobotsChangedEventDetail,
    StreamingEventDetail,
    SessionStoppedEventDetail,
    SessionRejectedEventDetail,
    StateEventDetail,
    VideoTrackEventDetail,
    MicSupportedEventDetail,
    ErrorEventDetail,
} from './lib/types';

import { ReachyMini } from './lib/reachy-mini';
export default ReachyMini;
