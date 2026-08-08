/**
 * Top-level host shell. Wires the SDK, OAuth, bridge, and the
 * per-phase views together into one component.
 *
 * Phases (cf. APP_CREATION_GUIDE §13.3 Mode A standalone flow):
 *   signing-in : SignInView visible (signed-out or post-OAuth
 *                splash). Host SDK alive for OAuth only.
 *   picking    : PickerView visible. Robots fetched via REST
 *                (`/api/robot-status`); the host never opens an
 *                SSE — that would break the iframe's WebRTC
 *                handshake later.
 *   embedded   : iframe mounted. ConnectingView overlay visible
 *                until embed reports phase=live.
 *   leaving    : `host:leaving` sent, waiting for tear-down
 *                deadline. Iframe still mounted.
 *   error      : ErrorView visible. SDK left as-is so a
 *                back-to-picker recovers in place.
 *
 * Strict Mode safety: every effect with side-effects ships with
 * an idempotent cleanup. `host:init` is sent exactly once per
 * `selectedRobotId`, gated by an `initSentForRef`.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { JSX } from 'react';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import GlobalStyles from '@mui/material/GlobalStyles';
import Stack from '@mui/material/Stack';
import { alpha } from '@mui/material/styles';

import { createLogger } from '@pollen-robotics/reachy-mini-sdk';
import type { ReachyMiniInstance, RobotInfo } from '../lib/sdk-types';
import {
  encodeCredsToHash,
  type AppConnectingStep,
  type AppPhase,
  type ConfigPayload,
  type CredsBundle,
  type LeavingReason,
  type ThemeMode,
} from '../lib/protocol';
import { resolveSignalingUrl } from '../lib/signalingUrl';
import {
  isTargetListed,
  sawTargetOffline,
  type RebootTarget,
} from '../lib/rebootWatch';
import { useLatestDaemonVersion } from '../hooks/useLatestDaemonVersion';
import { useHfProfile } from '../hooks/useHfProfile';
import { useOAuth } from '../hooks/useOAuth';
import { useRobots } from '../hooks/useRobots';
import {
  useHostBridge,
  type EmbedAppState,
  type EmbedUpdateProgress,
} from '../hooks/useHostBridge';

import { ConnectingView } from './ConnectingView';
import { DaemonUpdateGate } from './DaemonUpdateGate';
import { EmbedFrame } from './EmbedFrame';
import { ErrorView } from './ErrorView';
import { LeavingView } from './LeavingView';
import { PickerView } from './PickerView';
import { PostOAuthSplash } from './PostOAuthSplash';
import { SignInView } from './SignInView';
import { TopBar, type HostPhase } from './TopBar';
import { WelcomeBackOverlay } from './WelcomeBackOverlay';

const log = createLogger('host');

// Hard cap on how long the host stays on the leaving overlay waiting for the
// embed's `embed:left` ack (app cleanup + host-owned sleep/disable). The
// embed's worst case is SEQUENTIAL: the sleep race hard-cap (6.5 s) plus the
// motors-disabled confirmation window (1 s) = 7.5 s, so the cap leaves ~2 s
// for postMessage + a React commit - a well-behaved embed's ack always wins.
// A wedged/older embed that never acks just falls through here.
const LEAVING_ACK_CAP_MS = 9500;

/** Fresh `EmbedAppState` for a phase reset (boot, or the optimistic
 *  `connecting` shown while the iframe mounts). Single source of truth
 *  for the "everything unknown" shape - additive fields default to
 *  null here and nowhere else. */
function makeEmbedAppState(
  phase: AppPhase,
  connectingStep: AppConnectingStep | null = null,
): EmbedAppState {
  return {
    phase,
    connectingStep,
    message: null,
    rttMs: null,
    daemonVersion: null,
    sdkVersion: null,
  };
}

export interface ReachyHostShellProps {
  sdk: ReachyMiniInstance | null;

  appName: string;
  appIconUrl?: string;
  appEmoji?: string;
  hostName: string;

  theme: ThemeMode;
  initialConfig?: ConfigPayload;

  enableMicrophone: boolean;
  /** Path of the embedded app entry within the same origin.
   *  Defaults to `/?embedded=1`. */
  embedPath?: string;
  /** Surface a dev hint on the sign-in screen when no OAuth
   *  client ID is reachable and no dev token has been seeded. */
  isLocalDevMissingConfig?: boolean;
}

/* ─────────────────── Dev preview shortcut ─────────────────── */

type PreviewPhase = 'signing-in' | 'welcome' | 'picker' | 'connecting' | 'error';

/** Force-render a specific phase with mock data via
 *  `?host-preview=signing-in|welcome|picker|connecting|error`.
 *
 *  Pure visual harness: no SDK calls, no postMessage traffic.
 *  The query param is opt-in and never injected at runtime, so
 *  the surface area is limited to "developer typed the param" -
 *  the helper costs ~150 lines in the production bundle, which
 *  is worth paying for a one-URL design-review path. */
function readPreviewPhase(): PreviewPhase | null {
  try {
    const value = new URLSearchParams(window.location.search).get(
      'host-preview',
    );
    if (
      value === 'signing-in' ||
      value === 'welcome' ||
      value === 'picker' ||
      value === 'connecting' ||
      value === 'error'
    ) {
      return value;
    }
  } catch {
    /* ignore */
  }
  return null;
}

const MOCK_ROBOTS: RobotInfo[] = [
  {
    id: 'reachy-mini-a1b2c3d4',
    meta: { name: 'Tabouret' },
    busy: false,
    activeApp: null,
    transport: 'wifi',
    hardwareId: 'a1b2c',
  },
  {
    id: 'reachy-mini-e5f6g7h8',
    meta: { name: 'Sapin' },
    busy: true,
    activeApp: 'Emotions',
    transport: 'wifi',
    hardwareId: 'e5f6g',
  },
  {
    id: 'reachy-mini-i9j0k1l2',
    meta: { name: 'Robocop' },
    busy: false,
    activeApp: null,
    transport: 'usb',
    hardwareId: 'i9j0k',
  },
];

export function ReachyHostShell(
  props: ReachyHostShellProps,
): JSX.Element {
  // Preview harness is a DEV-only affordance. Gating on
  // `import.meta.env.DEV` lets the bundler tree-shake the whole
  // preview branch (ReachyHostShellPreview + MOCK_ROBOTS) out of the
  // production build, so a deployed Space can't be flipped into the
  // mock view via `?host-preview=...`.
  const previewPhase = import.meta.env.DEV ? readPreviewPhase() : null;
  if (previewPhase) {
    return <ReachyHostShellPreview phase={previewPhase} {...props} />;
  }
  return <ReachyHostShellNormal {...props} />;
}

function ReachyHostShellNormal({
  sdk,
  appName,
  appIconUrl,
  appEmoji,
  hostName,
  theme,
  initialConfig,
  enableMicrophone,
  embedPath = '/?embedded=1',
  isLocalDevMissingConfig = false,
}: ReachyHostShellProps): JSX.Element {
  /* ─────────────────── State ─────────────────── */

  const [hostPhase, setHostPhase] = useState<HostPhase>('signing-in');
  const [selectedRobotId, setSelectedRobotId] = useState<string | null>(null);
  /** Identity (name + transport) and stable hardware id of the picked
   *  robot, captured at selection time. `useRobots` is disabled the
   *  moment we leave `picking` (and clears its list to protect the 1:1
   *  token→peer handoff invariant), so we CANNOT look the robot up in
   *  `robots` during the session - it's empty. We snapshot what the
   *  topbar and the embed handoff need here instead. */
  const [selectedRobot, setSelectedRobot] = useState<{
    name: string | null;
    transport: string | null;
    hardwareId: string | null;
  } | null>(null);
  const [embedAppState, setEmbedAppState] = useState<EmbedAppState>(() =>
    makeEmbedAppState('boot'),
  );
  /** Daemon version of the robot in session, latched from the embed's
   *  app-state. Kept OUT of `embedAppState` because the update flow
   *  outlives the iframe: we tear the embed down when the daemon
   *  restarts, and the gate still needs to name the version it is
   *  replacing. Cleared on the next selection. */
  const [daemonVersion, setDaemonVersion] = useState<string | null>(null);
  /** Latest `embed:update-progress` payload, or `null` when no update
   *  has been asked for in this selection. */
  const [updateProgress, setUpdateProgress] =
    useState<EmbedUpdateProgress | null>(null);
  /** Set when we dropped the iframe because the daemon restarted mid
   *  update. Holds what we need to recognise the robot when it comes
   *  back on central (its peer id will have rotated). */
  const [awaitingReboot, setAwaitingReboot] = useState<RebootTarget | null>(
    null,
  );
  /** Offline-first latch over the central listing while `awaitingReboot`
   *  is set: the robot must be seen ABSENT once before its presence
   *  counts as "back online", so a stale pre-reboot listing can't
   *  complete the update gate prematurely (see `lib/rebootWatch.ts`).
   *  Held `false` whenever `awaitingReboot` is null: the advancing
   *  effect is gated on it, and every path touching `awaitingReboot`
   *  (`selectRobot`, `dismissDaemonUpdate`) resets it alongside. */
  const [sawOffline, setSawOffline] = useState<boolean>(false);
  /** Remounts `DaemonUpdateGate` on every selection so its "user
   *  already dismissed this" latch doesn't leak into the next robot. */
  const [gateKey, setGateKey] = useState(0);
  const [errorPayload, setErrorPayload] = useState<{
    message: string;
    detail?: unknown;
  } | null>(null);
  /** One-way welcome-back sequence, advanced `idle → showing → done`.
   *  `showing` mounts the WelcomeBackOverlay; `done` (set by its
   *  `onDone`) is terminal because `isPostOauthReturn` stays true for
   *  the rest of the page load - without the terminal position the
   *  post-OAuth splash would re-trigger in `picking` and loop
   *  "Signing you in…" forever. Reset to `idle` on sign-out so a
   *  re-sign-in replays the sequence. */
  const [welcomeBack, setWelcomeBack] =
    useState<'idle' | 'showing' | 'done'>('idle');
  /** Latches true once the initial boot has reached its first stable
   *  view (SignInView when unauthenticated, or the picker with its
   *  first central response settled). Lets the boot splash stay up
   *  continuously through the `signing-in → picking → list` handoff
   *  so the auto-login path goes splash → list with no bare
   *  picker-spinner frame in between. Only gates the FIRST boot: once
   *  latched, later returns to the picker use the picker's own
   *  spinner. Reset on sign-out so a re-sign-in replays the sequence. */
  const [bootSplashDone, setBootSplashDone] = useState<boolean>(false);

  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  /** Guards `host:init` sending: at most once per selected
   *  robot, and only after `embed:ready` arrived (StrictMode
   *  safe). */
  const initSentForRef = useRef<string | null>(null);
  /** Set when `embed:ready` is observed before the iframe ref
   *  is available, so we can flush as soon as the ref binds. */
  const embedReadyPendingRef = useRef<boolean>(false);
  /** Resolver for the in-flight `endSession` waiting on the embed's
   *  `embed:left` ack. Set while a leave is pending; the `onLeft`
   *  bridge callback calls it to let the host unmount immediately. */
  const leftAckResolveRef = useRef<(() => void) | null>(null);

  /* ─────────────────── External hooks ─────────────────── */

  const {
    isAuthenticated,
    userName,
    isPostOauthReturn,
    authResolved,
    signIn,
    signOut,
  } = useOAuth(sdk);
  const hfToken = isAuthenticated ? readToken() : null;
  const hfProfile = useHfProfile(hfToken);
  // Prefer the HF whoami payload's avatar + canonical username
  // once it lands; fall back to the OAuth-issued username (always
  // present once `isAuthenticated`) so the bar renders correctly
  // during the brief whoami-v2 in-flight window.
  const displayUserName = hfProfile.username ?? userName;
  const {
    robots,
    isLoading: robotsLoading,
    isRefreshing: robotsRefreshing,
    error: robotsError,
  } = useRobots({
    hfToken,
    enabled: isAuthenticated && hostPhase === 'picking',
  });
  const latestDaemonVersion = useLatestDaemonVersion();

  // Latch the welcome-back overlay on once the post-OAuth flag
  // fires AND the username is resolved. Gating on `userName`
  // matters because `isPostOauthReturn` can flip true a frame
  // or two before `setUserName(sdk.username)` lands (the OAuth
  // bootstrap effect updates them in separate batches in prod).
  // Without the gate the overlay mounts with "Welcome back" and
  // visibly flickers to "Hello, X" the next frame.
  //
  // The sequence only ever advances from `idle`, which makes it a
  // one-shot per auth session: after `onDone` moves it to `done`,
  // the overlay MUST NOT re-mount even though `isPostOauthReturn`
  // is still true.
  //
  // The 800 ms fallback timer is a defensive cap: if the SDK
  // somehow authenticates without resolving a username, we
  // still show the (generic) overlay rather than swallowing the
  // welcome moment forever.
  useEffect(() => {
    if (welcomeBack !== 'idle' || !isPostOauthReturn) return;

    if (userName) {
      setWelcomeBack('showing');
      return;
    }

    const t = window.setTimeout(() => {
      setWelcomeBack((prev) => (prev === 'idle' ? 'showing' : prev));
    }, 800);
    return () => window.clearTimeout(t);
  }, [isPostOauthReturn, welcomeBack, userName]);

  // Reset the sequence on sign-out so the next sign-in can play
  // the welcome anim again in the same tab.
  useEffect(() => {
    if (!isAuthenticated) {
      setWelcomeBack('idle');
      setBootSplashDone(false);
    }
  }, [isAuthenticated]);

  // Latch the boot splash off once the first stable view is ready.
  // The post-OAuth return leg runs its own splash → welcome-back
  // handoff, so we let it out immediately here and don't extend the
  // plain boot splash over it.
  useEffect(() => {
    if (bootSplashDone) return;
    if (isPostOauthReturn) {
      setBootSplashDone(true);
      return;
    }
    // Auth resolved but no session → SignInView is the stable view.
    if (authResolved && !isAuthenticated) {
      setBootSplashDone(true);
      return;
    }
    // Authenticated and the picker's first central fetch has settled
    // (robotsLoading is held true from the very first `picking` frame
    // by useRobots' initial-loading floor) → the list is the stable
    // view.
    if (
      isAuthenticated &&
      hostPhase === 'picking' &&
      !robotsLoading
    ) {
      setBootSplashDone(true);
    }
  }, [
    bootSplashDone,
    isPostOauthReturn,
    authResolved,
    isAuthenticated,
    hostPhase,
    robotsLoading,
  ]);

  /* ─────────────────── Bridge ─────────────────── */

  const flushInitIfReady = useCallback(() => {
    if (!iframeRef.current) return;
    if (!embedReadyPendingRef.current) return;
    if (selectedRobotId == null) return;
    if (initSentForRef.current === selectedRobotId) return;

    const bundle: Omit<CredsBundle, 'signalingUrl'> & {
      signalingUrl: string;
    } = {
      hfToken: readToken(),
      userName,
      robotPeerId: selectedRobotId,
      // Stable id so the embed re-resolves the live peer id at dial
      // time (the peer id above rotates on relay reconnects). Read from
      // the selection-time snapshot: this runs on `embed:ready`, by
      // which point `robots` has long been cleared.
      robotHardwareId: selectedRobot?.hardwareId ?? null,
      signalingUrl: resolveSignalingUrl(),
      theme,
      config: initialConfig ?? null,
      hostName,
      appName,
    };

    bridge.sendInit(iframeRef.current, {
      theme,
      signalingUrl: bundle.signalingUrl,
      hfToken: bundle.hfToken ?? undefined,
      userName: bundle.userName,
      robotPeerId: bundle.robotPeerId,
      robotHardwareId: bundle.robotHardwareId,
      config: bundle.config,
      hostName,
      appName,
    });
    initSentForRef.current = selectedRobotId;
    embedReadyPendingRef.current = false;
  }, [
    appName,
    hostName,
    initialConfig,
    selectedRobot,
    selectedRobotId,
    theme,
    userName,
  ]);

  const bridge = useHostBridge({
    onEmbedReady: () => {
      embedReadyPendingRef.current = true;
      flushInitIfReady();
    },
    onAppState: (state) => {
      setEmbedAppState(state);
      // Latch, never clear: the embed re-sends app-state without the
      // version on some transitions, and a momentary `null` would make
      // the gate blink out mid-decision.
      if (state.daemonVersion) setDaemonVersion(state.daemonVersion);
      if (state.phase === 'error') {
        setErrorPayload({
          message: state.message ?? 'The app reported an error.',
        });
        setHostPhase('error');
      }
    },
    onRequestLeave: () => {
      void endSession('session-stopped');
    },
    onLeft: () => {
      // Embed finished its host-owned sleep/disable; let the pending
      // endSession unmount now instead of waiting out the cap.
      leftAckResolveRef.current?.();
    },
    onError: ({ message, fatal, detail }) => {
      if (fatal) {
        setErrorPayload({ message, detail });
        setHostPhase('error');
      } else {
        log.warn(
          'embed reported non-fatal error:',
          message,
          detail,
        );
      }
    },
    onUpdateProgress: setUpdateProgress,
  });

  /* ─────────────────── Phase driver: auth ─────────────────── */

  // signing-in → picking transition: once the SDK reports auth.
  // The picker fetches robots via REST (no SDK / SSE needed).
  useEffect(() => {
    if (hostPhase === 'error') return;
    if (!isAuthenticated) {
      if (hostPhase !== 'signing-in') setHostPhase('signing-in');
      return;
    }
    if (hostPhase === 'signing-in') {
      setHostPhase('picking');
    }
  }, [hostPhase, isAuthenticated]);

  /* ─────────────────── Theme push ─────────────────── */

  useEffect(() => {
    if (hostPhase !== 'embedded' && hostPhase !== 'leaving') return;
    if (!iframeRef.current) return;
    bridge.sendThemeChanged(iframeRef.current, theme);
  }, [theme, bridge, hostPhase]);

  /* ─────────────────── Selection → handoff ─────────────────── */

  const selectRobot = useCallback(
    (robotId: string) => {
      if (!sdk) return;
      if (hostPhase !== 'picking') return;
      setSelectedRobotId(robotId);
      // Snapshot identity NOW, while `robots` is still populated -
      // it gets cleared as soon as we flip to `embedded` below.
      const picked = robots.find((r) => r.id === robotId);
      setSelectedRobot({
        name: picked?.meta?.name ?? null,
        transport: picked?.transport ?? null,
        hardwareId: picked?.hardwareId ?? null,
      });
      initSentForRef.current = null;
      embedReadyPendingRef.current = false;
      setDaemonVersion(null);
      setUpdateProgress(null);
      setAwaitingReboot(null);
      // Keep the documented invariant real: the latch is held `false`
      // whenever `awaitingReboot` is null. Without this, a latch left
      // pre-armed would let a STALE pre-reboot listing complete a later
      // watch in a single observation - the exact regression
      // rebootWatch.ts prevents.
      setSawOffline(false);
      setGateKey((k) => k + 1);
      // The host never opened an SSE (picker uses REST), so the
      // iframe's SDK gets a clean central slot with no prior peer
      // registered for this HF token. No releaseSdkForHandoff()
      // needed - that legacy hook tore down a connection we no
      // longer create.
      setEmbedAppState(makeEmbedAppState('connecting', 'link'));
      setHostPhase('embedded');
    },
    [hostPhase, robots, sdk],
  );

  /* ─────────────────── End session ─────────────────── */

  const endSession = useCallback(
    async (reason: LeavingReason): Promise<void> => {
      if (hostPhase !== 'embedded' && hostPhase !== 'error') return;
      if (iframeRef.current) {
        bridge.sendLeaving(iframeRef.current, reason, LEAVING_ACK_CAP_MS);
      }
      setHostPhase('leaving');

      // Wait for the embed to finish its host-owned tear-down (app onLeave +
      // gotoSleep + motors disabled) and ack via `embed:left`, so the robot is
      // actually asleep before we drop the session. Bounded by
      // `LEAVING_ACK_CAP_MS` so a wedged or older embed (which never acks)
      // still falls through and unmounts. `race` ignores the loser, so no
      // double-settle guard is needed.
      await Promise.race([
        new Promise<void>((resolve) => {
          leftAckResolveRef.current = resolve;
        }),
        sleep(LEAVING_ACK_CAP_MS),
      ]);
      leftAckResolveRef.current = null;

      // Unmount iframe (selectedRobotId = null) and clean up.
      // CRITICAL: do NOT call `wipeHfSessionStorage()` here. The
      // picker needs the HF token to keep its SSE listener open
      // and the REST safety-net polling alive. The token is wiped
      // only on full sign-out (`signOut`); its lifetime is otherwise
      // bounded by the SDK token-store's expiry + sliding idle window.
      setSelectedRobotId(null);
      setSelectedRobot(null);
      setEmbedAppState(makeEmbedAppState('boot'));
      initSentForRef.current = null;
      embedReadyPendingRef.current = false;

      // Back to picker. REST polling resumes via the
      // `enabled: hostPhase === 'picking'` gate in useRobots.
      setHostPhase('picking');
    },
    [bridge, hostPhase],
  );

  /* ─────────────────── Daemon update ─────────────────── */

  const startDaemonUpdate = useCallback((): void => {
    if (!iframeRef.current) return;
    setUpdateProgress(null);
    bridge.sendStartUpdate(iframeRef.current);
  }, [bridge]);

  /** The gate's stall timer gave up while the session is still alive:
   *  tell the embed to disarm its update-mode plumbing (translator +
   *  auto-reconnect stand-down) so a later normal end-session can't
   *  replay a stale `rebooting` frame. */
  const cancelDaemonUpdate = useCallback((): void => {
    if (!iframeRef.current) return;
    bridge.sendCancelUpdate(iframeRef.current);
  }, [bridge]);

  /** Escape from the gate's blocking tier. Clear `daemonVersion`
   *  FIRST: `endSession` doesn't reset it (only the next `selectRobot`
   *  does), so leaving it set would keep `severity === 'block'` true
   *  and repaint the full-screen gate over the picker we just returned
   *  to. */
  const exitBlockedApp = useCallback((): void => {
    setDaemonVersion(null);
    void endSession('user-action');
  }, [endSession]);

  /**
   * The daemon restarted, so the session is gone for good. Drop the
   * iframe WITHOUT the usual `host:leaving` handshake - there's no
   * robot left to sleep, and waiting on an ack that can't come would
   * just stall the overlay - then fall back to the picker, whose
   * central polling is how we find out the robot is back.
   */
  const handleUpdateSessionLost = useCallback((): void => {
    setAwaitingReboot((prev) =>
      prev ?? {
        hardwareId: selectedRobot?.hardwareId ?? null,
        name: selectedRobot?.name ?? null,
      },
    );
    setSelectedRobotId(null);
    setSelectedRobot(null);
    setEmbedAppState(makeEmbedAppState('boot'));
    initSentForRef.current = null;
    embedReadyPendingRef.current = false;
    setHostPhase('picking');
  }, [selectedRobot]);

  const dismissDaemonUpdate = useCallback((): void => {
    setAwaitingReboot(null);
    setSawOffline(false);
    setUpdateProgress(null);
  }, []);

  // Advance the offline latch on every settled view of the central
  // listing. Central may keep the robot's pre-reboot registration
  // listed for a while after the daemon goes down, so presence alone
  // is not "back online": the watch first requires one observation
  // WITHOUT the robot (the SSE offline event, typically) before its
  // reappearance completes it. Observations are skipped while
  // `robotsLoading` holds - in that window `robots` is empty because
  // there's no data yet, not because the robot is offline.
  useEffect(() => {
    if (awaitingReboot === null || hostPhase !== 'picking' || robotsLoading) {
      return;
    }
    const listed = isTargetListed(robots, awaitingReboot);
    setSawOffline((prev) => sawTargetOffline(prev, listed));
  }, [awaitingReboot, hostPhase, robotsLoading, robots]);

  const rebootedRobotBack =
    awaitingReboot !== null &&
    sawOffline &&
    isTargetListed(robots, awaitingReboot);

  /* ─────────────────── Iframe URL ─────────────────── */

  const iframeUrl = useMemo(() => {
    if (selectedRobotId == null) return null;
    const bundle: CredsBundle = {
      hfToken: readToken(),
      userName,
      robotPeerId: selectedRobotId,
      // Kept byte-identical with the `host:init` bundle so the embed
      // re-resolves the live peer id whichever channel it reads. Sourced
      // from the selection-time snapshot rather than `robots`: that list
      // is emptied one render after the handoff, which would rebuild this
      // URL with a null hardware id and re-navigate a live iframe.
      robotHardwareId: selectedRobot?.hardwareId ?? null,
      signalingUrl: resolveSignalingUrl(),
      theme,
      config: initialConfig ?? null,
      hostName,
      appName,
    };
    const hash = encodeCredsToHash(bundle);
    // Debug-only: the hash carries the user's HF token, so don't print
    // it in default console output.
    log.debug(
      `iframeUrl = ${window.location.origin}/?embedded=1#${hash}`,
    );
    // Hash creds carry the same data as host:init; the iframe
    // wipes the hash on its first tick (APP_CREATION_GUIDE §13.5.2). The
    // postMessage init is the canonical source once the bridge
    // is up; the hash exists so Mode B (mobile handoff) works
    // when there's no parent to talk to.
    const url = new URL(embedPath, window.location.origin);
    return `${url.toString()}#${hash}`;
  }, [
    appName,
    embedPath,
    hostName,
    initialConfig,
    selectedRobot,
    selectedRobotId,
    theme,
    userName,
  ]);

  /* ─────────────────── Pagehide cleanup ─────────────────── */

  useEffect(() => {
    const onPageHide = (): void => {
      // Best-effort: tell the embed it's leaving so it can
      // disconnect its SDK. We don't wait for an ack.
      //
      // Deliberately NOT wiping the HF token here anymore: pagehide
      // fires on every reload / navigation (it can't tell an F5 from a
      // tab close), so the wipe forced a re-login on every refresh.
      // The token-store's sliding idle window (SDK `token-store.ts`)
      // now covers what the wipe protected against - a tab resurrected
      // days later via session restore gets a stale-by-idle token that
      // reads as absent.
      if (iframeRef.current && (hostPhase === 'embedded' || hostPhase === 'leaving')) {
        bridge.sendLeaving(iframeRef.current, 'pagehide', 0);
      }
    };
    window.addEventListener('pagehide', onPageHide, { once: true });
    return () => window.removeEventListener('pagehide', onPageHide);
  }, [bridge, hostPhase]);

  /* ─────────────────── Render ─────────────────── */

  const showConnectingOverlay =
    hostPhase === 'embedded' &&
    (embedAppState.phase === 'boot' || embedAppState.phase === 'connecting');
  const showLeavingOverlay = hostPhase === 'leaving';

  // Neutral covering splash while auth is still settling, so neither
  // `SignInView` nor a naked `PickerView` flashes during the async
  // `authenticate()` + phase transition:
  //   - post-OAuth return: from boot through the flip to `picking`, until the
  //     WelcomeBackOverlay takes over (`welcomeBackShown`);
  //   - auto-login: the boot window while `authenticate()` resolves, plus the
  //     one-frame gap after it confirms auth but before the phase flips to
  //     `picking` (`isAuthenticated` true while still `signing-in`).
  const showPostOAuthSplash =
    isPostOauthReturn &&
    welcomeBack === 'idle' &&
    (hostPhase === 'signing-in' || hostPhase === 'picking');
  // Boot splash covers the whole first boot continuously, so the
  // auto-login path reads as splash → list with no intermediate bare
  // picker-spinner:
  //   - signing-in: while `authenticate()` resolves (`!authResolved`)
  //     and the one-frame gap after it confirms auth but before the
  //     phase flips to `picking` (`isAuthenticated` still `signing-in`);
  //   - picking: while the picker's first central fetch is in flight
  //     (`robotsLoading`, held true from frame 0 by useRobots).
  // `bootSplashDone` latches it off after the first stable view so
  // later returns to the picker use its own spinner instead.
  const showBootSplash =
    !isPostOauthReturn &&
    welcomeBack !== 'showing' &&
    !bootSplashDone &&
    ((hostPhase === 'signing-in' && (!authResolved || isAuthenticated)) ||
      (hostPhase === 'picking' && isAuthenticated && robotsLoading));

  return (
    <>
      <GlobalStyles
        styles={{
          ':root': {
            '--reachy-host-topbar-h': '56px',
          },
          'html, body, #root': { height: '100%' },
          body: { margin: 0 },
        }}
      />
      <Stack
        sx={{
          height: '100%',
          color: 'text.primary',
          bgcolor: 'background.default',
        }}
      >
        <TopBar
          appName={appName}
          appIconUrl={appIconUrl}
          appEmoji={appEmoji}
          hostPhase={hostPhase}
          userName={displayUserName}
          avatarUrl={hfProfile.avatarUrl}
          // Identity captured at selection time - `robots` is empty
          // during the session (useRobots disabled on handoff), so we
          // read the snapshot, falling back to the peer id only if the
          // robot never advertised a name.
          selectedRobotName={
            selectedRobotId
              ? (selectedRobot?.name ?? selectedRobotId)
              : null
          }
          selectedRobotTransport={selectedRobot?.transport ?? null}
          // True live latency reported by the embed (the host handed
          // its WebRTC slot to the iframe). `null` until the embed
          // reaches `live` and starts sampling.
          linkRttMs={embedAppState.rttMs}
          onSignOut={signOut}
          onEndSession={() => void endSession('user-action')}
        />

        <Box sx={{ flex: 1, position: 'relative', minHeight: 0 }}>
          {hostPhase === 'error' && errorPayload && (
            <ErrorView
              message={errorPayload.message}
              detail={errorPayload.detail}
              onReload={() => window.location.reload()}
              onBackToPicker={() => {
                setErrorPayload(null);
                setSelectedRobotId(null);
                setHostPhase('picking');
              }}
            />
          )}

          {hostPhase === 'signing-in' &&
            authResolved &&
            !isAuthenticated &&
            !isPostOauthReturn && (
              <SignInView
                appName={appName}
                isLocalDevMissingConfig={isLocalDevMissingConfig}
                onSignIn={signIn}
              />
            )}

          {hostPhase === 'picking' && (
            <PickerView
              robots={robots}
              isRefreshing={robotsLoading || robotsRefreshing}
              error={robotsError}
              preselectedRobotId={sdk?.preselectedRobotId ?? null}
              onSelect={selectRobot}
            />
          )}

          {(hostPhase === 'embedded' || hostPhase === 'leaving') &&
            iframeUrl && (
              <>
                <EmbedFrame
                  ref={iframeRef}
                  src={iframeUrl}
                  enableMicrophone={enableMicrophone}
                  title={appName}
                  visible={embedAppState.phase === 'live'}
                />
                {showConnectingOverlay && !showLeavingOverlay && (
                  <ConnectingView
                    step={embedAppState.connectingStep}
                    message={embedAppState.message}
                  />
                )}
                {showLeavingOverlay && <LeavingView />}
              </>
            )}
        </Box>
      </Stack>

      {/* Covering splashes for both boot legs, so the "Continue with
       *  Hugging Face" button and a naked picker never flash while
       *  `authenticate()` resolves and the phase transitions. The OAuth
       *  return leg - where the user really did just sign in - shows
       *  the branded PostOAuthSplash and hands off to WelcomeBackOverlay
       *  (zIndex 1300) once the username lands. The plain boot leg is a
       *  bare neutral spinner (PickerView's quiet-wait style): before
       *  `authResolved` we don't know whether a session exists, and an
       *  already-signed-in user must never see OAuth branding on a
       *  plain reload (nor a first-time visitor read "Signing you in…"
       *  right before landing on the sign-in button). */}
      {showPostOAuthSplash && <PostOAuthSplash />}
      {showBootSplash && (
        <Box
          sx={{
            position: 'fixed',
            inset: 0,
            zIndex: 1290,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'background.default',
          }}
        >
          <CircularProgress
            thickness={2.5}
            sx={{ color: (theme) => alpha(theme.palette.text.primary, 0.3) }}
          />
        </Box>
      )}

      {/* Web-only by construction: the mobile app never mounts this
       *  shell (it points its iframe straight at `?embedded=1` and runs
       *  its own gate), so there is nothing to suppress here. Rendered
       *  outside the phase switch because the update flow outlives the
       *  session it started in: the daemon reboots, we drop the iframe,
       *  and this overlay stays up over the picker until the robot is
       *  back. Renders null whenever there's nothing to say. */}
      <DaemonUpdateGate
        key={gateKey}
        currentVersion={daemonVersion}
        latestVersion={latestDaemonVersion}
        progress={updateProgress}
        robotBackOnline={rebootedRobotBack}
        appLive={hostPhase === 'embedded' && embedAppState.phase === 'live'}
        onStartUpdate={startDaemonUpdate}
        onCancelUpdate={cancelDaemonUpdate}
        onExitApp={exitBlockedApp}
        onSessionLost={handleUpdateSessionLost}
        onDismiss={dismissDaemonUpdate}
      />

      {welcomeBack === 'showing' && (
        <WelcomeBackOverlay
          userName={userName}
          onDone={() => setWelcomeBack('done')}
        />
      )}
    </>
  );
}

/* ─────────────────── helpers ─────────────────── */

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function readToken(): string | null {
  try {
    return window.sessionStorage.getItem('hf_token');
  } catch {
    return null;
  }
}

/* ─────────────────── Preview shell ───────────────────
 *
 * Pure visual harness gated by `?host-preview=...` (DEV only,
 * see `readPreviewPhase()`). Renders the same view tree as
 * `ReachyHostShellNormal` but with mock data and zero SDK
 * traffic, so reviewing the chrome doesn't require an OAuth
 * round-trip or a real robot.
 *
 * Switch via:
 *   ?host-preview=signing-in
 *   ?host-preview=picker
 *   ?host-preview=welcome
 *   ?host-preview=connecting
 *   ?host-preview=error
 */
function ReachyHostShellPreview({
  phase,
  appName,
  appIconUrl,
  appEmoji,
}: {
  phase: PreviewPhase;
} & ReachyHostShellProps): JSX.Element {
  const [welcomeShown, setWelcomeShown] = useState(phase === 'welcome');

  const fakeHostPhase: HostPhase =
    phase === 'signing-in'
      ? 'signing-in'
      : phase === 'error'
        ? 'error'
        : phase === 'picker' || phase === 'welcome'
          ? 'picking'
          : 'embedded';

  return (
    <>
      <GlobalStyles
        styles={{
          ':root': { '--reachy-host-topbar-h': '56px' },
          'html, body, #root': { height: '100%' },
          body: { margin: 0 },
        }}
      />
      <Stack
        sx={{
          height: '100%',
          color: 'text.primary',
          bgcolor: 'background.default',
        }}
      >
        <TopBar
          appName={appName}
          appIconUrl={appIconUrl}
          appEmoji={appEmoji}
          hostPhase={fakeHostPhase}
          userName={phase === 'signing-in' ? null : 'tfrere'}
          selectedRobotName={
            phase === 'connecting' ? 'Tabouret' : null
          }
          selectedRobotTransport={phase === 'connecting' ? 'wifi' : null}
          linkRttMs={phase === 'connecting' ? 38 : null}
          onSignOut={() => window.alert('Preview: sign-out')}
          onEndSession={() => {
            window.alert('Preview: end-session (no-op)');
          }}
        />

        <Box sx={{ flex: 1, position: 'relative', minHeight: 0 }}>
          {phase === 'signing-in' && (
            <SignInView
              appName={appName}
              isLocalDevMissingConfig={false}
              onSignIn={async () => {
                window.alert('Preview: sign-in (no-op)');
              }}
            />
          )}

          {(phase === 'picker' || phase === 'welcome') && (
            <PickerView
              robots={MOCK_ROBOTS}
              isRefreshing={false}
              preselectedRobotId={null}
              onSelect={(id) => window.alert(`Preview: selected ${id}`)}
            />
          )}

          {phase === 'connecting' && (
            <ConnectingView
              step="link"
              message="Opening secure link to Hugging Face"
            />
          )}

          {phase === 'error' && (
            <ErrorView
              message="The robot session ended unexpectedly."
              detail="WebRTC peer connection closed: peer left."
              onReload={() => window.location.reload()}
              onBackToPicker={() => {
                window.alert('Preview: back to picker');
              }}
            />
          )}
        </Box>
      </Stack>

      {welcomeShown && (
        <WelcomeBackOverlay
          userName="tfrere"
          onDone={() => setWelcomeShown(false)}
        />
      )}

      {/* Loud "PREVIEW MODE" badge so this harness can never be
       *  mistaken for a real session - the mock robots have
       *  caused that confusion at least once. */}
      <Box
        sx={{
          position: 'fixed',
          bottom: 12,
          right: 12,
          zIndex: 2000,
          px: 1.25,
          py: 0.5,
          borderRadius: 999,
          bgcolor: 'warning.main',
          color: 'warning.contrastText',
          fontSize: '0.65rem',
          fontWeight: 700,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          boxShadow: 2,
          pointerEvents: 'none',
        }}
      >
        Preview · {phase} · mock
      </Box>
    </>
  );
}
