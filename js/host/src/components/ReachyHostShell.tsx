/**
 * Top-level host shell. Wires the SDK, OAuth, bridge, and the
 * per-phase views together into one component.
 *
 * Phases (cf. SPEC §5):
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
import GlobalStyles from '@mui/material/GlobalStyles';
import Stack from '@mui/material/Stack';

import type { ReachyMiniInstance, RobotInfo } from '../lib/sdk-types';
import {
  encodeCredsToHash,
  type ConfigPayload,
  type CredsBundle,
  type LeavingReason,
  type ThemeMode,
} from '../lib/protocol';
import { resolveSignalingUrl } from '../lib/signalingUrl';
import { wipeHfSessionStorage } from '../lib/settings';
import { useHfProfile } from '../hooks/useHfProfile';
import { useOAuth } from '../hooks/useOAuth';
import { useRobots } from '../hooks/useRobots';
import { useHostBridge, type EmbedAppState } from '../hooks/useHostBridge';

import { ConnectingView } from './ConnectingView';
import { EmbedFrame } from './EmbedFrame';
import { ErrorView } from './ErrorView';
import { LeavingView } from './LeavingView';
import { PickerView } from './PickerView';
import { SignInView } from './SignInView';
import { TopBar, type HostPhase } from './TopBar';
import { WelcomeBackOverlay } from './WelcomeBackOverlay';

const LEAVING_TIMEOUT_MS = 1500;

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
  const previewPhase = readPreviewPhase();
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
  const [embedAppState, setEmbedAppState] = useState<EmbedAppState>({
    phase: 'boot',
    connectingStep: null,
    message: null,
  });
  const [errorPayload, setErrorPayload] = useState<{
    message: string;
    detail?: unknown;
  } | null>(null);
  /** Sticky flag: turns true once the boot-time OAuth-pending
   *  flag fires, stays true until the WelcomeBackOverlay's
   *  fade-out completes. Decoupled from `useOAuth`'s
   *  `isPostOauthReturn` (which flips false the moment the SDK
   *  reports authenticated) so the celebratory beat plays out
   *  even on a fast auth confirmation. */
  const [welcomeBackShown, setWelcomeBackShown] = useState<boolean>(false);

  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  /** Guards `host:init` sending: at most once per selected
   *  robot, and only after `embed:ready` arrived (StrictMode
   *  safe). */
  const initSentForRef = useRef<string | null>(null);
  /** Set when `embed:ready` is observed before the iframe ref
   *  is available, so we can flush as soon as the ref binds. */
  const embedReadyPendingRef = useRef<boolean>(false);
  /** One-shot latch for the welcome-back overlay. Without this,
   *  `isPostOauthReturn` can stay `true` after the overlay's
   *  fade-out (the dev sign-in path doesn't redirect, so nothing
   *  naturally clears the flag) and the effect re-fires the
   *  overlay in a loop. The ref is reset when the user signs out
   *  so a subsequent sign-in plays the anim again. */
  const welcomeBackShownOnceRef = useRef<boolean>(false);

  /* ─────────────────── External hooks ─────────────────── */

  const { isAuthenticated, userName, isPostOauthReturn, signIn, signOut } =
    useOAuth(sdk);
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
    refresh: refreshRobots,
  } = useRobots({
    hfToken,
    enabled: isAuthenticated && hostPhase === 'picking',
  });

  // Latch the welcome-back overlay on once the post-OAuth flag
  // fires AND the username is resolved. Gating on `userName`
  // matters because `isPostOauthReturn` can flip true a frame
  // or two before `setUserName(sdk.username)` lands (the OAuth
  // bootstrap effect updates them in separate batches in prod).
  // Without the gate the overlay mounts with "Welcome back" and
  // visibly flickers to "Hello, X" the next frame.
  //
  // The `welcomeBackShownOnceRef` guard makes this a one-shot
  // per auth session: after `onDone` flips `welcomeBackShown`
  // back to false, we MUST NOT re-mount the overlay even if
  // `isPostOauthReturn` is still true.
  //
  // The 800 ms fallback timer is a defensive cap: if the SDK
  // somehow authenticates without resolving a username, we
  // still show the (generic) overlay rather than swallowing the
  // welcome moment forever.
  useEffect(() => {
    if (welcomeBackShown || welcomeBackShownOnceRef.current) return;
    if (!isPostOauthReturn) return;

    if (userName) {
      welcomeBackShownOnceRef.current = true;
      setWelcomeBackShown(true);
      return;
    }

    const t = window.setTimeout(() => {
      if (welcomeBackShownOnceRef.current) return;
      welcomeBackShownOnceRef.current = true;
      setWelcomeBackShown(true);
    }, 800);
    return () => window.clearTimeout(t);
  }, [isPostOauthReturn, welcomeBackShown, userName]);

  // Reset the one-shot latch on sign-out so the next sign-in
  // can play the welcome anim again in the same tab.
  useEffect(() => {
    if (!isAuthenticated) {
      welcomeBackShownOnceRef.current = false;
    }
  }, [isAuthenticated]);

  /* ─────────────────── Bridge ─────────────────── */

  const flushInitIfReady = useCallback(() => {
    if (!iframeRef.current) return;
    if (!embedReadyPendingRef.current) return;
    if (selectedRobotId == null) return;
    if (initSentForRef.current === selectedRobotId) return;

    const robot = robots.find((r) => r.id === selectedRobotId);
    const bundle: Omit<CredsBundle, 'signalingUrl'> & {
      signalingUrl: string;
    } = {
      hfToken: readToken(),
      userName,
      robotPeerId: selectedRobotId,
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
      config: bundle.config,
      hostName,
      appName,
    });
    initSentForRef.current = selectedRobotId;
    embedReadyPendingRef.current = false;

    void robot; // referenced for clarity; could expose meta later
  }, [
    appName,
    hostName,
    initialConfig,
    robots,
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
    onError: ({ message, fatal, detail }) => {
      if (fatal) {
        setErrorPayload({ message, detail });
        setHostPhase('error');
      } else {
        console.warn(
          '[reachy-mini-sdk/host] embed reported non-fatal error:',
          message,
          detail,
        );
      }
    },
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
      initSentForRef.current = null;
      embedReadyPendingRef.current = false;
      // The host never opened an SSE (picker uses REST), so the
      // iframe's SDK gets a clean central slot with no prior peer
      // registered for this HF token. No releaseSdkForHandoff()
      // needed - that legacy hook tore down a connection we no
      // longer create.
      setEmbedAppState({
        phase: 'connecting',
        connectingStep: 'link',
        message: null,
      });
      setHostPhase('embedded');
    },
    [hostPhase, sdk],
  );

  /* ─────────────────── End session ─────────────────── */

  const endSession = useCallback(
    async (reason: LeavingReason): Promise<void> => {
      if (hostPhase !== 'embedded' && hostPhase !== 'error') return;
      if (iframeRef.current) {
        bridge.sendLeaving(iframeRef.current, reason, LEAVING_TIMEOUT_MS);
      }
      setHostPhase('leaving');

      await sleep(LEAVING_TIMEOUT_MS);

      // Unmount iframe (selectedRobotId = null) and clean up.
      // CRITICAL: do NOT call `wipeHfSessionStorage()` here. The
      // picker needs the HF token to keep its SSE listener open
      // and the REST safety-net polling alive. The token is wiped
      // only on full sign-out (`signOut`) and on `pagehide` (tab
      // close).
      setSelectedRobotId(null);
      setEmbedAppState({
        phase: 'boot',
        connectingStep: null,
        message: null,
      });
      initSentForRef.current = null;
      embedReadyPendingRef.current = false;

      // Back to picker. REST polling resumes via the
      // `enabled: hostPhase === 'picking'` gate in useRobots.
      setHostPhase('picking');
    },
    [bridge, hostPhase],
  );

  /* ─────────────────── Iframe URL ─────────────────── */

  const iframeUrl = useMemo(() => {
    if (selectedRobotId == null) return null;
    const bundle: CredsBundle = {
      hfToken: readToken(),
      userName,
      robotPeerId: selectedRobotId,
      signalingUrl: resolveSignalingUrl(),
      theme,
      config: initialConfig ?? null,
      hostName,
      appName,
    };
    const hash = encodeCredsToHash(bundle);
    console.info(
      `[host-debug] iframeUrl = ${window.location.origin}/?embedded=1#${hash}`,
    );
    // Hash creds carry the same data as host:init; the iframe
    // wipes the hash on its first tick (SPEC §8.2). The
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
    selectedRobotId,
    theme,
    userName,
  ]);

  /* ─────────────────── Pagehide cleanup ─────────────────── */

  useEffect(() => {
    const onPageHide = (): void => {
      // Best-effort: tell the embed it's leaving so it can
      // disconnect its SDK. We don't wait for an ack.
      if (iframeRef.current && (hostPhase === 'embedded' || hostPhase === 'leaving')) {
        bridge.sendLeaving(iframeRef.current, 'pagehide', 0);
      }
      wipeHfSessionStorage();
    };
    window.addEventListener('pagehide', onPageHide, { once: true });
    return () => window.removeEventListener('pagehide', onPageHide);
  }, [bridge, hostPhase]);

  /* ─────────────────── Render ─────────────────── */

  const showConnectingOverlay =
    hostPhase === 'embedded' &&
    (embedAppState.phase === 'boot' || embedAppState.phase === 'connecting');
  const showLeavingOverlay = hostPhase === 'leaving';

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
          selectedRobotName={
            selectedRobotId
              ? (robots.find((r) => r.id === selectedRobotId)?.meta?.name ??
                selectedRobotId)
              : null
          }
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

          {hostPhase === 'signing-in' && (
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
              onRefresh={refreshRobots}
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

      {welcomeBackShown && (
        <WelcomeBackOverlay
          userName={userName}
          onDone={() => setWelcomeBackShown(false)}
        />
      )}
    </>
  );
}

/* ─────────────────── helpers ─────────────────── */

function readToken(): string | null {
  try {
    return window.sessionStorage.getItem('hf_token');
  } catch {
    return null;
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
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
              onRefresh={() => window.alert('Preview: refresh')}
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
