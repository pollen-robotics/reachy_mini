/**
 * Daemon version gate for the standalone web shell. Two tiers: `block`
 * below `MIN_SUPPORTED_DAEMON_VERSION`, a dismissable `notice` when
 * merely behind latest - policy and rationale live in
 * `lib/daemonRelease.ts` (and APP_CREATION_GUIDE §13.6). Web-only by
 * construction: the mobile app points its iframe straight at the embed
 * entry and runs its own gate, so this component is never mounted there.
 *
 * How an update ends
 * ──────────────────
 * The install finishes with a daemon restart, which kills the session.
 * The embed reports that as `rebooting`, and from there the page has no
 * link left to observe the robot: the shell tears the iframe down and
 * hands us back over the picker, whose central polling tells us when
 * the robot is online again. That is why this component never tries to
 * resume the app itself - it returns the user to a picker that already
 * knows how to reconnect.
 */
import type { JSX } from 'react';
import { useCallback, useEffect, useRef, useState } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

import {
  MIN_SUPPORTED_DAEMON_VERSION,
  daemonUpdateSeverity,
  supportsSelfUpdate,
} from '../lib/daemonRelease';
import { FONT_WEIGHT, LAYOUT, RADIUS, TYPO } from '../lib/tokens';

/** Where to get the desktop app, which can update a daemon too old for
 *  OTA. Same target the mobile gate uses. */
const DESKTOP_APP_URL = 'https://huggingface.co/spaces/pollen-robotics/reachy-mini';

/** Public troubleshooting docs. */
const TROUBLESHOOTING_URL = 'https://huggingface.co/docs/reachy_mini/troubleshooting';

/** If the update goes SILENT - no progress frame at all - for this
 *  long, something failed; stop pretending it's still working. The
 *  timer re-arms on every progress line, so it measures silence, not
 *  total install time: a slow-but-reporting pip on weak wifi can run
 *  as long as it keeps talking. */
const UPDATE_STALL_TIMEOUT_MS = 180_000;

/** Upper bound on the post-restart wait. A Reachy is back on central
 *  well inside this; past it, telling the user to check on the robot
 *  beats an eternal spinner. */
const REBOOT_TIMEOUT_MS = 240_000;

/**
 * `updating` and later are driven by the flow; `prompt` is derived from
 * the version comparison, and `idle` means the user cleared it.
 */
type Phase = 'idle' | 'prompt' | 'updating' | 'rebooting' | 'done' | 'failed';

export interface DaemonUpdateProgress {
  status: 'in_progress' | 'rebooting' | 'done' | 'failed';
  line?: string | null;
  error?: string | null;
}

export interface DaemonUpdateGateProps {
  /** Daemon version the embed reported, or `null` while unknown. */
  currentVersion: string | null;
  /** Latest published release, or `null` when it can't be resolved. */
  latestVersion: string | null;
  /** Last `embed:update-progress` payload, or `null` before any. */
  progress: DaemonUpdateProgress | null;
  /** True once the robot we're updating is listed on central again
   *  AFTER having been seen absent at least once (the shell's
   *  offline-first reboot watch, see `lib/rebootWatch.ts` - a stale
   *  pre-reboot listing must not complete the gate). Only meaningful
   *  after the shell has torn the iframe down and resumed polling -
   *  `false` at every other point in the flow. */
  robotBackOnline: boolean;
  /** True once the embedded app is interactive. The soft notice waits
   *  for it so an optional offer doesn't land on top of the connecting
   *  splash; the blocking tier ignores it, since the whole point is to
   *  stop the user before the app opens. */
  appLive: boolean;
  /** Ask the embed to start the daemon update. */
  onStartUpdate(): void;
  /** The gate gave up on the update (its stall timer fired with the
   *  session still alive). The shell should relay `host:cancel-update`
   *  so the embed disarms its update-mode plumbing (sessionStopped
   *  translator + auto-reconnect stand-down) - otherwise the user's
   *  next NORMAL end-session replays a stale `rebooting` frame. */
  onCancelUpdate?(): void;
  /** Escape from the blocking tier: end the session and return to the
   *  picker. The block only paints AFTER `startSession()` succeeded
   *  (that's when `daemonVersion` arrives), so a live session and an
   *  awake robot sit behind the overlay - without this, closing the
   *  tab is the only way to release the app-slot lock. */
  onExitApp?(): void;
  /** The session is over (the daemon is restarting): drop the iframe
   *  and go back to the picker, keeping this overlay on top. */
  onSessionLost(): void;
  /** Dismiss the gate for the rest of this session. */
  onDismiss(): void;
}

export function DaemonUpdateGate({
  currentVersion,
  latestVersion,
  progress,
  robotBackOnline,
  appLive,
  onStartUpdate,
  onCancelUpdate,
  onExitApp,
  onSessionLost,
  onDismiss,
}: DaemonUpdateGateProps): JSX.Element | null {
  const [phase, setPhase] = useState<Phase>('idle');
  const [failure, setFailure] = useState<string | null>(null);
  /** Once cleared, stay cleared: a late version read must not reopen a
   *  card the user already dismissed. State, not a ref: the dismissed
   *  card must disappear on OUR render, without counting on the
   *  shell's onDismiss to happen to re-render the parent. */
  const [cleared, setCleared] = useState(false);
  /** `onSessionLost` is a one-way door (it unmounts the iframe), and
   *  its identity changes as the shell tears state down, which would
   *  otherwise re-run the progress effect and fire it again. */
  const sessionLostRef = useRef(false);

  const severity = daemonUpdateSeverity(currentVersion, latestVersion);
  const canSelfUpdate = supportsSelfUpdate(currentVersion);

  // Derive the prompt during render rather than from an effect: an
  // effect would leave one frame where the app is visible before the
  // gate mounts, which reads as a flash of content the user isn't
  // supposed to act on.
  const shouldPrompt =
    severity !== null && !cleared && (severity === 'block' || appLive);
  const effectivePhase: Phase =
    phase === 'idle' && shouldPrompt ? 'prompt' : phase;

  // Fold the progress stream into the local phase - but only while an
  // update WE drove is actually in flight. Frames arriving in any
  // other phase are stale: after a local stall timeout (`failed`) the
  // embed's sessionStopped translator may still fire on the user's
  // next NORMAL end-session, and folding that `rebooting` here would
  // resurrect a dismissed gate over the picker and call onSessionLost
  // in the middle of a clean leave.
  useEffect(() => {
    if (!progress) return;
    if (phase !== 'updating' && phase !== 'rebooting') return;
    // `in_progress` frames only matter as liveness: they re-arm the
    // stall timer below (which depends on `progress`).
    if (progress.status === 'in_progress') return;
    if (progress.status === 'failed') {
      setFailure(progress.error ?? null);
      setPhase('failed');
      return;
    }
    // `rebooting` / `done`: the install is out of our hands. Drop the
    // iframe so the shell can resume central polling - that's the only
    // way we get to see the robot come back.
    setPhase('rebooting');
    if (sessionLostRef.current) return;
    sessionLostRef.current = true;
    onSessionLost();
  }, [progress, phase, onSessionLost]);

  // The robot answered central again: the new daemon is up.
  useEffect(() => {
    if (phase !== 'rebooting' || !robotBackOnline) return;
    setCleared(true);
    setPhase('done');
  }, [phase, robotBackOnline]);

  // Safety nets. The update went silent, or the robot restarted and
  // never came back - either way, stop spinning and say so. `progress`
  // is a dependency ON PURPOSE: every frame re-arms the timer, so the
  // updating budget bounds silence since the last line, not the whole
  // install.
  useEffect(() => {
    if (phase !== 'updating' && phase !== 'rebooting') return;
    const budget =
      phase === 'updating' ? UPDATE_STALL_TIMEOUT_MS : REBOOT_TIMEOUT_MS;
    const t = window.setTimeout(() => {
      setFailure(
        phase === 'updating'
          ? 'The robot stopped responding during the update.'
          : 'The robot did not come back online after the restart.',
      );
      setPhase('failed');
      // The session is still alive in the updating case: tell the
      // embed we gave up so it disarms its translator and restores
      // auto-reconnect (fire-and-forget, older embeds ignore it).
      if (phase === 'updating') onCancelUpdate?.();
    }, budget);
    return () => window.clearTimeout(t);
  }, [phase, progress, onCancelUpdate]);

  const handleUpdateNow = useCallback(() => {
    setFailure(null);
    setPhase('updating');
    onStartUpdate();
  }, [onStartUpdate]);

  const handleDismiss = useCallback(() => {
    setCleared(true);
    setPhase('idle');
    onDismiss();
  }, [onDismiss]);

  if (effectivePhase === 'idle') return null;

  // The soft tier stays a card in the corner: the app behind it works,
  // and the user is entitled to ignore us.
  if (effectivePhase === 'prompt' && severity === 'notice') {
    return (
      <UpdateNoticeCard
        currentVersion={currentVersion}
        latestVersion={latestVersion}
        onUpdate={handleUpdateNow}
        onDismiss={handleDismiss}
      />
    );
  }

  // `idle` returned null above and the notice tier returned the card,
  // so only full-screen phases reach this point.
  const copy = COPY[effectivePhase as Exclude<Phase, 'idle'>];
  const copyCtx: CopyCtx = {
    canSelfUpdate,
    current: currentVersion,
    failure,
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        inset: 0,
        // Above the connecting / leaving overlays (1300): once an update
        // is running, its narrative outranks the reconnection churn it
        // is about to cause.
        zIndex: 1400,
        bgcolor: 'background.default',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 3,
      }}
    >
      <Stack
        spacing={2.5}
        sx={{
          alignItems: 'center',
          textAlign: 'center',
          width: '100%',
          maxWidth: LAYOUT.contentMaxWidth,
        }}
      >
        {copy.icon === 'spinner' ? (
          <CircularProgress size={32} sx={{ color: 'text.secondary' }} />
        ) : (
          <Box component="div" aria-hidden sx={{ fontSize: 56, lineHeight: 1 }}>
            {copy.icon}
          </Box>
        )}

        <Stack spacing={1} sx={{ alignItems: 'center' }}>
          <Typography
            component="h2"
            sx={{
              fontSize: TYPO.hero,
              fontWeight: FONT_WEIGHT.bold,
              letterSpacing: '-0.2px',
            }}
          >
            {copy.title(copyCtx)}
          </Typography>
          <Typography
            sx={{
              fontSize: TYPO.md,
              color: 'text.secondary',
              lineHeight: 1.5,
            }}
          >
            {copy.body(copyCtx)}
          </Typography>
        </Stack>

        <Stack spacing={1} sx={{ width: '100%', alignItems: 'center' }}>
          {effectivePhase === 'prompt' && canSelfUpdate && (
            <Button variant="contained" sx={PRIMARY_SX} onClick={handleUpdateNow}>
              Update now
            </Button>
          )}
          {(effectivePhase === 'done' || effectivePhase === 'failed') && (
            <Button variant="contained" sx={PRIMARY_SX} onClick={handleDismiss}>
              {effectivePhase === 'done' ? 'Back to my Reachies' : 'Close'}
            </Button>
          )}
          {((effectivePhase === 'prompt' && !canSelfUpdate) ||
            effectivePhase === 'failed') && (
            <>
              <Button
                href={DESKTOP_APP_URL}
                target="_blank"
                rel="noopener"
                sx={{ textTransform: 'none', fontSize: TYPO.sm }}
              >
                Get the desktop app
              </Button>
              <Button
                href={TROUBLESHOOTING_URL}
                target="_blank"
                rel="noopener"
                sx={{
                  textTransform: 'none',
                  fontSize: TYPO.sm,
                  color: 'text.secondary',
                }}
              >
                Troubleshooting
              </Button>
            </>
          )}
          {effectivePhase === 'prompt' && onExitApp && (
            // The blocking tier must not be a dead end: a live session
            // (and an awake robot) sits behind this overlay. Give the
            // user a way to release it without closing the tab.
            <Button
              onClick={onExitApp}
              variant="outlined"
              sx={{
                textTransform: 'none',
                fontSize: TYPO.sm,
                mt: 1,
              }}
            >
              Back to my Reachies
            </Button>
          )}
        </Stack>
      </Stack>
    </Box>
  );
}

/** Soft tier: the app works, we're only letting the user know. */
function UpdateNoticeCard({
  currentVersion,
  latestVersion,
  onUpdate,
  onDismiss,
}: {
  currentVersion: string | null;
  latestVersion: string | null;
  onUpdate(): void;
  onDismiss(): void;
}): JSX.Element {
  return (
    <Paper
      elevation={6}
      sx={{
        position: 'fixed',
        left: 16,
        bottom: 16,
        zIndex: 1200,
        p: 2,
        maxWidth: 340,
        borderRadius: `${RADIUS.lg}px`,
      }}
    >
      <Stack spacing={1.25}>
        <Typography sx={{ fontSize: TYPO.md, fontWeight: FONT_WEIGHT.semibold }}>
          A Reachy update is available
        </Typography>
        <Typography sx={{ fontSize: TYPO.sm, color: 'text.secondary', lineHeight: 1.5 }}>
          {currentVersion && latestVersion
            ? `This Reachy runs v${currentVersion}; v${latestVersion} is out. Updating takes about two minutes and reboots the robot.`
            : 'Updating takes about two minutes and reboots the robot.'}
        </Typography>
        <Stack direction="row" spacing={1} sx={{ justifyContent: 'flex-end' }}>
          <Button
            onClick={onDismiss}
            sx={{ textTransform: 'none', fontSize: TYPO.sm, color: 'text.secondary' }}
          >
            Not now
          </Button>
          <Button
            onClick={onUpdate}
            variant="contained"
            sx={{ textTransform: 'none', fontSize: TYPO.sm }}
          >
            Update
          </Button>
        </Stack>
      </Stack>
    </Paper>
  );
}

const PRIMARY_SX = {
  textTransform: 'none',
  fontWeight: FONT_WEIGHT.semibold,
  borderRadius: `${RADIUS.pill}px`,
  px: 3,
} as const;

interface CopyCtx {
  canSelfUpdate: boolean;
  current: string | null;
  failure: string | null;
}

/** Icon + title + body per full-screen phase, indexed once at render.
 *  `'spinner'` renders a CircularProgress instead of an emoji glyph.
 *  The `prompt` entries cover the blocking tier only - the soft tier
 *  renders as `UpdateNoticeCard`. */
const COPY: Record<
  Exclude<Phase, 'idle'>,
  { icon: string; title(ctx: CopyCtx): string; body(ctx: CopyCtx): string }
> = {
  prompt: {
    icon: '🤖',
    title: ({ canSelfUpdate }) =>
      canSelfUpdate ? 'Update required' : 'Update from the desktop app',
    body: ({ current }) =>
      current
        ? `This Reachy runs v${current}, which is too old to run web apps. Version v${MIN_SUPPORTED_DAEMON_VERSION} or newer is required. Install the Reachy desktop app to update it, then come back.`
        : `This Reachy needs version v${MIN_SUPPORTED_DAEMON_VERSION} or newer. Install the Reachy desktop app to update it, then come back.`,
  },
  updating: {
    icon: 'spinner',
    title: () => 'Updating your Reachy…',
    body: () =>
      'Installing the latest software. Keep this tab open - the robot reboots when it is done, which takes a minute or two.',
  },
  rebooting: {
    icon: 'spinner',
    title: () => 'Reachy is rebooting…',
    body: () =>
      'The robot is restarting to finish the update. It will show up in your list again as soon as it is back.',
  },
  done: {
    icon: '✅',
    title: () => 'Reachy is up to date',
    body: () => 'Pick your Reachy again to start the app.',
  },
  failed: {
    icon: '⚠️',
    title: () => "Update didn't finish",
    body: ({ failure }) =>
      failure
        ? `${failure} You can update it from the Reachy desktop app instead.`
        : 'You can update it from the Reachy desktop app instead.',
  },
};
