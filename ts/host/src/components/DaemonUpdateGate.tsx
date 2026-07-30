/**
 * Daemon version gate for the standalone web shell.
 *
 * Web-only by construction
 * ────────────────────────
 * This lives in the shell, and the shell only renders on a direct
 * browser visit: the mobile app points its iframe straight at the
 * embed entry (`?embedded=1`), plays the host role itself, and runs its
 * own gate before an app can even be opened. So there is nothing to
 * disable here for the embedded case - the component is never mounted.
 *
 * Two tiers, unlike the mobile app
 * ────────────────────────────────
 * The mobile gate blocks as soon as the daemon trails the latest GitHub
 * release. Applied to public Spaces that would turn every daemon
 * release into a global kill switch, so the web policy splits:
 *
 *   block  - below `MIN_SUPPORTED_DAEMON_VERSION`. Full-screen, no way
 *            past it. Such a daemon also predates the OTA command, so
 *            the only honest advice is the desktop app.
 *   notice - merely behind the latest release. A dismissable card; the
 *            app stays fully usable and the update is an offer.
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

/** If the daemon acked the job but never restarted within this window,
 *  something failed silently - stop pretending it's still working. */
const UPDATE_STALL_TIMEOUT_MS = 180_000;

/** Upper bound on the post-restart wait. A Reachy is back on central
 *  well inside this; past it, telling the user to check on the robot
 *  beats an eternal spinner. */
const REBOOT_TIMEOUT_MS = 240_000;

/** Tail of install log lines kept on screen. Enough to show progress is
 *  real without turning the overlay into a console. */
const LOG_TAIL = 6;

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
  onSessionLost,
  onDismiss,
}: DaemonUpdateGateProps): JSX.Element | null {
  const [phase, setPhase] = useState<Phase>('idle');
  const [logLines, setLogLines] = useState<string[]>([]);
  const [failure, setFailure] = useState<string | null>(null);
  /** Once cleared, stay cleared: a late version read must not reopen a
   *  card the user already dismissed. */
  const clearedRef = useRef(false);
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
    severity !== null &&
    !clearedRef.current &&
    (severity === 'block' || appLive);
  const effectivePhase: Phase =
    phase === 'idle' && shouldPrompt ? 'prompt' : phase;

  // Fold the progress stream into the local phase.
  useEffect(() => {
    if (!progress) return;
    if (progress.status === 'in_progress') {
      const line = progress.line?.trim();
      if (line) setLogLines((prev) => [...prev, line].slice(-LOG_TAIL));
      return;
    }
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
  }, [progress, onSessionLost]);

  // The robot answered central again: the new daemon is up.
  useEffect(() => {
    if (phase !== 'rebooting' || !robotBackOnline) return;
    clearedRef.current = true;
    setPhase('done');
  }, [phase, robotBackOnline]);

  // Safety nets. The daemon acked but never restarted, or it restarted
  // and never came back - either way, stop spinning and say so.
  useEffect(() => {
    if (phase !== 'updating' && phase !== 'rebooting') return;
    const budget =
      phase === 'updating' ? UPDATE_STALL_TIMEOUT_MS : REBOOT_TIMEOUT_MS;
    const t = window.setTimeout(() => {
      setFailure(
        phase === 'updating'
          ? 'The robot never started the update.'
          : 'The robot did not come back online after the restart.',
      );
      setPhase('failed');
    }, budget);
    return () => window.clearTimeout(t);
  }, [phase]);

  const handleUpdateNow = useCallback(() => {
    setLogLines([]);
    setFailure(null);
    setPhase('updating');
    onStartUpdate();
  }, [onStartUpdate]);

  const handleDismiss = useCallback(() => {
    clearedRef.current = true;
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
        <PhaseIcon phase={effectivePhase} />

        <Stack spacing={1} sx={{ alignItems: 'center' }}>
          <Typography
            component="h2"
            sx={{
              fontSize: TYPO.hero,
              fontWeight: FONT_WEIGHT.bold,
              letterSpacing: '-0.2px',
            }}
          >
            {titleFor(effectivePhase, canSelfUpdate)}
          </Typography>
          <Typography
            sx={{
              fontSize: TYPO.md,
              color: 'text.secondary',
              lineHeight: 1.5,
            }}
          >
            {bodyFor(effectivePhase, currentVersion, failure)}
          </Typography>
        </Stack>

        {effectivePhase === 'updating' && logLines.length > 0 && (
          <InstallLog lines={logLines} />
        )}

        <Stack spacing={1} sx={{ width: '100%', alignItems: 'center' }}>
          {effectivePhase === 'prompt' && canSelfUpdate && (
            <PrimaryButton onClick={handleUpdateNow}>Update now</PrimaryButton>
          )}
          {(effectivePhase === 'done' || effectivePhase === 'failed') && (
            <PrimaryButton onClick={handleDismiss}>
              {effectivePhase === 'done' ? 'Back to my Reachies' : 'Close'}
            </PrimaryButton>
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

function InstallLog({ lines }: { lines: string[] }): JSX.Element {
  return (
    <Box
      component="pre"
      aria-live="polite"
      sx={{
        width: '100%',
        m: 0,
        p: 1.5,
        textAlign: 'left',
        fontSize: TYPO.tiny,
        lineHeight: 1.6,
        color: 'text.secondary',
        bgcolor: 'action.hover',
        borderRadius: `${RADIUS.md}px`,
        overflowX: 'auto',
      }}
    >
      {lines.join('\n')}
    </Box>
  );
}

function PrimaryButton({
  onClick,
  children,
}: {
  onClick(): void;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <Button
      onClick={onClick}
      variant="contained"
      sx={{
        textTransform: 'none',
        fontWeight: FONT_WEIGHT.semibold,
        borderRadius: `${RADIUS.pill}px`,
        px: 3,
      }}
    >
      {children}
    </Button>
  );
}

function PhaseIcon({ phase }: { phase: Phase }): JSX.Element {
  if (phase === 'updating' || phase === 'rebooting') {
    return <CircularProgress size={32} sx={{ color: 'text.secondary' }} />;
  }
  return (
    <Box component="div" aria-hidden sx={{ fontSize: 56, lineHeight: 1 }}>
      {phase === 'done' ? '✅' : phase === 'failed' ? '⚠️' : '🤖'}
    </Box>
  );
}

function titleFor(phase: Phase, canSelfUpdate: boolean): string {
  switch (phase) {
    case 'updating':
      return 'Updating your Reachy…';
    case 'rebooting':
      return 'Reachy is rebooting…';
    case 'done':
      return 'Reachy is up to date';
    case 'failed':
      return "Update didn't finish";
    default:
      return canSelfUpdate ? 'Update required' : 'Update from the desktop app';
  }
}

function bodyFor(
  phase: Phase,
  current: string | null,
  failure: string | null,
): string {
  switch (phase) {
    case 'updating':
      return 'Installing the latest software. Keep this tab open - the robot reboots when it is done, which takes a minute or two.';
    case 'rebooting':
      return 'The robot is restarting to finish the update. It will show up in your list again as soon as it is back.';
    case 'done':
      return 'Pick your Reachy again to start the app.';
    case 'failed':
      return failure
        ? `${failure} You can update it from the Reachy desktop app instead.`
        : 'You can update it from the Reachy desktop app instead.';
    default:
      // `prompt`, blocking tier only: the soft tier renders as a card.
      return current
        ? `This Reachy runs v${current}, which is too old to run web apps. Version v${MIN_SUPPORTED_DAEMON_VERSION} or newer is required. Install the Reachy desktop app to update it, then come back.`
        : `This Reachy needs version v${MIN_SUPPORTED_DAEMON_VERSION} or newer. Install the Reachy desktop app to update it, then come back.`;
  }
}
