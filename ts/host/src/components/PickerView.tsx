/**
 * Robot picker, aligned with `reachy_mini_mobile_app/src/ui/
 * screens/ScanScreen.tsx`.
 *
 * Layout (the shared TopBar - identity + sign-out - sits ABOVE
 * this view and is owned by `ReachyHostShell`):
 *   ┌──────────────────────────────────────────┐
 *   │              (reachy-buste)              │
 *   │                                          │
 *   │        Your Reachies      ( ↻ )         │  ← discreet auto-spinner,
 *   │   N online · linked to your HF account   │    right of the title
 *   │                                          │
 *   │   ┌────────────────────────────────┐     │
 *   │   │ [reachy] ● Name             >  │     │
 *   │   └────────────────────────────────┘     │
 *   │   ┌────────────────────────────────┐     │
 *   │   │ [reachy] ● Other            🔒 │     │  (busy)
 *   │   └────────────────────────────────┘     │
 *   └──────────────────────────────────────────┘
 *
 *  - Hero illustration: `reachy-buste.svg`, same asset as the
 *    splash, gives the screen a brand identity beyond the cards.
 *  - Robot cards: avatar + name + hardware-id tag + trailing
 *    chevron (or lock when busy).
 *  - Refresh: NOT a button anymore. The list refreshes on its own
 *    (realtime SSE + a 60 s safety-net poll in `useRobots`), so the
 *    former sticky "Refresh" button was pure redundancy. Mirroring
 *    the mobile ScanScreen, it's now a discreet NON-interactive
 *    spinner to the right of the title, shown only while a fetch is
 *    in flight (with a min-visible floor + trailing debounce so a
 *    fast fetch isn't clipped and a burst reads as one spin).
 *  - Quiet initial load: the very first fetch (no data yet) collapses
 *    the whole screen to a single centered spinner - no hero, no
 *    header - so the first paint is calm while we wait on central.
 */
import type { JSX } from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Box,
  CircularProgress,
  ListItemButton,
  Stack,
  Tooltip,
  Typography,
  alpha,
  keyframes,
} from '@mui/material';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import LockIcon from '@mui/icons-material/Lock';
import RefreshIcon from '@mui/icons-material/Refresh';

import { reachyBusteSvg, reachyStandardSvg } from '../assets';
import type { RobotInfo } from '../lib/sdk-types';
import { FONT_WEIGHT, LAYOUT, TYPO } from '../lib/tokens';
import { VariantTag } from '../ui/design/MetaPill';

export interface PickerViewProps {
  robots: RobotInfo[];
  /** Any in-flight central fetch (initial load, safety-net poll, or
   *  SSE-triggered refetch). Drives the discreet auto-spinner to the
   *  right of the title. */
  isRefreshing: boolean;
  /** `true` only on the very first fetch with no data yet. Drives the
   *  quiet initial paint (a bare centered spinner, no hero / header).
   *  Distinct from `isRefreshing`, which is also `true` on background
   *  polls once the list is populated. */
  isLoading?: boolean;
  /** Last error message from the central listener / REST fetch,
   *  or `null` if everything's healthy. Surfaces as an error state
   *  card when the list is empty so the user knows the screen is
   *  silent for a reason. */
  error?: string | null;
  preselectedRobotId: string | null;
  onSelect(robotId: string): void;
}

export function PickerView({
  robots,
  isRefreshing,
  isLoading = false,
  error,
  preselectedRobotId,
  onSelect,
}: PickerViewProps): JSX.Element {
  const hasRobots = robots.length > 0;
  // First-ever fetch, nothing to show yet → quiet, chrome-free paint.
  const isInitialLoading = !hasRobots && isLoading;

  // Auto-select a preselected robot when it appears free.
  useEffect(() => {
    if (!preselectedRobotId) return;
    const target = robots.find((r) => r.id === preselectedRobotId);
    if (target && !target.busy) onSelect(target.id);
  }, [preselectedRobotId, robots, onSelect]);

  return (
    <Stack
      sx={{
        height: '100%',
        width: '100%',
        bgcolor: 'background.default',
      }}
    >
      <Stack
        sx={{
          flex: 1,
          minHeight: 0,
          width: '100%',
          overflowY: 'auto',
        }}
      >
        <Stack
          spacing={3}
          sx={{
            m: 'auto',
            width: '100%',
            maxWidth: LAYOUT.contentMaxWidth,
            px: 3,
            py: 4,
          }}
        >
          {isInitialLoading ? (
            <Box
              sx={{
                width: '100%',
                display: 'flex',
                justifyContent: 'center',
                py: 6,
              }}
            >
              <CircularProgress
                thickness={2.5}
                sx={{ color: (theme) => alpha(theme.palette.text.primary, 0.3) }}
              />
            </Box>
          ) : (
            <>
              <Stack spacing={2} sx={{ alignItems: 'center' }}>
                <HeroBuste />
                <RobotsHeader
                  isRefreshing={isRefreshing}
                  hasError={Boolean(error)}
                  count={robots.length}
                  hasRobots={hasRobots}
                />
              </Stack>

              {hasRobots ? (
                <Stack
                  spacing={2.5}
                  sx={{ width: '100%' }}
                  role="list"
                  aria-label="Available Reachies"
                >
                  {robots.map((robot) => (
                    <RemoteRobotCard
                      key={robot.id}
                      robot={robot}
                      disabled={Boolean(robot.busy)}
                      onTap={() => onSelect(robot.id)}
                    />
                  ))}
                </Stack>
              ) : error ? (
                <CenteredMessageState
                  title="Couldn't reach Hugging Face"
                  subtitle={error}
                />
              ) : (
                <CenteredMessageState title="No Reachy online" />
              )}
            </>
          )}
        </Stack>
      </Stack>
    </Stack>
  );
}

/* ─────────────────── Hero ─────────────────── */

function HeroBuste(): JSX.Element {
  return (
    <Box
      sx={{
        width: 144,
        height: 144,
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <img
        src={reachyBusteSvg}
        alt=""
        aria-hidden
        style={{
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          userSelect: 'none',
          pointerEvents: 'none',
        }}
      />
    </Box>
  );
}

/* ─────────────────── Header ─────────────────── */

function RobotsHeader({
  isRefreshing,
  hasError,
  count,
  hasRobots,
}: {
  isRefreshing: boolean;
  hasError: boolean;
  count: number;
  hasRobots: boolean;
}): JSX.Element {
  const subtitle = useMemo(() => {
    if (!hasRobots && isRefreshing) return 'Looking for your Reachies…';
    if (!hasRobots && hasError) return 'Connection lost - retrying';
    if (!hasRobots) return 'None linked to your Hugging Face account are online';
    if (count === 1) return '1 online · linked to your Hugging Face account';
    return `${count} online · linked to your Hugging Face account`;
  }, [hasRobots, hasError, isRefreshing, count]);

  return (
    <Stack
      spacing={0.5}
      sx={{
        alignItems: 'center',
        width: '100%',
      }}
    >
      {/* Title row: the title stays optically centered, flanked by two
          equal fixed-width slots - a mirror spacer on the left and the
          discreet refresh indicator on the right. Both keep their width
          whether or not the spinner shows, so no layout shift when a
          refresh starts / ends. */}
      <Box
        sx={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1,
        }}
      >
        <Box sx={{ width: 40, flexShrink: 0 }} aria-hidden />
        <Typography
          component="h1"
          sx={{
            m: 0,
            textAlign: 'center',
            fontSize: TYPO.display,
            fontWeight: FONT_WEIGHT.semibold,
            color: 'text.primary',
            letterSpacing: '-0.3px',
          }}
        >
          Your Reachies
        </Typography>
        <RefreshIndicator isRefreshing={isRefreshing} />
      </Box>
      <Typography
        sx={{
          fontSize: TYPO.sm,
          color: 'text.secondary',
          textAlign: 'center',
          lineHeight: 1.5,
          minHeight: '3em',
        }}
      >
        {subtitle}
      </Typography>
    </Stack>
  );
}

/* ─────────────────── Refresh activity indicator ─────────────────── */

// Floor we keep the indicator on screen once a refresh starts, so an
// instant (cached) fetch still shows a complete, smooth turn instead of
// a one-frame flash that reads as "cut off".
const MIN_VISIBLE_MS = 900;

// Trailing debounce: a single logical refresh often fires two fetches
// back-to-back (a poll plus an SSE-driven refetch), so `isRefreshing`
// flips true→false→true. We hold the spinner for this long after
// activity STOPS; any follow-up fetch within the window re-arms it,
// coalescing the burst into one continuous spinner.
const TAIL_DEBOUNCE_MS = 600;

// Opacity cross-fade on enter / leave so the glyph eases in and out
// instead of popping. It stays mounted (still spinning) through the
// fade-out, then unmounts once fully transparent.
const FADE_MS = 320;
const REST_OPACITY = 0.32;

const refreshSpinKeyframes = keyframes`
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
`;

/**
 * Discreet, NON-interactive refresh indicator to the right of the
 * title. There's nothing to tap: the list refreshes on its own
 * (realtime SSE push + a 60 s safety-net poll in `useRobots`), so this
 * is pure feedback - the slowly-spinning arrows show ONLY while a fetch
 * is in flight and are absent otherwise. Lives in a fixed-width slot so
 * the title stays optically centered whether or not it shows. Mirrors
 * the mobile `ScanScreen` `RefreshIndicator`.
 */
function RefreshIndicator({ isRefreshing }: { isRefreshing: boolean }): JSX.Element {
  const [visible, setVisible] = useState(false);
  const startedAtRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (isRefreshing) {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
      if (!visible) {
        startedAtRef.current = Date.now();
        setVisible(true);
      }
    } else if (visible && !timerRef.current) {
      const sinceStart = Date.now() - startedAtRef.current;
      const delay = Math.max(MIN_VISIBLE_MS - sinceStart, TAIL_DEBOUNCE_MS);
      timerRef.current = setTimeout(() => {
        setVisible(false);
        timerRef.current = null;
      }, delay);
    }
  }, [isRefreshing, visible]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // Fade layer: `mounted` keeps the glyph in the DOM through its
  // fade-out, `shown` drives the opacity target. Toggling `shown` one
  // frame after mount lets the CSS transition animate the enter.
  const [mounted, setMounted] = useState(false);
  const [shown, setShown] = useState(false);
  const fadeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (fadeTimerRef.current) {
      clearTimeout(fadeTimerRef.current);
      fadeTimerRef.current = null;
    }
    if (visible) {
      setMounted(true);
      const id = requestAnimationFrame(() => setShown(true));
      return () => cancelAnimationFrame(id);
    }
    setShown(false);
    fadeTimerRef.current = setTimeout(() => setMounted(false), FADE_MS);
    return () => {
      if (fadeTimerRef.current) clearTimeout(fadeTimerRef.current);
    };
  }, [visible]);

  return (
    <Box
      sx={{
        width: 40,
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      aria-hidden={!visible}
    >
      {mounted && (
        <RefreshIcon
          aria-label="Refreshing robot list"
          sx={{
            fontSize: 22,
            color: 'text.disabled',
            opacity: shown ? REST_OPACITY : 0,
            transition: `opacity ${FADE_MS}ms ease`,
            transformOrigin: 'center',
            animation: `${refreshSpinKeyframes} 1.4s linear infinite`,
          }}
        />
      )}
    </Box>
  );
}

/* ─────────────────── Robot card ─────────────────── */

function RemoteRobotCard({
  robot,
  disabled,
  onTap,
}: {
  robot: RobotInfo;
  disabled: boolean;
  onTap(): void;
}): JSX.Element {
  const name = robot.meta?.name ?? robot.id;
  // Mobile parity: prefer the daemon-provided hardware id (a short
  // human-friendly serial) over the longer central peer id when
  // available. Slice to 5 chars - enough to disambiguate without
  // dominating the row, same trim as the mobile card.
  const rawTag = robot.hardwareId ?? robot.id ?? '';
  const idTag = rawTag.slice(0, 5);
  const idLabel = idTag ? `#${idTag}` : '—';
  const busy = Boolean(robot.busy);
  const transport = robot.transport ?? null;

  return (
    <ListItemButton
      disabled={disabled}
      onClick={onTap}
      sx={{
        p: 2,
        pr: 2.5,
        // Same min-height as the loading / empty / error state
        // cards below, so the body slot doesn't snap between
        // states as the user transitions between them.
        minHeight: STATE_CARD_MIN_HEIGHT,
        borderRadius: '14px',
        bgcolor: 'background.paper',
        border: (theme) => `1px solid ${theme.palette.divider}`,
        boxShadow: (theme) =>
          theme.palette.mode === 'dark'
            ? '0 1px 0 rgba(255,255,255,0.04) inset, 0 2px 6px rgba(0,0,0,0.35)'
            : '0 1px 0 rgba(255,255,255,0.6) inset, 0 1px 2px rgba(15,23,42,0.04), 0 2px 6px rgba(15,23,42,0.05)',
        transition: (theme) =>
          theme.transitions.create(['transform'], {
            duration: theme.transitions.duration.shortest,
          }),
        // Mobile parity: no hover override. The press feedback
        // (`scale(0.99)` on `:active`) is what users expect; a
        // hover colour shift made every card feel "noisy" once
        // the cursor settled on the list.
        '&:hover': {
          bgcolor: 'background.paper',
        },
        '&:active': {
          transform: 'scale(0.99)',
        },
      }}
    >
      <Stack
        direction="row"
        spacing={2}
        sx={{
          alignItems: 'center',
          width: '100%'
        }}>
        <CardAvatar />
        {/* Two-row identity grid: name + transport chip, then id.
            Mirrors the mobile `RemoteRobotCard` so users moving
            between mobile and desktop pick out the same
            elements. */}
        <Stack sx={{ flex: 1, minWidth: 0 }} spacing={0.25}>
          <Stack
            direction="row"
            spacing={1}
            sx={{
              alignItems: 'center',
              minWidth: 0
            }}>
            <Typography
              sx={{
                minWidth: 0,
                fontSize: TYPO.lg,
                fontWeight: FONT_WEIGHT.bold,
                color: 'text.primary',
                letterSpacing: '-0.1px',
                lineHeight: 1.2,
                flexShrink: 1,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              noWrap
            >
              {name}
            </Typography>
            {transport ? (
              <Box sx={{ flexShrink: 0 }}>
                <VariantTag transport={transport} />
              </Box>
            ) : null}
          </Stack>
          <Typography
            component="span"
            title="Hardware id"
            sx={{
              fontSize: TYPO.xs,
              fontFamily: 'monospace',
              color: (theme) =>
                theme.palette.mode === 'dark'
                  ? 'rgba(255,255,255,0.40)'
                  : 'rgba(0,0,0,0.36)',
              whiteSpace: 'nowrap',
            }}
          >
            {idLabel}
          </Typography>
        </Stack>
        {/* Trailing affordance: chevron when tappable, lock when
            the robot already has an active session. The lock
            tooltip surfaces `activeApp` when the consumer
            advertised a meta.name so a curious user can read
            who's holding it without us blowing up the card with
            a chip. Placement="left" matches mobile. */}
        {busy ? (
          <Tooltip
            title={robot.activeApp ? `In use · ${robot.activeApp}` : 'In use'}
            placement="left"
          >
            <LockIcon
              aria-label={
                robot.activeApp ? `In use - ${robot.activeApp}` : 'In use'
              }
              sx={{
                color: 'text.disabled',
                flexShrink: 0,
                fontSize: 20,
              }}
            />
          </Tooltip>
        ) : (
          <ChevronRightIcon
            sx={{
              color: 'primary.main',
              flexShrink: 0,
              fontSize: 22,
            }}
          />
        )}
      </Stack>
    </ListItemButton>
  );
}

/**
 * Card-sized avatar mirroring `RobotAvatar` from the mobile shell.
 *
 * The reachy-standard SVG (720 × 721) is not visually balanced:
 *   - antennas live in the upper ~17%
 *   - the head body fills the middle ~66%
 *   - the lower ~17% is whitespace
 *
 * To centre the *head body* (not the SVG's geometric centre)
 * inside the disc, we render the SVG at 155% of the disc width
 * and shift it up by 60% of its own height. The antennas
 * naturally peek a few pixels above the rim, the head fills the
 * disc, and the empty bottom of the SVG is invisible
 * (transparent background, `overflow: visible` on the disc so
 * antennas don't get clipped).
 */
function CardAvatar(): JSX.Element {
  return (
    <Box
      sx={{
        width: 72,
        height: 72,
        flexShrink: 0,
        position: 'relative',
        borderRadius: '50%',
        bgcolor: (theme) =>
          theme.palette.mode === 'dark'
            ? 'rgba(255,255,255,0.04)'
            : 'rgba(0,0,0,0.03)',
        border: (theme) =>
          `1px solid ${
            theme.palette.mode === 'dark'
              ? 'rgba(255,255,255,0.06)'
              : 'rgba(0,0,0,0.04)'
          }`,
        // Antennas must break the disc silhouette.
        overflow: 'visible',
      }}
    >
      <Box
        component="img"
        src={reachyStandardSvg}
        alt=""
        aria-hidden
        sx={{
          position: 'absolute',
          width: '155%',
          height: 'auto',
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -60%)',
          userSelect: 'none',
          pointerEvents: 'none',
        }}
      />
    </Box>
  );
}

/* ─────────────────── State cards (empty / error) ───── */

/**
 * Shared minimum height for every "single-card" state.
 *
 * Tuned to match the natural height of a populated
 * `RemoteRobotCard` so the body slot doesn't snap between
 * states (empty → error → 1 robot → N robots).
 */
const STATE_CARD_MIN_HEIGHT = 104;

/**
 * Card chrome shared by the empty / error states.
 *
 * Mirrors the surface used by `RemoteRobotCard` (paper bg,
 * theme divider border, same dual inset + drop shadow) so the
 * states form a coherent visual family with the actual robot
 * rows below them.
 */
function StateCard({
  children,
}: {
  children: React.ReactNode;
}): JSX.Element {
  return (
    <Box
      sx={{
        width: '100%',
        minHeight: STATE_CARD_MIN_HEIGHT,
        px: 3,
        py: 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: '14px',
        bgcolor: 'background.paper',
        border: (theme) => `1px solid ${theme.palette.divider}`,
        boxShadow: (theme) =>
          theme.palette.mode === 'dark'
            ? '0 1px 0 rgba(255,255,255,0.04) inset, 0 2px 6px rgba(0,0,0,0.35)'
            : '0 1px 0 rgba(255,255,255,0.6) inset, 0 1px 2px rgba(15,23,42,0.04), 0 2px 6px rgba(15,23,42,0.05)',
      }}
    >
      {children}
    </Box>
  );
}

function CenteredMessageState({
  title,
  subtitle,
}: {
  title: string;
  subtitle?: string;
}): JSX.Element {
  return (
    <StateCard>
      <Stack
        spacing={0.75}
        sx={{
          alignItems: 'center',
          textAlign: 'center',
          maxWidth: 280
        }}>
        <Typography
          sx={{
            fontSize: TYPO.lg,
            fontWeight: FONT_WEIGHT.semibold,
            color: 'text.primary',
          }}
        >
          {title}
        </Typography>
        {subtitle ? (
          <Typography
            sx={{
              fontSize: TYPO.sm,
              color: 'text.secondary',
              lineHeight: 1.5,
            }}
          >
            {subtitle}
          </Typography>
        ) : null}
      </Stack>
    </StateCard>
  );
}
