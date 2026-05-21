/**
 * Robot picker, aligned with `reachy_mini_mobile_app/src/ui/
 * screens/ScanScreen.tsx`.
 *
 * Layout (the shared TopBar - identity + sign-out - sits ABOVE
 * this view and is owned by `ReachyHostShell`):
 *   ┌──────────────────────────────────────────┐
 *   │              (reachy-buste)              │
 *   │                                          │
 *   │            Your Reachies                 │
 *   │   N online · linked to your HF account   │
 *   │                                          │
 *   │   ┌────────────────────────────────┐     │
 *   │   │ [reachy] ● Name             >  │     │
 *   │   └────────────────────────────────┘     │
 *   │   ┌────────────────────────────────┐     │
 *   │   │ [reachy] ● Other            🔒 │     │  (busy)
 *   │   └────────────────────────────────┘     │
 *   │                                          │
 *   │  ────────────────────────────────────────│
 *   │              ↻ Refresh                   │  Sticky bottom
 *   └──────────────────────────────────────────┘
 *
 *  - Hero illustration: `reachy-buste.svg`, same asset as the
 *    splash, gives the screen a brand identity beyond the cards.
 *  - Robot cards: avatar + name + hardware-id tag + trailing
 *    chevron (or lock when busy).
 *  - Sticky refresh: pinned to the bottom, out of the scrollable
 *    area, so it's always one tap away.
 */
import type { JSX } from 'react';
import { useEffect, useMemo, useState } from 'react';
import {
  Box,
  Button,
  Chip,
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
import UsbIcon from '@mui/icons-material/Usb';
import WifiIcon from '@mui/icons-material/Wifi';

import { reachyBusteSvg, reachyStandardSvg } from '../assets';
import type { RobotInfo } from '../lib/sdk-types';
import { FONT_WEIGHT, LAYOUT, TYPO } from '../lib/tokens';

export interface PickerViewProps {
  robots: RobotInfo[];
  /** `true` while the picker is fetching its first robot list or
   *  the user just clicked "Refresh" - drives the header spinner. */
  isRefreshing: boolean;
  /** Last error message from the central listener / REST fetch,
   *  or `null` if everything's healthy. Surfaces as an error state
   *  card when the list is empty so the user knows the screen is
   *  silent for a reason. */
  error?: string | null;
  preselectedRobotId: string | null;
  onSelect(robotId: string): void;
  /** Asks the picker source to re-fetch the robot list. Defaults
   *  to a no-op if the host has nothing to refresh; the spinner
   *  still plays for ~1 s as visual feedback. */
  onRefresh?(): void;
}

export function PickerView({
  robots,
  isRefreshing,
  error,
  preselectedRobotId,
  onSelect,
  onRefresh,
}: PickerViewProps): JSX.Element {
  const hasRobots = robots.length > 0;

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
          <Stack alignItems="center" spacing={2}>
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
          ) : isRefreshing ? (
            <LoadingState />
          ) : error ? (
            <CenteredMessageState
              title="Couldn't reach Hugging Face"
              subtitle={error}
            />
          ) : (
            <CenteredMessageState title="No Reachy online" />
          )}
        </Stack>
      </Stack>

      <StickyRefreshBar
        onRefresh={onRefresh}
        isRefreshing={isRefreshing}
      />
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
    <Stack alignItems="center" spacing={0.5} sx={{ width: '100%' }}>
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
        alignItems="center"
        spacing={2}
        sx={{ width: '100%' }}
      >
        <CardAvatar />
        {/* Two-row identity grid: name + transport chip, then id.
            Mirrors the mobile `RemoteRobotCard` so users moving
            between mobile and desktop pick out the same
            elements. */}
        <Stack sx={{ flex: 1, minWidth: 0 }} spacing={0.25}>
          <Stack
            direction="row"
            alignItems="center"
            spacing={1}
            sx={{ minWidth: 0 }}
          >
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
                <TransportChip transport={transport} />
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

/* ─────────────────── Transport chip ─────────────────── */

/**
 * Compact transport tag for a robot listing. Ported 1-to-1 from
 * the mobile shell's `TransportChip`. Two well-known values get
 * an icon + typed label (`usb`, `wifi`); anything else falls
 * through to a plain capitalised label so a future daemon
 * advertising `ethernet` / `sim` / `mockup` still renders without
 * a component update.
 */
function TransportChip({ transport }: { transport: string }): JSX.Element {
  if (transport === 'usb') {
    return (
      <Chip
        size="small"
        icon={<UsbIcon sx={{ fontSize: 14 }} />}
        label="USB"
        variant="outlined"
        sx={{
          height: 20,
          fontSize: TYPO.tiny,
          '.MuiChip-icon': { ml: 0.5 },
        }}
      />
    );
  }
  if (transport === 'wifi') {
    return (
      <Chip
        size="small"
        icon={<WifiIcon sx={{ fontSize: 14 }} />}
        label="Wi-Fi"
        variant="outlined"
        sx={{
          height: 20,
          fontSize: TYPO.tiny,
          '.MuiChip-icon': { ml: 0.5 },
        }}
      />
    );
  }
  return (
    <Chip
      size="small"
      label={transport}
      variant="outlined"
      sx={{
        height: 20,
        fontSize: TYPO.tiny,
        textTransform: 'capitalize',
      }}
    />
  );
}

/* ─────────────────── Sticky refresh ─────────────────── */

const refreshSpinKeyframes = keyframes`
  0%   { transform: rotate(0deg) scale(1); }
  50%  { transform: rotate(180deg) scale(0.92); }
  100% { transform: rotate(360deg) scale(1); }
`;

const refreshTapKeyframes = keyframes`
  0%   { transform: rotate(0deg) scale(1); }
  12%  { transform: rotate(-32deg) scale(0.94); }
  100% { transform: rotate(360deg) scale(1); }
`;

function StickyRefreshBar({
  onRefresh,
  isRefreshing,
}: {
  onRefresh?(): void;
  isRefreshing: boolean;
}): JSX.Element {
  const [tapCounter, setTapCounter] = useState(0);

  const handleClick = (): void => {
    setTapCounter((c) => c + 1);
    onRefresh?.();
  };

  const iconAnimation = isRefreshing
    ? `${refreshSpinKeyframes} 1.1s cubic-bezier(0.45, 0.05, 0.55, 0.95) infinite`
    : tapCounter > 0
      ? `${refreshTapKeyframes} 0.55s cubic-bezier(0.34, 1.56, 0.64, 1)`
      : 'none';

  return (
    <Stack
      alignItems="center"
      sx={{
        width: '100%',
        flexShrink: 0,
        pt: 1.5,
        pb: 2,
        px: 2,
        bgcolor: 'background.default',
        borderTop: (theme) => `1px solid ${theme.palette.divider}`,
      }}
    >
      <Button
        variant="text"
        color="primary"
        disabled={isRefreshing}
        startIcon={
          <RefreshIcon
            key={`refresh-icon-${tapCounter}-${isRefreshing}`}
            sx={{
              fontSize: 22,
              transformOrigin: 'center',
              animation: iconAnimation,
              color: 'inherit',
            }}
          />
        }
        onClick={handleClick}
        sx={{
          textTransform: 'none',
          fontSize: TYPO.md,
          fontWeight: FONT_WEIGHT.semibold,
          borderRadius: 999,
          px: 3,
          py: 1,
          transition: (theme) =>
            theme.transitions.create(['background-color', 'color'], {
              duration: theme.transitions.duration.short,
            }),
          ...(isRefreshing && {
            bgcolor: (theme) => alpha(theme.palette.primary.main, 0.08),
          }),
          '&.Mui-disabled': {
            color: 'primary.main',
            opacity: 1,
          },
        }}
      >
        Refresh
      </Button>
    </Stack>
  );
}

/* ─────────────────── State cards (loading / empty / error) ───── */

/**
 * Shared minimum height for every "single-card" state.
 *
 * Tuned to match the natural height of a populated
 * `RemoteRobotCard` so the body slot doesn't snap between
 * states (loading → empty → 1 robot → N robots). Without this,
 * the spinner would visibly resize the moment central returns
 * the list.
 */
const STATE_CARD_MIN_HEIGHT = 104;

/**
 * Card chrome shared by the loading / empty states.
 *
 * Mirrors the surface used by `RemoteRobotCard` (paper bg,
 * theme divider border, same dual inset + drop shadow) so the
 * three states form a coherent visual family with the actual
 * robot rows below them.
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

function LoadingState(): JSX.Element {
  return (
    <StateCard>
      <Stack
        alignItems="center"
        spacing={1.5}
        sx={{ color: 'text.secondary' }}
      >
        <CircularProgress size={24} sx={{ color: 'text.secondary' }} />
        <Typography sx={{ fontSize: TYPO.sm, fontWeight: FONT_WEIGHT.medium }}>
          Asking Hugging Face for your robots…
        </Typography>
      </Stack>
    </StateCard>
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
        alignItems="center"
        spacing={0.75}
        sx={{ textAlign: 'center', maxWidth: 280 }}
      >
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
