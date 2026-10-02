/**
 * Persistent host top bar.
 *
 * Provides the same chrome as the pre-migration standalone host:
 *
 * Layout (signed in, no session):
 *
 *   [ logo ]  Telepresence                  [ avatar @user ▾ ]
 *
 * Layout (signed in, streaming):
 *
 *   [ logo ]  Telepresence  [ End session ⏻ ]  [ avatar @user ▾ ]
 *
 * UX rationale
 * ────────────
 * The destructive actions are visually distinct:
 *   - "End session" is a labeled primary outlined button that ONLY renders
 *     while a session is live. When there's no session it disappears, so the
 *     bar isn't cluttered with a permanent disabled glyph.
 *   - "Sign out" is a primary-colored logout icon button sitting to the
 *     right of the (borderless) account chip, mirroring the mobile app.
 *
 * Signed-out
 * ──────────
 * The bar STAYS visible on the sign-in screen, with the embedded app's
 * logo + name on the left and an empty right slot - no avatar, no
 * End-session button, since neither makes sense before the user has
 * authenticated. Anchors "where am I" for users landing on the host
 * through `huggingface.co/spaces/<app>`.
 */
import { useEffect, useState, type JSX } from 'react';
import {
  Avatar,
  Box,
  Button,
  CircularProgress,
  IconButton,
  Stack,
  Typography,
} from '@mui/material';
import LogoutIcon from '@mui/icons-material/Logout';
import PowerSettingsNewIcon from '@mui/icons-material/PowerSettingsNew';

import { reachyHeadSvg } from '../assets';
import { RADIUS } from '../lib/tokens';
import { IdentityChipBar } from '../ui/design/IdentityChipBar';

export type HostPhase =
  | 'signing-in'
  | 'picking'
  | 'embedded'
  | 'leaving'
  | 'error';

export interface TopBarProps {
  appName: string;
  /** Resolved app icon URL probed from `${embedUrl}/icon.svg`. Best
   *  signal: the app shipped a real icon. */
  appIconUrl?: string | null;
  /** Fallback emoji from the HF Spaces frontmatter (`cardData.emoji`).
   *  Used when no `appIconUrl` is available. */
  appEmoji?: string | null;
  hostPhase: HostPhase;
  userName: string | null;
  /** Resolved HF avatar URL from `/api/whoami-v2`, or `null` while in
   *  flight / failed. The chip falls back to a first-letter glyph. */
  avatarUrl?: string | null;
  selectedRobotName?: string | null;
  /** Physical transport of the selected robot (`wifi` / `usb` / …),
   *  surfaced as the Lite/Wireless tag on the in-session identity
   *  sub-line. `null` when no robot is selected. */
  selectedRobotTransport?: string | null;
  /** Rolling-min RTT (ms) the embed reports once live, or `null` while
   *  not yet measured. Drives the latency pill on the sub-line. */
  linkRttMs?: number | null;
  onSignOut(): void;
  onEndSession(): void;
}

export function TopBar({
  appName,
  appIconUrl = null,
  appEmoji = null,
  hostPhase,
  userName,
  avatarUrl = null,
  selectedRobotName = null,
  selectedRobotTransport = null,
  linkRttMs = null,
  onSignOut,
  onEndSession,
}: TopBarProps): JSX.Element {
  const sessionOpen =
    hostPhase === 'embedded' || hostPhase === 'leaving';
  const isSignedIn = Boolean(userName);
  // Local "session tear-down in flight" flag, mirrors the
  // pre-migration TopBar. The host's `endSession` flips
  // `hostPhase === 'leaving'`, so we treat that as the canonical
  // signal AND also latch on a click so the spinner lands on the
  // very first frame after the click (the phase flip arrives a
  // tick later).
  const [isStoppingLocal, setIsStoppingLocal] = useState(false);
  const isStopping = isStoppingLocal || hostPhase === 'leaving';

  // Clear the local spinner flag once the host leaves `embedded`/
  // `leaving` (back to picker or error). Defensive against the case
  // where the parent flips away from those phases for an unrelated
  // reason while we were waiting.
  useEffect(() => {
    if (!sessionOpen) setIsStoppingLocal(false);
  }, [sessionOpen]);

  const handleEndSession = (): void => {
    if (isStopping) return;
    setIsStoppingLocal(true);
    onEndSession();
  };

  const showEndSession = isSignedIn && (sessionOpen || isStopping);

  return (
    <Box
      component="header"
      sx={(theme) => ({
        position: 'sticky',
        top: 0,
        zIndex: 10,
        backdropFilter: 'saturate(180%) blur(12px)',
        WebkitBackdropFilter: 'saturate(180%) blur(12px)',
        backgroundColor:
          theme.palette.mode === 'dark'
            ? 'rgba(16, 16, 19, 0.78)'
            : 'rgba(250, 250, 250, 0.82)',
        borderBottom: `1px solid ${theme.palette.divider}`,
        paddingTop: 'env(safe-area-inset-top, 0px)',
        flexShrink: 0,
      })}
    >
      <Stack
        direction="row"
        spacing={1.25}
        sx={{
          alignItems: 'center',
          py: 1,
          px: 2,
          // Same fixed height as the rest of the host shell expects
          // (the iframe layout reserves `--reachy-host-topbar-h`
          // px above it).
          minHeight: 'var(--reachy-host-topbar-h)'
        }}>
        <AppLogo iconUrl={appIconUrl} emoji={appEmoji} />
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography
            variant="body1"
            sx={{ fontWeight: 700, lineHeight: 1.2 }}
            noWrap
          >
            {appName}
          </Typography>
          {/* In-session identity sub-line: the running app is the
              headline (above); the robot it's driving + its link
              facts (Lite/Wireless, live latency) read as the smaller
              context line, mirroring the mobile app overlay. Only the
              embed measures the live RTT (the host handed off its
              WebRTC slot), so the latency pill is gated on a reported
              value to avoid a misleading `0 ms`. */}
          {sessionOpen && selectedRobotName && (
            <Box sx={{ mt: 0.25 }}>
              <IdentityChipBar
                robotName={selectedRobotName}
                transport={selectedRobotTransport}
                linkRttMs={linkRttMs}
                showLatency={linkRttMs !== null}
                variant="secondary"
              />
            </Box>
          )}
        </Box>

        {showEndSession && (
          <Button
            variant="outlined"
            color="primary"
            size="small"
            onClick={handleEndSession}
            disabled={isStopping}
            endIcon={
              // Power glyph sits AFTER the label. Wrap both it and the
              // spinner in a fixed 18x18 slot so the button width does
              // NOT shift when we swap glyphs during teardown.
              // `CircularProgress` and `PowerSettingsNewIcon` otherwise
              // render at slightly different intrinsic sizes.
              <Box
                sx={{
                  width: 18,
                  height: 18,
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                {isStopping ? (
                  <CircularProgress
                    size={15}
                    thickness={5}
                    sx={{ color: 'inherit' }}
                  />
                ) : (
                  <PowerSettingsNewIcon sx={{ fontSize: 18 }} />
                )}
              </Box>
            }
            sx={{
              fontSize: 13.5,
              fontWeight: 700,
              py: 0.7,
              px: 2,
              // Crisp, squared-off corners matching the Reachy ecosystem
              // (px value, not the sx multiplier which would scale off
              // theme.shape.borderRadius = 12 and read as ~18px).
              borderRadius: `${RADIUS.md}px`,
              textTransform: 'none',
              lineHeight: 1.1,
            }}
          >
            {isStopping ? 'Ending…' : 'End session'}
          </Button>
        )}

        {/* Account menu (avatar + sign-out) is a PICKER-only affordance:
            sign-out lives in the robot list, not mid-session. Once a
            session is open we swap it for the "End session" button
            above, so the right cluster shows exactly one primary action
            for the current phase. */}
        {isSignedIn && !showEndSession && (
          <AccountChip
            username={userName}
            avatarUrl={avatarUrl}
            disabled={isStopping}
            onLogout={onSignOut}
          />
        )}
      </Stack>
    </Box>
  );
}

/* ─────────────────── Account chip ─────────────────── */

/**
 * Borderless identity row (avatar + "Signed in as @handle"), mirroring the
 * mobile app's HfAccountBar - no outline, just the account read at a glance -
 * with a primary-colored logout icon button on the right that signs out
 * directly (no intermediate menu).
 */
function AccountChip({
  username,
  avatarUrl,
  disabled = false,
  onLogout,
}: {
  username: string | null;
  avatarUrl: string | null;
  disabled?: boolean;
  onLogout(): void;
}): JSX.Element {
  const initial = (username ?? '').slice(0, 1).toUpperCase() || null;

  return (
    <Stack
      direction="row"
      spacing={0.75}
      sx={{ alignItems: 'center', pl: 0.5, py: 0.25, minWidth: 0 }}
    >
      <Avatar
        src={avatarUrl ?? undefined}
        alt={username ?? 'Hugging Face user'}
        sx={(theme) => ({
          width: 24,
          height: 24,
          fontSize: 11,
          fontWeight: 600,
          bgcolor:
            theme.palette.mode === 'dark'
              ? 'rgba(255,255,255,0.08)'
              : 'rgba(0,0,0,0.06)',
          color: 'text.secondary',
        })}
      >
        {initial}
      </Avatar>
      {/* Two-line identity, aligned with the mobile ScanScreen's
          HfAccountBar: a small "Signed in as" kicker over the `@handle`. */}
      <Stack spacing={0} sx={{ minWidth: 0, maxWidth: 140, alignItems: 'flex-start' }}>
        <Typography
          sx={{
            fontSize: 9.5,
            fontWeight: 600,
            color: 'text.secondary',
            letterSpacing: 0.5,
            textTransform: 'uppercase',
            lineHeight: 1.15,
          }}
          noWrap
        >
          Signed in as
        </Typography>
        <Typography
          variant="body2"
          sx={{
            fontWeight: 600,
            color: 'text.primary',
            lineHeight: 1.2,
            maxWidth: '100%',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          noWrap
        >
          @{username}
        </Typography>
      </Stack>
      {/* One-click sign-out, deliberately: mobile parity, and the cost
          of a mis-click is one silent-auth round trip, not data loss.
          (The previous menu-behind-avatar guard traded that for a
          hidden action nobody found.) */}
      <IconButton
        onClick={onLogout}
        disabled={disabled}
        color="primary"
        size="small"
        aria-label={username ? `Sign out @${username}` : 'Sign out'}
        sx={{ ml: 0.25 }}
      >
        <LogoutIcon sx={{ fontSize: 18 }} />
      </IconButton>
    </Stack>
  );
}

/* ─────────────────── App logo ─────────────────── */

/**
 * Logo slot with a 3-step fallback:
 *
 *   1. `iconUrl` - the app shipped an `icon.svg` probed upstream with
 *      a Content-Type check (only non-null when it's a real image).
 *   2. `emoji`   - `cardData.emoji` from the Space's frontmatter.
 *   3. The bundled `reachy-head` SVG - the generic host fallback.
 *
 * The defensive `<img onError>` covers the rare case where the
 * resolved icon URL stops working between probe + render (CDN blip,
 * region change, etc.) so the bar never displays a broken-image
 * glyph: it falls through to step 2 / 3.
 */
function AppLogo({
  iconUrl,
  emoji,
}: {
  iconUrl: string | null;
  emoji: string | null;
}): JSX.Element {
  const [errored, setErrored] = useState(false);
  useEffect(() => {
    setErrored(false);
  }, [iconUrl]);

  const SLOT_SX = {
    width: 28,
    height: 28,
    display: 'flex',
    flexShrink: 0,
    alignItems: 'center',
    justifyContent: 'center',
    lineHeight: 1,
  } as const;

  if (iconUrl && !errored) {
    return (
      <Box
        component="img"
        src={iconUrl}
        alt=""
        draggable={false}
        onError={() => setErrored(true)}
        sx={{ ...SLOT_SX, objectFit: 'contain' }}
      />
    );
  }

  if (emoji) {
    return (
      <Box sx={SLOT_SX} aria-hidden>
        <Box
          component="span"
          sx={{
            fontSize: 22,
            lineHeight: 1,
            userSelect: 'none',
          }}
        >
          {emoji}
        </Box>
      </Box>
    );
  }

  return (
    <Box
      component="img"
      src={reachyHeadSvg}
      alt=""
      draggable={false}
      sx={{ ...SLOT_SX, objectFit: 'contain' }}
    />
  );
}
