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
 *   [ logo ]  Telepresence  [ ⏻ End session ]  [ avatar @user ▾ ]
 *
 * UX rationale
 * ────────────
 * The destructive actions are visually distinct:
 *   - "End session" is a labeled red button that ONLY renders while a
 *     session is live. When there's no session it disappears, so the
 *     bar isn't cluttered with a permanent disabled glyph.
 *   - "Sign out" lives inside an account menu opened by clicking the
 *     avatar+username chip. Standard "click your face to see account
 *     actions" pattern, prevents accidental sign-out clicks.
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
  ButtonBase,
  CircularProgress,
  Divider,
  ListItemIcon,
  ListItemText,
  Menu,
  MenuItem,
  Stack,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import LogoutIcon from '@mui/icons-material/Logout';
import PowerSettingsNewIcon from '@mui/icons-material/PowerSettingsNew';

import { reachyHeadSvg } from '../assets';

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
  selectedRobotName: _selectedRobotName,
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
        alignItems="center"
        spacing={1.25}
        sx={{
          py: 1,
          px: 2,
          // Same fixed height as the rest of the host shell expects
          // (the iframe layout reserves `--reachy-host-topbar-h`
          // px above it).
          minHeight: 'var(--reachy-host-topbar-h)',
        }}
      >
        <AppLogo iconUrl={appIconUrl} emoji={appEmoji} />
        <Box sx={{ minWidth: 0, flex: 1 }}>
          <Typography
            variant="body1"
            sx={{ fontWeight: 700, lineHeight: 1.2 }}
            noWrap
          >
            {appName}
          </Typography>
        </Box>

        {showEndSession && (
          <Button
            variant="outlined"
            color="error"
            size="small"
            onClick={handleEndSession}
            disabled={isStopping}
            startIcon={
              isStopping ? (
                <CircularProgress
                  size={14}
                  thickness={5}
                  sx={{ color: 'inherit' }}
                />
              ) : (
                <PowerSettingsNewIcon sx={{ fontSize: 16 }} />
              )
            }
            sx={{
              fontSize: 12.5,
              fontWeight: 600,
              py: 0.5,
              px: 1.25,
              minWidth: 0,
              borderRadius: 999,
              textTransform: 'none',
              lineHeight: 1.1,
            }}
          >
            {isStopping ? 'Ending…' : 'End session'}
          </Button>
        )}

        {isSignedIn && (
          <AccountMenu
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

/* ─────────────────── Account menu ─────────────────── */

/**
 * Avatar + username chip that opens a small Menu with the "Sign out"
 * action. The chip itself does not perform any destructive action -
 * it's a disclosure trigger - so it's safe to keep prominent in the
 * bar without risking an accidental sign-out click.
 */
function AccountMenu({
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
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);
  const open = Boolean(anchorEl);
  const initial = (username ?? '').slice(0, 1).toUpperCase() || null;

  // Close the menu defensively if the parent flips us into the
  // disabled (stopping) state while it's open. Otherwise the user
  // could keep mashing "Sign out" inside an already-detached
  // teardown flow.
  useEffect(() => {
    if (disabled) setAnchorEl(null);
  }, [disabled]);

  return (
    <>
      <ButtonBase
        disabled={disabled}
        onClick={(e) => setAnchorEl(e.currentTarget)}
        focusRipple
        aria-label={`Account menu for @${username}`}
        aria-haspopup="menu"
        aria-expanded={open}
        sx={(theme) => ({
          display: 'inline-flex',
          alignItems: 'center',
          gap: 0.75,
          pl: 0.5,
          pr: 0.75,
          py: 0.25,
          borderRadius: 999,
          border: `1px solid ${theme.palette.divider}`,
          transition: theme.transitions.create(
            ['background-color', 'border-color'],
            { duration: theme.transitions.duration.shortest },
          ),
          '&:hover': {
            backgroundColor:
              theme.palette.mode === 'dark'
                ? 'rgba(255,255,255,0.04)'
                : 'rgba(0,0,0,0.03)',
          },
          '&:focus-visible': {
            outline: `2px solid ${theme.palette.primary.main}`,
            outlineOffset: 2,
          },
        })}
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
        <Typography
          variant="body2"
          sx={{
            fontWeight: 600,
            color: 'text.primary',
            lineHeight: 1.2,
            maxWidth: 120,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          noWrap
        >
          {username}
        </Typography>
        <ExpandMoreIcon
          sx={{
            fontSize: 16,
            color: 'text.secondary',
            transition: 'transform 120ms ease',
            transform: open ? 'rotate(180deg)' : 'none',
          }}
        />
      </ButtonBase>

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: {
            sx: {
              mt: 0.75,
              minWidth: 220,
              borderRadius: 1.5,
              border: (theme) => `1px solid ${theme.palette.divider}`,
              boxShadow:
                '0 6px 24px rgba(15,23,42,0.10), 0 1px 2px rgba(15,23,42,0.06)',
            },
          },
        }}
      >
        {/* Read-only identity header. Not a MenuItem on purpose - it
            shouldn't be focusable / clickable, just informative. */}
        <Box sx={{ px: 2, py: 1.25 }}>
          <Typography
            variant="caption"
            sx={{
              display: 'block',
              color: 'text.disabled',
              fontWeight: 600,
              letterSpacing: 0.4,
              textTransform: 'uppercase',
              fontSize: 10.5,
              lineHeight: 1.2,
            }}
          >
            Signed in as
          </Typography>
          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              color: 'text.primary',
              lineHeight: 1.3,
              mt: 0.25,
            }}
            noWrap
          >
            @{username}
          </Typography>
        </Box>
        <Divider />
        <MenuItem
          onClick={() => {
            setAnchorEl(null);
            onLogout();
          }}
          sx={{ py: 1 }}
        >
          <ListItemIcon sx={{ minWidth: 32 }}>
            <LogoutIcon sx={{ fontSize: 18 }} />
          </ListItemIcon>
          <ListItemText
            primary="Sign out"
            primaryTypographyProps={{ fontSize: 14, fontWeight: 500 }}
          />
        </MenuItem>
      </Menu>
    </>
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
