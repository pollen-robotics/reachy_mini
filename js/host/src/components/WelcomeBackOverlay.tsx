/**
 * Brief celebratory transition shown after a successful HF
 * sign-in.
 *
 * Mirrors `reachy_mini_mobile_app/src/ui/screens/
 * WelcomeBackScreen.tsx` so the desktop host and the mobile shell
 * use the same beat between OAuth return and the picker reveal.
 *
 * Bridges two states the user otherwise experiences as a hard cut:
 *  - the browser OAuth flow returns focus to the tab,
 *  - the host suddenly switches from SignInView to PickerView.
 *
 * Without this, the user lands on a list of robots with no
 * acknowledgment of who they are. This overlay surfaces their
 * username for ~3.4 s with a subtle entrance animation, then
 * fades out on top of the (already-mounted) picker.
 *
 * Pure presentational, no network calls. The picker behind us
 * is already warming its data via `useRobots`, so the user
 * lands on a populated list when the overlay clears.
 */
import type { JSX } from 'react';
import { useEffect, useState } from 'react';
import { Box, Fade, Stack, Typography, keyframes } from '@mui/material';

import { hfLogoSvg } from '../assets';
import { DURATION, FONT_WEIGHT, TYPO } from '../lib/tokens';

/** How long the welcome sits at full opacity before fading out.
 *  Tuned so the user has time to read the username and feel the
 *  celebratory beat land before the picker takes over. */
const VISIBLE_MS = 3400;

/** Pop-in keyframes used to stagger the entrance of the logo and
 *  headline. Slight overshoot via the cubic-bezier gives the
 *  motion an "earned" feel rather than a dry fade. */
const popInKeyframes = keyframes`
  0% {
    transform: scale(0.7);
    opacity: 0;
  }
  60% {
    transform: scale(1.06);
    opacity: 1;
  }
  100% {
    transform: scale(1);
    opacity: 1;
  }
`;

export interface WelcomeBackOverlayProps {
  userName: string | null;
  onDone(): void;
}

export function WelcomeBackOverlay({
  userName,
  onDone,
}: WelcomeBackOverlayProps): JSX.Element {
  const [show, setShow] = useState(true);

  useEffect(() => {
    const t = window.setTimeout(() => setShow(false), VISIBLE_MS);
    return () => window.clearTimeout(t);
  }, []);

  return (
    <Fade
      in={show}
      timeout={DURATION.base}
      onExited={onDone}
      // Critical: no initial fade-in. The overlay covers a
      // still-visible sign-in / picker screen during the auth
      // → connect handshake; if we let MUI fade us in from
      // opacity 0, the user briefly sees the underlying screen
      // bleed through. The inner logo / text keyframes already
      // provide the entrance motion, so we only animate on the
      // way out.
      appear={false}
    >
      <Box
        sx={{
          position: 'fixed',
          inset: 0,
          zIndex: 1300,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'background.default',
          color: 'text.primary',
          touchAction: 'none',
        }}
      >
        <Box
          component="img"
          src={hfLogoSvg}
          alt=""
          aria-hidden
          sx={{
            width: 72,
            height: 72,
            mb: 3,
            display: 'block',
            animation: `${popInKeyframes} 0.55s cubic-bezier(0.34, 1.56, 0.64, 1) both`,
          }}
        />
        <Stack alignItems="center" spacing={0.5}>
          <Typography
            component="h1"
            sx={{
              fontSize: TYPO.hero,
              fontWeight: FONT_WEIGHT.bold,
              letterSpacing: '-0.3px',
              textAlign: 'center',
              m: 0,
              animation: `${popInKeyframes} 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both 0.08s`,
            }}
          >
            {userName ? `Hello, ${userName}` : 'Welcome back'}
          </Typography>
          <Typography
            sx={{
              fontSize: TYPO.md,
              color: 'text.secondary',
              textAlign: 'center',
              animation: `${popInKeyframes} 0.6s cubic-bezier(0.34, 1.56, 0.64, 1) both 0.16s`,
            }}
          >
            Looking up your Reachies…
          </Typography>
        </Stack>
      </Box>
    </Fade>
  );
}
