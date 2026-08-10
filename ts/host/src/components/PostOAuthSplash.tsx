/**
 * Covering splash shown on the OAuth return leg, while the SDK
 * resolves the cached token via `authenticate()`.
 *
 * Without it the host would render `SignInView` (the "Continue
 * with Hugging Face" button) for the ~30-500 ms the in-flight
 * `authenticate()` takes, so the user briefly bounces back to the
 * sign-in button after already authorising on huggingface.co.
 *
 * This splash deliberately mirrors `WelcomeBackOverlay`'s frame
 * (same `background.default` fill, same centred HF logo) so the
 * transition reads as one continuous beat:
 *
 *   PostOAuthSplash (logo + spinner, no name)
 *     → WelcomeBackOverlay ("Hello, <name>")
 *       → PickerView
 *
 * It carries NO username on purpose: the name only appears in the
 * `WelcomeBackOverlay`, which keeps the "Welcome back" → "Hello,
 * X" text from ever flickering (cf. the welcome-back latch in
 * `ReachyHostShell`).
 */
import type { JSX } from 'react';
import { Box, CircularProgress, Stack, Typography, keyframes } from '@mui/material';

import { hfLogoSvg } from '../assets';
import { FONT_WEIGHT, TYPO } from '../lib/tokens';

/** Reserved height of the content block BELOW the logo (heading +
 *  spinner here, "Hello, X" + subtitle in `WelcomeBackOverlay`).
 *  Both overlays pin the SAME value and centre their content within
 *  it, so the logo above sits at the exact same Y in both. Without
 *  this the splash (heading + spinner) and the welcome (two text
 *  lines) have different natural heights, and the vertically-centred
 *  column makes the logo visibly jump at the splash → welcome cut.
 *  KEEP IN LOCKSTEP with `WelcomeBackOverlay`'s constant. */
const CONTENT_MIN_HEIGHT = 72;

/** Gentle breathing pulse on the logo so the wait reads as "working"
 *  rather than "stalled", without the harshness of a spinner alone. */
const pulseKeyframes = keyframes`
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(0.94);
    opacity: 0.75;
  }
`;

export function PostOAuthSplash(): JSX.Element {
  return (
    <Box
      sx={{
        position: 'fixed',
        inset: 0,
        // One below WelcomeBackOverlay (1300) so that, once the
        // username resolves, the welcome beat fades in cleanly on
        // top of this splash with no underlying screen bleed.
        zIndex: 1290,
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
          animation: `${pulseKeyframes} 1.6s ease-in-out infinite`,
        }}
      />
      <Stack
        spacing={1}
        sx={{
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: CONTENT_MIN_HEIGHT
        }}>
        <Typography
          component="h1"
          sx={{
            fontSize: TYPO.hero,
            fontWeight: FONT_WEIGHT.bold,
            letterSpacing: '-0.3px',
            textAlign: 'center',
            m: 0,
          }}
        >
          Signing you in…
        </Typography>
        <CircularProgress size={22} thickness={5} color="primary" />
      </Stack>
    </Box>
  );
}
