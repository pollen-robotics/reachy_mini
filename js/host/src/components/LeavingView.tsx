/**
 * Transition view rendered while the host is tearing the iframe
 * session down (user clicked "End session" / power-off).
 *
 * 1-to-1 port of
 * `reachy_mini_mobile_app/src/ui/screens/session/LeavingView.tsx`
 * — the desktop host and the mobile shell deliberately share the
 * exact same disconnection visual so a user moving between the
 * two clients sees a consistent "Putting Reachy to sleep…" beat.
 *
 * Visual posture
 * ──────────────
 * Mirrors `<ConnectingView>`'s anatomy (stack centred on the
 * vertical axis, primary visual on top, title below) so the
 * leave-screen reads as the symmetric counterpart of the join-
 * screen. But where Connecting carries a 3-dot stepper + bold
 * headline + a two-line caption to narrate a multi-phase
 * bring-up, Leaving is intentionally one beat lighter:
 *
 *   - a small, low-contrast spinner instead of the stepper
 *     (the user already decided to leave - we don't need to
 *     dramatise the wait),
 *   - a single short headline, no caption (the goto-sleep
 *     trajectory is short enough that a sub-line of explanation
 *     reads as filler by the time it lands).
 *
 * The trailing ellipsis on the headline is enough of a
 * "something's still happening" cue without needing a second line
 * of copy.
 */
import type { JSX } from 'react';
import { CircularProgress, Stack, Typography } from '@mui/material';

import { FONT_WEIGHT, TYPO } from '../lib/tokens';

export function LeavingView(): JSX.Element {
  return (
    <Stack
      sx={{
        position: 'absolute',
        inset: 0,
        bgcolor: 'background.default',
        zIndex: 5,
      }}
      role="status"
      aria-live="polite"
    >
      <Stack
        alignItems="center"
        justifyContent="center"
        spacing={2}
        sx={{ flex: 1, minHeight: 0, width: '100%', px: 3 }}
      >
        {/* Discreet, thin-stroked spinner. Size + thickness tuned
            to read as "a small ambient activity indicator" rather
            than "the focal point of the screen". `text.secondary`
            colouring keeps it muted against both light and dark
            backgrounds; the underlying CSS animation already gives
            enough motion to register as a loading state. */}
        <CircularProgress
          size={22}
          thickness={2.4}
          sx={{ color: 'text.secondary' }}
        />
        <Typography
          sx={{
            fontSize: TYPO.md,
            fontWeight: FONT_WEIGHT.medium,
            color: 'text.primary',
            textAlign: 'center',
            letterSpacing: '-0.1px',
          }}
        >
          Putting Reachy to sleep…
        </Typography>
      </Stack>
    </Stack>
  );
}
