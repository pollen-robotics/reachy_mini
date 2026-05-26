/**
 * Connection screen rendered while the host is bringing the
 * embedded app online: the iframe is mounted but hidden until it
 * reports `phase: live`. Disconnection (`leaving` phase) is
 * handled by the sibling `LeavingView` so each beat owns its
 * dedicated visual language.
 *
 * 1-to-1 port of `reachy_mini_mobile_app/src/ui/screens/session/
 * ConnectingView.tsx`. The two clients deliberately share the
 * exact same connection animation so a user moving from the
 * mobile companion to a desktop tab lands on a screen that feels
 * built by the same hand:
 *
 *   - Horizontal `StepsProgressIndicator` (Link / Session /
 *     Wake-up) with progress bar BEHIND the dots, check pop-in
 *     on completed steps, pulsing inner ring on the current
 *     step.
 *   - "Connecting to your Reachy" headline above a fixed
 *     two-line caption that morphs with the current step so the
 *     layout never shifts under the user.
 *
 * Difference with the mobile shell: the mobile shell drives this
 * from a conversation engine FSM. The host doesn't run a
 * conversation engine; the pre-`live` cluster collapses to a
 * single signal coming from the embedded app via `embed:app-state`
 * messages, so we map directly from `AppConnectingStep`
 * (`link | session | wake`) to the same three-dot indicator.
 */
import { useEffect, useState } from 'react';
import type { JSX } from 'react';
import { Box, Stack, Typography } from '@mui/material';

import { connectionSvg } from '../assets';
import type { AppConnectingStep } from '../lib/protocol';
import { FONT_WEIGHT, TYPO } from '../lib/tokens';
import { StepsProgressIndicator } from './StepsProgressIndicator';

export interface ConnectingViewProps {
  /** Sub-step inside `connecting`. Maps to the 3-dot stepper. */
  step: AppConnectingStep | null;
  /** Optional caption override (apps can send one via
   *  `embed:app-state.message`). When provided it wins over the
   *  default per-step caption. */
  message?: string | null;
}

/**
 * After this many ms in the same `wake` step without any forward
 * signal we surface a small "taking a moment" hint so the user
 * knows we're not stuck. Mirrors the mobile shell's 6 s window.
 */
const SLOW_HINT_DELAY_MS = 6_000;

const STEPS = [
  { id: 'link', label: 'Link' },
  { id: 'session', label: 'Session' },
  { id: 'wake', label: 'Wake-up' },
] as const;

const STEP_ORDER: AppConnectingStep[] = ['link', 'session', 'wake'];

export function ConnectingView({
  step,
  message,
}: ConnectingViewProps): JSX.Element {
  const currentStep = stepIndexFor(step);
  const slowHintVisible = useStepElapsedPast(step, 'wake', SLOW_HINT_DELAY_MS);

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
        spacing={3.5}
        sx={{ flex: 1, minHeight: 0, width: '100%', px: 3 }}
      >
        {/* Hero illustration: the "Reachy connecting" hand-drawn SVG
            from the design system. Sits above the stepper to anchor
            the screen visually before the eye drops to the
            progression bar. Sized at 160 px so it reads as a hero
            without competing with the stepper for attention. */}
        <Box
          aria-hidden
          sx={{
            width: 160,
            height: 160,
            flexShrink: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Box
            component="img"
            src={connectionSvg}
            alt=""
            draggable={false}
            sx={{
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              userSelect: 'none',
              pointerEvents: 'none',
            }}
          />
        </Box>

        {/* Stepper takes the visual lead under the illustration;
            capped width so the dots stay close enough together to
            read as a single progression. */}
        <Stack sx={{ width: '100%', maxWidth: 340 }}>
          <StepsProgressIndicator
            steps={[...STEPS]}
            currentStep={currentStep}
          />
        </Stack>

        <Stack alignItems="center" spacing={0.75}>
          <Typography
            sx={{
              fontSize: TYPO.lg,
              fontWeight: FONT_WEIGHT.semibold,
              textAlign: 'center',
            }}
          >
            Connecting to your Reachy
          </Typography>
          <Typography
            sx={{
              fontSize: TYPO.sm,
              color: 'text.secondary',
              textAlign: 'center',
              maxWidth: 280,
              // Reserve a stable two-line height so the slow-hint
              // appearance doesn't shift the layout under the
              // user.
              minHeight: '2.6em',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {message ?? captionFor({ step, slowHintVisible })}
          </Typography>
        </Stack>
      </Stack>
    </Stack>
  );
}

/**
 * Map the embed's `AppConnectingStep` to the 0/1/2 step index of
 * the 3-step indicator. `null` collapses to "Link" so the very
 * first paint already shows the first dot active instead of an
 * empty track.
 */
function stepIndexFor(step: AppConnectingStep | null): 0 | 1 | 2 {
  if (step === 'session') return 1;
  if (step === 'wake') return 2;
  return 0;
}

interface CaptionInputs {
  step: AppConnectingStep | null;
  slowHintVisible: boolean;
}

function captionFor({
  step,
  slowHintVisible,
}: CaptionInputs): string {
  switch (step) {
    case 'session':
      return 'Establishing the WebRTC session';
    case 'wake':
      if (slowHintVisible) {
        return 'Waking up your Reachy - taking a moment to settle in…';
      }
      return 'Enabling motors and waking up your Reachy';
    case 'link':
    default:
      return 'Opening secure link to Hugging Face';
  }
}

/**
 * Returns `true` once we've been on `targetStep` for longer than
 * `delayMs`. Resets to `false` whenever the step changes. Used to
 * surface a slow-progress hint after ~6 s, never to drive the
 * stepper itself.
 */
function useStepElapsedPast(
  step: AppConnectingStep | null,
  targetStep: AppConnectingStep,
  delayMs: number,
): boolean {
  const [past, setPast] = useState(false);

  useEffect(() => {
    if (step !== targetStep) {
      setPast(false);
      return;
    }
    const handle = window.setTimeout(() => setPast(true), delayMs);
    return () => {
      window.clearTimeout(handle);
    };
  }, [step, targetStep, delayMs]);

  return past;
}

// Re-export the step order so consumers can derive labels /
// counts without re-declaring the order locally.
export const CONNECTING_STEP_ORDER = STEP_ORDER;
