/**
 * Sign-in landing screen.
 *
 * Visual + copy mirror of `reachy_mini_mobile_app/src/ui/screens/
 * RemoteSignInScreen.tsx` so a user moving between the mobile
 * companion and a desktop browser tab lands on the same gateway.
 *
 * Composition:
 *  - h1 "Sign in to Hugging Face" (NOT the app name; the app
 *    name lives in the document title and the picker).
 *  - Subtitle explaining why signing in is the gate.
 *  - Single outlined CTA "Continue with Hugging Face" that
 *    morphs to "Waiting for Hugging Face…" + spinner during the
 *    OAuth redirect.
 *
 * Post-OAuth return is handled by the host-level
 * `WelcomeBackOverlay`, NOT this screen, so we never need an
 * inline "Restoring your session…" sub-state here.
 */
import type { JSX, ReactNode } from 'react';
import { useState } from 'react';
import {
  Alert,
  AlertTitle,
  Box,
  Button,
  CircularProgress,
  Link,
  Stack,
  Typography,
} from '@mui/material';

import { hfLogoSvg } from '../assets';
import { FONT_WEIGHT, LAYOUT, TYPO } from '../lib/tokens';

export interface SignInViewProps {
  /** App name (used only in subtitle copy; not the heading). */
  appName: string;
  /** When `true`, the page has no obvious way to obtain an OAuth
   *  client ID (no `?clientId=` prop, no `window.huggingface`
   *  injection, no cached settings). Surfaces a dev-hint alert
   *  pointing at the local `.env.local` workflow. */
  isLocalDevMissingConfig: boolean;
  onSignIn(): Promise<void>;
}

export function SignInView({
  appName,
  isLocalDevMissingConfig,
  onSignIn,
}: SignInViewProps): JSX.Element {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClick = async (): Promise<void> => {
    if (busy) return;
    setError(null);
    setBusy(true);
    try {
      await onSignIn();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Sign-in failed';
      setError(friendlyAuthError(message));
      setBusy(false);
    }
    // No `finally`: on success the host transitions out of
    // `signing-in` and unmounts this component, so the spinner
    // stays up until the page navigates away (no flash).
  };

  return (
    <Stack
      sx={{
        height: '100%',
        width: '100%',
        bgcolor: 'background.default',
      }}
    >
      <Stack
        spacing={3}
        sx={{
          p: 3,
          flex: 1,
          alignItems: 'center',
          justifyContent: 'center',
          maxWidth: LAYOUT.contentMaxWidth,
          mx: 'auto',
          width: '100%',
        }}
      >
        <Typography
          component="h1"
          sx={{
            fontSize: '1.875rem',
            fontWeight: FONT_WEIGHT.bold,
            lineHeight: 1.15,
            textAlign: 'center',
            letterSpacing: '-0.5px',
            color: 'text.primary',
            m: 0,
          }}
        >
          Sign in to Hugging Face
        </Typography>

        <Typography
          sx={{
            fontSize: TYPO.md,
            color: 'text.secondary',
            textAlign: 'center',
            maxWidth: 340,
            lineHeight: 1.5,
          }}
        >
          This is how <EmphasizedSpan>{appName}</EmphasizedSpan> will
          detect the <EmphasizedSpan>Reachies</EmphasizedSpan> linked
          to your <EmphasizedSpan>Hugging Face account</EmphasizedSpan>
          {' '}and let you control them from this browser.
        </Typography>

        {error ? (
          <Alert severity="error" sx={{ width: '100%', maxWidth: 420 }}>
            {error}
          </Alert>
        ) : isLocalDevMissingConfig ? (
          <Alert
            severity="info"
            variant="outlined"
            sx={{ width: '100%', maxWidth: 480, textAlign: 'left' }}
          >
            <AlertTitle sx={{ fontWeight: FONT_WEIGHT.semibold }}>
              Local dev: no OAuth client ID detected
            </AlertTitle>
            On a deployed HF Space the client ID is injected
            automatically. To sign in from `npm run dev`, create a{' '}
            <code>.env.local</code> at the app root and fill in one
            of the two options described in{' '}
            <code>.env.example</code>:
            <Box component="ul" sx={{ mt: 0.5, mb: 0, pl: 2 }}>
              <li>
                <code>VITE_HF_TOKEN</code> +{' '}
                <code>VITE_HF_USERNAME</code> - skip the OAuth
                redirect entirely (recommended).
              </li>
              <li>
                <code>VITE_HF_OAUTH_CLIENT_ID</code> - exercise the
                real OAuth flow against a{' '}
                <Link
                  href="https://huggingface.co/settings/applications/new"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  personal HF OAuth app
                </Link>
                .
              </li>
            </Box>
            Then restart the dev server.
          </Alert>
        ) : null}

        <Button
          variant="outlined"
          color="primary"
          size="large"
          disabled={busy}
          startIcon={
            busy ? (
              <CircularProgress size={18} thickness={5} color="primary" />
            ) : (
              <Box
                component="img"
                src={hfLogoSvg}
                alt=""
                aria-hidden
                sx={{
                  width: 20,
                  height: 20,
                  display: 'block',
                  transform: 'translateY(-1px)',
                }}
              />
            )
          }
          onClick={() => void handleClick()}
          sx={{
            textTransform: 'none',
            fontSize: TYPO.md,
            fontWeight: FONT_WEIGHT.semibold,
            borderWidth: 1.5,
            borderRadius: 2,
            px: 2.5,
            py: 1,
            minWidth: 260,
            '&.Mui-disabled': {
              borderColor: (theme) => theme.palette.primary.main,
              color: (theme) => theme.palette.primary.main,
              opacity: 0.85,
              borderWidth: 1.5,
            },
            '&:hover': {
              borderWidth: 1.5,
            },
            '& .MuiButton-startIcon': {
              mr: 1.25,
            },
          }}
        >
          {busy ? 'Waiting for Hugging Face…' : 'Continue with Hugging Face'}
        </Button>

      </Stack>
    </Stack>
  );
}

function EmphasizedSpan({ children }: { children: ReactNode }): JSX.Element {
  return (
    <Box
      component="span"
      sx={{
        fontWeight: FONT_WEIGHT.semibold,
        color: 'text.primary',
      }}
    >
      {children}
    </Box>
  );
}

function friendlyAuthError(raw: string): string {
  if (/Missing clientId/i.test(raw)) {
    return (
      'OAuth client ID not configured. On a deployed HF Space ' +
      'it is injected automatically; in local dev fill in ' +
      '`.env.local` (see `.env.example`).'
    );
  }
  if (/cancelled/i.test(raw)) return 'Sign-in cancelled.';
  if (/timeout/i.test(raw)) return 'Sign-in took too long, please retry.';
  if (/state.?mismatch/i.test(raw)) {
    return 'Sign-in security check failed, please retry.';
  }
  return raw;
}
