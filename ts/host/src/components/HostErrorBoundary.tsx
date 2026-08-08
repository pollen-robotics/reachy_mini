/**
 * Top-level React error boundary for the host shell.
 *
 * A render-time throw anywhere in `ReachyHostShell` (a bad payload, a
 * component bug, a null deref) would otherwise white-screen the whole
 * host with no recovery path. This boundary catches it, logs the stack,
 * and shows a themed fallback with a single outlined "Reload" action -
 * a full page reload is the only safe recovery once the React tree is
 * in an unknown state (the shell's own `error` phase can't be trusted
 * after an uncaught render error).
 *
 * Mounted INSIDE the `ThemeProvider` (see `ReachyHost`) so the fallback
 * inherits the host theme (colors, dark mode).
 */
import { Component, type ErrorInfo, type ReactNode } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

import { createLogger } from '@pollen-robotics/reachy-mini-sdk';
import { RADIUS } from '../lib/tokens';

const log = createLogger('host');

export interface HostErrorBoundaryProps {
  children: ReactNode;
}

interface HostErrorBoundaryState {
  error: Error | null;
}

export class HostErrorBoundary extends Component<
  HostErrorBoundaryProps,
  HostErrorBoundaryState
> {
  state: HostErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): HostErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    log.error(
      'uncaught render error:',
      error,
      info.componentStack,
    );
  }

  private readonly handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <Box
        sx={{
          minHeight: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 4,
          bgcolor: 'background.default',
          color: 'text.primary',
        }}
      >
        <Stack
          spacing={2}
          sx={{ alignItems: 'center', textAlign: 'center', maxWidth: 520 }}
        >
          <Box component="div" sx={{ fontSize: 56, lineHeight: 1 }} aria-hidden>
            ⚠️
          </Box>
          <Typography variant="h5">Something went wrong</Typography>
          <Typography variant="body2" sx={{ color: 'text.secondary' }}>
            The app hit an unexpected error and needs to reload.
          </Typography>
          <Button
            variant="contained"
            onClick={this.handleReload}
            sx={{ borderRadius: `${RADIUS.md}px` }}
          >
            Reload
          </Button>
        </Stack>
      </Box>
    );
  }
}
