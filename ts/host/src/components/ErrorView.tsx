/**
 * Terminal error screen. Reached when:
 *  - The SDK throws on `connect()` / `startSession()`.
 *  - The embed reports `embed:error { fatal: true }`.
 *  - The host catches an uncaught render error (via the React
 *    error boundary in `ReachyHost`).
 *
 * Always offers a "Reload" action (full page reload) and a
 * "Back to picker" action that resets the host phase to picking
 * without losing the OAuth session.
 */
import type { JSX } from 'react';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

export interface ErrorViewProps {
  title?: string;
  message: string;
  detail?: unknown;
  onReload(): void;
  onBackToPicker(): void;
}

export function ErrorView({
  title = 'Something went wrong',
  message,
  detail,
  onReload,
  onBackToPicker,
}: ErrorViewProps): JSX.Element {
  return (
    <Box
      sx={{
        minHeight: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        p: 4,
      }}
    >
      <Stack
        spacing={3}
        sx={{ alignItems: 'center', textAlign: 'center', maxWidth: 520 }}
      >
        <Box
          component="div"
          sx={{
            fontSize: 56,
            lineHeight: 1,
            filter: 'grayscale(0.4)',
          }}
          aria-hidden
        >
          ⚠️
        </Box>
        <Stack spacing={1} sx={{ alignItems: 'center' }}>
          <Typography variant="h5">{title}</Typography>
          <Typography variant="body2" color="text.secondary">
            {message}
          </Typography>
        </Stack>
        {detail !== undefined && detail !== null && (
          <Box
            component="pre"
            sx={{
              maxWidth: '100%',
              overflowX: 'auto',
              fontSize: 12,
              p: 2,
              bgcolor: 'action.hover',
              borderRadius: 1,
              textAlign: 'left',
              fontFamily:
                '"JetBrains Mono", "Fira Code", ui-monospace, monospace',
            }}
          >
            {formatDetail(detail)}
          </Box>
        )}
        <Stack direction="row" spacing={1.5}>
          <Button variant="outlined" onClick={onBackToPicker}>
            Back to picker
          </Button>
          <Button variant="contained" onClick={onReload}>
            Reload
          </Button>
        </Stack>
      </Stack>
    </Box>
  );
}

function formatDetail(detail: unknown): string {
  if (typeof detail === 'string') return detail;
  try {
    return JSON.stringify(detail, null, 2);
  } catch {
    return String(detail);
  }
}
