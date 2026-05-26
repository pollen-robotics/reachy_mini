/**
 * Host shell MUI themes (light + dark).
 *
 * Aligned 1-to-1 with `reachy_mini_mobile_app/src/theme.ts` so a
 * user moving between the mobile companion and the desktop host
 * lands on the same palette, typography, and radii. Drift between
 * the two is what makes the product feel like two apps stitched
 * together; keep them in sync.
 *
 * These themes are bundled with `@pollen-robotics/reachy-mini-sdk/host/auto`
 * and are not user-overridable. The host owns its chrome; apps own
 * theirs inside the iframe.
 */
import { createTheme } from '@mui/material/styles';
import type { Theme } from '@mui/material/styles';

import { RADIUS } from './tokens';

/** Pollen orange. The single accent colour shared with the
 *  desktop daemon UI and the mobile shell. */
const ACCENT = '#FF9500';

const FONT_FAMILY =
  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, ' +
  '"Helvetica Neue", sans-serif';

function buildTheme(mode: 'light' | 'dark'): Theme {
  const isDark = mode === 'dark';
  return createTheme({
    palette: {
      mode,
      primary: { main: ACCENT },
      background: {
        // Soft canvas vs paper contrast: cards (paper) pop without
        // the body feeling "hard" grey in light, or crushed to
        // OLED black in dark.
        default: isDark ? '#101013' : '#fafafa',
        paper: isDark ? '#1a1a1a' : '#ffffff',
      },
      text: {
        primary: isDark ? '#f5f5f5' : '#111111',
        secondary: isDark ? 'rgba(255,255,255,0.72)' : 'rgba(0,0,0,0.65)',
      },
      divider: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
    },
    typography: {
      fontFamily: FONT_FAMILY,
      button: { textTransform: 'none', fontWeight: 600 },
    },
    shape: { borderRadius: RADIUS.lg },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          html: {
            backgroundColor: isDark ? '#101013' : '#fafafa',
            minHeight: '100vh',
          },
          body: {
            backgroundColor: isDark ? '#101013' : '#fafafa',
            minHeight: '100vh',
            margin: 0,
            // App-style "unselectable" baseline: long-press shouldn't
            // pop iOS' text-selection / share callout, double-tap
            // shouldn't highlight chrome labels. The host is a control
            // surface, not a document. Form fields opt back in below
            // so future inputs keep their native copy / paste UX.
            userSelect: 'none',
            WebkitUserSelect: 'none',
            WebkitTouchCallout: 'none',
            WebkitTapHighlightColor: 'transparent',
            '& input, & textarea, & [contenteditable="true"]': {
              userSelect: 'text',
              WebkitUserSelect: 'text',
              WebkitTouchCallout: 'default',
            },
          },
          '#root': {
            minHeight: '100vh',
          },
        },
      },
      MuiButton: {
        defaultProps: { disableElevation: true },
        styleOverrides: {
          root: {
            borderRadius: RADIUS.lg,
            paddingInline: 20,
            paddingBlock: 10,
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: { backgroundImage: 'none' },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: { borderRadius: RADIUS.lg },
        },
      },
    },
  });
}

export const lightTheme: Theme = buildTheme('light');
export const darkTheme: Theme = buildTheme('dark');
