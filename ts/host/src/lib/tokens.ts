/**
 * Shared design tokens.
 *
 * Mirrors `reachy_mini_mobile_app/src/ui/design/tokens.ts` so the
 * host shell, the mobile shell, and the desktop app share one
 * visual vocabulary (typography scale, weight scale, radii,
 * status colours, motion durations). Keep this file in lockstep
 * with the mobile one - drift is what gets flagged at design
 * review.
 *
 * Safe-area insets are NOT exposed here: the host shell ships on
 * desktop browsers and HF Spaces, where notch / home-indicator
 * insets are not a concern. The mobile shell keeps that layer in
 * its own tokens module.
 */

export const TYPO = {
  micro: '0.65rem',
  tiny: '0.7rem',
  xs: '0.75rem',
  sm: '0.8rem',
  body: '0.85rem',
  md: '0.9rem',
  lg: '1rem',
  xl: '1.1rem',
  xxl: '1.25rem',
  hero: '1.5rem',
  display: '1.625rem',
} as const;

export const FONT_WEIGHT = {
  regular: 400,
  medium: 500,
  semibold: 600,
  bold: 700,
} as const;

export const RADIUS = {
  xs: 4,
  sm: 6,
  md: 8,
  lg: 12,
  xl: 16,
  xxl: 20,
  pill: 9999,
  circle: '50%',
} as const;

export const STATUS = {
  success: '#22c55e',
  successSoft: '#16a34a',
  error: '#ef4444',
  warning: '#f59e0b',
  info: '#6366f1',
} as const;

export const DURATION = {
  fast: 150,
  base: 250,
  slow: 400,
} as const;

export const LAYOUT = {
  /** Max width of any centred "card-in-screen" content (sign-in,
   *  picker, error). Matches the mobile shell. */
  contentMaxWidth: 420,
  heroSize: 160,
  heroSizeSmall: 120,
} as const;
