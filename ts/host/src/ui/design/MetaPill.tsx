/**
 * Shared meta-pill primitives for the robot identity row.
 *
 * Ported from `reachy_mini_mobile_app/src/ui/design/MetaPill.tsx`.
 * A meta pill is a small, light-bordered tag (icon/glyph + text) used
 * both in the picker list and in the in-session topbar so a robot
 * keeps the same visual taxonomy before and after the user picks it.
 *
 *   [⌁ Lite]   [▮▮▮ 38 ms]
 *
 * `VariantTag` is the always-on first pill: the USB / Wi-Fi icon plus
 * the product SKU it maps onto (`Lite` / `Wireless`).
 */
import type { JSX, ReactNode } from 'react';
import { Box, Typography } from '@mui/material';

import { TransportChip, transportLabelOf } from './TransportChip';
import { FONT_WEIGHT, RADIUS, TYPO } from '../../lib/tokens';

/** Uniform pill height so the meta row lines up cleanly. */
export const META_PILL_HEIGHT_PX = 24;

/**
 * Self-contained, light-bordered pill holding one meta module
 * (icon/glyph + text). An optional `tone` tints the border + content;
 * omitted = neutral.
 */
export function MetaPill({
  children,
  tone,
  pl = 0.875,
}: {
  children: ReactNode;
  tone?: string;
  /** Left padding override - the icon-led variant pill tightens this
   *  so the glyph's own whitespace doesn't read as extra padding. */
  pl?: number;
}): JSX.Element {
  return (
    <Box
      sx={(theme) => ({
        display: 'inline-flex',
        alignItems: 'center',
        gap: 0.5,
        height: META_PILL_HEIGHT_PX,
        pl,
        pr: 0.875,
        flexShrink: 0,
        borderRadius: `${RADIUS.sm}px`,
        border: `1px solid ${tone ?? theme.palette.divider}`,
        bgcolor:
          theme.palette.mode === 'dark'
            ? 'rgba(255,255,255,0.03)'
            : 'rgba(0,0,0,0.02)',
        color: tone ?? 'text.secondary',
      })}
    >
      {children}
    </Box>
  );
}

/** Text inside a meta pill, inheriting the pill's (possibly toned) colour. */
export function TagLabel({ children }: { children: ReactNode }): JSX.Element {
  return (
    <Typography
      component="span"
      sx={{
        fontSize: TYPO.micro,
        fontWeight: FONT_WEIGHT.medium,
        letterSpacing: '0.2px',
        color: 'inherit',
        whiteSpace: 'nowrap',
        lineHeight: 1,
      }}
    >
      {children}
    </Typography>
  );
}

/**
 * Product-variant pill: USB / Wi-Fi icon + the SKU it maps onto
 * (`Lite` / `Wireless`). Always-on, neutral - it's stable identity,
 * not a health signal.
 */
export function VariantTag({ transport }: { transport: string }): JSX.Element {
  return (
    <MetaPill pl={0.5}>
      <TransportChip transport={transport} iconOnly />
      <TagLabel>{transportLabelOf(transport)}</TagLabel>
    </MetaPill>
  );
}
