/**
 * Tiny "signal bars" glyph that reads the live link QUALITY at a
 * glance, driven by latency (RTT).
 *
 * Ported from the mobile shell
 * (`reachy_mini_mobile_app/src/ui/design/LinkQualityBars.tsx` +
 * the `linkQualityLevel` mapping from its `transport-monitor`). The
 * host has no `TransportMonitor` of its own - it consumes the RTT the
 * embedded app reports over `embed:app-state` - so the level mapping
 * lives here, self-contained.
 *
 *   < 40 ms   ▮▮▮  excellent
 *   < 150 ms  ▮▮▯  good
 *   ≥ 150 ms  ▮▯▯  poor
 */
import type { JSX } from 'react';
import { Box, Stack } from '@mui/material';

import { STATUS } from '../../lib/tokens';

/** Discrete signal level, 0 (measuring / empty) to 3 (excellent). */
export type LinkQuality = 0 | 1 | 2 | 3;

/** RTT (ms) cut-offs for the 3 → 2 and 2 → 1 bar transitions. */
const RTT_GOOD_MS = 40; // < 40 ms  : excellent → 3 bars
const RTT_OK_MS = 150; // < 150 ms : usable     → 2 bars
//                        ≥ 150 ms : laggy      → 1 bar

/**
 * Map a rolling-min RTT (ms) to a 0-3 quality level. `null` (RTT not
 * yet measured / unavailable) reads as `0` (muted/empty bars).
 */
export function linkQualityLevel(rttMs: number | null): LinkQuality {
  if (rttMs === null) return 0;
  if (rttMs < RTT_GOOD_MS) return 3;
  if (rttMs < RTT_OK_MS) return 2;
  return 1;
}

/** Lit-bar colour + human label per quality level. */
const LEVEL_META: Record<LinkQuality, { color: string | null; label: string }> = {
  0: { color: null, label: 'measuring…' },
  1: { color: STATUS.warning, label: 'poor' },
  2: { color: 'text.secondary', label: 'good' },
  3: { color: STATUS.success, label: 'excellent' },
};

/** Bar heights from shortest to tallest, in px. */
const BAR_HEIGHTS_PX = [5, 8, 11] as const;
const BAR_WIDTH_PX = 3;

interface LinkQualityBarsProps {
  /** Pre-computed signal level (0-3), via `linkQualityLevel`. */
  level: LinkQuality;
  /** Accessible label / tooltip; defaults to the level's word. */
  title?: string;
}

export function LinkQualityBars({ level, title }: LinkQualityBarsProps): JSX.Element {
  const meta = LEVEL_META[level];
  const label = title ?? `Link quality: ${meta.label}`;

  return (
    <Stack
      role="img"
      aria-label={label}
      title={label}
      direction="row"
      spacing="2px"
      sx={{
        alignItems: 'flex-end',
        height: BAR_HEIGHTS_PX[BAR_HEIGHTS_PX.length - 1],
      }}
    >
      {BAR_HEIGHTS_PX.map((h, i) => {
        const lit = i < level;
        return (
          <Box
            key={i}
            sx={{
              width: BAR_WIDTH_PX,
              height: h,
              borderRadius: '1px',
              bgcolor: lit
                ? (meta.color ?? 'text.disabled')
                : (theme) =>
                    theme.palette.mode === 'dark'
                      ? 'rgba(255,255,255,0.18)'
                      : 'rgba(0,0,0,0.16)',
            }}
          />
        );
      })}
    </Stack>
  );
}
