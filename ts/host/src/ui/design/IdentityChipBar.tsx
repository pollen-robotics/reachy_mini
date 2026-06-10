/**
 * Robot identity block: robot name + transport tag (+ optional live
 * latency pill). Ported from the mobile shell's `IdentityChipBar`
 * (`reachy_mini_mobile_app/src/ui/screens/session/IdentityChipBar.tsx`),
 * trimmed for the host:
 *
 *   - No session-phase spinner: the host renders this only while the
 *     embedded app is live, so the transient Connecting/Reconnecting
 *     cue from the mobile session FSM has no equivalent here.
 *   - Latency comes from the embed-reported `rttMs` (the embed owns
 *     the live WebRTC pair; the host released its slot to the iframe),
 *     so the bars are driven purely by RTT.
 *
 * Layout:
 *
 *   Reachy_mini  [⌁ Lite]  [▮▮▮ 38 ms]
 *
 * The `secondary` variant renders the name as a muted sub-line (used
 * under the bigger app title in the host topbar).
 */
import type { JSX } from 'react';
import { Stack, Typography } from '@mui/material';

import { LinkQualityBars, linkQualityLevel } from './LinkQualityBars';
import { MetaPill, TagLabel, VariantTag } from './MetaPill';
import { FONT_WEIGHT, TYPO } from '../../lib/tokens';

/**
 * Format the rolling-min RTT as a compact integer `… ms` tag. An
 * unknown / not-yet-measured RTT reads as `0 ms`.
 */
function formatLatencyTag(rttMs: number | null): string {
  const ms = rttMs !== null && Number.isFinite(rttMs) && rttMs > 0 ? Math.round(rttMs) : 0;
  return `${ms} ms`;
}

export interface IdentityChipBarProps {
  robotName: string;
  /** Physical transport string from central (`wifi` / `usb` / …),
   *  rendered via `<VariantTag>` as the stable Lite/Wireless tag.
   *  `null` when central didn't advertise `meta.transport` for this
   *  robot - the tag is then omitted, but the robot name (and the
   *  latency pill, if any) still render. */
  transport: string | null;
  /** Rolling-min RTT (ms) reported by the embed, or `null` when not
   *  yet measured / unavailable. Drives the latency pill value + bars. */
  linkRttMs: number | null;
  /** Whether to show the live latency pill. Defaults to `true`; pass
   *  `false` (or rely on it being gated upstream) when no RTT is
   *  available so we don't paint a misleading `0 ms`. */
  showLatency?: boolean;
  /**
   * Visual weight of the robot name.
   *  - `'primary'` (default): bold, `text.primary`, `TYPO.md`.
   *  - `'secondary'`: medium, `text.secondary`, `TYPO.sm` - a sub-line
   *    under a bigger headline (the host topbar's app title).
   */
  variant?: 'primary' | 'secondary';
}

export function IdentityChipBar({
  robotName,
  transport,
  linkRttMs,
  showLatency = true,
  variant = 'primary',
}: IdentityChipBarProps): JSX.Element {
  const secondary = variant === 'secondary';
  const latencyText = formatLatencyTag(linkRttMs);

  return (
    <Stack direction="row" spacing={1} sx={{ alignItems: 'center', minWidth: 0, flex: 1 }}>
      <Typography
        sx={{
          minWidth: 0,
          fontSize: secondary ? TYPO.sm : TYPO.md,
          fontWeight: secondary ? FONT_WEIGHT.medium : FONT_WEIGHT.bold,
          color: secondary ? 'text.secondary' : 'text.primary',
          letterSpacing: '-0.1px',
          lineHeight: 1.2,
          flexShrink: 1,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
          '&::first-letter': { textTransform: 'uppercase' },
        }}
        noWrap
      >
        {robotName}
      </Typography>

      <Stack direction="row" spacing={0.75} sx={{ alignItems: 'center', flexShrink: 0 }}>
        {transport && <VariantTag transport={transport} />}
        {showLatency && (
          <MetaPill>
            <LinkQualityBars
              level={linkQualityLevel(linkRttMs)}
              title={`Link quality (${latencyText})`}
            />
            <TagLabel>{latencyText}</TagLabel>
          </MetaPill>
        )}
      </Stack>
    </Stack>
  );
}
