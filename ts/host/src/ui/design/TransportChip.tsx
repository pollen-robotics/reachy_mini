/**
 * Compact transport tag for a robot listing.
 *
 * Ported from `reachy_mini_mobile_app/src/ui/design/TransportChip.tsx`
 * so the host shell and the mobile shell read the same `Lite` /
 * `Wireless` taxonomy. The host is a separate package (no cross-import
 * with the mobile app), so this is an assumed duplication - same
 * stance as the shared `lib/tokens.ts`.
 *
 * Two well-known values get an icon + typed label (`usb`, `wifi`);
 * anything else falls through to a plain label so a future daemon
 * advertising `ethernet` / `sim` / `mockup` still renders without a
 * component update.
 */
import type { JSX } from 'react';
import { Box, Chip } from '@mui/material';
import UsbIcon from '@mui/icons-material/Usb';
import WifiIcon from '@mui/icons-material/Wifi';

import { TYPO } from '../../lib/tokens';

/**
 * Map a transport string onto the product variant it implies. The
 * link type IS the SKU: `usb` -> Reachy Mini Lite (wired, needs a host
 * computer), `wifi` -> Reachy Mini Wireless (onboard compute +
 * battery). Anything else falls through to the raw string.
 */
export function transportLabelOf(transport: string): string {
  if (transport === 'usb') return 'Lite';
  if (transport === 'wifi') return 'Wireless';
  return transport;
}

export interface TransportChipProps {
  transport: string;
  /** Pixel height of the labelled chip. Defaults to 20px. */
  height?: number;
  /** Font size override. Defaults to `TYPO.tiny`. */
  fontSize?: string | number;
  /**
   * Icon-only variant for the well-known transports (`usb` /
   * `wifi`): drops the chip border + text label and renders just
   * the glyph in a muted colour. Used inside `<VariantTag>` so the
   * pill carries the label itself. Unknown transports still fall
   * through to the labelled chip.
   */
  iconOnly?: boolean;
}

export function TransportChip({
  transport,
  height = 20,
  fontSize = TYPO.tiny,
  iconOnly = false,
}: TransportChipProps): JSX.Element {
  const knownIcon =
    transport === 'usb' ? UsbIcon : transport === 'wifi' ? WifiIcon : null;

  // Icon-only variant: glyph alone, muted, no chrome.
  if (iconOnly && knownIcon) {
    const Icon = knownIcon;
    return (
      <Box
        aria-label={transport === 'usb' ? 'USB' : 'Wi-Fi'}
        sx={{
          display: 'inline-flex',
          alignItems: 'center',
          color: (theme) =>
            theme.palette.mode === 'dark'
              ? 'rgba(255,255,255,0.45)'
              : 'rgba(0,0,0,0.40)',
        }}
      >
        <Icon sx={{ fontSize: 16 }} />
      </Box>
    );
  }

  if (transport === 'usb') {
    return (
      <Chip
        size="small"
        icon={<UsbIcon sx={{ fontSize: 14 }} />}
        label="USB"
        variant="outlined"
        sx={{ height, fontSize, '.MuiChip-icon': { ml: 0.5 } }}
      />
    );
  }
  if (transport === 'wifi') {
    return (
      <Chip
        size="small"
        icon={<WifiIcon sx={{ fontSize: 14 }} />}
        label="Wi-Fi"
        variant="outlined"
        sx={{ height, fontSize, '.MuiChip-icon': { ml: 0.5 } }}
      />
    );
  }
  return (
    <Chip
      size="small"
      label={transport}
      variant="outlined"
      sx={{ height, fontSize, textTransform: 'capitalize' }}
    />
  );
}
