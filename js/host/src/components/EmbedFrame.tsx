/**
 * Iframe wrapper for the embedded app.
 *
 * Responsibilities:
 *  - Forward the parent ref so the host can call
 *    `bridge.sendInit(iframe, ...)` once it sees `embed:ready`.
 *  - Build the embed URL from a base path + `?embedded=1` flag.
 *    Hash creds are injected separately on the live iframe via
 *    `EmbedFrame`'s `pendingHash` prop (we mutate `iframe.src`
 *    once with the hash, then leave it alone).
 *  - Provide a `<noscript>`-style copy when JS is disabled
 *    inside the iframe (unlikely on HF but cheap).
 */
import { forwardRef } from 'react';
import Box from '@mui/material/Box';

export interface EmbedFrameProps {
  /** Same-origin URL of the embedded app's `index.html`.
   *  Defaults to `${origin}/?embedded=1`. */
  src: string;
  /** Allow microphone capture inside the iframe (apps that need
   *  WebRTC mic input). Maps to the `allow="microphone; ..."`
   *  attribute. */
  enableMicrophone: boolean;
  /** Title attribute for a11y. */
  title: string;
  /** Set `visibility` so the iframe can stay mounted (and keep
   *  the SDK socket) while the ConnectingView overlay is on top. */
  visible: boolean;
}

export const EmbedFrame = forwardRef<HTMLIFrameElement, EmbedFrameProps>(
  ({ src, enableMicrophone, title, visible }, ref) => {
    const allow = [
      enableMicrophone ? 'microphone' : null,
      'camera',
      'autoplay',
      'fullscreen',
    ]
      .filter(Boolean)
      .join('; ');

    return (
      <Box
        sx={{
          position: 'absolute',
          inset: 0,
          bgcolor: 'background.default',
        }}
      >
        <Box
          component="iframe"
          ref={ref}
          src={src}
          title={title}
          allow={allow}
          sx={{
            width: '100%',
            height: '100%',
            border: 'none',
            display: 'block',
            visibility: visible ? 'visible' : 'hidden',
          }}
        />
      </Box>
    );
  },
);

EmbedFrame.displayName = 'EmbedFrame';
