/**
 * CDN "embed" entry: vanilla TS, no React / MUI, ~5 KB gz
 * (plus the SDK runtime bundled in).
 *
 * Loaded inside the embedded app's iframe:
 *
 *   <script type="module"
 *     src="https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/embed.js">
 *   </script>
 *
 * The bundle exposes `connectToHost` on
 * `window.ReachyMiniHostEmbed` for legacy callers. ESM consumers
 * import normally.
 *
 * SDK auto-install
 * ────────────────
 * The embed client instantiates `new window.ReachyMini(...)`. Now
 * that the SDK ships in the same npm package we import it here
 * and self-assign onto `window.ReachyMini` (when unset) so apps
 * no longer need the old jsdelivr `<script type="module">` tag.
 * Existing apps that set `window.ReachyMini` themselves keep
 * working — we only assign when the global is missing.
 */
import { ReachyMini } from '@pollen-robotics/reachy-mini-sdk';
import { connectToHost } from '../embed';
import type {
  ConnectedHandle,
  ConnectToHostOptions,
} from '../embed';

declare global {
  interface Window {
    ReachyMiniHostEmbed?: {
      connectToHost<TConfig = unknown>(
        opts?: ConnectToHostOptions,
      ): Promise<ConnectedHandle<TConfig>>;
    };
  }
}

if (typeof window !== 'undefined') {
  if (!window.ReachyMini) {
    window.ReachyMini = ReachyMini;
    window.dispatchEvent(new Event('reachymini:ready'));
  }
  window.ReachyMiniHostEmbed = { connectToHost };
}

export { connectToHost };
export type { ConnectedHandle, ConnectToHostOptions };
