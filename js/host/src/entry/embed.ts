/**
 * CDN "embed" entry: vanilla TS, no React / MUI, ~5 KB gz.
 *
 * Loaded inside the embedded app's iframe alongside the SDK:
 *
 *   <script type="module"
 *     src="https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/embed.js">
 *   </script>
 *
 * The bundle exposes `connectToHost` on
 * `window.ReachyMiniHostEmbed` for legacy callers. ESM consumers
 * import normally.
 */
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
  window.ReachyMiniHostEmbed = { connectToHost };
}

export { connectToHost };
export type { ConnectedHandle, ConnectToHostOptions };
