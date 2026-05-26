/**
 * CDN "auto" entry: the single script tag loaded from an app's
 * `index.html` to bring up the host shell.
 *
 *   <script type="module"
 *     src="https://cdn.jsdelivr.net/npm/@pollen-robotics/reachy-mini-sdk@1/host/dist/entry/auto.js">
 *   </script>
 *   <script type="module" src="/src/dispatch.ts"></script>
 *
 * The auto bundle exposes `mountHost` on `window.ReachyMiniHost`
 * for legacy / non-module callers. ESM consumers import it the
 * normal way.
 *
 * SDK auto-install
 * ────────────────
 * The host depends on `window.ReachyMini` for its OAuth helpers
 * (`useSdk`) and the embed client. Now that the SDK lives in the
 * same npm package, we import it directly and assign it to
 * `window.ReachyMini` here as a side effect, then dispatch the
 * `reachymini:ready` event the host's wait loops listen for.
 * Existing apps that still set `window.ReachyMini` themselves
 * (e.g. via the old jsdelivr `<script type="module">` tag) are
 * untouched: we only assign when the global is unset.
 */
import { ReachyMini } from '@pollen-robotics/reachy-mini-sdk';
import { mountHost } from '../mountHost';
import type { MountHostOptions, MountedHost } from '../mountHost';

declare global {
  interface Window {
    ReachyMiniHost?: {
      mountHost(options: MountHostOptions): MountedHost;
    };
  }
}

if (typeof window !== 'undefined') {
  if (!window.ReachyMini) {
    window.ReachyMini = ReachyMini;
    window.dispatchEvent(new Event('reachymini:ready'));
  }
  window.ReachyMiniHost = { mountHost };
}

export { mountHost };
export type { MountHostOptions, MountedHost };
