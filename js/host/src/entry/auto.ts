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
 */
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
  window.ReachyMiniHost = { mountHost };
}

export { mountHost };
export type { MountHostOptions, MountedHost };
