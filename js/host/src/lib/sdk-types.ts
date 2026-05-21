/**
 * Re-export of the SDK type surface, sourced from the canonical
 * declarations next to the SDK runtime
 * (`../../../reachy-mini-sdk.d.ts`). Kept as a thin barrel so existing
 * `import type ... from '../lib/sdk-types'` call sites in the host
 * code (`useSdk`, `useRobots`, `embed/index`, picker view, etc.)
 * don't need to reach across the package layout themselves.
 *
 * If you're touching the type surface, edit the canonical file at
 * `js/reachy-mini-sdk.d.ts` — it ships in the npm tarball as
 * the companion `.d.ts` next to `reachy-mini-sdk.js`.
 */

export type {
  RobotInfo,
  RobotState,
  ReachyMiniOptions,
  ReachyMiniInstance,
  ReachyMiniConstructor,
  AutoConnectOptions,
  AutoConnectRobotChoice,
  AutoConnectResult,
  MotionAwaitOptions,
  SubscribeLogsOptions,
} from '../../../reachy-mini-sdk';

import type { ReachyMiniConstructor } from '../../../reachy-mini-sdk';

/**
 * Host-internal global augmentation: the host shell instantiates
 * `new window.ReachyMini(opts)` (see `useSdk`, `embed/index.ts`)
 * after either the npm entry points or the CDN bundles have
 * auto-assigned the constructor at module-load time. We declare
 * the global as non-optional here so those use sites stay
 * non-null-asserted on the type level; the runtime guarantee is
 * upheld by the side-effect imports in `entry/auto.ts`,
 * `entry/embed.ts` and `index.ts`.
 *
 * The SDK package's companion `reachy-mini-sdk.d.ts` deliberately
 * does NOT augment `Window.ReachyMini` so external consumers who
 * import only the SDK don't have a phantom non-optional global
 * appear in their type-checker.
 */
declare global {
  interface Window {
    ReachyMini: ReachyMiniConstructor;
    /** Injected by HF Spaces at container start when
     *  `hf_oauth: true` is set in the Space frontmatter. */
    huggingface?: {
      variables?: {
        OAUTH_CLIENT_ID?: string;
        OAUTH_SCOPES?: string;
        SPACE_HOST?: string;
        SPACE_ID?: string;
      };
    };
  }
}

export {};
