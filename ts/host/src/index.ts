/**
 * npm entry point for the `./host` subpath.
 *
 *   import { mountHost, connectToHost } from '@pollen-robotics/reachy-mini-sdk/host';
 *
 * works for IDE autocomplete and for app authors who prefer the
 * package directly over the CDN `host/auto` bundle.
 *
 * Importing this module also auto-installs the SDK constructor on
 * `window.ReachyMini` (when unset) and dispatches the
 * `reachymini:ready` event the host's wait loops listen for.
 * Apps consuming the host via npm therefore no longer need a
 * separate jsdelivr `<script type="module">` to expose the SDK
 * on the global — installing this package is enough.
 */

import { ReachyMini } from '@pollen-robotics/reachy-mini-sdk';

if (typeof window !== 'undefined' && !window.ReachyMini) {
  window.ReachyMini = ReachyMini;
  window.dispatchEvent(new Event('reachymini:ready'));
}

export { ReachyHost } from './ReachyHost';
export type { ReachyHostProps } from './ReachyHost';

export { mountHost } from './mountHost';
export type { MountHostOptions, MountedHost } from './mountHost';

export { connectToHost } from './embed';
export type { ConnectedHandle, ConnectToHostOptions } from './embed';

// Protocol types (useful for app authors implementing custom
// flows or unit-testing their dispatcher).
export type {
  AppConnectingStep,
  AppPhase,
  ConfigPayload,
  CredsBundle,
  EmbedToHostMsg,
  HostToEmbedMsg,
  LeavingReason,
  ThemeMode,
} from './lib/protocol';
export {
  PROTOCOL_VERSION,
  decodeCredsFromHash,
  encodeCredsToHash,
  isProtocolMessage,
} from './lib/protocol';

// SDK types (re-export so app authors can write
// `import type { ReachyMiniInstance } from '@pollen-robotics/reachy-mini-sdk/host'`.
export type {
  ReachyMiniInstance,
  ReachyMiniOptions,
  ReachyMiniConstructor,
  RobotInfo,
} from './lib/sdk-types';
