/**
 * npm entry point. Re-exports the public surface so:
 *
 *   import { mountHost, connectToHost } from '@pollen-robotics/reachy-mini-sdk/host';
 *
 * works for IDE autocomplete and for app authors who prefer the
 * package directly over the CDN auto bundle.
 */

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
