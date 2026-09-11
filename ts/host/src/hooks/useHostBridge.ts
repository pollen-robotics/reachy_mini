/**
 * Host-side postMessage bridge to the embedded app.
 *
 * Responsibilities:
 *  - Listen for `embed:*` envelopes on `window` and route them
 *    to caller-provided handlers.
 *  - Provide imperative `sendInit / sendThemeChanged /
 *    sendConfigChanged / sendLeaving` helpers that target a
 *    specific iframe.
 *  - Drop unrelated traffic (DevTools, MUI portals, browser
 *    extensions) via the `source` / `version` discriminator.
 *  - Validate the iframe's content origin against
 *    `window.location.origin` (same-origin deployment).
 */
import { useCallback, useEffect, useRef } from 'react';
import { createLogger } from '@pollen-robotics/reachy-mini-sdk';

import {
  PROTOCOL_SOURCE,
  PROTOCOL_VERSION,
  isProtocolMessage,
} from '../lib/protocol';
import type {
  AppConnectingStep,
  AppPhase,
  ConfigPayload,
  EmbedToHostMsg,
  EmbedUpdateProgressMsg,
  HostInitMsg,
  LeavingReason,
  ThemeMode,
} from '../lib/protocol';

const log = createLogger('host');

export interface EmbedAppState {
  phase: AppPhase;
  connectingStep: AppConnectingStep | null;
  message: string | null;
  /** Rolling-min round-trip time (ms) the embed measures on its own
   *  WebRTC pair and reports over `embed:app-state` once `live`. The
   *  host released its session slot to the iframe, so this is the only
   *  TRUE link-latency signal available; `null` when not yet measured
   *  or unavailable (e.g. iOS WKWebView). */
  rttMs: number | null;
  /** Daemon version the embed read over its data channel, or `null`
   *  while unknown (not yet resolved, or a daemon predating
   *  `get_version`). The shell has no channel of its own, so this is
   *  its only version signal - see `DaemonUpdateGate`. */
  daemonVersion: string | null;
  /** Build version of the SDK bundle the app loaded, or `null` when
   *  the SDK predates the field (see `sdkVersion` in protocol.ts).
   *  The web shell itself has no use for it (same bundle as the
   *  embed); parsed for consumers that update independently. */
  sdkVersion: string | null;
}

export interface UseHostBridgeOptions {
  /** Called once per session, when the embed posts
   *  `embed:ready`. The host MUST respond with `host:init`
   *  via `sendInit` before resolving the picker phase. */
  onEmbedReady(): void;
  /** Called on every `embed:app-state` envelope. */
  onAppState(state: EmbedAppState): void;
  /** App requests a clean leave (in-app button, etc.). Host
   *  should run the same tear-down as a user-initiated end. */
  onRequestLeave(): void;
  /** App finished its `host:leaving` tear-down (onLeave + the SDK's
   *  host-owned sleep/disable). The host can unmount now instead of
   *  waiting out its safety cap. Optional, like every additive
   *  callback: implementers outside this repo stay source-compatible. */
  onLeft?(): void;
  /** App reported an error. Fatal errors should switch the host
   *  to ErrorView; non-fatal can be toasted or logged. */
  onError(payload: { message: string; fatal: boolean; detail?: unknown }): void;
  /** Progress of a daemon update the host asked for via
   *  `sendStartUpdate`. Optional: hosts that never trigger one will
   *  never see these. */
  onUpdateProgress?(payload: EmbedUpdateProgress): void;
}

/** Payload of an `embed:update-progress` envelope. */
export interface EmbedUpdateProgress {
  status: EmbedUpdateProgressMsg['status'];
  line: string | null;
  error: string | null;
}

export interface HostBridge {
  /** `host:init`. Sends credentials + initial state to the embed. */
  sendInit(
    iframe: HTMLIFrameElement,
    payload: Omit<HostInitMsg, 'source' | 'type' | 'version'>,
  ): void;
  /** `host:theme-changed`. Push a theme update without reload. */
  sendThemeChanged(iframe: HTMLIFrameElement, theme: ThemeMode): void;
  /** `host:config-changed`. Push a config update without reload. */
  sendConfigChanged(
    iframe: HTMLIFrameElement,
    config: ConfigPayload,
  ): void;
  /** `host:leaving`. Notify the embed to tear down. The host
   *  unmounts the iframe after `timeoutMs` regardless. */
  sendLeaving(
    iframe: HTMLIFrameElement,
    reason: LeavingReason,
    timeoutMs: number,
  ): void;
  /** `host:start-update`. Ask the embed to run the daemon's PyPI
   *  self-update on our behalf (only it holds a data channel).
   *  Fire-and-forget: an embed too old to know the message ignores it,
   *  so callers MUST have their own timeout. */
  sendStartUpdate(iframe: HTMLIFrameElement, preRelease?: boolean): void;
  /** `host:cancel-update`. The host gave up on the update (stall
   *  timeout): the embed should disarm its update-mode plumbing.
   *  Fire-and-forget, additive - older embeds ignore it. */
  sendCancelUpdate(iframe: HTMLIFrameElement): void;
}

export function useHostBridge(opts: UseHostBridgeOptions): HostBridge {
  // Store callbacks in a ref so identity changes don't reattach
  // the message listener every render. The listener reads the
  // latest callbacks on every dispatch.
  const callbacks = useRef(opts);
  callbacks.current = opts;

  useEffect(() => {
    const expectedOrigin = window.location.origin;
    const onMessage = (event: MessageEvent): void => {
      if (event.origin !== expectedOrigin) return;
      if (!isProtocolMessage(event.data)) return;
      const data = event.data as EmbedToHostMsg;
      switch (data.type) {
        case 'embed:ready':
          callbacks.current.onEmbedReady();
          return;
        case 'embed:app-state':
          callbacks.current.onAppState({
            phase: data.phase,
            connectingStep: data.connectingStep ?? null,
            message: data.message ?? null,
            rttMs: data.rttMs ?? null,
            daemonVersion: data.daemonVersion ?? null,
            sdkVersion: data.sdkVersion ?? null,
          });
          return;
        case 'embed:request-leave':
          callbacks.current.onRequestLeave();
          return;
        case 'embed:left':
          callbacks.current.onLeft?.();
          return;
        case 'embed:error':
          callbacks.current.onError({
            message: data.message,
            fatal: data.fatal,
            detail: data.detail,
          });
          return;
        case 'embed:update-progress':
          callbacks.current.onUpdateProgress?.({
            status: data.status,
            line: data.line ?? null,
            error: data.error ?? null,
          });
          return;
        default: {
          const msg = data as unknown as {
            type?: string;
            tag?: string;
            payload?: unknown;
          };
          if (msg?.type === 'embed:debug') {
            let asJson = '';
            try {
              asJson = JSON.stringify(msg.payload ?? {});
            } catch {
              asJson = '<unserializable>';
            }
            // Mirror of the embed's diagnostic channel, one level up so it
            // shows in the parent page's console. Same leveling as the
            // embed side: lifecycle tags at info, traffic at debug.
            const tag = msg.tag ?? '?';
            const line = `embed ${tag} ${asJson}`;
            if (tag.startsWith('boot:') || tag.startsWith('leave:')) log.info(line);
            else log.debug(line);
          }
          return;
        }
      }
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, []);

  const sendInit = useCallback<HostBridge['sendInit']>(
    (iframe, payload) => {
      postToFrame(iframe, {
        ...payload,
        source: PROTOCOL_SOURCE,
        type: 'host:init',
        version: PROTOCOL_VERSION,
      });
    },
    [],
  );

  const sendThemeChanged = useCallback<HostBridge['sendThemeChanged']>(
    (iframe, theme) => {
      postToFrame(iframe, {
        source: PROTOCOL_SOURCE,
        type: 'host:theme-changed',
        version: PROTOCOL_VERSION,
        theme,
      });
    },
    [],
  );

  const sendConfigChanged = useCallback<HostBridge['sendConfigChanged']>(
    (iframe, config) => {
      postToFrame(iframe, {
        source: PROTOCOL_SOURCE,
        type: 'host:config-changed',
        version: PROTOCOL_VERSION,
        config,
      });
    },
    [],
  );

  const sendLeaving = useCallback<HostBridge['sendLeaving']>(
    (iframe, reason, timeoutMs) => {
      postToFrame(iframe, {
        source: PROTOCOL_SOURCE,
        type: 'host:leaving',
        version: PROTOCOL_VERSION,
        reason,
        timeoutMs,
      });
    },
    [],
  );

  const sendStartUpdate = useCallback<HostBridge['sendStartUpdate']>(
    (iframe, preRelease = false) => {
      postToFrame(iframe, {
        source: PROTOCOL_SOURCE,
        type: 'host:start-update',
        version: PROTOCOL_VERSION,
        preRelease,
      });
    },
    [],
  );

  const sendCancelUpdate = useCallback<HostBridge['sendCancelUpdate']>(
    (iframe) => {
      postToFrame(iframe, {
        source: PROTOCOL_SOURCE,
        type: 'host:cancel-update',
        version: PROTOCOL_VERSION,
      });
    },
    [],
  );

  return {
    sendInit,
    sendThemeChanged,
    sendConfigChanged,
    sendLeaving,
    sendStartUpdate,
    sendCancelUpdate,
  };
}

function postToFrame(iframe: HTMLIFrameElement, msg: unknown): void {
  if (!iframe.contentWindow) return;
  try {
    iframe.contentWindow.postMessage(msg, window.location.origin);
  } catch (err) {
    log.warn('postMessage to embed failed', err);
  }
}
