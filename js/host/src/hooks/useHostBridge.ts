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
  HostInitMsg,
  LeavingReason,
  ThemeMode,
} from '../lib/protocol';

export interface EmbedAppState {
  phase: AppPhase;
  connectingStep: AppConnectingStep | null;
  message: string | null;
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
  /** App reported an error. Fatal errors should switch the host
   *  to ErrorView; non-fatal can be toasted or logged. */
  onError(payload: { message: string; fatal: boolean; detail?: unknown }): void;
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
          });
          return;
        case 'embed:request-leave':
          callbacks.current.onRequestLeave();
          return;
        case 'embed:error':
          callbacks.current.onError({
            message: data.message,
            fatal: data.fatal,
            detail: data.detail,
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
            console.info(`[host-debug] embed:${msg.tag ?? '?'} ${asJson}`);
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

  return { sendInit, sendThemeChanged, sendConfigChanged, sendLeaving };
}

function postToFrame(iframe: HTMLIFrameElement, msg: unknown): void {
  if (!iframe.contentWindow) return;
  try {
    iframe.contentWindow.postMessage(msg, window.location.origin);
  } catch (err) {
    console.warn('[reachy-mini-sdk/host] postMessage to embed failed', err);
  }
}
