/**
 * Wire-compat tests for the host bridge, against GOLDEN ENVELOPES.
 *
 * Why golden fixtures
 * ───────────────────
 * A published Space bundles the embed runtime at build time and never
 * rebuilds: whatever we ship in the shell must keep understanding the
 * messages that OLD embeds emit, forever. The `LEGACY_*` fixtures
 * below are transcribed field-for-field from the published
 * `@pollen-robotics/reachy-mini-sdk@1.9.0` bundle
 * (`host/dist/chunks/index-*.js`, functions `b(...)` / `w(...)`) - the
 * oldest wire shape still in the wild. If a refactor stops accepting
 * them, THIS file must go red before a user does.
 *
 * The reverse direction matters too: an old embed drops anything that
 * fails its own `isProtocolMessage` (source === 'reachy-mini' &&
 * version === 1), so every message the bridge sends is asserted to
 * keep that envelope.
 */
import { describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';

import { useHostBridge, type UseHostBridgeOptions } from './useHostBridge';

/* ─────────────────── Golden fixtures (1.9.0) ─────────────────── */

/** `embed:ready` - bare envelope, no payload. */
const LEGACY_READY = {
  source: 'reachy-mini',
  type: 'embed:ready',
  version: 1,
};

/** `w('connecting', 'link')` - note: no `daemonVersion`, no
 *  `sdkVersion` (fields didn't exist), `message`/`rttMs` default null. */
const LEGACY_APP_STATE_CONNECTING = {
  source: 'reachy-mini',
  type: 'embed:app-state',
  version: 1,
  phase: 'connecting',
  connectingStep: 'link',
  message: null,
  rttMs: null,
};

/** `w('live', null)` at the end of boot. */
const LEGACY_APP_STATE_LIVE = {
  source: 'reachy-mini',
  type: 'embed:app-state',
  version: 1,
  phase: 'live',
  connectingStep: null,
  message: null,
  rttMs: null,
};

/** The RTT monitor's periodic `w('live', null, null, minRtt)`. */
const LEGACY_APP_STATE_RTT = {
  source: 'reachy-mini',
  type: 'embed:app-state',
  version: 1,
  phase: 'live',
  connectingStep: null,
  message: null,
  rttMs: 42,
};

const LEGACY_REQUEST_LEAVE = {
  source: 'reachy-mini',
  type: 'embed:request-leave',
  version: 1,
};

const LEGACY_ERROR = {
  source: 'reachy-mini',
  type: 'embed:error',
  version: 1,
  message: 'boom',
  fatal: true,
  detail: { stack: 'Error: boom' },
};

/** Diagnostics side-channel; must be swallowed, not routed. */
const LEGACY_DEBUG = {
  source: 'reachy-mini',
  type: 'embed:debug',
  version: 1,
  tag: 'boot:link:start',
  payload: { robotPeerId: 'peer-1' },
};

/* ─────────────────── Harness ─────────────────── */

function makeCallbacks() {
  return {
    onEmbedReady: vi.fn<Required<UseHostBridgeOptions>['onEmbedReady']>(),
    onAppState: vi.fn<Required<UseHostBridgeOptions>['onAppState']>(),
    onRequestLeave: vi.fn<Required<UseHostBridgeOptions>['onRequestLeave']>(),
    onLeft: vi.fn<Required<UseHostBridgeOptions>['onLeft']>(),
    onError: vi.fn<Required<UseHostBridgeOptions>['onError']>(),
    onUpdateProgress:
      vi.fn<Required<UseHostBridgeOptions>['onUpdateProgress']>(),
  } satisfies UseHostBridgeOptions;
}

function mountBridge() {
  const callbacks = makeCallbacks();
  const { result, unmount } = renderHook(() => useHostBridge(callbacks));
  return { callbacks, bridge: result.current, unmount };
}

/** Deliver an envelope the way the browser would: a `message` event
 *  on `window`, from the same origin (our deployment contract). */
function deliver(data: unknown, origin = window.location.origin): void {
  window.dispatchEvent(new MessageEvent('message', { data, origin }));
}

/** Fake iframe capturing outbound `postMessage` payloads. */
function makeIframe(): {
  iframe: HTMLIFrameElement;
  sent: () => unknown[];
} {
  const postMessage = vi.fn();
  const iframe = {
    contentWindow: { postMessage },
  } as unknown as HTMLIFrameElement;
  return {
    iframe,
    sent: () => postMessage.mock.calls.map((c) => c[0] as unknown),
  };
}

/** What a frozen 1.9.0 embed applies before trusting a message. */
function accepted19(msg: unknown): boolean {
  if (!msg || typeof msg !== 'object') return false;
  const m = msg as Record<string, unknown>;
  return (
    m.source === 'reachy-mini' &&
    typeof m.type === 'string' &&
    m.version === 1
  );
}

/* ─────────────────── Inbound: legacy envelopes ─────────────────── */

describe('useHostBridge accepts every envelope a 1.9.0 embed emits', () => {
  it('embed:ready → onEmbedReady', () => {
    const { callbacks, unmount } = mountBridge();
    deliver(LEGACY_READY);
    expect(callbacks.onEmbedReady).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('embed:app-state (connecting) → normalised with null version fields', () => {
    const { callbacks, unmount } = mountBridge();
    deliver(LEGACY_APP_STATE_CONNECTING);
    expect(callbacks.onAppState).toHaveBeenCalledWith({
      phase: 'connecting',
      connectingStep: 'link',
      message: null,
      rttMs: null,
      // The 1.9.0 wire has neither field: the bridge must default
      // both to null, never undefined (state initialisers rely on it).
      daemonVersion: null,
      sdkVersion: null,
    });
    unmount();
  });

  it('embed:app-state (live + periodic RTT) → phase and rtt flow through', () => {
    const { callbacks, unmount } = mountBridge();
    deliver(LEGACY_APP_STATE_LIVE);
    deliver(LEGACY_APP_STATE_RTT);
    expect(callbacks.onAppState).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ phase: 'live', rttMs: null }),
    );
    expect(callbacks.onAppState).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ phase: 'live', rttMs: 42 }),
    );
    unmount();
  });

  it('embed:request-leave → onRequestLeave', () => {
    const { callbacks, unmount } = mountBridge();
    deliver(LEGACY_REQUEST_LEAVE);
    expect(callbacks.onRequestLeave).toHaveBeenCalledTimes(1);
    unmount();
  });

  it('embed:error → onError with the full payload', () => {
    const { callbacks, unmount } = mountBridge();
    deliver(LEGACY_ERROR);
    expect(callbacks.onError).toHaveBeenCalledWith({
      message: 'boom',
      fatal: true,
      detail: { stack: 'Error: boom' },
    });
    unmount();
  });

  it('embed:debug is swallowed without routing to any handler', () => {
    const { callbacks, unmount } = mountBridge();
    const info = vi.spyOn(console, 'info').mockImplementation(() => {});
    deliver(LEGACY_DEBUG);
    for (const fn of Object.values(callbacks)) {
      expect(fn).not.toHaveBeenCalled();
    }
    info.mockRestore();
    unmount();
  });
});

/* ─────────────────── Inbound: current + hostile ─────────────────── */

describe('useHostBridge with current and malformed envelopes', () => {
  it('forwards the additive version fields when present', () => {
    const { callbacks, unmount } = mountBridge();
    deliver({
      ...LEGACY_APP_STATE_LIVE,
      daemonVersion: '1.9.0',
      sdkVersion: '1.10.0',
    });
    expect(callbacks.onAppState).toHaveBeenCalledWith(
      expect.objectContaining({
        daemonVersion: '1.9.0',
        sdkVersion: '1.10.0',
      }),
    );
    unmount();
  });

  it('routes embed:left and embed:update-progress (post-1.9.0 types)', () => {
    const { callbacks, unmount } = mountBridge();
    deliver({ source: 'reachy-mini', type: 'embed:left', version: 1 });
    expect(callbacks.onLeft).toHaveBeenCalledTimes(1);
    deliver({
      source: 'reachy-mini',
      type: 'embed:update-progress',
      version: 1,
      status: 'in_progress',
      line: 'Downloading...',
    });
    expect(callbacks.onUpdateProgress).toHaveBeenCalledWith({
      status: 'in_progress',
      line: 'Downloading...',
      error: null,
    });
    unmount();
  });

  it('ignores an unknown FUTURE embed:* type without crashing', () => {
    const { callbacks, unmount } = mountBridge();
    deliver({
      source: 'reachy-mini',
      type: 'embed:telemetry-v9',
      version: 1,
      blob: new Array(10).fill('x'),
    });
    for (const fn of Object.values(callbacks)) {
      expect(fn).not.toHaveBeenCalled();
    }
    unmount();
  });

  it('drops cross-origin traffic even with a perfect envelope', () => {
    const { callbacks, unmount } = mountBridge();
    deliver(LEGACY_READY, 'https://evil.example');
    expect(callbacks.onEmbedReady).not.toHaveBeenCalled();
    unmount();
  });

  it('drops foreign / versionless / junk messages', () => {
    const { callbacks, unmount } = mountBridge();
    deliver({ source: 'react-devtools-bridge', type: 'embed:ready' });
    deliver({ ...LEGACY_READY, version: 2 });
    deliver('a plain string');
    deliver(null);
    for (const fn of Object.values(callbacks)) {
      expect(fn).not.toHaveBeenCalled();
    }
    unmount();
  });
});

/* ─────────────────── Outbound: must stay 1.9.0-readable ─────────────────── */

describe('useHostBridge outbound messages keep the v1 envelope', () => {
  it('sendInit carries the envelope + the fields a 1.9.0 embed reads', () => {
    const { bridge, unmount } = mountBridge();
    const { iframe, sent } = makeIframe();
    bridge.sendInit(iframe, {
      theme: 'dark',
      signalingUrl: 'https://central.example/api',
      hfToken: 'hf_x',
      userName: 'thibaud',
      robotPeerId: 'peer-1',
      robotHardwareId: 'hw-1',
      config: null,
      hostName: 'Reachy Mini',
      appName: 'Demo',
    });
    const [msg] = sent();
    expect(accepted19(msg)).toBe(true);
    expect(msg).toEqual(
      expect.objectContaining({
        type: 'host:init',
        theme: 'dark',
        robotPeerId: 'peer-1',
        signalingUrl: 'https://central.example/api',
      }),
    );
    unmount();
  });

  it('sendThemeChanged / sendConfigChanged / sendLeaving / sendStartUpdate', () => {
    const { bridge, unmount } = mountBridge();
    const { iframe, sent } = makeIframe();
    bridge.sendThemeChanged(iframe, 'light');
    bridge.sendConfigChanged(iframe, { volume: 1 });
    bridge.sendLeaving(iframe, 'user-action', 5000);
    bridge.sendStartUpdate(iframe);

    const msgs = sent();
    expect(msgs).toHaveLength(4);
    for (const msg of msgs) expect(accepted19(msg)).toBe(true);
    expect(msgs[0]).toEqual(
      expect.objectContaining({ type: 'host:theme-changed', theme: 'light' }),
    );
    expect(msgs[1]).toEqual(
      expect.objectContaining({
        type: 'host:config-changed',
        config: { volume: 1 },
      }),
    );
    expect(msgs[2]).toEqual(
      expect.objectContaining({
        type: 'host:leaving',
        reason: 'user-action',
        timeoutMs: 5000,
      }),
    );
    expect(msgs[3]).toEqual(
      expect.objectContaining({ type: 'host:start-update', preRelease: false }),
    );
    unmount();
  });

  it('does not throw on an iframe with no contentWindow', () => {
    const { bridge, unmount } = mountBridge();
    const orphan = { contentWindow: null } as unknown as HTMLIFrameElement;
    expect(() => bridge.sendThemeChanged(orphan, 'dark')).not.toThrow();
    unmount();
  });
});
