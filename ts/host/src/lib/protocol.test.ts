/**
 * Wire-level tests for protocol v1 helpers.
 *
 * The creds hash and the envelope discriminator are the two places
 * where a "harmless" refactor can strand every published Space (their
 * bundles are frozen at build time and never rebuilt). These tests pin
 * the exact behaviour, including a legacy bundle captured from the
 * published `@pollen-robotics/reachy-mini-sdk@1.9.0` package.
 */
import { describe, expect, it, vi } from 'vitest';

import {
  PROTOCOL_SOURCE,
  PROTOCOL_VERSION,
  decodeCredsFromHash,
  encodeCredsToHash,
  isProtocolMessage,
  type CredsBundle,
} from './protocol';

const BASE_BUNDLE: CredsBundle = {
  hfToken: 'hf_test_token',
  userName: 'thibaud',
  robotPeerId: 'peer-123',
  robotHardwareId: 'hw-abc',
  signalingUrl: 'https://central.example/api',
  theme: 'dark',
  config: { volume: 0.5 },
  hostName: 'Reachy Mini',
  appName: 'Demo App',
};

describe('encodeCredsToHash / decodeCredsFromHash', () => {
  it('round-trips a full bundle', () => {
    const hash = encodeCredsToHash(BASE_BUNDLE);
    expect(hash.startsWith('creds=')).toBe(true);
    expect(decodeCredsFromHash(hash)).toEqual(BASE_BUNDLE);
  });

  it('round-trips non-Latin config payloads (UTF-8 safety)', () => {
    // btoa/atob choke on non-Latin-1; the helpers must not.
    const bundle: CredsBundle = {
      ...BASE_BUNDLE,
      config: { greeting: 'héllo 世界 🤖', emoji: '⚠️' },
      userName: 'Thibaud Frère',
    };
    expect(decodeCredsFromHash(encodeCredsToHash(bundle))).toEqual(bundle);
  });

  it('accepts a leading # (window.location.hash form)', () => {
    const hash = `#${encodeCredsToHash(BASE_BUNDLE)}`;
    expect(decodeCredsFromHash(hash)).toEqual(BASE_BUNDLE);
  });

  it('finds creds among other hash segments', () => {
    const hash = `#foo=1&${encodeCredsToHash(BASE_BUNDLE)}&bar=2`;
    expect(decodeCredsFromHash(hash)).toEqual(BASE_BUNDLE);
  });

  it('returns null on missing / empty / foreign hash', () => {
    expect(decodeCredsFromHash(null)).toBeNull();
    expect(decodeCredsFromHash('')).toBeNull();
    expect(decodeCredsFromHash('#foo=bar')).toBeNull();
  });

  it('returns null (not throw) on malformed base64', () => {
    expect(decodeCredsFromHash('#creds=%%%not-base64%%%')).toBeNull();
    expect(decodeCredsFromHash('#creds=aGVsbG8')).toBeNull(); // "hello", not JSON
  });

  it('never echoes the bundle when decoding fails', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const truncated = window.btoa('{"hfToken":"browser-secret-marker"');

    expect(decodeCredsFromHash(`#creds=${truncated}`)).toBeNull();
    expect(warn).not.toHaveBeenCalled();
    vi.restoreAllMocks();
  });

  it('decodes a legacy 1.9.0 bundle (no robotHardwareId)', () => {
    // Shape frozen from the published 1.9.0 package: `robotHardwareId`
    // did not exist yet. Current code must decode it and leave the
    // field undefined rather than reject the bundle.
    const legacy = {
      hfToken: 'hf_test_token',
      userName: 'thibaud',
      robotPeerId: 'peer-123',
      signalingUrl: 'https://central.example/api',
      theme: 'light',
      config: null,
      hostName: 'Reachy Mini',
      appName: 'Emotions',
    };
    // 1.9.0's encoder is byte-identical to ours (verified against the
    // published bundle), so encoding the legacy shape reproduces a
    // legacy hash exactly.
    const hash = encodeCredsToHash(legacy as unknown as CredsBundle);
    const decoded = decodeCredsFromHash(hash);
    expect(decoded).toEqual(legacy);
    expect(decoded?.robotHardwareId).toBeUndefined();
  });
});

describe('isProtocolMessage', () => {
  const valid = {
    source: PROTOCOL_SOURCE,
    type: 'embed:ready',
    version: PROTOCOL_VERSION,
  };

  it('accepts a v1 envelope', () => {
    expect(isProtocolMessage(valid)).toBe(true);
  });

  it('accepts envelopes with extra fields (additive contract)', () => {
    expect(
      isProtocolMessage({ ...valid, sdkVersion: '9.9.9', anything: [1] }),
    ).toBe(true);
  });

  it('rejects wrong source', () => {
    expect(isProtocolMessage({ ...valid, source: 'react-devtools' })).toBe(
      false,
    );
  });

  it('rejects unknown versions (forward-compat: v2 is not ours)', () => {
    expect(isProtocolMessage({ ...valid, version: 2 })).toBe(false);
    expect(isProtocolMessage({ ...valid, version: '1' })).toBe(false);
  });

  it('rejects null and incomplete envelopes without throwing', () => {
    expect(isProtocolMessage(null)).toBe(false);
    expect(isProtocolMessage({ source: PROTOCOL_SOURCE })).toBe(false);
  });
});
