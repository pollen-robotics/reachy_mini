import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_CENTRAL_SIGNALING_URL,
  resolveSignalingUrl,
} from './signalingUrl';

interface BrowserConfig {
  origin?: string;
  search?: string;
  signalingUrl?: string;
}

function setBrowser(config: BrowserConfig): void {
  vi.stubGlobal('window', {
    location: {
      hostname: new URL(config.origin ?? 'https://app.example').hostname,
      search: config.search ?? '',
    },
    huggingface: { variables: { SIGNALING_URL: config.signalingUrl } },
  });
}

const LOOPBACK_PAGE = 'http://localhost:5173';
const STAGING = '?signaling_url=https%3A%2F%2Fstaging-central.hf.space';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('resolveSignalingUrl', () => {
  it('ignores a hosted query override and warns once without echoing it', async () => {
    vi.resetModules();
    const { resolveSignalingUrl } = await import('./signalingUrl');
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    setBrowser({ search: STAGING });
    expect(resolveSignalingUrl()).toBe(DEFAULT_CENTRAL_SIGNALING_URL);
    expect(resolveSignalingUrl()).toBe(DEFAULT_CENTRAL_SIGNALING_URL);
    expect(warn).toHaveBeenCalledExactlyOnceWith(
      'Ignoring ?signaling_url= outside local development.',
    );
  });

  it.each([LOOPBACK_PAGE, 'http://LOCALHOST:5173', 'http://[::1]:5173'])(
    'honours a query override on %s',
    (origin) => {
      setBrowser({ origin, search: STAGING });
      expect(resolveSignalingUrl()).toBe('https://staging-central.hf.space');
    },
  );

  it('still validates a query override on a loopback page', () => {
    setBrowser({
      origin: LOOPBACK_PAGE,
      search: '?signaling_url=http%3A%2F%2Fcentral.example',
    });
    expect(resolveSignalingUrl()).toBe(DEFAULT_CENTRAL_SIGNALING_URL);
  });

  it.each([
    [
      'keeps a hosted HTTPS central and strips a trailing slash',
      { signalingUrl: 'https://central.example/reachy/v1/' },
      'https://central.example/reachy/v1',
    ],
    [
      'keeps loopback HTTP on a local page',
      { origin: LOOPBACK_PAGE, signalingUrl: 'http://127.0.0.1:9000/central/' },
      'http://127.0.0.1:9000/central',
    ],
    [
      'treats an empty variable as unset',
      { signalingUrl: '' },
      DEFAULT_CENTRAL_SIGNALING_URL,
    ],
  ])('author configuration %s', (_name, config, expected) => {
    setBrowser(config);
    expect(resolveSignalingUrl()).toBe(expected);
  });

  it.each([
    ['remote plaintext', 'http://central.example'],
    ['loopback from a hosted page', 'http://127.0.0.1:9000'],
    ['embedded credentials', 'https://user:secret-marker@central.example'],
    ['a query', 'https://central.example?next=attacker.example'],
    ['a hash', 'https://central.example#secret'],
    ['surrounding whitespace', ' https://central.example'],
    ['a backslash', 'https://central.example\\attacker.example'],
    ['a non-HTTP scheme', 'ftp://central.example'],
    ['a bare hostname', 'central.example'],
  ])('falls back to the default on author configuration with %s', (_name, value) => {
    setBrowser({ signalingUrl: value });
    expect(resolveSignalingUrl()).toBe(DEFAULT_CENTRAL_SIGNALING_URL);
  });

  it('falls back when the page has no hostname', () => {
    setBrowser({ origin: 'about:blank', search: STAGING });
    expect(resolveSignalingUrl()).toBe(DEFAULT_CENTRAL_SIGNALING_URL);
  });
});
