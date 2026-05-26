/**
 * Reactive view of the Hugging Face account behind the current
 * token (avatar URL + canonical username).
 *
 * Ported from the Reachy Mini mobile app's `useHfProfile`, without
 * the TanStack Query dependency: the host package wants a minimal
 * deps surface, so we use a bare `useEffect` + `useState`. The
 * fetch is one-shot per token mount, with the result cached in a
 * module-level map so navigating between phases doesn't refetch.
 *
 * Hits `/api/whoami-v2` once per (token, mount) and caches the
 * result for the rest of the session. The auth gate already
 * validated the token, so a 4xx here is surfaced as "no profile"
 * rather than an error: the fallback avatar (initial of the
 * username we already have in memory) keeps the top bar
 * functional.
 */
import { useEffect, useState } from 'react';

export interface HfProfile {
  username: string | null;
  /** Fully-qualified avatar URL ready to drop into an `<img>` `src`. */
  avatarUrl: string | null;
}

interface WhoamiV2Response {
  name?: string;
  fullname?: string;
  /** May be path-relative (`/avatars/abc.svg`) or absolute. */
  avatarUrl?: string;
}

const HF_BASE = 'https://huggingface.co';
const REQUEST_TIMEOUT_MS = 8000;
const profileCache = new Map<string, HfProfile>();

function absolutiseAvatarUrl(raw: string | undefined): string | null {
  if (!raw) return null;
  if (raw.startsWith('http://') || raw.startsWith('https://')) return raw;
  if (raw.startsWith('//')) return `https:${raw}`;
  if (raw.startsWith('/')) return `${HF_BASE}${raw}`;
  return `${HF_BASE}/${raw}`;
}

async function fetchHfProfile(token: string): Promise<HfProfile> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const resp = await fetch(`${HF_BASE}/api/whoami-v2`, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/json',
      },
      signal: controller.signal,
    });
    if (!resp.ok) {
      throw new Error(`HF whoami HTTP ${resp.status}`);
    }
    const data = (await resp.json()) as WhoamiV2Response;
    return {
      username: data.name ?? data.fullname ?? null,
      avatarUrl: absolutiseAvatarUrl(data.avatarUrl),
    };
  } finally {
    clearTimeout(timeout);
  }
}

const EMPTY: HfProfile = { username: null, avatarUrl: null };

export function useHfProfile(token: string | null): HfProfile {
  const [profile, setProfile] = useState<HfProfile>(() =>
    token && profileCache.has(token) ? profileCache.get(token)! : EMPTY,
  );

  useEffect(() => {
    if (!token) {
      setProfile(EMPTY);
      return;
    }
    const cached = profileCache.get(token);
    if (cached) {
      setProfile(cached);
      return;
    }
    let cancelled = false;
    void fetchHfProfile(token)
      .then((p) => {
        if (cancelled) return;
        profileCache.set(token, p);
        setProfile(p);
      })
      .catch(() => {
        // Swallow: fallback (initial avatar) keeps the bar functional.
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  return profile;
}
