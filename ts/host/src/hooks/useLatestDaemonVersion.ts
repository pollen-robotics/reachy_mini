/**
 * React binding for `fetchLatestDaemonVersion`. Split from
 * `lib/daemonRelease.ts` so that module stays React-free - the embed
 * runtime imports the same policy helpers and is bundled without React.
 */
import { useEffect, useState } from 'react';

import { fetchLatestDaemonVersion } from '../lib/daemonRelease';

/**
 * Latest daemon version, or `null` until it resolves / when it can't be
 * determined. Kicks off a single fetch on mount; `fetchLatestDaemonVersion`
 * serves its localStorage cache when fresh, so this is usually instant.
 */
export function useLatestDaemonVersion(): string | null {
  const [latest, setLatest] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetchLatestDaemonVersion().then((value) => {
      if (!cancelled) setLatest(value);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return latest;
}
