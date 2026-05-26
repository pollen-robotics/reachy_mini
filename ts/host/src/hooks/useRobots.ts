/**
 * Reactive view of "what robots does HF central know about for
 * this user, right now?" - REST snapshot + SSE realtime patches.
 *
 * Ported from the Reachy Mini mobile app's `useRemoteRobots`,
 * adapted to the host's data model (`RobotInfo`) and to React
 * without TanStack Query.
 *
 * Two data planes
 * ───────────────
 *  1. `fetchRobotsFromCentral` REST call (`/api/robot-status`):
 *     authoritative initial load, safety-net poll every 60 s, and
 *     re-fetch after an SSE reconnect.
 *
 *  2. `openCentralListener` SSE stream (`/events`):
 *     realtime push of busy/free and online/offline transitions.
 *     We update React state directly on each event so the UI
 *     reflects central within ~50 ms instead of one poll cycle.
 *
 * Critical handoff invariant
 * ──────────────────────────
 * The SSE listener registers a peer at central using the user's
 * HF token. Central enforces a 1:1 `token → peer_id` mapping, so
 * leaving the listener open across an iframe handoff would force
 * the iframe's SDK to clobber it during its own welcome, which
 * has been observed to break WebRTC SDP routing.
 *
 * The hook protects against this with the `enabled` flag: the
 * host shell passes `enabled: hostPhase === 'picking'`, so when
 * the user selects a robot and the phase flips to `embedded`, the
 * effect's cleanup fires (`handle.close()`) BEFORE React renders
 * the iframe and the embed's SDK opens its own SSE.
 *
 * Polling cadence
 * ───────────────
 * The SSE handles realtime; REST is the safety net for missed
 * events (transient SSE drop, central restart between two events,
 * future schema fields we don't yet reduce). 60 s mirrors the
 * mobile app's lease window.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

import { fetchRobotsFromCentral } from '../lib/centralRest';
import {
  openCentralListener,
  type CentralListenerHandle,
  type CentralStreamProducer,
} from '../lib/centralListener';
import { resolveSignalingUrl } from '../lib/signalingUrl';
import type { RobotInfo } from '../lib/sdk-types';

const POLL_INTERVAL_MS = 60_000;

export interface RobotsState {
  robots: RobotInfo[];
  /** True on the very first fetch (no data yet). */
  isLoading: boolean;
  /** True while a refresh is in-flight (poll or manual). */
  isRefreshing: boolean;
  /** Last error message or null. */
  error: string | null;
  /** Trigger an immediate REST refresh. */
  refresh(): void;
}

export function useRobots(opts: {
  hfToken: string | null;
  enabled: boolean;
}): RobotsState {
  const { hfToken, enabled } = opts;

  const [robots, setRobots] = useState<RobotInfo[]>([]);
  const [isLoading, setLoading] = useState<boolean>(false);
  const [isRefreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const hasLoadedRef = useRef<boolean>(false);
  const inflightRef = useRef<AbortController | null>(null);
  const pollHandleRef = useRef<number | null>(null);

  const cancelInflight = useCallback((): void => {
    if (inflightRef.current) {
      inflightRef.current.abort();
      inflightRef.current = null;
    }
  }, []);

  const doFetch = useCallback(async (): Promise<void> => {
    if (!hfToken) return;
    cancelInflight();
    const controller = new AbortController();
    inflightRef.current = controller;
    if (!hasLoadedRef.current) setLoading(true);
    setRefreshing(true);
    try {
      const result = await fetchRobotsFromCentral({
        signalingUrl: resolveSignalingUrl(),
        hfToken,
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      if (result.ok) {
        setRobots(result.robots);
        setError(null);
      } else {
        setError(result.reason ?? 'Unknown error');
      }
      hasLoadedRef.current = true;
    } finally {
      if (inflightRef.current === controller) inflightRef.current = null;
      setLoading(false);
      setRefreshing(false);
    }
  }, [cancelInflight, hfToken]);

  // REST plane: initial load + 60 s safety-net poll. Tied to
  // `enabled` so the listener and REST share the same lifetime
  // (which is what guarantees the close-before-iframe invariant).
  useEffect(() => {
    if (!enabled || !hfToken) {
      hasLoadedRef.current = false;
      setRobots([]);
      setError(null);
      return;
    }

    void doFetch();
    pollHandleRef.current = window.setInterval(() => {
      void doFetch();
    }, POLL_INTERVAL_MS);

    return () => {
      if (pollHandleRef.current != null) {
        window.clearInterval(pollHandleRef.current);
        pollHandleRef.current = null;
      }
      cancelInflight();
    };
  }, [cancelInflight, doFetch, enabled, hfToken]);

  // SSE plane: open the listener for the lifetime of "enabled".
  // The cleanup MUST run before the embed's SDK opens its own
  // SSE on the same token (see hook docstring).
  const droppedAtRef = useRef<number | null>(null);
  useEffect(() => {
    if (!enabled || !hfToken) return;

    const signalingUrl = resolveSignalingUrl();
    const handle: CentralListenerHandle = openCentralListener({
      token: hfToken,
      signalingUrl,
      onConnect: () => {
        if (droppedAtRef.current !== null) {
          droppedAtRef.current = null;
          void doFetch();
        }
      },
      onDisconnect: () => {
        droppedAtRef.current = Date.now();
      },
      onList: (event) => {
        setRobots(event.producers.map(producerToRobotInfo));
        setError(null);
        hasLoadedRef.current = true;
      },
      onPeerStatusChanged: (event) => {
        if (event.roles.length === 0) {
          setRobots((prev) => prev.filter((r) => r.id !== event.peerId));
        } else {
          // (Re)appearance: meta isn't in the event payload, so
          // pull fresh data from REST. Cheaper than a parallel
          // reducer that has to know every field central emits.
          void doFetch();
        }
      },
      onSessionStateChanged: (event) => {
        setRobots((prev) =>
          prev.map((r) =>
            r.id === event.peerId
              ? { ...r, busy: event.busy, activeApp: event.activeApp }
              : r,
          ),
        );
      },
      onError: (err) => {
        console.warn(
          '[reachy-mini-sdk/host] central listener error:',
          err.message,
        );
      },
    });

    return () => handle.close();
  }, [doFetch, enabled, hfToken]);

  const refresh = useCallback((): void => {
    if (!enabled || !hfToken) return;
    void doFetch();
  }, [doFetch, enabled, hfToken]);

  return { robots, isLoading, isRefreshing, error, refresh };
}

function producerToRobotInfo(p: CentralStreamProducer): RobotInfo {
  const meta = (p.meta ?? {}) as {
    name?: string;
    transport?: string;
    hardware_id?: string;
  };
  const transport =
    typeof meta.transport === 'string' && meta.transport.length > 0
      ? meta.transport
      : null;
  const hardwareId =
    typeof meta.hardware_id === 'string' && meta.hardware_id.length > 0
      ? meta.hardware_id
      : null;
  return {
    id: p.id,
    meta: meta.name ? { name: meta.name } : undefined,
    busy: p.busy === true,
    activeApp:
      typeof p.activeApp === 'string' && p.activeApp.trim().length > 0
        ? p.activeApp
        : null,
    transport,
    hardwareId,
  };
}
