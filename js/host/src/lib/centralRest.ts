/**
 * Direct REST access to the HF central signaling server.
 *
 * The picker needs a list of robots, but opening an SSE
 * connection from the host shell (via `sdk.connect()`) breaks
 * the iframe's WebRTC handshake: central can't reliably route
 * the daemon's SDP offer once two SSE streams have been
 * registered for the same HF token, even after one disconnects.
 *
 * Central exposes a public `GET /api/robot-status` endpoint
 * authenticated by Bearer token that returns the same producer
 * list as `setPeerStatus → list`, without requiring an SSE
 * registration. The mobile app uses this for "remote mode" and
 * we reuse the exact same wire shape here.
 */
import type { RobotInfo } from './sdk-types';

const REQUEST_TIMEOUT_MS = 8_000;

interface CentralRobotEntry {
  id?: string;
  peerId?: string;
  peer_id?: string;
  busy?: boolean;
  activeApp?: string | null;
  meta?: {
    name?: string;
    transport?: string;
    hardware_id?: string;
    simulation?: boolean;
  };
  name?: string;
}

export interface FetchRobotsResult {
  ok: boolean;
  robots: RobotInfo[];
  reason?: string;
}

/** Coerce any of central's id field shapes into the SDK shape. */
function extractRobotId(entry: CentralRobotEntry | undefined): string | null {
  if (!entry) return null;
  const raw = entry.id ?? entry.peerId ?? entry.peer_id;
  return typeof raw === 'string' && raw.length > 0 ? raw : null;
}

function toRobotInfo(entry: CentralRobotEntry): RobotInfo | null {
  const id = extractRobotId(entry);
  if (!id) return null;
  const name = entry.meta?.name ?? entry.name;
  const transport =
    typeof entry.meta?.transport === 'string' && entry.meta.transport.length > 0
      ? entry.meta.transport
      : null;
  const hardwareId =
    typeof entry.meta?.hardware_id === 'string' && entry.meta.hardware_id.length > 0
      ? entry.meta.hardware_id
      : null;
  return {
    id,
    meta: name ? { name } : undefined,
    busy: entry.busy === true,
    activeApp:
      typeof entry.activeApp === 'string' && entry.activeApp.trim().length > 0
        ? entry.activeApp
        : null,
    transport,
    hardwareId,
  };
}

/**
 * Fetch the user's robot list directly from central.
 *
 * @param signalingUrl Base URL of the central server (e.g.
 *                     `https://pollen-robotics-reachy-mini-central.hf.space`)
 * @param hfToken      Hugging Face user token
 * @param signal       Optional AbortSignal for cancellation
 */
export async function fetchRobotsFromCentral(opts: {
  signalingUrl: string;
  hfToken: string;
  signal?: AbortSignal;
}): Promise<FetchRobotsResult> {
  const { signalingUrl, hfToken } = opts;
  if (!hfToken) return { ok: false, robots: [], reason: 'No HF token' };
  if (!signalingUrl) {
    return { ok: false, robots: [], reason: 'No signaling URL' };
  }

  const url = `${signalingUrl.replace(/\/$/, '')}/api/robot-status`;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  const signal = opts.signal
    ? mergeAbort([controller.signal, opts.signal])
    : controller.signal;

  try {
    const resp = await fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${hfToken}`,
        Accept: 'application/json',
      },
      signal,
    });

    if (resp.status === 401 || resp.status === 403) {
      return { ok: false, robots: [], reason: 'Token rejected by Hugging Face' };
    }
    if (!resp.ok) {
      return {
        ok: false,
        robots: [],
        reason: `Central returned HTTP ${resp.status}`,
      };
    }

    const data = (await resp.json()) as { robots?: CentralRobotEntry[] };
    const entries = Array.isArray(data.robots) ? data.robots : [];
    const robots = entries
      .map(toRobotInfo)
      .filter((r): r is RobotInfo => r != null);
    return { ok: true, robots };
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') {
      return { ok: false, robots: [], reason: 'Central request timed out' };
    }
    const message = err instanceof Error ? err.message : String(err);
    return { ok: false, robots: [], reason: `Network error: ${message}` };
  } finally {
    clearTimeout(timeout);
  }
}

function mergeAbort(signals: AbortSignal[]): AbortSignal {
  const controller = new AbortController();
  for (const s of signals) {
    if (s.aborted) {
      controller.abort();
      return controller.signal;
    }
    s.addEventListener('abort', () => controller.abort(), { once: true });
  }
  return controller.signal;
}
