/**
 * Leveled, namespaced console logging shared by the SDK core, the
 * embed runtime, and the host shell.
 *
 * Every log line is prefixed `[reachy:<ns>]` so a devtools filter on
 * `reachy:` shows the whole story (or `reachy:session` just the
 * reconnect layer), and gated by a single global level:
 *
 *   - `debug` — per-message traffic (commands, SSE events, media cache
 *     ops). Off by default; also hidden behind devtools' "Verbose"
 *     filter since it maps to `console.debug`.
 *   - `info`  — lifecycle transitions (boot phases, wake/sleep steps,
 *     reconnects). The default: a handful of lines per session, not
 *     per second.
 *   - `warn` / `error` — always worth seeing.
 *   - `silent` — nothing at all.
 *
 * The level persists in `localStorage["reachy-log"]` so it survives
 * reloads mid-debugging. Flip it from the console without touching
 * code:
 *
 *   localStorage.setItem('reachy-log', 'debug'); location.reload();
 *
 * or at runtime via the exported `setLogLevel('debug')`.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'silent';

const ORDER: Record<LogLevel, number> = {
    debug: 0,
    info: 1,
    warn: 2,
    error: 3,
    silent: 4,
};

const STORAGE_KEY = 'reachy-log';

function initialLevel(): LogLevel {
    try {
        const stored = globalThis.localStorage?.getItem(STORAGE_KEY);
        if (stored && stored in ORDER) return stored as LogLevel;
    } catch {
        /* SSR or storage-denied context */
    }
    return 'info';
}

let currentLevel: LogLevel = initialLevel();

/** Set the global log level (persisted to localStorage when available). */
export function setLogLevel(level: LogLevel): void {
    currentLevel = level;
    try {
        globalThis.localStorage?.setItem(STORAGE_KEY, level);
    } catch {
        /* ignore */
    }
}

export function getLogLevel(): LogLevel {
    return currentLevel;
}

export interface Logger {
    debug: (...args: unknown[]) => void;
    info: (...args: unknown[]) => void;
    warn: (...args: unknown[]) => void;
    error: (...args: unknown[]) => void;
}

/** Create a logger whose lines are prefixed `[reachy:<ns>]`. */
export function createLogger(ns: string): Logger {
    const prefix = `[reachy:${ns}]`;
    return {
        debug: (...args) => {
            if (ORDER[currentLevel] <= ORDER.debug) console.debug(prefix, ...args);
        },
        info: (...args) => {
            if (ORDER[currentLevel] <= ORDER.info) console.info(prefix, ...args);
        },
        warn: (...args) => {
            if (ORDER[currentLevel] <= ORDER.warn) console.warn(prefix, ...args);
        },
        error: (...args) => {
            if (ORDER[currentLevel] <= ORDER.error) console.error(prefix, ...args);
        },
    };
}
