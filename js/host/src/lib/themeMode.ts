/**
 * Resolve and react to the host shell's theme mode.
 *
 * Priority:
 *   1. `?theme=dark|light` URL parameter (set by mobile handoff
 *      or by a parent shell that wants to force a palette).
 *   2. `prefers-color-scheme` media query (OS / browser default).
 *
 * Also writes `data-theme="dark|light"` on `<html>` so the theme
 * bootstrap in `index.html` and any consumer CSS can react to
 * runtime changes (e.g. user toggles OS palette mid-session).
 */
import { useEffect, useState } from 'react';

import type { ThemeMode } from './protocol';

function readQueryThemeOverride(): ThemeMode | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = new URLSearchParams(window.location.search).get('theme');
    if (raw === 'dark' || raw === 'light') return raw;
  } catch {
    /* ignore malformed URLs */
  }
  return null;
}

function detectPreferred(): ThemeMode {
  if (typeof window === 'undefined' || !window.matchMedia) return 'dark';
  return window.matchMedia('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light';
}

function resolveInitial(): ThemeMode {
  return readQueryThemeOverride() ?? detectPreferred();
}

function applyToDocument(mode: ThemeMode): void {
  if (typeof document === 'undefined') return;
  document.documentElement.setAttribute('data-theme', mode);
}

/**
 * React hook returning the current theme mode and reacting to OS
 * palette changes. Stable across React StrictMode double-mounts:
 * the `matchMedia` listener is registered in an effect with a
 * proper cleanup.
 */
export function useThemeMode(): ThemeMode {
  const [mode, setMode] = useState<ThemeMode>(resolveInitial);

  useEffect(() => {
    applyToDocument(mode);
  }, [mode]);

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    // Don't follow the OS if the URL override is in effect; the
    // mobile shell sets `?theme=` and expects to keep control.
    if (readQueryThemeOverride() != null) return;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = (e: MediaQueryListEvent): void => {
      setMode(e.matches ? 'dark' : 'light');
    };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  return mode;
}
