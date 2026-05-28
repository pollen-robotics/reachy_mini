/**
 * Public React component for the host shell.
 *
 * Wraps `ReachyHostShell` with the bundled MUI theme provider,
 * CssBaseline, and the React-friendly hooks. App authors who
 * want a single `<ReachyHost />` JSX node import this; everyone
 * else uses `mountHost()`.
 *
 * Strict Mode safety: the underlying `useSdk` hook stores its
 * SDK instance at module scope, so a double-mount in dev does
 * NOT create two SDK instances. See SPEC §8.1 / §8.4.
 */
import type { JSX } from 'react';
import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider } from '@mui/material/styles';

import { ReachyHostShell } from './components/ReachyHostShell';
import { useSdk } from './hooks/useSdk';
import { resolveSignalingUrl } from './lib/signalingUrl';
import {
  hasCachedDevToken,
  resolveClientIdSource,
  readUrlConfig,
} from './lib/settings';
import { useThemeMode } from './lib/themeMode';
import { darkTheme, lightTheme } from './lib/theme';
import type { ConfigPayload } from './lib/protocol';

export interface ReachyHostProps {
  /** App's display name. Required - shown in top bar, passed to
   *  the SDK, and surfaced to other apps that may collide on a
   *  busy robot. */
  appName: string;
  /** Top-bar icon. Recommended size 32×32. */
  appIconUrl?: string;
  /** Emoji fallback if no icon. */
  appEmoji?: string;
  /** Allow microphone capture inside the iframe. */
  enableMicrophone?: boolean;
  /** HF OAuth client ID. Falls back to
   *  `window.huggingface.variables.OAUTH_CLIENT_ID` then
   *  `localStorage`. */
  clientId?: string;
  /** Embed entry path (defaults to `/?embedded=1`). */
  embedPath?: string;
  /** Host display name (e.g. "Reachy Mini Hub"). */
  hostName?: string;
  /** Initial config payload (typically decoded from
   *  `?config=`). */
  initialConfig?: ConfigPayload;
}

export function ReachyHost({
  appName,
  appIconUrl,
  appEmoji,
  enableMicrophone = false,
  clientId,
  embedPath = '/?embedded=1',
  hostName = 'Reachy Mini',
  initialConfig,
}: ReachyHostProps): JSX.Element {
  const theme = useThemeMode();
  const { clientId: resolvedClientId, source: clientIdSource } =
    resolveClientIdSource(clientId);
  const signalingUrl = resolveSignalingUrl();

  const { sdk } = useSdk({
    appName,
    signalingUrl,
    enableMicrophone,
    clientId: resolvedClientId,
  });

  const config = initialConfig ?? readUrlConfig() ?? null;

  // No clientId anywhere AND no dev-token cached from `mountHost()`
  // means clicking "Sign in" is going to throw `Missing clientId` -
  // surface a dev hint pointing at `.env.local` before the user
  // discovers it the hard way. We deliberately read the cached
  // dev token (not the session storage flag, which gets wiped by
  // `signOut()`) so the banner stays hidden after a logout.
  const isLocalDevMissingConfig =
    clientIdSource === 'none' && !hasCachedDevToken();

  return (
    <ThemeProvider theme={theme === 'dark' ? darkTheme : lightTheme}>
      <CssBaseline />
      <ReachyHostShell
        sdk={sdk}
        appName={appName}
        appIconUrl={appIconUrl}
        appEmoji={appEmoji}
        hostName={hostName}
        theme={theme}
        initialConfig={config}
        enableMicrophone={enableMicrophone}
        embedPath={embedPath}
        isLocalDevMissingConfig={isLocalDevMissingConfig}
      />
    </ThemeProvider>
  );
}

