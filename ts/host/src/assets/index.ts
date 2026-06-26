/**
 * Asset imports re-exported as data URIs (Vite's
 * `assetsInlineLimit: Infinity` inlines every SVG / image into
 * the JS bundle, so the host package ships as a single self-
 * contained CDN file - no per-app `public/` shipping).
 *
 * Consume these via `import { connectionSvg } from '@/assets'`
 * (or `../assets`) and pass them straight to `<img src={...}>`.
 */
import connectionSvg from './connection.svg';
import hfLogoSvg from './hf-logo.svg';
import reachyBusteSvg from './reachy-buste.svg';
import reachyHeadSvg from './reachy-head.svg';
import reachyStandardSvg from './reachy-standard.svg';

export {
  connectionSvg,
  hfLogoSvg,
  reachyBusteSvg,
  reachyHeadSvg,
  reachyStandardSvg,
};
