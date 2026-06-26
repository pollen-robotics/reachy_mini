# Building a Web UI for the Reachy Mini Desktop App

When your app defines a `custom_app_url`, its web interface can be displayed directly inside the Reachy Mini desktop app's right panel. This guide covers how to build a web UI that looks great in this embedded context.

## Layout constraints

The embedded panel is an iframe with the following dimensions:

| Property | Value |
|----------|-------|
| Width | **~440px** (fixed) |
| Height | **~600px** (fixed, not scrollable by default) |
| Position | Right side of the 900×670px desktop window |

Design for a **narrow, vertical layout** — similar to a mobile app in portrait mode.

## Theme integration

When your app runs inside the desktop panel, the iframe URL includes query parameters with the current theme. Your app can read these to match the desktop app's look and feel:

| Parameter | Example | Description |
|-----------|---------|-------------|
| `embedded` | `1` | Always `"1"` when inside the desktop panel |
| `theme` | `dark` or `light` | Current theme |
| `accent` | `FF9500` | Accent color (hex, no `#`) |
| `bg` | `1a1a1a` or `fafafc` | Background color |
| `fg` | `f5f5f5` or `333333` | Foreground/text color |

### Reading theme params in JavaScript

```js
const params = new URLSearchParams(window.location.search);
const isEmbedded = params.get('embedded') === '1';
const theme = params.get('theme') || 'light';
const accent = '#' + (params.get('accent') || 'FF9500');
const bg = '#' + (params.get('bg') || 'fafafc');
const fg = '#' + (params.get('fg') || '333333');

// Apply as CSS custom properties
document.documentElement.style.setProperty('--bg', bg);
document.documentElement.style.setProperty('--fg', fg);
document.documentElement.style.setProperty('--accent', accent);
```

### Using CSS custom properties

```css
:root {
    --bg: #fafafc;
    --fg: #333333;
    --accent: #FF9500;
}

body {
    background: var(--bg);
    color: var(--fg);
}

button {
    background: var(--accent);
}
```

The CSS defines defaults, and the JS overrides them with the actual theme values at runtime. This way your app also looks fine when opened directly in a browser (without the query params).

## File structure

Your app's web UI lives in a `static/` directory next to `main.py`:

```
your_app/
├── main.py
└── static/
    ├── index.html      ← served at /
    ├── style.css        ← your styles
    ├── main.js          ← your logic
    └── fonts/           ← optional, local fonts
```

The base class `ReachyMiniApp` automatically:
- Serves `static/index.html` at `/`
- Mounts the `static/` directory at `/static`
- Starts a uvicorn server on the port from `custom_app_url`

## Best practices

### Keep it self-contained

- **Bundle all assets locally** (CSS, JS, fonts, images) — don't rely on external CDNs (Google Fonts, unpkg, etc.)
- The robot may not have internet access, and the desktop app's security policy may block external resources

### Use `defer` on script tags

```html
<!-- Good: script runs after DOM is parsed -->
<script src="/static/main.js" defer></script>

<!-- Bad: script may run before DOM elements exist -->
<script src="/static/main.js"></script>
```

Without `defer`, your JS runs as soon as it's downloaded. If it references DOM elements that haven't been parsed yet, you'll get `null is not an object` errors — especially in iframe contexts where timing can differ from a regular browser tab.

### Design for ~440px width

- Use a **single-column layout**
- Avoid horizontal scrolling
- Use `box-sizing: border-box` and constrain widths with `max-width: 100%`
- Test at 440×600px in your browser's responsive mode

### Minimal CSS reset

```css
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    padding: 16px;
    line-height: 1.5;
}
```

Using `-apple-system, system-ui, sans-serif` gives you the native system font on every OS — no need to bundle web fonts.

### API calls go to the same origin

Your FastAPI routes (defined via `self.settings_app`) are on the same origin as the web UI, so API calls are straightforward:

```js
// This just works — same origin, no CORS issues
const resp = await fetch('/api/my-endpoint', { method: 'POST', ... });
```

### Detect embedded mode

If your app should behave differently when embedded (e.g., hide a header, simplify layout):

```js
const isEmbedded = new URLSearchParams(window.location.search).get('embedded') === '1';
if (isEmbedded) {
    document.body.classList.add('embedded');
}
```

```css
/* Hide elements that don't fit in the small panel */
.embedded .full-page-header { display: none; }
```

## Frameworks

You don't need a framework. **Vanilla HTML/CSS/JS works great** for the typical app UI (toggles, buttons, status displays). The panel is small and the interactions are simple.

If you prefer a framework, keep it lightweight:
- **Preact** (~3KB) or **Alpine.js** (~15KB) are good choices
- Avoid heavy frameworks (React, Vue, Angular) unless your UI genuinely needs them — they add bundle size and complexity for a ~440px panel

If using a build tool (Vite, esbuild, etc.), make sure the output goes into the `static/` directory and uses **relative paths** (not absolute `/assets/...` paths that might conflict with the mount point).

## Testing

1. **In a browser**: Run your app locally and open `http://localhost:<port>` — verify it works standalone
2. **With theme params**: Open `http://localhost:<port>/?embedded=1&theme=dark&accent=FF9500&bg=1a1a1a&fg=f5f5f5` — verify dark mode looks correct
3. **At panel size**: Use your browser's responsive mode at **440×600px**
4. **In the desktop app**: Install and launch your app — verify it opens in the right panel
