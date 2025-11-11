# Architecture Comparison: Automatic1111 vs Reachy Control Center

## Automatic1111 (Stable Diffusion WebUI)

### Core Architecture

```
┌────────────────────────────────────────────┐
│  webui.py (Main Process)                   │
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │  Gradio Web Interface                 │ │
│  │  (HTML/CSS/JavaScript)                │ │
│  └──────────────────────────────────────┘ │
│                ↓                           │
│  ┌──────────────────────────────────────┐ │
│  │  modules/ (Core Python code)          │ │
│  │  - ui.py                              │ │
│  │  - scripts.py                         │ │
│  │  - extensions.py                      │ │
│  └──────────────────────────────────────┘ │
│                ↓                           │
│  ┌──────────────────────────────────────┐ │
│  │  Extensions (Python modules)          │ │
│  │                                       │ │
│  │  extensions-builtin/                  │ │
│  │  ├── Lora/                            │ │
│  │  │   ├── scripts/lora.py              │ │
│  │  │   ├── ui_*.py (Gradio UI)          │ │
│  │  │   └── metadata.ini                 │ │
│  │  │                                    │ │
│  │  extensions/ (user-installed)         │ │
│  │  └── custom-extension/                │ │
│  │      ├── scripts/                     │ │
│  │      └── metadata.ini                 │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
     (Everything in ONE Python process)
```

### Key Characteristics

1. **Web-Based UI (Gradio)**
   - All UI is HTML/CSS/JavaScript
   - Runs in browser
   - Gradio auto-generates UI from Python code

2. **Extension Loading**
   ```python
   # modules/extensions.py
   def list_extensions():
       for dirname in [extensions_builtin_dir, extensions_dir]:
           for extension_dirname in os.listdir(dirname):
               metadata = ExtensionMetadata(path, name)
               extension = Extension(name, path, metadata=metadata)
               extensions.append(extension)
   ```
   - Scans `extensions/` and `extensions-builtin/` directories
   - Loads `metadata.ini` for each extension
   - Imports Python modules directly into main process

3. **Extension Structure**
   ```
   extensions/my-extension/
   ├── metadata.ini          # Extension info
   ├── scripts/
   │   └── my_script.py      # Script that runs
   ├── javascript/
   │   └── my_ui.js          # Frontend code
   └── style.css             # Styling
   ```

4. **Metadata Format (`metadata.ini`)**
   ```ini
   [Extension]
   Name = My Extension
   Description = Does cool stuff

   [callbacks/my_extension.ui]
   Before = other_extension
   After = yet_another_extension

   Requires = some_dependency
   ```

5. **Integration Method**
   - Extensions add Gradio components directly to UI
   - Use callbacks/hooks to inject functionality
   - All Python code runs in same process
   - Extensions can import and modify core modules

6. **Communication**
   - Direct Python function calls (same process)
   - Shared global state
   - Gradio event handlers

---

## Your Current Desktop Viewer

### Current State

```
desktop_viewer.py (Single monolithic file ~2000+ lines)
├── ImGui UI rendering
├── MuJoCo 3D viewer
├── Choreography builder (hardcoded)
├── Move library (hardcoded)
└── Direct SDK imports
```

### Characteristics

- **Desktop app** (GLFW + ImGui + OpenGL)
- **Everything hardcoded** in one file
- **Direct SDK usage** (imports ReachyMini)
- **No extension system** yet

---

## Your Proposed Reachy Control Center

### Proposed Architecture

```
┌────────────────────────────────────────────┐
│  Control Center (Desktop App)              │
│  - ImGui UI                                │
│  - GLFW window                             │
│  - OpenGL rendering                        │
│                                            │
│  ┌──────────────────────────────────────┐ │
│  │  Core Panels (built-in)               │ │
│  │  - Daemon status                      │ │
│  │  - Manual control                     │ │
│  │  - Move library                       │ │
│  └──────────────────────────────────────┘ │
│                ↓                           │
│  ┌──────────────────────────────────────┐ │
│  │  Extension Manager                    │ │
│  │  - Scans extensions/ directory        │ │
│  │  - Reads manifest.json files          │ │
│  │  - Auto-generates UI from manifest    │ │
│  │  - Manages subprocess lifecycle       │ │
│  └──────────────────────────────────────┘ │
│                ↓                           │
│  ┌──────────────────────────────────────┐ │
│  │  Viewport Manager                     │ │
│  │  - MuJoCo 3D viewer                   │ │
│  │  - HTML/WebView embed                 │ │
│  │  - Extension displays                 │ │
│  └──────────────────────────────────────┘ │
└────────────────────────────────────────────┘
         ↓ HTTP REST API
┌────────────────────────────────────────────┐
│  Extensions (Separate Processes)           │
│                                            │
│  Process 1: Dance Dance Reachy :5050       │
│  ├── Python app with FastAPI               │
│  ├── Serves REST API                       │
│  ├── Serves HTML at /display               │
│  └── manifest.json                         │
│                                            │
│  Process 2: Choreography Builder :5100     │
│  ├── Python app with FastAPI               │
│  ├── Serves REST API                       │
│  ├── Serves HTML at /display               │
│  └── manifest.json                         │
│                                            │
│  Process 3: Conversation App :5300         │
│  └── ... same pattern ...                  │
└────────────────────────────────────────────┘
```

### Key Characteristics

1. **Desktop UI (ImGui)**
   - Native desktop app, not web-based
   - GLFW + OpenGL for rendering
   - ImGui for panels/controls

2. **Extension Loading**
   ```python
   # extension_manager.py (to be written)
   def discover_extensions():
       for ext_dir in Path("extensions").iterdir():
           manifest = json.load(open(ext_dir / "manifest.json"))

           # Check if daemon has required endpoints
           if check_daemon_compatibility(manifest):
               extensions.append(Extension(manifest))
   ```

3. **Extension Structure**
   ```
   extensions/dance-dance-reachy/
   ├── manifest.json         # UI declaration + lifecycle
   ├── main.py              # Extension app with REST API
   ├── requirements.txt
   └── templates/
       └── display.html     # HTML served at /display
   ```

4. **Manifest Format (`manifest.json`)**
   ```json
   {
     "extension": {
       "name": "Dance Dance Reachy",
       "api_base_url": "http://localhost:5050"
     },
     "sidebar_panel": {
       "controls": [
         {"type": "button", "label": "Start", "endpoint": "/start"}
       ]
     },
     "lifecycle": {
       "on_start": "python main.py --port 5050"
     }
   }
   ```

5. **Integration Method**
   - Extensions run as **separate processes**
   - Control center starts/stops them via subprocess
   - Communication via **HTTP REST API**
   - UI auto-generated from manifest declarations
   - Extensions serve HTML for viewport display

6. **Communication**
   - HTTP REST calls
   - Process isolation
   - No shared state
   - WebSocket for real-time updates (optional)

---

## Side-by-Side Comparison

| Aspect | Automatic1111 | Reachy Control Center |
|--------|---------------|----------------------|
| **UI Framework** | Gradio (web) | ImGui (desktop) |
| **Extension Execution** | Same process (Python modules) | Separate processes (subprocesses) |
| **Extension Discovery** | Scan directories, load `.py` files | Scan directories, read `manifest.json` |
| **Extension Config** | `metadata.ini` | `manifest.json` |
| **UI Generation** | Python code → Gradio components | JSON manifest → ImGui controls |
| **Communication** | Direct function calls | HTTP REST API |
| **State Sharing** | Global shared state | Isolated processes |
| **Extension Install** | Git clone to `extensions/` | Git clone to `extensions/` |
| **Viewport** | Web page panels | 3D viewer / HTML webview |
| **Running Extensions** | Imported modules (always loaded) | Started/stopped on demand |

---

## Detailed Comparison: Extension Loading

### Automatic1111 Way

1. **Scan directories:**
   ```python
   for dirname in [extensions_builtin_dir, extensions_dir]:
       for extension_dirname in os.listdir(dirname):
           # Found extension folder
   ```

2. **Load metadata.ini:**
   ```python
   metadata = ExtensionMetadata(path, canonical_name)
   config = configparser.ConfigParser()
   config.read(os.path.join(path, "metadata.ini"))
   ```

3. **Import Python modules:**
   ```python
   # Extensions have scripts/xyz.py files
   # These get imported and executed
   # They can call gradio functions to add UI
   ```

4. **Extension adds UI directly:**
   ```python
   # Inside extension's script
   import gradio as gr

   def on_ui():
       with gr.Accordion("My Extension"):
           button = gr.Button("Do Thing")
           button.click(my_function)
   ```

### Your Way (Proposed)

1. **Scan directories:**
   ```python
   for ext_dir in Path("extensions").iterdir():
       # Found extension folder
   ```

2. **Load manifest.json:**
   ```python
   manifest = json.load(open(ext_dir / "manifest.json"))
   ```

3. **Start subprocess:**
   ```python
   # Start extension as separate process
   process = subprocess.Popen(manifest["lifecycle"]["on_start"])
   ```

4. **Auto-generate UI from manifest:**
   ```python
   # Control center reads manifest controls
   for control in manifest["sidebar_panel"]["controls"]:
       if control["type"] == "button":
           if imgui.button(control["label"]):
               requests.post(f"{api_url}{control['endpoint']}")
   ```

5. **Extension serves its own UI:**
   ```python
   # Inside extension's main.py
   from fastapi import FastAPI

   app = FastAPI()

   @app.post("/start")
   def start():
       # Do the thing
       return {"status": "started"}

   @app.get("/display")
   def display():
       return HTMLResponse(open("templates/display.html").read())
   ```

---

## Key Architectural Decisions

### Why Automatic1111 Uses Same-Process Modules

**Pros:**
- ✅ Fast (no HTTP overhead)
- ✅ Can share GPU/models efficiently
- ✅ Direct access to core functionality
- ✅ Simpler for extension developers (just Python)

**Cons:**
- ❌ Extensions can crash the whole app
- ❌ Can't use different languages (must be Python)
- ❌ Memory leaks in extensions affect everything
- ❌ Hard to sandbox/isolate extensions

**Why it works for them:**
- Stable Diffusion is computationally heavy
- All extensions need GPU access
- Performance critical (image generation)
- Extensions are mostly UI + model tweaks

### Why You Should Use Separate Processes

**Pros:**
- ✅ Extension crash doesn't kill control center
- ✅ Can use any language (Python, Node, Rust, etc.)
- ✅ Each extension owns its resources
- ✅ Can restart extensions without restarting UI
- ✅ Better security/sandboxing
- ✅ Extensions can be developed independently

**Cons:**
- ❌ HTTP overhead (negligible for robot control)
- ❌ More complex (need to manage processes)
- ❌ Can't directly share memory

**Why it works for you:**
- Robot control isn't computationally heavy
- Each extension has different needs (webcam, audio, AI)
- Extensions should be independent apps
- Beta testers will create diverse extensions
- Want to support community contributions safely

---

## Installation Flow Comparison

### Automatic1111

```bash
# User goes to Extensions tab in web UI
# Enters GitHub URL
# Clicks Install

# Behind the scenes:
cd extensions/
git clone https://github.com/user/extension-name
cd extension-name
pip install -r requirements.txt

# Restart webui
# Extension auto-loaded on next launch
```

### Your Approach (Proposed)

```bash
# User enters GitHub URL in Extensions panel
# Clicks Install

# Control center does:
cd extensions/
git clone https://github.com/user/extension-name
cd extension-name

# Read manifest.json
manifest = json.load("manifest.json")

# Run install commands
for cmd in manifest["lifecycle"]["on_install"]["commands"]:
    subprocess.run(cmd)

# Extension now appears in sidebar
# Click "Start" to launch it
```

---

## What You Can Learn from Automatic1111

### Good Ideas to Adopt

1. **Directory scanning for auto-discovery**
   - They scan `extensions/` and `extensions-builtin/`
   - You should do the same

2. **Metadata files**
   - They use `metadata.ini`
   - You use `manifest.json` (better for your use case)

3. **Built-in vs User extensions**
   - `extensions-builtin/` ships with app
   - `extensions/` for user installs
   - Good pattern to copy

4. **Extension enable/disable**
   - They track enabled state
   - Let users disable without uninstalling

5. **Dependency checking**
   - They check `Requires` field in metadata
   - You should validate daemon endpoints before loading

### Bad Ideas to Avoid

1. **❌ Don't load extensions as Python modules**
   - Keep them as separate processes

2. **❌ Don't share global state**
   - Use REST API for communication

3. **❌ Don't let extensions import core code**
   - Keep clear boundaries

4. **❌ Don't use callbacks/hooks**
   - Use declarative manifests instead

---

## Recommended Architecture

Based on both approaches, here's what you should build:

```
Reachy Control Center
├── Core (always running)
│   ├── main.py (entry point)
│   ├── extension_manager.py (discover + lifecycle)
│   ├── ui_generator.py (manifest → ImGui)
│   ├── viewport_manager.py (3D viewer + webview)
│   └── daemon_client.py (REST API wrapper)
│
├── extensions-builtin/ (ship with app)
│   ├── manual-control/
│   │   └── manifest.json (no subprocess, just UI)
│   └── move-library/
│       └── manifest.json (no subprocess, just UI)
│
└── extensions/ (user installs)
    ├── dance-dance-reachy/
    │   ├── manifest.json
    │   ├── main.py (FastAPI app)
    │   └── requirements.txt
    │
    ├── choreography-builder/
    │   └── ... same pattern ...
    │
    └── conversation-app/
        └── ... same pattern ...
```

### Extension Types

**Type 1: UI-Only Extensions (built-in)**
- Just manifest.json
- No subprocess
- UI talks directly to daemon
- Example: Manual control sliders

**Type 2: Full Extensions (user-installed)**
- manifest.json + Python app
- Runs as subprocess
- Serves REST API + HTML
- Example: Dance Dance Reachy

---

## Next Steps

1. **Study Automatic1111's extension discovery** (you've done this)
2. **Adapt their directory scanning** (good pattern)
3. **Keep your subprocess approach** (better for your use case)
4. **Build extension_manager.py** (similar to their extensions.py)
5. **Build ui_generator.py** (reads manifest, generates ImGui)
6. **Test with one extension** (DDR is perfect)

---

*Automatic1111 taught us how to discover and manage extensions, but your subprocess + REST API approach is better suited for a robot control ecosystem where extensions are diverse independent applications.*
