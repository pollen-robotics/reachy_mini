# Skill: Create App

## When to Use

- User wants to create a new Reachy Mini application
- User asks how to structure an app
- User wants to publish an app to Hugging Face

## Important: Always Python Apps

**Always create Python apps using the app assistant.** Python apps are:
- Easily discoverable on Hugging Face
- Shareable via the robot's app store
- Can include web GUIs (via the `static/` folder)

JS-only apps are not yet supported for discovery/sharing. If the user needs a web UI, create a Python app with a web frontend in `static/`.

## Quick Check

If an app folder already exists with `README.md` containing `reachy_mini_python_app` tag, the app structure is probably already set up. In doubt, double check.

---

## Procedure

### Step 1: Use the App Assistant

**CRITICAL: Never create app folders manually.** Always use the assistant - it handles boilerplate, metadata tags, entry points, and proper structure. Manual creation leads to subtle issues that are hard to debug.

**IMPORTANT: Always use `--publish` unless the user explicitly requests a local-only app.** Publishing immediately creates a Git repo on Hugging Face, enabling proper version control and sharing from the start.

**If the command fails for any reason:** Ask the user to run it manually in their terminal rather than attempting complex workarounds.

```bash
# Default template - minimal blank app (recommended for most cases):
reachy-mini-app-assistant create <app_name> <path> --publish

# Conversation template - fork of the conversation app:
reachy-mini-app-assistant create --template conversation <app_name> <path> --publish

# Only if user explicitly wants local-only (no HF repo):
reachy-mini-app-assistant create <app_name> <path>
```

#### Which template to choose?

| Template | Use when |
|----------|----------|
| **default** | Most apps. Gives you a minimal working structure to build from scratch. |
| **conversation** | App needs LLM integration, speech, or making the robot talk. Forks the conversation app with a locked profile - includes audio pipeline, LLM tools, and all the plumbing already set up. |

**IMPORTANT: Both `app_name` AND `path` are required for non-interactive mode.** If either is omitted, the command will prompt interactively (which fails in non-TTY environments like Claude Code).

Options:
- `--template conversation` - Fork the conversation app (for LLM/speech apps)
- `--publish` - Create a Git repo on Hugging Face immediately **(always use this)**
- `--private` - Make the HF Space private (default is public)

**Prerequisite for `--publish`:** Must be logged in to Hugging Face first:
```bash
hf auth login
```

Example:
```bash
reachy-mini-app-assistant create my_app_name . --publish
reachy-mini-app-assistant create --template conversation my_assistant . --publish
```

### Step 2: Understand the Generated Structure

```
my_app/
├── index.html              # HuggingFace Space landing page
├── style.css               # Landing page styles
├── pyproject.toml          # Package config (includes reachy_mini tag!)
├── README.md               # Must contain reachy_mini tag in YAML frontmatter
├── .gitignore
└── my_app/
    ├── __init__.py
    ├── main.py             # Your app code (run method)
    └── static/             # Optional web UI
        ├── index.html
        ├── style.css
        └── main.js
```

### Step 3: Development Workflow

1. **Create and publish immediately** with `--publish` to get a Git repo
2. **Develop iteratively** using standard git: `git add`, `git commit`, `git push`
3. **Validate** with `reachy-mini-app-assistant check /path/to/app`

### Step 4: Before Writing Code

Create a `plan.md` file in the app directory with:
- Your understanding of what the user wants
- Technical approach (components, patterns)
- Questions that need clarification

Wait for user to answer questions before implementing.

---

## Full Tutorial

For detailed guide with screenshots: https://huggingface.co/blog/pollen-robotics/make-and-publish-your-reachy-mini-apps

---

## Common Patterns to Consider

When planning the app, consider which patterns apply:

| Pattern | When to use | Reference app |
|---------|-------------|---------------|
| Web UI | User needs visual interface | Most apps have optional static/ folder |
| No-GUI (antenna trigger) | Simple apps, kiosk mode | `reachy_mini_simon` |
| Control loop | Real-time reactivity needed | `reachy_mini_conversation_app/moves.py` |
| Head as controller | Games, recording | `fire_nation_attacked`, `marionette` |
| LLM integration | AI-powered behavior | `reachy_mini_conversation_app` |

---

## Creating a Beautiful Landing Page (index.html)

The root `index.html` is the landing page shown on Hugging Face Spaces. A well-designed landing page makes your app look professional and helps users understand what it does.

**Reference template:** Use the [Marionette app](https://huggingface.co/spaces/RemiFabre/marionette) as a template for the structure and styling.

### Structure

A good landing page has three sections:

1. **Hero Section** - Video/image + title + description
2. **Technical Section** - "How it works" steps + features
3. **Footer** - Resources, links, social media

### Key Elements

```
index.html
├── Hero Section
│   ├── Demo video (autoplay, loop, muted)
│   ├── App emoji + title
│   ├── Tags (categories)
│   └── Short description (1-2 sentences)
│
├── Technical Section
│   ├── "How it works" (numbered steps)
│   └── "Features" or additional info
│
└── Footer
    ├── Resources (docs, troubleshooting)
    ├── Reachy Mini Apps links
    └── Social media icons
```

### Assets

Put demo videos/images in `<app_name>/assets/`:

```
my_app/
├── index.html              # References my_app/assets/demo.mp4
├── my_app/
│   ├── assets/
│   │   └── demo.mp4        # Demo video
│   ├── main.py
│   └── static/             # Web UI (if any)
```

Video tag example:
```html
<video autoplay loop muted playsinline>
  <source src="my_app/assets/demo.mp4" type="video/mp4" />
</video>
```

### Styling

Use the Marionette CSS as a starting point:
- Inter font from Google Fonts
- CSS variables for colors (`--primary: #FF9900` for Pollen orange)
- Responsive grid layout (`grid-template-columns: 1fr 1fr` on desktop)
- Numbered steps with orange circles
- Footer with social media SVG icons

### Quick Checklist

- [ ] Hero with video/image showing the app in action
- [ ] Clear title and short tagline
- [ ] Tags describing the app's purpose
- [ ] "How it works" numbered steps (4 steps max)
- [ ] Footer with standard Pollen/Reachy links
- [ ] Responsive design (works on mobile)
- [ ] All CSS inline in `<style>` tag (no external stylesheet needed)
