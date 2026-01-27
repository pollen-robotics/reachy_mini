# Skill: Create App

## When to Use

- User wants to create a new Reachy Mini application
- User asks how to structure an app
- User wants to publish an app to Hugging Face

## Quick Check

If an app folder already exists with `README.md` containing `reachy_mini_python_app` tag, the app structure is probably already set up. In doubt, double check.

---

## Procedure

### Step 1: Use the App Assistant

**Never manually create app folders.** Always use the assistant to get proper structure, metadata, and git setup:

```bash
reachy-mini-app-assistant create my_app_name /path/to/create --publish
```

- `--publish` creates a Git repo on Hugging Face immediately (public by default)
- The assistant handles all boilerplate, metadata tags, and proper structure

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
