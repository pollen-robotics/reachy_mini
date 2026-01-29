# Skill: Setup Environment

## When to Use

- First conversation with a new user
- `agents.local.md` does not exist
- Reference apps folder does not exist
- User explicitly asks to set up or reset their environment

## Quick Check

```bash
# If BOTH exist, setup is likely done - read agents.local.md to confirm
ls ~/reachy_mini_resources/ 2>/dev/null && ls agents.local.md 2>/dev/null
```

If `agents.local.md` exists, read it. Look for "Setup complete" or similar confirmation.

---

## Procedure

### Step 1: Explain What You're About to Do

Tell the user:

> "To help you develop Reachy Mini apps, I'll set up a folder with example apps and reference code. This gives me access to proven patterns.
>
> The default location is `~/reachy_mini_resources/`. Is this OK, or would you prefer a different location?
> (Another option: if you already have reachy_mini cloned, I can put resources inside it)"

**Important:** Always use absolute paths for robustness. The location should be permanent to avoid re-downloading on each session. Store the chosen path in `agents.local.md`.

### Step 2: Check for Missing Tools

Before proceeding, verify these tools are available:

| Tool | Check command | Why needed |
|------|---------------|------------|
| git | `git --version` | Clone repositories |
| python | `python --version` or `python3 --version` | Run apps |
| pip or uv | `pip --version` or `uv --version` | Install packages |

If any are missing:
1. List ALL missing tools
2. Explain why each is needed (briefly)
3. Ask permission once before installing
4. Adapt installation commands to user's OS

### Step 3: Create Virtual Environment

Ask user preference, or use defaults:

**With uv (preferred, faster):**
```bash
cd ~/reachy_mini_resources
uv venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
uv pip install reachy-mini
```

**With standard venv (fallback):**
```bash
cd ~/reachy_mini_resources
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows
pip install reachy-mini
```

**Installing uv** (if user agrees):
- Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

### Step 4: Clone Resources

```bash
mkdir -p ~/reachy_mini_resources
cd ~/reachy_mini_resources

# Main SDK (essential)
git clone https://github.com/pollen-robotics/reachy_mini

# Example apps (each demonstrates different patterns)
git clone https://github.com/pollen-robotics/reachy_mini_conversation_app
git clone https://github.com/pollen-robotics/reachy_mini_dances_library
git clone https://huggingface.co/spaces/RemiFabre/marionette
git clone https://huggingface.co/spaces/RemiFabre/fire_nation_attacked
git clone https://huggingface.co/spaces/apirrone/spaceship_game
git clone https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio
git clone https://huggingface.co/spaces/apirrone/reachy_mini_simon
git clone https://huggingface.co/spaces/pollen-robotics/hand_tracker_v2
```

### Step 5: Ask About Robot Hardware

> "What type of Reachy Mini do you have?
> - **Lite**: Connected via USB to your computer
> - **Wireless**: Has onboard computer (CM4), connects via WiFi
> - **Neither yet**: You're just exploring"

### Step 6: Create agents.local.md

After setup completes, create `agents.local.md` in the working directory:

```markdown
# Reachy Mini Local Configuration

## Setup Status
Setup complete: YES

## User Environment
- Robot type: [Lite / Wireless / None yet]
- OS: [Linux / macOS / Windows]
- Shell: [bash / zsh / fish / PowerShell]
- Python env tool: [uv / venv]
- Resources path: ~/reachy_mini_resources/

## Notes for Future Sessions
[Agent can add useful notes here for continuity]
```

### Step 7: Verify Setup Works

Run a quick test to confirm the SDK is installed:

```bash
python -c "from reachy_mini import ReachyMini; print('SDK installed successfully')"
```

If this fails, debug before marking setup complete.

---

## Key Reference Paths (After Setup)

**In this repository (always available):**

| Purpose | Path |
|---------|------|
| SDK source | `src/reachy_mini/reachy_mini.py` |
| App base class | `src/reachy_mini/apps/app.py` |
| Documentation | `docs/source/` |
| Examples | `examples/` |

**In ~/reachy_mini_resources/ (after setup):**

| Purpose | Path |
|---------|------|
| Conversation app | `~/reachy_mini_resources/reachy_mini_conversation_app/` |
| Marionette | `~/reachy_mini_resources/marionette/` |
| Other example apps | See `skills/deep-dive-docs.md` for full list |
