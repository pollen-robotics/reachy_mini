# Skill: Testing Apps

## When to Use

- Before considering an app "done"
- After making significant changes
- When debugging issues

---

## Know the User's Setup First

If not recorded in `agents.local.md`, ask:

> "What type of Reachy Mini do you have: **Lite** or **Wireless**?"

This affects how testing works.

---

## Step 1: Prepare the Robot

**The daemon must be running before your app can connect.**

Ask the user how they want to test:

| Test mode | Lite | Wireless |
|-----------|------|----------|
| **Simulation** | Start daemon: `reachy-mini-daemon --sim` | Start daemon: `reachy-mini-daemon --sim` |
| **Physical robot** | Start daemon: `reachy-mini-daemon` | Turn on robot (daemon runs automatically) |

**Important:**
- Make sure no other daemon is already running
- For Wireless, the daemon starts automatically on boot - just power on and wait for WiFi

Wait for user to confirm they're ready.

---

## Step 2: Run the App

Run it the same way the dashboard will launch it:

```bash
cd ~/reachy_mini_apps/my_app
python my_app/main.py
```

Or if the app has a different entry point, use that.

---

## Step 3: Check for Issues

### Immediate crashes

- Import errors?
- Missing dependencies?
- Syntax errors?

### Runtime issues

- Does the control loop run without exceptions?
- Are there type errors or None values?
- Does motion look correct?

### Behavioral issues

- Does the app do what the user requested?
- Are there edge cases not handled?
- Does it exit cleanly?

---

## Testing Checklist

- [ ] App starts without crashing
- [ ] No import errors
- [ ] No missing dependencies
- [ ] Control loop runs (if applicable)
- [ ] Basic functionality works
- [ ] App exits cleanly (Ctrl+C or natural end)
- [ ] No error messages in console

---

## Simulation Limitations

Simulation can't test:
- Camera-based tracking
- Audio input/output
- Physical antenna button presses
- Real motor behavior

For these features, tell the user they'll need to test on physical hardware.

---

## Quick Smoke Test

If you just want to verify the app doesn't crash immediately:

```bash
# Run with timeout (Linux/macOS)
timeout 5 python my_app/main.py

# Or on Windows PowerShell
# Start-Process python -ArgumentList "my_app/main.py" -Wait -Timeout 5
```

---

## Fix and Repeat

If issues are found:
1. Fix the issue
2. Run the app again
3. Repeat until it runs cleanly

Only mark the app as done when testing passes.
