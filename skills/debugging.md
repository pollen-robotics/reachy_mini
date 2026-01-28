# Skill: Debugging

## When to Use

- App crashes or doesn't behave as expected
- Robot doesn't respond to commands
- Connection issues
- Before complex debugging, verify basics work

---

## First: Verify Basic Connectivity

Before debugging complex issues, always check that basic examples work.

### Step 1: Check Daemon Status

**For Lite (USB connection):**
```bash
# Start daemon if not running
reachy-mini-daemon

# Or in simulation mode
reachy-mini-daemon --sim
```
- Access dashboard to verify robot is awake


**For Wireless:**
- Ensure robot is powered on
- Check WiFi connection
- Access dashboard to verify robot is awake

### Step 2: Test Basic Motion

Run the minimal demo to verify connectivity and motion:

```bash
python examples/minimal_demo.py
```

This connects to the robot and makes the head nod with antenna movement. If this fails, the problem is connectivity, not your app.

---

## Common Issues

### "Connection refused" or timeout

1. Is the daemon running?
2. Is another app already connected?
3. For Wireless: Is robot on the same network?

**Fix:** Kill other connections, restart daemon.

### Robot doesn't move

1. Are motors enabled?
2. Is another goto_target still running?
3. Are you sending valid poses?

**Check:**
```python
print(mini.get_motor_status())  # Check if motors enabled
```

**Fix:**
```python
mini.enable_motors()
```

### Jerky motion

1. Are you calling set_target from multiple places?
2. Is your control loop too slow?
3. Are you mixing goto_target and set_target?

**Fix:** Single control point, maintain 30Hz+, use one method at a time.

### Import errors

1. Is `reachy-mini` installed in your environment?
2. Are you in the right virtual environment?

### "Motors in different states" issues

See `safe-torque.md` for the workaround when enabling/disabling motor subsets.

---

## Simulation vs Physical

| Issue | Simulation | Physical |
|-------|------------|----------|
| No motion | Check daemon is in `--sim` mode | Check motors enabled, power on |
| Camera fails | Expected - no camera in sim | Check USB connection |
| Audio fails | May not work in sim | Check microphone permissions |

---

## Reading Logs

### Daemon logs (Lite)
```bash
# If running in terminal, logs appear there
# Otherwise run with verbose flag
reachy-mini-daemon --verbose
```

### Daemon logs (Wireless)
SSH into the robot and check system logs or even restart the daemon:
```bash
ssh pollen@reachy-mini.local  # password: root

# View daemon logs
journalctl -u reachy-mini-daemon.service

# Restart the daemon
systemctl restart reachy-mini-daemon.service # password: root

```

### App logs
Add logging to your app:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Current pose: {mini.get_current_head_pose()}")
```

---

## Debugging Checklist

Before asking for help or doing complex debugging:

- [ ] Daemon is running
- [ ] No other apps connected to robot
- [ ] Basic connection test passes
- [ ] Basic motion test passes
- [ ] Motors are enabled
- [ ] Correct virtual environment activated
- [ ] `reachy-mini` package installed

---

## Debugging by Dichotomy (Git Bisect)

When your app breaks and you're not sure which change caused it, use git history to narrow down the problem efficiently.

### The Strategy

1. **First, verify basics work** - run `python examples/minimal_demo.py`
2. **If basics work**, the problem is in your app code, not connectivity
3. **Find a known-good commit** - go back in git history until your app works
4. **Binary search** - narrow down between "works" and "broken" commits

### Typical Workflow

```bash
# 1. Check current (broken) state
git log --oneline -10

# 2. Go back to a commit you know worked
git checkout <older-commit-hash>
python my_app/main.py  # Test it

# 3. If it works, check the diff to the broken version
git diff <this-commit> <broken-commit>

# 4. Decide: is the diff small enough to debug directly?
#    - YES → Read the diff and find the bug
#    - NO  → Binary search further (pick a commit in the middle)
```

The key insight: you don't always need to bisect all the way down. At each step, check `git diff` and judge if the changes are small enough to spot the bug directly.

### When to Use This

- App was working before, now it's broken
- You've made many changes and lost track
- Error messages aren't helpful
- Faster than reading all code changes manually

### Tips

- **Commit often** - small commits make bisect more useful
- **Write descriptive commit messages** - helps identify suspicious commits
- **Don't mix features in one commit** - isolates problems better
- Once you find the breaking diff, the bug is usually obvious

---

## Full Troubleshooting Reference

For comprehensive troubleshooting (hardware issues, motor errors, audio problems, etc.), see:
**`docs/source/troubleshooting.md`**

This covers:
- Motor diagnosis and "Overload Error" fixes
- Connection issues (WiFi, USB, dashboard)
- Audio/camera problems
- Common error messages and solutions

---

## Getting Help

If basics work but your app doesn't:
1. Isolate the problem (which part fails?)
2. Use git bisect to find when it broke
3. Check reference apps for similar functionality
4. Add logging around the failing code
5. Ask on Discord: https://discord.gg/Y7FgMqHsub
