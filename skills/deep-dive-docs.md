# Skill: Deep Dive into Documentation

## When to Use

- Need detailed understanding of a specific feature
- Looking for edge cases or advanced usage
- User asks "how does X work internally"
- Standard approaches aren't working

---

## SDK Documentation Structure

Located in `docs/source/` (in this repository).

### Start Here (Recommended Order)

1. **`docs/source/SDK/quickstart.md`** - Basic setup and first moves
2. **`docs/source/SDK/core-concept.md`** - Architecture, coordinate systems, safety limits
3. **`docs/source/SDK/python-sdk.md`** - Python API overview (movement, sensors, media)

### Detailed References

| Topic | File |
|-------|------|
| Installation | `docs/source/SDK/installation.md` |
| AI/LLM integration | `docs/source/SDK/integration.md` |
| Media architecture | `docs/source/SDK/media-architecture.md` |
| GStreamer setup | `docs/source/SDK/gstreamer-installation.md` |
| Troubleshooting | `docs/source/troubleshooting.md` |

### API Reference (auto-generated from docstrings)

| Module | File |
|--------|------|
| ReachyMini class | `docs/source/API/reachymini.mdx` |
| Motion module | `docs/source/API/motion.mdx` |
| Media module | `docs/source/API/media.mdx` |
| Utils | `docs/source/API/utils.mdx` |

### Platform-Specific Guides

| Platform | Getting Started | Hardware |
|----------|-----------------|----------|
| Wireless | `docs/source/platforms/reachy_mini/get_started.md` | `docs/source/platforms/reachy_mini/hardware.md` |
| Lite | `docs/source/platforms/reachy_mini_lite/get_started.md` | `docs/source/platforms/reachy_mini_lite/hardware.md` |
| Simulation | `docs/source/platforms/simulation/get_started.md` | N/A |

---

## Source Code as Documentation

When docstrings aren't enough, read the source:

### Key Source Files

| Purpose | Path |
|---------|------|
| **All SDK methods** | `src/reachy_mini/reachy_mini.py` |
| **App base class** | `src/reachy_mini/apps/app.py` |
| **Utils (create_head_pose, etc.)** | `src/reachy_mini/utils/` |
| **Motion interpolation** | `src/reachy_mini/motion/` |
| **REST API routers** | `src/reachy_mini/daemon/app/routers/` |

### Reading reachy_mini.py

This file contains the main `ReachyMini` class. When you need to understand:
- What methods exist
- What parameters they accept
- What they return

**Tip:** Skim the file focusing on docstrings. Don't read every line - look for the method you need.

```python
# Example: Find all public methods
# Look for 'def ' lines that don't start with '_'
```

---

## Example Apps as Documentation

Often the best documentation is working code. Local examples in this repo:

| Pattern | File |
|---------|------|
| Basic motion + antennas | `examples/minimal_demo.py` |
| Look-at with camera | `examples/look_at_image.py` |
| IMU usage | `examples/imu_example.py` |
| Recorded moves playback | `examples/recorded_moves_example.py` |
| Compliant mode | `examples/reachy_compliant_demo.py` |

External reference apps (run `skills/setup-environment.md` to clone, or browse online):

| Pattern | App | Source |
|---------|-----|--------|
| Control loops, LLM tools | reachy_mini_conversation_app | [GitHub](https://github.com/pollen-robotics/reachy_mini_conversation_app) |
| Safe torque, recording | marionette | [HF Space](https://huggingface.co/spaces/RemiFabre/marionette) |
| Head as controller | fire_nation_attacked | [HF Space](https://huggingface.co/spaces/RemiFabre/fire_nation_attacked) |
| Symbolic motion | reachy_mini_dances_library | [GitHub](https://github.com/pollen-robotics/reachy_mini_dances_library) |
| Antenna interaction | reachy_mini_radio | [HF Space](https://huggingface.co/spaces/pollen-robotics/reachy_mini_radio) |
| No-GUI pattern | reachy_mini_simon | [HF Space](https://huggingface.co/spaces/apirrone/reachy_mini_simon) |

---

## When Docs Are Insufficient

If you find something unclear or missing:

1. **Check source code** - The implementation is the ultimate truth
2. **Document your finding** - Add to `~/reachy_mini_resources/insights_for_reachy_mini_maintainers.md
3. **Tell the user** - Encourage them to submit a PR or issue on GitHub

---

## Never Invent Functions

Before using any SDK function:
1. Verify it exists in `reachy_mini.py`
2. Check the signature and return type
3. Read the docstring for usage notes

Don't guess or assume - check the source.
