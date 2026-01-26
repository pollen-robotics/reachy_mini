# Skill: Deep Dive into Documentation

## When to Use

- Need detailed understanding of a specific feature
- Looking for edge cases or advanced usage
- User asks "how does X work internally"
- Standard approaches aren't working

---

## SDK Documentation Structure

Located in `~/reachy_mini_resources/reachy_mini/docs/`

### Start Here (Recommended Order)

1. **`docs/SDK/quickstart.md`** - Basic setup and first moves
2. **`docs/SDK/core-concept.md`** - Motion control concepts, safety limits
3. **`docs/SDK/python-sdk.md`** - Python API overview

### Detailed References

| Topic | File |
|-------|------|
| App development | `docs/SDK/app.md` |
| Motion control details | `docs/SDK/core-concept.md` |
| REST API | `docs/old_doc/rest-api.md` |

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

Often the best documentation is working code:

| Pattern | Reference App | Key File |
|---------|---------------|----------|
| Control loops | `reachy_mini_conversation_app` | `src/.../moves.py` |
| Safe torque | `marionette` | `marionette/main.py` |
| Head as input | `fire_nation_attacked` | `fire_nation_attacked/main.py` |
| Symbolic motion | `reachy_mini_dances_library` | `src/.../rhythmic_motion.py` |
| Antenna interaction | `reachy_mini_radio` | Main file |
| No-GUI pattern | `reachy_mini_simon` | Main file |

---

## When Docs Are Insufficient

If you find something unclear or missing:

1. **Check source code** - The implementation is the ultimate truth
2. **Try it** - Run a quick test to see what happens
3. **Document your finding** - Add to `~/reachy_mini_resources/insights_for_reachy_mini_maintainers.md`
4. **Tell the user** - Encourage them to submit a PR or issue

---

## Never Invent Functions

Before using any SDK function:
1. Verify it exists in `reachy_mini.py`
2. Check the signature and return type
3. Read the docstring for usage notes

Don't guess or assume - check the source.
