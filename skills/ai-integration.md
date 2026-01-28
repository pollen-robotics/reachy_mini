# Skill: AI Agent Integration

## When to Use

- Building an app that uses LLMs to control the robot
- Integrating speech recognition or voice interaction
- Creating an autonomous conversational robot

---

## Reference Implementation

**Primary reference:** `~/reachy_mini_resources/reachy_mini_conversation_app/`

> If this folder doesn't exist, run `skills/setup-environment.md` to clone reference apps.

This app demonstrates how to turn Reachy Mini into an intelligent, autonomous robot using LLMs.

---

## Architecture Overview

The LLM doesn't control motors directly. Instead:

```
LLM decides → Tool call (e.g., "dance") → Queue move → Control loop executes smoothly
```

This separation:
- Keeps the control loop clean
- Ensures smooth motion regardless of LLM latency
- Prevents race conditions

---

## Key Components

### 1. LLM Tools

The conversation app exposes these tools to the LLM:

| Tool | What it does |
|------|--------------|
| `move_head` | Queue head pose changes |
| `dance` | Play dances from library |
| `play_emotion` | Play recorded emotions |
| `camera` | Capture and analyze images |
| `head_tracking` | Enable/disable face tracking |

### 2. Custom Profiles

Different personalities with different instructions and enabled tools. Profiles let you create variations (helpful assistant, playful character, etc.) without code changes.

### 3. Vision Processing

Two options for image analysis (e.g., "what do you see?"):
- **Cloud (e.g. gpt_realtime)**: Higher quality, requires API key
- **Local (SmolVLM2)**: Runs on device, no external calls, lower quality

---

## Implementing LLM Tools

Basic pattern for a tool that queues robot actions:

```python
from queue import Queue

class RobotController:
    def __init__(self):
        self.move_queue = Queue()

    # This is called by the LLM
    def tool_move_head(self, yaw: float, pitch: float, duration: float = 1.0):
        """Move the robot's head to a position."""
        self.move_queue.put({
            "type": "goto",
            "yaw": yaw,
            "pitch": pitch,
            "duration": duration
        })
        return "Head movement queued"

    # This runs in the control loop
    def process_queue(self, mini):
        while not self.move_queue.empty():
            move = self.move_queue.get()
            if move["type"] == "goto":
                pose = create_head_pose(yaw=move["yaw"], pitch=move["pitch"], degrees=True)
                mini.goto_target(head=pose, duration=move["duration"])
```

---

## Real-Time API Integration

The conversation app uses OpenAI's Realtime API (compatible with Grok and others):
- Low-latency audio streaming
- Function calling for robot control
- Handles interruptions gracefully

Key files to study:
- `src/reachy_mini_conversation_app/realtime/` - API integration
- `src/reachy_mini_conversation_app/tools/` - Tool definitions
- `src/reachy_mini_conversation_app/profiles/` - Personality profiles

---

## Tips for LLM Integration

1. **Keep tool responses short** - LLMs work better with concise feedback
2. **Use queues, not direct calls** - Prevents blocking and race conditions
3. **Provide context in system prompt** - Tell the LLM about the robot's capabilities
4. **Handle failures gracefully** - Robot might be busy, tool might fail
5. **Test with simple commands first** - "nod", "shake head", before complex behaviors
