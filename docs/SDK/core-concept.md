# Core Concepts & Architecture

Understanding how Reachy Mini works under the hood will help you build robust applications and debug issues.

## Software Architecture

Reachy Mini uses a **Client-Server** architecture:

1.  **The Daemon (Server):** 
    * Runs on the computer connected to the robot (or the simulation).
    * Handles hardware I/O (USB/Serial), safety checks, and sensor reading.
    * Exposes a REST API (`localhost:8000`) and WebSocket.
    
3.  **The SDK (Client):**
    * Your Python code (`reachy_mini` package).
    * Connects to the Daemon over the network.
    * *Advantage:* You can run your AI code on a powerful server while the Daemon runs on a Raspberry Pi connected to the robot.

## Coordinate Systems

When moving the robot, you will work with two main reference frames:

### 1. Head Frame
Located at the base of the head. Used for `goto_target` commands.

[![Reachy Mini Head Frame](/docs/assets/head_frame.png)]()

### 2. World Frame
Fixed relative to the robot's base. Used for `look_at_world` commands.

[![Reachy Mini World Frame](/docs/assets/world_frame.png)]()

## Safety Limits ⚠️

Reachy Mini has physical and software limits to prevent self-collision and damage. The SDK will automatically clamp values to the closest valid position.

| Joint / Axis | Limit Range |
| :--- | :--- |
| **Head Pitch/Roll** | [-40°, +40°] |
| **Head Yaw** | [-180°, +180°] |
| **Body Yaw** | [-180°, +180°] |
| **Yaw Delta** | Max 65° difference between Head and Body Yaw |

## Motor Modes

You can change how the motors behave:
* **`reachy_mini.enable_motors()`**: Stiff. Holds position.
* **`reachy_mini.disable_motors()`**: Limp. No power.
* **`reachy_mini.enable_gravity_compensation()`**: "Soft" mode. You can move the head with your hands and it will stay in place where you left it. (Only works with Placo Kinematics backend)


## Next Steps
* **[Quickstart Guide](quickstart.md)**: Run your first behavior on Reachy Mini
* **[Python SDK](python-sdk.md)**: Learn to move, see, speak, and hear.
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
