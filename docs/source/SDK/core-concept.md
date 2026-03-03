# Core Concepts & Architecture

Understanding how Reachy Mini works under the hood will help you build robust applications and debug issues.

## Software Architecture

Reachy Mini uses a **Client-Server** architecture:

1.  **The Daemon (Server):** 
    * Runs on the computer connected to the robot (or the simulation).
    * Handles hardware I/O (USB/Serial), safety checks, and sensor reading.
    * Exposes a REST API (`localhost:8000`) and WebSocket.
    
2.  **The SDK (Client):**
    * Your Python code (`reachy_mini` package).
    * Connects to the Daemon over the network.
    * *Advantage:* You can run your AI code on a powerful server while the Daemon runs on a Raspberry Pi connected to the robot.

## Coordinate Systems

When moving the robot, you will work with two main reference frames:

### 1. Head Frame
Located at the base of the head. Used for `goto_target` and `set_target` commands.

[![Reachy Mini Head Frame](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/head_frame.png)]()

### 2. World Frame
Fixed relative to the robot's base. Used for `look_at_world` commands.

[![Reachy Mini World Frame](https://github.com/pollen-robotics/reachy_mini/raw/develop/docs/assets/world_frame.png)]()

## Safety Limits ⚠️

Reachy Mini has physical and software limits to prevent self-collision and damage. The SDK will automatically clamp values to the closest valid position.

| Joint / Axis | Limit Range |
| :--- | :--- |
| **Head cone (tilt)** | Max 35° from vertical (pitch/roll coupled) |
| **Head Yaw** | [-179°, +179°] |
| **Body Yaw** | [-160°, +160°] |
| **Yaw Delta** | Max 55° difference between Head and Body Yaw |

You can read these limits in code via `mini.limits` and check reachability with `mini.is_reachable(head_pose, body_yaw)`.

## Motor Modes

You can change how the motors behave:
* **`mini.enable_motors()`**: Stiff. Holds position.
* **`mini.disable_motors()`**: Limp. No power.
* **`mini.enable_gravity_compensation()`**: "Soft" mode. You can move the head by hand, and it will stay where you leave it. (Only works with the Placo kinematics backend.)


## Next Steps
* **[Quickstart Guide](quickstart.md)**: Run your first behavior on Reachy Mini
* **[Python SDK](python-sdk.md)**: Learn to move, see, speak, and hear.
* **[AI Integrations](integration.md)**: Connect LLMs, build Apps, and publish to Hugging Face.
