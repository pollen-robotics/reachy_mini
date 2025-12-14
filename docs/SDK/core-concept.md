# Core Concepts & Architecture

Understanding how Reachy Mini works under the hood will help you build robust applications and debug issues.

## Software Architecture

Reachy Mini uses a **Client-Server** architecture:

1.  **The Daemon (Server):** * Runs on the computer connected to the robot (or the simulation).
    * Handles hardware I/O (USB/Serial), safety checks, and sensor reading.
    * Exposes a REST API (`localhost:8000`) and WebSocket.
    
2.  **The SDK (Client):**
    * Your Python code (`reachy_mini` package).
    * Connects to the Daemon over the network.
    * *Advantage:* You can run your AI code on a powerful server while the Daemon runs on a Raspberry Pi connected to the robot.

## Coordinate Systems

When moving the robot, you will work with two main reference frames:

### 1. Head Frame
Located at the base of the head. Used for `goto_target` commands.
* **X:** Forward
* **Y:** Left
* **Z:** Up

### 2. World Frame
Fixed relative to the robot's base. Used for `look_at_world` commands.

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
* **`enable_motors()`**: Stiff. Holds position.
* **`disable_motors()`**: Limp. No power.
* **`make_motors_compliant()`**: "Soft" mode. Motors are on but yield to external force. Useful for teaching-by-demonstration (moving the robot by hand).
