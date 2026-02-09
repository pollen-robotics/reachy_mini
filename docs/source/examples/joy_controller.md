# Joystick Controller

This example demonstrates how to control Reachy Mini's head yaw angle using a joystick (PS4 or Xbox controller). The left joystick controls the head's left-right rotation, providing intuitive real-time control of the robot.

**Controls:**
- **LEFT JOYSTICK (Left/Right)**: Control head yaw angle
- **CIRCLE / B BUTTON**: Quit the application safely
- **CTRL-C**: Quit the application

**Requirements:**
- Install pygame: `pip install pygame`
- Connect a PS4 or Xbox controller to your computer

**Controller mappings:**
- PS4: Button 1 = Circle (O), Axis 0 = Left Stick Horizontal
- Xbox: Button 1 = B, Axis 0 = Left Stick Horizontal

<literalinclude>
{"path": "../../../examples/joy_controller.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
