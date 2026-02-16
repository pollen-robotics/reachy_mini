# Head Position GUI

This example provides a graphical user interface (GUI) using Tkinter to interactively control Reachy Mini's head position and orientation. You can adjust the head's roll, pitch, yaw angles, and X, Y, Z position using sliders, as well as control the body yaw and see animated antenna movements.

**Features:**
- Real-time control of head orientation (roll, pitch, yaw)
- Real-time control of head position (X, Y, Z)
- Body yaw control
- Animated antenna movements (sine wave oscillation)
- Interactive sliders with live updates at 50Hz

**Usage:**
```bash
python mini_head_position_gui.py
```

A GUI window will open with sliders for all controllable parameters. Adjust the sliders to see the robot respond in real-time.

<literalinclude>
{"path": "../../../examples/mini_head_position_gui.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
