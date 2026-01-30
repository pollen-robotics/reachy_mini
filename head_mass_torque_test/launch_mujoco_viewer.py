"""
Launch MuJoCo viewer with interactive controls
You can control actuators by clicking and using the GUI
"""

import mujoco
import mujoco.viewer
from pathlib import Path
import numpy as np

# Path to the scene XML
XML_PATH = (
    Path(__file__).parent.parent
    / "src/reachy_mini/descriptions/reachy_mini/mjcf/scenes/empty.xml"
)

print("=" * 80)
print("LAUNCHING MUJOCO INTERACTIVE VIEWER")
print("=" * 80)
print()
print(f"Loading: {XML_PATH}")
print()

# Load model
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

print("=" * 80)
print("VIEWER CONTROLS:")
print("=" * 80)
print()
print("CAMERA:")
print("  - Left mouse button + drag:  Rotate view")
print("  - Right mouse button + drag: Move view")
print("  - Scroll:                    Zoom in/out")
print()
print("SIMULATION:")
print("  - SPACE:     Pause/unpause simulation")
print("  - BACKSPACE: Reset simulation to initial state")
print("  - RIGHT ARROW: Step forward one step (when paused)")
print()
print("ACTUATORS (GUI):")
print("  - Double-click on robot parts to select")
print("  - Use sliders in right panel to control actuators")
print("  - Or: Press TAB to open control panel with sliders")
print()
print("APPLY FORCES:")
print("  - CTRL + Right mouse button: Apply force at click point")
print("  - CTRL + Left mouse button:  Apply torque")
print()
print("OTHER:")
print("  - Press H:   Show help overlay")
print("  - Press F1:  Show rendering options")
print("  - ESC:       Close viewer")
print()
print("=" * 80)
print()
print("Opening viewer... (close window or press ESC to exit)")
print()

# Launch interactive viewer
mujoco.viewer.launch(model, data)

print("\n✓ Viewer closed")
