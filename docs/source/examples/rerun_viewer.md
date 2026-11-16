# Rerun Viewer

This example shows how to use the Rerun utility to log and visualize Reachy Mini's state in real-time. The robot will be in compliant mode with gravity compensation, making it easy to move around while visualizing its configuration.

Requirements:
- Install with: `pip install reachy-mini[rerun,placo_kinematics]`
- For the compliant head, start the daemon with: `reachy-mini-daemon --kinematics-engine Placo`. Without it (e.g. on a Wireless), the viewer still shows the model and the camera, but the head is not compliant.

<literalinclude>
{"path": "../../../examples/rerun_viewer.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
