# Rerun Viewer

This example shows how to use the Rerun utility to log and visualize Reachy Mini's state in real-time. The robot will be in compliant mode with gravity compensation, making it easy to move around while visualizing its configuration.

Requirements:
- Install with: `pip install reachy-mini[rerun,placo_kinematics]`
- Start the daemon with: `reachy-mini-daemon --kinematics-engine Placo`

<literalinclude>
{"path": "../../../examples/rerun_viewer.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
