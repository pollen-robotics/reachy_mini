# Compliant Mode Demo

This demo turns the Reachy Mini into compliant mode and compensates for the gravity of the robot platform to prevent it from falling down.

You can gently push the robot and it will follow your movements. When you stop pushing it, it will stay in place. This is useful for applications like human-robot interaction, where you want the robot to be compliant and follow the user's movements.

Note: This demo currently only works with Placo as the kinematics engine.

<literalinclude>
{"path": "../../../examples/reachy_compliant_demo.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
