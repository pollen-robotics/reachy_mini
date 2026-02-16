# Sound Direction of Arrival (DoA)

This example demonstrates how to use the microphone array to detect the Direction of Arrival (DoA) of speech. The robot uses the FastAPI endpoint to get DoA information, calculates the position of the sound source, transforms it into world coordinates, and automatically looks towards the speaker.

**How it works:**
1. Continuously polls the `/api/state/doa` endpoint to get speech direction
2. When speech is detected, calculates the 3D position of the sound source
3. Transforms the position from head coordinates to world coordinates
4. Commands the robot to look at the speaker using `look_at_world()`

**Features:**
- Automatic detection of robot IP (local or wireless)
- Threshold-based filtering to avoid excessive head movements
- Real-time transformation from head to world coordinates


<literalinclude>
{"path": "../../../examples/sound_doa.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
