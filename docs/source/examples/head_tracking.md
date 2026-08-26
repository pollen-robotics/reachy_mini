# Head Tracking

This example enables daemon-side head tracking: Reachy Mini turns its head to follow the closest face, aiming at the nose. Detection runs inside the daemon (YuNet on ONNX Runtime), so the script only toggles tracking and polls the latest face target.

Run with:
```bash
python head_tracking.py
```

`start_head_tracking(weight=...)` accepts a blend factor: `1.0` lets tracking own the head, `0.0` pauses detection (freeing the head and CPU) without stopping the tracker — useful to hand the head back to an application between conversation turns. `mode=` chooses when the robot re-aims: `"continuous"` (default), `"periodic"` (a glance every `glance_interval_s=(min, max)` seconds, holding in between), or `"speaking"` (follows the face only while the robot is speaking, detected daemon-side from audio playback and speech offsets, with `speaking_hold_s` of hold after speech ends).

<literalinclude>
{"path": "../../../examples/head_tracking.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
