# Head Tracking

This example enables daemon-side head tracking: Reachy Mini turns its head to follow the closest face, aiming at the nose. Detection runs inside the daemon (YuNet on ONNX Runtime), so the script only toggles tracking and polls the latest face target.

Run with:
```bash
python head_tracking.py
```

`start_head_tracking(weight=...)` accepts a blend factor: `1.0` lets tracking own the head, `0.0` pauses detection (freeing the head and CPU) without stopping the tracker — useful to hand the head back to an application between conversation turns.

<literalinclude>
{"path": "../../../examples/head_tracking.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
