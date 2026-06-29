# Take Picture

This example demonstrates how to capture a single frame from Reachy Mini's camera and save it as an image file.

Run with:
```bash
python take_picture.py --backend [default|local|webrtc]
```

The captured image will be saved as `reachy_mini_picture.jpg` in the current directory.

<literalinclude>
{"path": "../../../examples/take_picture.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
