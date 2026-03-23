# Custom Media Manager

This example demonstrates how to deactivate the built-in media manager and access the camera and microphone directly using OpenCV and sounddevice.

**Why?** The daemon normally owns the camera and audio hardware. If you need raw access (e.g. custom OpenCV pipelines, sounddevice recording, or a third-party vision library), you must first tell the daemon to release the hardware. See [Media Architecture - Disabling Media](../SDK/media-architecture.md#disabling-media--direct-hardware-access) for details.

**How it works:**
1. Connects with `media_backend="no_media"` — this automatically tells the daemon to release camera and audio hardware
2. Uses OpenCV to capture a frame directly from the camera
3. Uses sounddevice to record audio from the microphone
4. On exit, the daemon automatically re-acquires the hardware

> **💡 Tip:** Robot control (head, antennas, body) keeps working normally while media is released. Only camera and audio are affected.

**Requirements:**
```bash
pip install opencv-python sounddevice soundfile
```

**Usage:**
```bash
python custom_media_manager.py
```

<literalinclude>
{"path": "../../../examples/custom_media_manager.py",
"language": "python",
"start-after": "START doc_example",
"end-before": "END doc_example"
}
</literalinclude>
