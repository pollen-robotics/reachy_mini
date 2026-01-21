# Motion Hooks Plugin System

## Overview

The reachy_mini SDK now includes a plugin system that allows external packages to receive notifications when moves (emotions, dances, etc.) start and end. This enables community extensions like LED control, logging, or custom behaviors without modifying the SDK code.

## Architecture

The plugin system uses Python's standard entry points mechanism for automatic discovery. When reachy_mini initializes, it automatically loads all registered plugins.

### How It Works

1. **Plugin Discovery**: During initialization, reachy_mini searches for entry points in the `reachy_mini.motion.hooks` group
2. **Hook Loading**: Each discovered plugin is instantiated and registered
3. **Event Notification**: When moves are played, all registered hooks receive `on_move_start` and `on_move_end` callbacks
4. **Graceful Errors**: Plugin errors are logged but don't crash the main application

## Creating a Plugin

### Step 1: Implement the Hook Class

Create a class that implements the `MoveHook` protocol:

```python
# my_plugin/hook.py
import logging

logger = logging.getLogger(__name__)

class MyCustomHook:
    """Example plugin hook."""
    
    def __init__(self):
        # Initialize your plugin (hardware, connections, etc.)
        logger.info("MyCustomHook initialized")
    
    def on_move_start(self, move_name: str | None, dataset: str | None) -> None:
        """Called when a move starts playing.
        
        Args:
            move_name: Name/description of the move (e.g., "happy", "confused")
            dataset: HuggingFace dataset name (e.g., "pollen-robotics/reachy-mini-emotions-library")
        """
        if dataset and "emotions" in dataset.lower():
            logger.info(f"Emotion '{move_name}' starting")
            # Your custom logic here
    
    def on_move_end(self, move_name: str | None, dataset: str | None) -> None:
        """Called when a move finishes.
        
        Args:
            move_name: Name/description of the move
            dataset: HuggingFace dataset name
        """
        if dataset and "emotions" in dataset.lower():
            logger.info(f"Emotion '{move_name}' completed")
            # Your cleanup logic here
```

### Step 2: Register via Entry Point

In your plugin's `pyproject.toml`, register the hook:

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "my-reachy-plugin"
version = "0.1.0"
dependencies = [
    # Your dependencies
]

# THIS IS THE KEY PART: Register your plugin
[project.entry-points."reachy_mini.motion.hooks"]
my_plugin = "my_plugin.hook:MyCustomHook"

[tool.setuptools]
package-dir = { "" = "src" }
```

### Step 3: Install and Use

Users install your plugin alongside reachy_mini:

```bash
pip install reachy_mini my-reachy-plugin
```

That's it! The plugin is automatically discovered and loaded when reachy_mini initializes. No additional configuration needed.

## Example: LED Eye Color Plugin

Here's a complete example of a plugin that controls LED eye colors based on emotions:

### Project Structure

```
reachy_mini_led_eyes/
├── pyproject.toml
└── src/
    └── reachy_mini_led_eyes/
        ├── __init__.py
        ├── hook.py
        └── led_controller.py
```

### pyproject.toml

```toml
[project]
name = "reachy-mini-led-eyes"
version = "0.1.0"
description = "LED eye color control for Reachy Mini emotions"

[project.entry-points."reachy_mini.motion.hooks"]
led_eyes = "reachy_mini_led_eyes.hook:LEDEyeHook"
```

### led_controller.py

```python
"""Hardware control for LED eyes."""
import logging

logger = logging.getLogger(__name__)

EMOTION_COLORS = {
    "happy": (255, 220, 0),      # Warm yellow
    "sad": (0, 50, 200),          # Blue
    "angry": (255, 0, 0),         # Red
    "surprised": (255, 150, 0),   # Orange
    "confused": (150, 100, 255),  # Purple
    "neutral": (255, 255, 255),   # White
}

class LEDController:
    def __init__(self):
        # Initialize your LED hardware (GPIO, I2C, etc.)
        logger.info("LED controller initialized")
    
    def set_color(self, r: int, g: int, b: int) -> None:
        """Set LED color."""
        # Your hardware control code
        logger.debug(f"LED color: ({r}, {g}, {b})")
    
    def set_emotion(self, emotion_name: str | None) -> None:
        """Set LED color based on emotion."""
        if not emotion_name:
            return
        base = emotion_name.split("_")[0].lower()
        color = EMOTION_COLORS.get(base, EMOTION_COLORS["neutral"])
        self.set_color(*color)
```

### hook.py

```python
"""Plugin hook for LED eye control."""
import logging
from .led_controller import LEDController

logger = logging.getLogger(__name__)

class LEDEyeHook:
    def __init__(self):
        self.led = LEDController()
        logger.info("LED eye hook registered")
    
    def on_move_start(self, move_name: str | None, dataset: str | None) -> None:
        if dataset and "emotions" in dataset.lower():
            self.led.set_emotion(move_name)
    
    def on_move_end(self, move_name: str | None, dataset: str | None) -> None:
        if dataset and "emotions" in dataset.lower():
            self.led.set_color(255, 255, 255)  # Reset to white
```

## Testing Your Plugin

You can test your hook implementation locally:

```python
from reachy_mini.motion import MoveHook

# Verify your class implements the protocol
from my_plugin.hook import MyCustomHook
hook = MyCustomHook()
assert isinstance(hook, MoveHook)

# Test the methods
hook.on_move_start("happy", "pollen-robotics/reachy-mini-emotions-library")
hook.on_move_end("happy", "pollen-robotics/reachy-mini-emotions-library")
```

## API Reference

### MoveHook Protocol

All plugins must implement this protocol:

```python
from reachy_mini.motion import MoveHook

class MyHook:
    def on_move_start(self, move_name: str | None, dataset: str | None) -> None:
        """Called when a move starts playing."""
        pass
    
    def on_move_end(self, move_name: str | None, dataset: str | None) -> None:
        """Called when a move finishes playing."""
        pass
```

### Hook Parameters

- **move_name**: The name/description of the move being played (e.g., "happy", "confused")
  - Comes from `RecordedMove.description` or `Move.description` attribute
  - Can be `None` for moves without a description

- **dataset**: The HuggingFace dataset name the move comes from
  - For recorded moves: e.g., "pollen-robotics/reachy-mini-emotions-library"
  - For custom moves: Can be `None` or any string you set

### Entry Point Group

Plugins must register in the `reachy_mini.motion.hooks` entry point group:

```toml
[project.entry-points."reachy_mini.motion.hooks"]
my_plugin_name = "my_package.module:MyHookClass"
```

## Best Practices

1. **Defensive Programming**: Your hook may receive `None` values. Always check:
   ```python
   if dataset and "emotions" in dataset.lower():
       # Process emotion
   ```

2. **Error Handling**: Wrap risky operations in try/except. Errors are logged but won't crash reachy_mini:
   ```python
   def on_move_start(self, move_name, dataset):
       try:
           self.risky_operation()
       except Exception as e:
           logger.error(f"Hook error: {e}")
   ```

3. **Performance**: Keep hooks fast. They run on the main thread before/after move playback.

4. **Resource Management**: Clean up resources in `__del__` if needed:
   ```python
   def __del__(self):
       self.cleanup_hardware()
   ```

5. **Logging**: Use the standard `logging` module for debugging:
   ```python
   import logging
   logger = logging.getLogger(__name__)
   logger.info("Hook event")
   ```

## Troubleshooting

### Hook Not Loading

Check that:
1. Entry point is correctly registered in `pyproject.toml`
2. Package is installed: `pip list | grep my-plugin`
3. Check logs: `import logging; logging.basicConfig(level=logging.INFO)`

### Multiple Plugins Conflict

Hooks are called in the order they were loaded. If order matters, you may need to coordinate between plugins or use a single plugin that handles multiple concerns.

### Testing Without Hardware

Create a mock mode in your plugin:

```python
class MyHook:
    def __init__(self, mock=False):
        self.mock = mock
        if not mock:
            self.init_hardware()
```

## Contributing

If you create a useful plugin, consider:
- Sharing it on PyPI for others to use
- Opening a GitHub discussion to showcase your plugin
- Contributing to the reachy_mini community

## Questions?

- Check the [reachy_mini documentation](https://docs.pollen-robotics.com/)
- Open an issue on the [reachy_mini GitHub repository](https://github.com/pollen-robotics/reachy_mini)
