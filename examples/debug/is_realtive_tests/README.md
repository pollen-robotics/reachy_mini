# Relative Motion Tests

This folder contains test scripts for the `is_relative` motion feature, which allows layering relative offsets on top of absolute robot movements.

## Test Scripts

### Basic Functionality Tests

**`test_relative_pitch.py`** - Minimal relative motion test
```bash
python test_relative_pitch.py
```
Simple 5° pitch sine wave using `set_target(is_relative=True)`. Basic functionality verification.

**`test_relative_timeout.py`** - Timeout and decay behavior
```bash
python test_relative_timeout.py
```
Tests automatic timeout (1s) and smooth decay (1s) when relative commands stop. Multi-DOF motion with 2cm translations and 15° rotations.

### Advanced Feature Tests

**`test_goto_relative.py`** - Relative interpolated motion
```bash
python test_goto_relative.py
```
Absolute pitch motion + relative rectangular pattern using `goto_target(is_relative=True)`. Rectangle: Y=±2cm, Z=±1cm with smooth interpolation.

**`test_play_on_relative.py`** - Relative dance moves
```bash
python test_play_on_relative.py
```
Absolute pitch motion + relative dance moves using `play_on(is_relative=True)`. Tests multiple dance moves as expressive offsets.

**`test_combined_absolute_relative.py`** - Full integration
```bash
python test_combined_absolute_relative.py
```
Absolute rectangular choreography (12s cycle) + relative dance expressions (8s interval). Demonstrates real-world performance scenario.

### Demo Scripts

**`dance_demo_with_relative.py`** - Enhanced dance demo with cycling relative modes
```bash
# Default settings
python dance_demo_with_relative.py

# With choreography file
python dance_demo_with_relative.py --choreography ../../choreographies/another_one_bites_the_dust.json --no-keyboard

# Custom relative motion parameters
python dance_demo_with_relative.py --relative-amplitude-deg 3.0 --relative-amplitude-mm 1.5
```
Original dance demo with 9 cycling relative motion modes (pitch, roll, yaw, x, y, z, combinations). Changes every 4 seconds with different prime frequencies.

## Key Features Tested

- **Timeout System**: 1s timeout + 1s smooth decay when relative commands stop
- **Matrix Composition**: Proper combination of absolute poses + relative offsets  
- **goto_target() and play_on() integration**
- **Multi-threading**: Independent absolute and relative motion streams
- **Recovery**: System continues working after timeout/decay cycles

## Usage Notes

- All scripts include automatic wake up and sleep sequences
- Use `Ctrl+C` to interrupt any test safely
- Scripts demonstrate both simple and complex relative motion patterns
- Timeout behavior is observable in real-time with status messages