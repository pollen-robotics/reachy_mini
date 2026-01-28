# Skill: Interaction Patterns

## When to Use

- Designing how users will interact with the robot
- Building games or interactive experiences
- Creating apps without a traditional GUI

---

## Antennas as Buttons

The antenna motors use low P in PID - they're semi-passive and safe to push. This makes them natural physical buttons.

```python
ANTENNA_THRESHOLD = 0.3  # radians, tune as needed

def check_antenna_press(mini):
    _, antennas = mini.get_current_joint_positions()
    if antennas is None:
        return None

    left, right = antennas
    if abs(left) > ANTENNA_THRESHOLD:
        return "left"
    if abs(right) > ANTENNA_THRESHOLD:
        return "right"
    return None
```

### Use Cases

- **Start/stop**: Press antenna to begin game
- **Selection**: Left = option A, Right = option B
- **Confirmation**: Any antenna = "yes"

**Reference:** `~/reachy_mini_resources/reachy_mini_radio/` (change radio station with antennas)

> If `~/reachy_mini_resources/` doesn't exist, run `skills/setup-environment.md` to clone reference apps.

---

## Head as Controller

The head has 6 DOF - it's a powerful input device for games or recording.

### Reading Head Position

```python
def get_head_as_joystick(mini):
    """Map head orientation to joystick-like values."""
    pose = mini.get_current_head_pose()
    # Extract orientation (implementation depends on pose format)
    # Typically you'd extract yaw and pitch

    # Normalize to -1 to 1 range
    yaw_normalized = pose_yaw / 45.0  # Assuming ±45° range
    pitch_normalized = pose_pitch / 30.0  # Assuming ±30° range

    return {
        "x": np.clip(yaw_normalized, -1, 1),
        "y": np.clip(pitch_normalized, -1, 1)
    }
```

### Use Cases

- **Games**: Head tilt controls spaceship, character, cursor
- **Recording**: Capture head motion for playback
- **Puppeteering**: Control another robot or avatar

**References:**
- `~/reachy_mini_resources/fire_nation_attacked/` - Head as joystick in game
- `~/reachy_mini_resources/spaceship_game/` - Head controls spaceship
- `~/reachy_mini_resources/marionette/` - Record and playback head motion

---

## No-GUI Pattern

For simple apps, skip the web UI. Use antenna interactions:

### Basic Flow

```python
def run(self, mini):
    # 1. Signal readiness (twitch antennas)
    self.twitch_antennas(mini)

    # 2. Wait for user to press antenna
    print("Press an antenna to start...")
    while True:
        press = check_antenna_press(mini)
        if press:
            break
        time.sleep(0.1)

    # 3. Run the actual app
    self.main_loop(mini)
```

### Signaling States

Use antenna motion to communicate without GUI:

| State | Antenna behavior |
|-------|------------------|
| Ready/waiting | Gentle twitch |
| Processing | Slow wave |
| Success | Quick double bounce |
| Error | Shake rapidly |

**Reference:** `~/reachy_mini_resources/reachy_mini_simon/` (full game using only antennas)

---

## Combining Patterns

Many apps combine multiple patterns:

```python
class InteractiveApp:
    def run(self, mini):
        # Start with antenna press (no-GUI)
        self.wait_for_start(mini)

        while self.running:
            # Use head as input
            joystick = self.get_head_as_joystick(mini)

            # Check for antenna presses
            antenna = check_antenna_press(mini)
            if antenna == "left":
                self.action_a()
            elif antenna == "right":
                self.action_b()

            # Update game/app state based on head position
            self.update(joystick)

            time.sleep(0.01)
```

---

## Tips

- **Debounce antenna presses** - Add cooldown to prevent multiple triggers
- **Provide feedback** - Move antennas or play emotion when input detected
- **Consider accessibility** - Not everyone can push antennas easily
- **Test threshold values** - Different users push with different force
