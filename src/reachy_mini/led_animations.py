"""
LED Ring Animation Sequences
Contains pre-defined animation steps for 12-LED ring animations.

Usage:
    import led_animations as animations

    # Play wake up animation
    for step in animations.wake_spiral:
        self.set_led_colors(step)

    # Play sleep animation
    for step in animations.sleep_fade:
        self.set_led_colors(step)
"""

# ============================================================================
# WAKE UP ANIMATIONS
# ============================================================================

wake_spiral = [
    # Spiral outward with blue to white transition
    ({0: (50, 50, 255)}, 0.15),
    ({0: (50, 50, 255), 1: (67, 67, 240)}, 0.15),
    ({0: (50, 50, 255), 1: (67, 67, 240), 11: (84, 84, 225)}, 0.15),
    ({0: (50, 50, 255), 1: (67, 67, 240), 11: (84, 84, 225), 2: (101, 101, 210)}, 0.15),
    (
        {
            0: (50, 50, 255),
            1: (67, 67, 240),
            11: (84, 84, 225),
            2: (101, 101, 210),
            10: (118, 118, 195),
        },
        0.15,
    ),
    (
        {
            0: (50, 50, 255),
            1: (67, 67, 240),
            11: (84, 84, 225),
            2: (101, 101, 210),
            10: (118, 118, 195),
            3: (135, 135, 180),
        },
        0.15,
    ),
    (
        {
            0: (50, 50, 255),
            1: (67, 67, 240),
            11: (84, 84, 225),
            2: (101, 101, 210),
            10: (118, 118, 195),
            3: (135, 135, 180),
            9: (152, 152, 165),
        },
        0.15,
    ),
    (
        {
            0: (50, 50, 255),
            1: (67, 67, 240),
            11: (84, 84, 225),
            2: (101, 101, 210),
            10: (118, 118, 195),
            3: (135, 135, 180),
            9: (152, 152, 165),
            4: (169, 169, 150),
        },
        0.15,
    ),
    (
        {
            0: (50, 50, 255),
            1: (67, 67, 240),
            11: (84, 84, 225),
            2: (101, 101, 210),
            10: (118, 118, 195),
            3: (135, 135, 180),
            9: (152, 152, 165),
            4: (169, 169, 150),
            8: (186, 186, 135),
        },
        0.15,
    ),
    (
        {
            0: (50, 50, 255),
            1: (67, 67, 240),
            11: (84, 84, 225),
            2: (101, 101, 210),
            10: (118, 118, 195),
            3: (135, 135, 180),
            9: (152, 152, 165),
            4: (169, 169, 150),
            8: (186, 186, 135),
            5: (203, 203, 120),
        },
        0.15,
    ),
    (
        {
            0: (50, 50, 255),
            1: (67, 67, 240),
            11: (84, 84, 225),
            2: (101, 101, 210),
            10: (118, 118, 195),
            3: (135, 135, 180),
            9: (152, 152, 165),
            4: (169, 169, 150),
            8: (186, 186, 135),
            5: (203, 203, 120),
            7: (220, 220, 105),
        },
        0.15,
    ),
    (
        {
            0: (50, 50, 255),
            1: (67, 67, 240),
            11: (84, 84, 225),
            2: (101, 101, 210),
            10: (118, 118, 195),
            3: (135, 135, 180),
            9: (152, 152, 165),
            4: (169, 169, 150),
            8: (186, 186, 135),
            5: (203, 203, 120),
            7: (220, 220, 105),
            6: (237, 237, 100),
        },
        0.15,
    ),
    # Final bright white flash
    ({i: (255, 255, 255) for i in range(12)}, 0.5),
]

wake_pulse = []
# Build pulse animation - 4 pulses getting brighter
for pulse in range(4):
    # Fade up
    for brightness in range(0, 200, 20):
        blue_val = min(255, brightness + 55)
        color = (brightness // 3, brightness // 2, blue_val)
        wake_pulse.append(({i: color for i in range(12)}, 0.05))

    # Fade down
    for brightness in range(200, 0, -30):
        blue_val = min(255, brightness + 55)
        color = (brightness // 3, brightness // 2, blue_val)
        wake_pulse.append(({i: color for i in range(12)}, 0.03))

# Final wake state
wake_pulse.append(({i: (150, 150, 200) for i in range(12)}, 0.5))

wake_scanner = []
# Build scanner animation - 3 sweeps
for sweep in range(3):
    # Forward sweep
    for pos in range(12):
        colors = {}
        colors[pos] = (0, 255, 255)  # Cyan main spot
        if pos > 0:
            colors[(pos - 1) % 12] = (0, 150, 150)  # Trail
        if pos > 1:
            colors[(pos - 2) % 12] = (0, 75, 75)  # Fading trail
        wake_scanner.append((colors, 0.08))

    # Backward sweep
    for pos in range(10, -1, -1):
        colors = {}
        colors[pos] = (0, 255, 255)
        if pos < 11:
            colors[(pos + 1) % 12] = (0, 150, 150)
        if pos < 10:
            colors[(pos + 2) % 12] = (0, 75, 75)
        wake_scanner.append((colors, 0.08))

# ============================================================================
# SLEEP ANIMATIONS
# ============================================================================

sleep_fade = [
    # Note: This one needs current state, see sleep_fade_from_state() function below
    # Starting with gentle glow if nothing is on
    ({i: (100, 100, 150) for i in range(12)}, 0.5),
]

# Add fade out steps - 20 steps from bright to off
base_colors = {i: (100, 100, 150) for i in range(12)}
for step in range(19, 0, -1):  # 19 down to 1
    fade_colors = {}
    for led_id, (r, g, b) in base_colors.items():
        fade_colors[led_id] = (
            int(r * step / 20),
            int(g * step / 20),
            int(b * step / 20),
        )
    sleep_fade.append((fade_colors, 0.1))

sleep_spiral_in = [
    # Start with warm glow
    ({i: (255, 150, 50) for i in range(12)}, 0.5),
]

# Spiral inward sequence
spiral_sequence = [6, 7, 5, 8, 4, 9, 3, 10, 2, 11, 1, 0]
for i, led_to_turn_off in enumerate(spiral_sequence):
    remaining_colors = {}
    for j in range(12):
        if j not in spiral_sequence[: i + 1]:  # Not yet turned off
            brightness = int(255 * (1 - i * 0.08))
            remaining_colors[j] = (
                max(0, brightness),
                max(0, brightness // 2),
                max(0, brightness // 4),
            )
    sleep_spiral_in.append((remaining_colors, 0.2))

sleep_heartbeat = []
# Slowing heartbeat pattern
beat_delays = [0.3, 0.4, 0.5, 0.7]

for i, delay in enumerate(beat_delays):
    intensity = int(200 - i * 30)  # Getting dimmer

    # First beat (quick pulse)
    sleep_heartbeat.append(
        ({j: (intensity, intensity // 3, intensity // 3) for j in range(12)}, 0.1)
    )
    sleep_heartbeat.append(
        ({j: (intensity // 3, intensity // 6, intensity // 6) for j in range(12)}, 0.1)
    )

    # Second beat
    sleep_heartbeat.append(
        ({j: (intensity, intensity // 3, intensity // 3) for j in range(12)}, 0.1)
    )
    sleep_heartbeat.append(
        (
            {j: (intensity // 4, intensity // 8, intensity // 8) for j in range(12)},
            delay,
        )
    )
sleep_heartbeat.append(({i: (0, 0, 0) for i in range(12)}, 0.5))

# ============================================================================
# ANIMATION COLLECTIONS
# ============================================================================

# Easy access to all animations
wake = {
    "spiral": wake_spiral,
    "pulse": wake_pulse,
    "scanner": wake_scanner,
}

sleep = {
    "fade": sleep_fade,
    "spiral_in": sleep_spiral_in,
    "heartbeat": sleep_heartbeat,
}

# Default choices
wake_default = wake_spiral
sleep_default = sleep_fade
