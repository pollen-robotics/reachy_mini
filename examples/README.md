# Example Choreographies

This directory contains sample choreography JSON files demonstrating different moods, tempos, and use cases for Reachy Mini.

---

## 📁 Files

### `simple_greeting.json`
**BPM:** 100 (Moderate tempo)
**Moves:** 3 (7 total cycles)
**Theme:** Welcoming interaction

A short, friendly greeting sequence perfect for starting conversations or demos.

**Sequence:**
1. `welcoming1` - Friendly welcoming gesture (×2, amp 1.2)
2. `cheerful1` - Happy, upbeat expression (×3, amp 1.0)
3. `yes1` - Affirmative nod (×2, amp 1.0)

**Use Cases:**
- Demo introduction
- User onboarding
- First interaction with robot

---

### `dance_party.json`
**BPM:** 140 (Fast, upbeat)
**Moves:** 8 (25 total cycles)
**Theme:** High-energy dance performance

An extended dance routine showcasing rhythmic and dynamic movements from the dance library.

**Sequence:**
1. `side_to_side_sway` - Gentle rhythmic sway (×4, amp 1.0)
2. `groovy_sway_and_roll` - Compound roll motion (×3, amp 1.3)
3. `dizzy_spin` - Spinning head movement (×2, amp 1.0)
4. `headbanger_combo` - Energetic headbanging (×4, amp 1.5)
5. `jackson_square` - Precise geometric motion (×3, amp 1.2)
6. `polyrhythm_combo` - Complex multi-axis movement (×2, amp 1.0)
7. `interwoven_spirals` - Smooth spiral patterns (×3, amp 1.1)
8. `sharp_side_tilt` - Quick tilt movements (×4, amp 1.4)

**Use Cases:**
- Entertainment demos
- Music synchronization testing
- Showcasing motion capabilities

---

### `emotional_journey.json`
**BPM:** 90 (Slower, contemplative)
**Moves:** 9 (23 total cycles)
**Theme:** Emotional storytelling

A narrative sequence progressing through different emotional states, demonstrating expressive range.

**Emotional Arc:**
1. `curious1` - Initial curiosity (×2, amp 1.0)
2. `thoughtful1` - Deep thinking (×3, amp 0.9)
3. `confused1` - Confusion sets in (×2, amp 1.1)
4. `frustrated1` - Frustration builds (×2, amp 1.3)
5. `understanding1` - Moment of clarity (×3, amp 1.0)
6. `relief1` - Relief at solution (×2, amp 1.2)
7. `success1` - Success celebration (×3, amp 1.4)
8. `proud1` - Pride in achievement (×2, amp 1.2)
9. `cheerful1` - Happy conclusion (×4, amp 1.3)

**Use Cases:**
- Storytelling applications
- Emotion recognition demos
- Human-robot interaction research

---

### `subtle_conversation.json`
**BPM:** 80 (Slow, measured)
**Moves:** 8 (21 total cycles)
**Theme:** Natural conversation behaviors

Subtle, lower-amplitude movements mimicking active listening and conversational gestures.

**Sequence:**
1. `attentive1` - Focused attention (×3, amp 0.8)
2. `simple_nod` - Minimal acknowledgment (×2, amp 0.7)
3. `inquiring1` - Gentle questioning (×2, amp 0.9)
4. `uh_huh_tilt` - Backchannel response (×3, amp 0.8)
5. `understanding2` - Comprehension signal (×2, amp 0.9)
6. `yeah_nod` - Agreement nod (×2, amp 0.7)
7. `thoughtful2` - Contemplative pause (×3, amp 0.8)
8. `serenity1` - Calm, centered state (×4, amp 0.6)

**Use Cases:**
- Conversational AI demos
- Active listening demonstrations
- Subtle interaction research
- Background ambient behavior

**Note:** Lower amplitudes (0.6-0.9) create more natural, less exaggerated movements.

---

### `energetic_performance.json`
**BPM:** 160 (Very fast, high energy)
**Moves:** 9 (34 total cycles)
**Theme:** Maximum expressiveness

High-intensity performance pushing amplitude and tempo limits for dramatic effect.

**Sequence:**
1. `enthusiastic1` - Extreme excitement (×3, amp 1.6)
2. `electric1` - Electrified energy (×4, amp 1.8)
3. `headbanger_combo` - Intense headbanging (×5, amp 1.7)
4. `laughing1` - Exuberant laughter (×3, amp 1.5)
5. `dizzy_spin` - Fast spinning (×3, amp 1.4)
6. `success2` - Major celebration (×4, amp 1.6)
7. `polyrhythm_combo` - Complex rhythms (×3, amp 1.5)
8. `cheerful1` - High-energy happiness (×4, amp 1.7)
9. `dance3` - Dynamic dance finale (×5, amp 1.8)

**Use Cases:**
- Entertainment performances
- Stress testing motion system
- Maximum expressiveness demos
- High-energy interactions

**Warning:** High amplitudes (1.4-1.8) may stress motors. Monitor for overheating during extended use.

---

## 🎮 How to Use

### Via Choreography Builder Interface

1. Open `move_controller.html` in your browser
2. Manually recreate the sequence:
   - Select each move from the Pre-Recorded Moves panel
   - Set cycles and amplitude to match the JSON
   - Click "Add to Routine"
   - Set BPM to match
3. Or: Import functionality (planned for future version)

### Via Python SDK

```python
from reachy_mini import ReachyMini

with ReachyMini() as reachy:
    reachy.play_choreography('examples/simple_greeting.json')
```

### Via REST API

```bash
curl -X POST http://localhost:8100/api/choreography/play \
  -H "Content-Type: application/json" \
  -d @examples/simple_greeting.json
```

---

## 🎵 BPM Guide

**BPM** (Beats Per Minute) controls the overall tempo of the choreography.

| BPM Range | Tempo | Example Use Cases |
|-----------|-------|-------------------|
| 40-60 | Very Slow | Meditation, calm states, sleep mode |
| 60-80 | Slow | Conversation, contemplation, sad emotions |
| 80-100 | Moderate | Greetings, neutral interactions, thinking |
| 100-120 | Medium | General purpose, balanced energy |
| 120-140 | Upbeat | Dancing, excitement, celebrations |
| 140-160 | Fast | High energy, entertainment, intense emotions |
| 160-200 | Very Fast | Maximum excitement, stress testing |

**Recommendation:** Start with 100-120 BPM for balanced, natural-looking choreographies.

---

## 🎭 Amplitude Guide

**Amplitude** controls movement intensity (0.1-2.0).

| Amplitude | Intensity | Visual Effect | Use Cases |
|-----------|-----------|---------------|-----------|
| 0.1-0.5 | Minimal | Barely visible | Idle animations, sleep mode |
| 0.6-0.9 | Subtle | Noticeable, gentle | Conversation, active listening |
| 1.0 | Normal | Default intensity | General purpose, balanced |
| 1.1-1.3 | Enhanced | More expressive | Emphasis, clarity |
| 1.4-1.6 | Strong | Very pronounced | Performances, excitement |
| 1.7-2.0 | Maximum | Extreme movements | Entertainment, stress testing |

**Recommendation:** Use 0.8-1.2 for most applications. Reserve >1.5 for special effects.

**Warning:** Amplitudes >1.5 may stress motors over extended periods.

---

## 🔧 Cycles Guide

**Cycles** determines how many times each move repeats (1-10).

| Cycles | Duration | Use Cases |
|--------|----------|-----------|
| 1 | Brief | Quick gestures, transitions |
| 2-3 | Short | Acknowledgments, small responses |
| 4-5 | Medium | Standard expressions, balanced timing |
| 6-7 | Long | Emphasis, extended states |
| 8-10 | Very Long | Ambient behaviors, waiting states |

**Calculation:**
Total choreography duration ≈ `(sum of all cycles) × (60 / BPM)` seconds

**Example:**
- Total cycles: 25
- BPM: 140
- Duration: 25 × (60/140) ≈ 10.7 seconds

---

## 📝 Creating Your Own

### Using the Choreography Builder

1. Open `move_controller.html`
2. Set desired BPM (40-200)
3. For each move in your sequence:
   - Select move from library
   - Set cycles (1-10)
   - Set amplitude (0.1-2.0)
   - Click "Add to Routine"
4. Click "💾 Export JSON"
5. Save to `examples/` directory

### Manual JSON Creation

```json
{
    "bpm": 120,
    "sequence": [
        {
            "move": "move_name_from_library",
            "cycles": 4,
            "amplitude": 1.0
        },
        {
            "move": "another_move",
            "cycles": 2,
            "amplitude": 1.2
        }
    ]
}
```

**Valid Move Names:**
- See `moves.json` for complete list
- 20 dance moves (e.g., `dizzy_spin`, `groovy_sway_and_roll`)
- 81 emotion moves (e.g., `cheerful1`, `frustrated1`, `loving1`)

---

## 🎨 Choreography Design Tips

### Theme Consistency

**Do:**
- Group similar emotions (all happy, all sad, or a clear progression)
- Match BPM to mood (slow for sad, fast for happy)
- Use consistent amplitude ranges

**Don't:**
- Mix random unrelated moves
- Jump between extreme amplitudes
- Use conflicting tempos

### Pacing

**Build gradually:**
1. Start with lower amplitudes (0.8-1.0)
2. Build to peak (1.3-1.5)
3. Cool down at end (1.0-0.8)

**Vary cycles:**
- Short cycles (2-3) for quick transitions
- Long cycles (4-6) for emphasis
- Very long cycles (7-10) for ambient states

### Emotional Arc

**Storytelling structure:**
1. **Introduction** - Establish baseline emotion
2. **Rising action** - Build intensity
3. **Climax** - Peak emotional moment
4. **Resolution** - Return to calm

**Example:** See `emotional_journey.json`

### Musical Choreography

**Sync to music:**
1. Determine song BPM
2. Match choreography BPM
3. Align move changes to musical phrases
4. Use amplitude for dynamics (loud = high amp)

**Tip:** Most pop music is 120-140 BPM.

### Conversation Behaviors

**Active listening:**
- Use subtle moves (0.6-0.9 amplitude)
- Slow tempo (70-90 BPM)
- Short cycles (2-3) for responsiveness
- Mix: attentive, nods, understanding

**Example:** See `subtle_conversation.json`

---

## 🧪 Testing Choreographies

### Pre-Testing Checklist

- [ ] Total duration reasonable (under 2 minutes for initial tests)
- [ ] BPM appropriate for theme
- [ ] Amplitudes not all at extremes
- [ ] Cycles varied (not all the same)
- [ ] Move names match library exactly (case-sensitive)
- [ ] JSON syntax valid (use validator)

### Testing Process

1. **Start simple:** Test with 2-3 moves first
2. **Verify timing:** Does BPM feel right?
3. **Check amplitude:** Too subtle? Too intense?
4. **Monitor hardware:** Any unusual sounds or stuttering?
5. **Iterate:** Adjust values and re-test
6. **Extend:** Add more moves once core timing works

### Common Issues

**Moves too fast:**
- Decrease BPM
- Increase cycles (slower transitions)

**Moves too subtle:**
- Increase amplitude
- Choose more expressive moves

**Jerky transitions:**
- Lower BPM for smoother execution
- Reduce amplitude if hitting limits

**Duration too long:**
- Reduce total cycles
- Remove some moves
- Increase BPM (faster completion)

---

## 📚 Move Library Reference

**Full list available in:** `moves.json`

**Dance Moves (20):**
Rhythmic, choreographed movements. Best for entertainment and synchronization.

**Emotion Moves (81):**
Expressive poses and gestures. Best for storytelling and interaction.

**Tip:** Test individual moves before choreographing to understand their character.

---

## 🤝 Contributing Examples

Have a great choreography? Share it with the community!

**Submission Guidelines:**

1. **Descriptive filename:** `theme_mood_bpm.json`
2. **Test thoroughly:** Works on both simulator and real robot
3. **Document clearly:** Add entry to this README
4. **Reasonable parameters:** BPM 60-160, amp 0.5-1.5 for safety
5. **Attribution:** Include your name/username in pull request

**Submission Process:**

1. Create choreography JSON in `examples/` directory
2. Add description to this README
3. Test on simulator and (if possible) real robot
4. Submit pull request to Reachy Mini repository

---

## 📄 License

These example choreographies are provided under the same license as the Reachy Mini software (Apache 2.0).

Feel free to use, modify, and share!

---

**Questions?** Open an issue on GitHub or ask in the Pollen Robotics community forum.

**Happy choreographing! 🎭🤖**
