"""LLM adapter for choreography generation - supports multiple providers."""

import json
import os
from datetime import datetime
import anthropic

# Import API keys from api_keys file
try:
    from api_keys import ANTHROPIC_API_KEY
except ImportError:
    ANTHROPIC_API_KEY = None

# Import context builder
from choreography.context_builder import ChoreographyContext


class ChoreographyLLM:
    """Adapter for generating choreography recommendations via LLM."""

    def __init__(self, provider="anthropic", model=None):
        """
        Initialize LLM adapter.

        Args:
            provider: LLM provider ("anthropic", "ollama", "huggingface")
            model: Specific model name (None uses default for provider)
        """
        self.provider = provider
        self.model = model or self._default_model()

        # Initialize context builder with real move metadata
        print(f"[ChoreographyLLM] Loading move metadata for context...")
        self.context = ChoreographyContext()

        # Initialize provider-specific clients
        if self.provider == "anthropic":
            api_key = ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found (check api_keys.py or environment)")
            self.client = anthropic.Anthropic(api_key=api_key)

    def _default_model(self):
        """Get default model for provider."""
        defaults = {
            "anthropic": "claude-haiku-4-5",
            "ollama": "llama3.2",
            "huggingface": "custom-choreography-model"  # Future: Clem's model
        }
        return defaults.get(self.provider, "claude-haiku-4-5")

    def generate_recommendation(self, audio_features, moves_library, save_to=None):
        """
        Generate choreography recommendation from audio analysis.

        Args:
            audio_features: Dict from AudioAnalyzer
            moves_library: Dict with 'dances' and 'emotions' lists
            save_to: Optional path to save recommendation JSON

        Returns:
            dict: Choreography recommendation
        """
        # Build prompt
        prompt = self._build_prompt(audio_features, moves_library)

        # Generate based on provider
        if self.provider == "anthropic":
            response_text = self._anthropic_inference(prompt)
        elif self.provider == "ollama":
            response_text = self._ollama_inference(prompt)
        elif self.provider == "huggingface":
            response_text = self._huggingface_inference(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        # Parse JSON response
        try:
            recommendation = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response_text[:500]}")
            return None

        # Validate and FIX choreography to match audio duration exactly
        recommendation = self._validate_and_fix_choreography(recommendation, audio_features, moves_library)

        # Add metadata
        recommendation['recommendation_id'] = f"rec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        recommendation['generated_by'] = f"{self.provider}/{self.model}"
        recommendation['timestamp'] = datetime.now().isoformat()

        # Save if requested
        if save_to:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            with open(save_to, 'w') as f:
                json.dump(recommendation, f, indent=2)
            print(f"Saved recommendation to {save_to}")

        return recommendation

    def _validate_and_fix_choreography(self, recommendation: dict, audio_features: dict, moves_library: dict):
        """
        Validate and automatically fix choreography to match audio duration exactly.

        Args:
            recommendation: Generated choreography JSON (with bpm and sequence)
            audio_features: Original audio analysis
            moves_library: Dict with 'dances' and 'emotions' lists

        Returns:
            Fixed recommendation with exact duration match
        """
        sequence = recommendation.get('sequence', [])
        bpm = recommendation.get('bpm', audio_features['bpm'])

        if not sequence:
            print("[ChoreographyLLM] Warning: Empty sequence generated")
            return

        # Validate move names
        all_moves = set(moves_library.get('dances', [])) | set(moves_library.get('emotions', []))
        all_moves.add('idle')  # idle is always valid
        all_moves.add('manual')  # manual is always valid

        invalid_moves = []
        for i, entry in enumerate(sequence):
            move_name = entry.get('move')
            if move_name and move_name not in all_moves:
                invalid_moves.append((i, move_name))

        if invalid_moves:
            print("[ChoreographyLLM] WARNING: Invalid move names detected!")
            for idx, name in invalid_moves:
                print(f"  Move {idx}: '{name}' not found in library")
                # Try to find similar names
                similar = [m for m in all_moves if name.lower() in m.lower() or m.lower() in name.lower()]
                if similar:
                    print(f"    Did you mean: {', '.join(similar[:3])}?")

            # REMOVE invalid moves from sequence
            invalid_indices = {idx for idx, _ in invalid_moves}
            sequence = [entry for i, entry in enumerate(sequence) if i not in invalid_indices]
            recommendation['sequence'] = sequence
            print(f"[ChoreographyLLM] Removed {len(invalid_moves)} invalid moves, {len(sequence)} remain")

        # Convert sequence format to internal format for duration calculation
        choreography_for_calc = []
        for entry in sequence:
            move_name = entry.get('move')
            if move_name == "idle":
                choreography_for_calc.append({
                    "move_name": "idle",
                    "duration": entry.get('duration', 1.0)
                })
            elif move_name == "manual":
                # Manual moves have explicit duration
                choreography_for_calc.append({
                    "move_name": "manual",
                    "duration": entry.get('duration', 1.0)
                })
            else:
                # Determine move type from our metadata
                move_type = "dance" if move_name in self.context.dance_metadata else "emotion"
                choreography_for_calc.append({
                    "move_name": move_name,
                    "move_type": move_type,
                    "cycles": entry.get('cycles', 1)
                })

        # Calculate actual total duration using BPM formula
        actual_duration = self.context.calculate_total_duration(choreography_for_calc, bpm)
        expected_duration = audio_features['duration']
        beat_duration = 60.0 / bpm

        # Check total coverage
        duration_diff = abs(actual_duration - expected_duration)
        coverage = actual_duration / expected_duration if expected_duration > 0 else 0

        print(f"[ChoreographyLLM] Choreography validation:")
        print(f"  BPM: {bpm:.1f} (beat = {beat_duration:.3f}s)")
        print(f"  Expected duration: {expected_duration:.1f}s")
        print(f"  Actual duration: {actual_duration:.1f}s")
        print(f"  Coverage: {coverage*100:.1f}%")
        print(f"  Difference: {duration_diff:.1f}s")
        print(f"  Total moves: {len(sequence)}")

        if duration_diff > 2.0:
            print(f"[ChoreographyLLM] WARNING: Duration mismatch > 2 seconds!")

        # Detailed move breakdown
        if duration_diff > 5.0:
            print(f"[ChoreographyLLM] Move breakdown:")
            for i, entry in enumerate(sequence):
                move_name = entry.get('move')
                if move_name == "idle":
                    dur = entry.get('duration', 0)
                    print(f"  {i}: idle = {dur:.2f}s")
                elif move_name == "manual":
                    dur = entry.get('duration', 0)
                    body_yaw = entry.get('body_yaw', 0)
                    head_pose = entry.get('head_pose', {})
                    print(f"  {i}: manual (body_yaw={body_yaw:.0f}°, head_yaw={head_pose.get('yaw', 0):.0f}°) = {dur:.2f}s")
                else:
                    cycles = entry.get('cycles', 1)
                    move_type = "dance" if move_name in self.context.dance_metadata else "emotion"
                    beat_count = self.context.get_beat_count(move_name, move_type)
                    dur = cycles * beat_count * beat_duration if beat_count else 0
                    print(f"  {i}: {move_name} ({cycles}×{beat_count}beats) = {dur:.2f}s")

        # AUTOMATIC CORRECTION: Fix duration to match audio exactly
        if duration_diff > 1.0:  # Allow 1s tolerance
            print(f"[ChoreographyLLM] Auto-correcting duration mismatch...")

            if actual_duration > expected_duration:
                # TOO LONG: Trim moves from the end
                print(f"[ChoreographyLLM] Choreography too long, trimming moves...")
                cumulative_duration = 0
                trimmed_sequence = []

                for entry in sequence:
                    move_name = entry.get('move')

                    # Calculate this move's duration
                    if move_name == "idle" or move_name == "manual":
                        move_duration = entry.get('duration', 0.1)
                    else:
                        move_type = "dance" if move_name in self.context.dance_metadata else "emotion"
                        beat_count = self.context.get_beat_count(move_name, move_type)
                        cycles = entry.get('cycles', 1)
                        move_duration = cycles * beat_count * beat_duration if beat_count else 0

                    # Check if adding this move would exceed duration
                    if cumulative_duration + move_duration <= expected_duration:
                        trimmed_sequence.append(entry)
                        cumulative_duration += move_duration
                    else:
                        # Check if we can reduce cycles to fit
                        if move_name != "idle" and beat_count and beat_count > 0:
                            remaining_time = expected_duration - cumulative_duration
                            max_cycles = int(remaining_time / (beat_count * beat_duration))
                            if max_cycles > 0:
                                entry['cycles'] = max_cycles
                                trimmed_sequence.append(entry)
                                cumulative_duration += max_cycles * beat_count * beat_duration
                        break

                # Fill remaining time with manual neutral pause if needed
                remaining = expected_duration - cumulative_duration
                if remaining > 0.1:
                    trimmed_sequence.append({
                        "move": "manual",
                        "duration": round(remaining, 2),
                        "section": "outro",
                        "reasoning": "Auto-added neutral pause to fill remaining duration"
                    })

                sequence = trimmed_sequence
                recommendation['sequence'] = sequence
                recommendation['final_duration'] = cumulative_duration # Store the actual final duration
                print(f"[ChoreographyLLM] Trimmed to {len(sequence)} moves, new duration: {expected_duration:.1f}s")

            else:
                # TOO SHORT: Scale up existing moves instead of adding pauses
                shortfall = expected_duration - actual_duration
                print(f"[ChoreographyLLM] WARNING: Choreography {shortfall:.1f}s too short!")
                print(f"[ChoreographyLLM] Continuing without auto-fill (let it be short)")

        return recommendation

    def _suggest_moves_for_energy(self, audio_features):
        """Suggest appropriate moves based on energy level."""
        energy = audio_features['energy']
        danceability = audio_features['danceability']

        if energy > 0.7 or danceability > 0.7:
            return """HIGH ENERGY detected!
CREATE VARIETY - Use ALL these moves, mix them up, 1-2 cycles each:
- groovy_sway_and_roll (1-2 cycles)
- dizzy_spin (1-2 cycles)
- jackson_square (1-2 cycles)
- polyrhythm_combo (1-2 cycles)
- interwoven_spirals (1-2 cycles)
- headbanger_combo (1-2 cycles)
- pendulum_swing (1-2 cycles)
- side_to_side_sway (1-2 cycles)
- head_tilt_roll (1-2 cycles)
- Manual moves with dramatic body_yaw ±120-160° spins (1.0-1.5s each)

IMPORTANT: Use 20-30 total moves. Don't use the same move more than twice!"""
        elif energy > 0.4:
            return """MODERATE ENERGY - Mix these moves (1-2 cycles each):
- pendulum_swing
- side_to_side_sway
- head_tilt_roll
- side_peekaboo
- simple_nod
- Manual moves with body_yaw ±45-90° turns"""
        else:
            return """LOW ENERGY - Gentle moves (1 cycle each):
- simple_nod
- side_glance_flick
- chin_lead
- Calm emotions: thoughtful, serenity, calm
- Manual moves with gentle head tilts"""

    def _build_prompt(self, audio_features, moves_library):
        """Build universal prompt for choreography generation with actual move durations."""
        # Format segments for readability
        segments_text = "\n".join([
            f"  - {seg['start']:.1f}s to {seg['end']:.1f}s: {seg['label']} ({seg['end']-seg['start']:.1f}s duration)"
            for seg in audio_features.get('segments', [])
        ])

        # Format mood
        mood = audio_features.get('mood', {})
        mood_text = ", ".join([f"{k}={v:.2f}" for k, v in mood.items()])

        # Get formatted move context with REAL DURATIONS from datasets
        moves_context = self.context.format_for_prompt()

        prompt = f"""Generate a robot choreography for this audio file using BPM-based timing.

AUDIO ANALYSIS:
- Duration: {audio_features['duration']:.1f} seconds
- BPM: {audio_features['bpm']:.1f}
- Energy: {audio_features['energy']:.2f} (0-1 scale)
- Danceability: {audio_features['danceability']:.2f} (0-1 scale)
- Mood: {mood_text}

SONG STRUCTURE:
{segments_text}

{moves_context}

MANUAL MOVES FOR DYNAMIC CHOREOGRAPHY:

MANUAL POSITION CONTROL:
Create dynamic custom poses using "manual" moves with these parameters:
- head_pose: Use REAL rotation values for actual movement!
  - yaw (degrees): [-180 to 180] - left/right head turn
  - pitch (degrees): [-40 to 40] - up/down head tilt
  - roll (degrees): [-40 to 40] - side-to-side head tilt
  - position usually stays at defaults (x=0, y=0, z=0)
- body_yaw (degrees): [-160 to 160] - CRITICAL for dynamic turns and spins!
- antennas: [left, right] in radians [-3 to 3] - optional
- duration (seconds): How long to execute the movement

BODY YAW is extremely important for dynamic choreography:
- Use body_yaw liberally to create turning, spinning, orientation changes
- Combine body_yaw with head_yaw for complex movements
- Safety: Head yaw - body yaw must stay within ±65° (auto-enforced)
- **CRITICAL**: Manual moves MUST have non-zero values! Don't use all zeros (that's idle!)

Example manual moves (use REAL values like these):
- Look left: {{"move": "manual", "head_pose": {{"yaw": -45}}, "body_yaw": -20, "duration": 1.0}}
- Turn body right: {{"move": "manual", "body_yaw": 90, "head_pose": {{"yaw": 45}}, "duration": 2.0}}
- Dramatic spin: {{"move": "manual", "body_yaw": -120, "head_pose": {{"yaw": -80}}, "duration": 2.5}}
- Tilt head: {{"move": "manual", "head_pose": {{"pitch": 25, "roll": 20}}, "body_yaw": 15, "duration": 1.5}}

**NEVER use manual with all zeros - that defeats the purpose!**

TIMING FORMULA:
Move Duration = cycles × beat_count × (60/BPM)

With BPM={audio_features['bpm']:.1f}:
- 1 beat = {60.0/audio_features['bpm']:.3f} seconds
- Example: simple_nod (1 beat) × 4 cycles = {4 * 1 * (60.0/audio_features['bpm']):.2f}s
- Example: jackson_square (4 beats) × 2 cycles = {2 * 4 * (60.0/audio_features['bpm']):.2f}s

REQUIREMENTS:
1. **MUST FILL ENTIRE {audio_features['duration']:.1f}s**: Add moves until you hit {audio_features['duration']:.1f}s exactly!
2. **Track running total as you go**: After each move, calculate: "total so far = X.Xs, remaining = Y.Ys"
3. **Keep adding moves until duration is filled** - don't stop at 20s if audio is 30s!
4. Use BPM={audio_features['bpm']:.1f} to match the music's tempo
5. Calculate move durations using: cycles × beat_count × (60/{audio_features['bpm']:.1f})
6. **MATCH ENERGY LEVEL**: High energy audio = high energy moves ONLY (see energy matching above)!
7. **NO IDLE/PAUSES**: Use only recorded moves and manual position moves - keep moving!
8. **Manual moves MUST have movement**: body_yaw OR head_yaw/pitch/roll must be non-zero!
9. Each recorded move needs: move (name), cycles (repetitions), optional amplitude
10. Each manual move needs: move="manual", head_pose OR body_yaw (not all zeros!), duration
11. **CRITICAL**: Use EXACT move names from the lists above - do not shorten or modify names!
    - Example: Use "serenity1" NOT "serene1"
    - Example: Use "welcoming1" NOT "welcome1"
12. **FINAL CHECK**: Count up all move durations - they MUST sum to {audio_features['duration']:.1f}s!

DURATION BUDGETING for {audio_features['duration']:.1f}s audio:
- Target: 20-30 total moves (MORE variety, LESS repetition)
- Average move length: 1.0-1.5 seconds per move
- **CRITICAL**: No single move should exceed 3 seconds total!
- **VARIETY REQUIREMENT**: Use each move MAX 1-2 times in entire routine!
- **Max cycles: 1-2 cycles per move** (prevents boring repetition)
- Calculate as you go: track running total to stay within {audio_features['duration']:.1f}s

ENERGY-TO-MOVE MATCHING (CRITICAL - FOLLOW THIS):

**Energy: {audio_features['energy']:.2f}, Danceability: {audio_features['danceability']:.2f}**

IF Energy > 0.7 OR Danceability > 0.7 (HIGH ENERGY):
- Use ALL available high-energy dances, MIX them up!
- Available: groovy_sway_and_roll, dizzy_spin, jackson_square, polyrhythm_combo, interwoven_spirals, headbanger_combo, pendulum_swing, side_to_side_sway, head_tilt_roll
- 80% dances, 20% manual moves with dramatic spins (body_yaw ±120-160°)
- **Cycles: 1-2 MAX** (variety over repetition!)
- **Use DIFFERENT moves each time** - don't repeat the same move back-to-back
- NO calm emotions! NO thoughtful/serenity/calming!

IF Energy 0.4-0.7 (MODERATE):
- Mix dances: pendulum_swing, side_to_side_sway, head_tilt_roll, side_peekaboo, simple_nod
- 60% dances, 30% manual, 10% emotions
- Cycles: 1-2 MAX

IF Energy < 0.4 (LOW):
- Gentle dances + emotions: simple_nod, side_glance_flick, chin_lead
- Emotions: thoughtful, serenity, calm allowed here
- 40% dances, 40% manual, 20% emotions
- Cycles: 1 only

**FOR THIS AUDIO (Energy={audio_features['energy']:.2f}):**
{self._suggest_moves_for_energy(audio_features)}

CHOREOGRAPHY STRATEGY:
- Match EVERY move to the current energy level
- High energy music = high energy moves ONLY
- Don't use calming emotions for upbeat music!

OUTPUT FORMAT (JSON only, no other text):
{{
  "bpm": {audio_features['bpm']},
  "sequence": [
    {{
      "move": "welcoming1",
      "cycles": 1,
      "amplitude": 0.8,
      "section": "intro",
      "reasoning": "Gentle greeting to open performance"
    }},
    {{
      "move": "manual",
      "head_pose": {{"yaw": -35, "pitch": 15}},
      "body_yaw": -25,
      "duration": 1.5,
      "section": "intro",
      "reasoning": "Curious head tilt with body turn to engage audience"
    }},
    {{
      "move": "simple_nod",
      "cycles": 3,
      "amplitude": 0.6,
      "section": "verse",
      "reasoning": "Calm nodding, 3 cycles = moderate pacing"
    }},
    {{
      "move": "manual",
      "body_yaw": 75,
      "head_pose": {{"yaw": 50, "pitch": -12}},
      "duration": 1.8,
      "section": "verse",
      "reasoning": "Dramatic body turn with head looking up-right for verse energy"
    }},
    {{
      "move": "pendulum_swing",
      "cycles": 4,
      "amplitude": 1.0,
      "section": "verse",
      "reasoning": "Rhythmic swaying for verse groove, 4 cycles keeps it under 5s"
    }}
  ]
}}

Generate the complete choreography sequence now:"""

        return prompt

    def _anthropic_inference(self, prompt):
        """Generate using Anthropic Claude."""
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Extract text from response
            response_text = message.content[0].text

            # Claude might wrap JSON in markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            return response_text

        except Exception as e:
            print(f"Anthropic API error: {e}")
            return None

    def _ollama_inference(self, prompt):
        """Generate using Ollama (local)."""
        try:
            import requests

            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"Ollama API error: {response.status_code}")
                return None

        except Exception as e:
            print(f"Ollama connection error: {e}")
            print("Make sure Ollama is running: ollama serve")
            return None

    def _huggingface_inference(self, prompt):
        """Generate using HuggingFace model (future: Clem's custom model)."""
        # Placeholder for future custom model
        print("HuggingFace inference not yet implemented")
        print("This will load Clem's fine-tuned choreography model")
        return None
