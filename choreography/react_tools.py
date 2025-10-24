"""
Tool Registry for ReAct Choreographer

All tools that the LLM can call during choreography generation.
Each tool is designed to provide precise, actionable information.
"""

import json
from typing import List, Dict, Any, Optional
from .move_metadata_cache import load_cache, get_move_duration as _get_move_duration


class ChoreographyTools:
    """Tools for the ReAct agent to use during choreography generation."""

    def __init__(self, audio_analysis: Dict[str, Any]):
        """
        Initialize tools with audio analysis data and move metadata.

        Args:
            audio_analysis: Full audio analysis dict from AudioAnalyzer
        """
        self.audio_analysis = audio_analysis
        self.move_metadata = load_cache()
        print(f"[Tools] Loaded {len(self.move_metadata)} moves from cache")

    def get_move_duration(self, move_name: str) -> Optional[float]:
        """
        Get the actual SDK duration for a specific move.

        Args:
            move_name: Name of the move (e.g., "groovy_sway_and_roll")

        Returns:
            Duration in seconds, or None if move not found
        """
        duration = _get_move_duration(move_name, self.move_metadata)
        if duration is None:
            return None
        return float(duration)

    def calculate_sequence_duration(self, sequence: List[Dict[str, Any]]) -> float:
        """
        Calculate the total duration of a choreography sequence using actual SDK durations.

        Args:
            sequence: List of moves, each dict with 'move' (name) and optional 'cycles'

        Returns:
            Total duration in seconds
        """
        total = 0.0
        for move_info in sequence:
            move_name = move_info.get('move') or move_info.get('move_name')

            # Skip manual moves (they have variable duration)
            if move_name == 'manual' or move_name is None:
                continue

            cycles = move_info.get('cycles', 1)
            duration = self.get_move_duration(move_name)

            if duration is not None:
                total += duration * cycles

        return total

    def get_moves_by_duration(self, min_dur: float, max_dur: float) -> List[str]:
        """
        Find all moves within a specific duration range.

        Args:
            min_dur: Minimum duration in seconds
            max_dur: Maximum duration in seconds

        Returns:
            List of move names that fall within the duration range
        """
        matching_moves = []
        for move_name, data in self.move_metadata.items():
            if min_dur <= data['duration'] <= max_dur:
                matching_moves.append(move_name)

        return matching_moves

    def get_moves_by_type(self, move_type: str) -> List[str]:
        """
        Get all moves of a specific type.

        Args:
            move_type: Either "dance" or "emotion"

        Returns:
            List of move names of the specified type
        """
        return [name for name, data in self.move_metadata.items()
                if data['type'] == move_type]

    def get_move_info(self, move_name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive information about a specific move.

        Args:
            move_name: Name of the move to query

        Returns:
            {
                'name': str,
                'duration': float,
                'type': str ('dance' or 'emotion'),
                'library': str ('dances' or 'emotions'),
                'duration_category': str ('short', 'medium', 'long'),
                'exists': bool
            } or None if move doesn't exist
        """
        if move_name not in self.move_metadata:
            return {
                'exists': False,
                'message': f"Move '{move_name}' not found in library"
            }

        data = self.move_metadata[move_name]
        duration = data['duration']

        # Categorize by duration and type
        if data['type'] == 'dance':
            if duration < 2.5:
                category = 'short'
            elif duration < 4.0:
                category = 'medium'
            else:
                category = 'long'
        else:  # emotion
            if duration < 3.0:
                category = 'quick'
            elif duration < 6.0:
                category = 'short'
            elif duration < 10.0:
                category = 'medium'
            else:
                category = 'long'

        return {
            'exists': True,
            'name': move_name,
            'duration': duration,
            'type': data['type'],
            'library': data['library'],
            'duration_category': category
        }

    def get_music_structure(self) -> Dict[str, Any]:
        """
        Get the music structure analysis with labeled segments.

        Returns detailed segmentation with per-segment features for section-based
        choreography planning. Use this to understand the song's structure before
        creating choreography.

        Returns:
            {
                'total_duration': float,  # Total track duration in seconds
                'bpm': float,  # Global BPM
                'total_segments': int,  # Number of segments
                'segments': [
                    {
                        'label': str,  # 'intro', 'verse', 'chorus', 'bridge', 'outro'
                        'start': float,  # Start time in seconds
                        'end': float,  # End time in seconds
                        'duration': float,  # Segment duration in seconds
                        'energy': float,  # RMS energy (0.0-1.0+)
                        'spectral_centroid': float,  # Brightness in Hz
                        'spectral_rolloff': float,  # Frequency content in Hz
                        'beats_count': int  # Number of beats in segment
                    },
                    ...
                ]
            }

        Example Usage:
            structure = tools.get_music_structure()
            for segment in structure['segments']:
                if segment['label'] == 'chorus' and segment['energy'] > 0.15:
                    # Use high-energy dance moves for this chorus
                    ...
        """
        segments = self.audio_analysis.get('segments', [])

        return {
            'total_duration': self.audio_analysis.get('duration', 0.0),
            'bpm': self.audio_analysis.get('bpm', 0.0),
            'total_segments': len(segments),
            'segments': segments
        }

    def suggest_moves_for_context(
        self,
        bpm_range: Optional[str] = None,
        energy_range: Optional[str] = None,
        move_type: Optional[str] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Suggest moves that fit a specific musical context.

        Args:
            bpm_range: "slow" (<100), "medium" (100-140), "fast" (>140), or None
            energy_range: "low" (0-0.4), "moderate" (0.4-0.7), "high" (>0.7), or None
            move_type: "dance", "emotion", or None for both
            max_results: Maximum number of moves to return per category

        Returns:
            {
                'context': dict (parameters used),
                'recommendations': {
                    'primary': list (best matches),
                    'alternates': list (other good options)
                },
                'rationale': str (why these moves fit)
            }
        """
        # Parse parameters
        bpm = self.audio_analysis.get('bpm', 120)
        energy = self.audio_analysis.get('energy', 0.5)

        # Filter moves by type
        candidate_moves = []
        for name, data in self.move_metadata.items():
            if move_type and data['type'] != move_type:
                continue
            candidate_moves.append((name, data))

        # BPM-based filtering
        if bpm_range:
            if bpm_range == "slow":  # <100 BPM
                # Prefer longer moves, sustained holds
                candidate_moves = [(n, d) for n, d in candidate_moves if d['duration'] >= 3.0]
                rationale = f"Slow tempo ({bpm:.0f} BPM) suits sustained, longer moves"
            elif bpm_range == "medium":  # 100-140
                # Balanced mix
                rationale = f"Medium tempo ({bpm:.0f} BPM) allows balanced variety"
            elif bpm_range == "fast":  # >140
                # Prefer shorter, rapid moves
                candidate_moves = [(n, d) for n, d in candidate_moves if d['duration'] <= 3.0]
                rationale = f"Fast tempo ({bpm:.0f} BPM) needs quick, rapid-fire moves"
            else:
                rationale = "General musical context"
        else:
            rationale = "General choreography context"

        # Energy-based filtering and sorting
        if energy_range:
            if energy_range == "low":
                # Prefer dance moves for precision, or calm emotion moves
                if move_type != "emotion":
                    candidate_moves = [(n, d) for n, d in candidate_moves
                                     if d['type'] == 'dance' or
                                        n in ['serenity1', 'calming1', 'tired1', 'sleep1']]
                rationale += ", low energy suits subtle movements"
            elif energy_range == "moderate":
                rationale += ", moderate energy allows expressive variety"
            elif energy_range == "high":
                # Prefer dance moves or energetic emotions
                if move_type != "dance":
                    candidate_moves = [(n, d) for n, d in candidate_moves
                                     if d['type'] == 'dance' or
                                        n in ['enthusiastic1', 'enthusiastic2', 'cheerful1',
                                              'laughing1', 'success1', 'electric1']]
                rationale += ", high energy needs dynamic, vigorous moves"

        # Sort by duration (shorter first for variety)
        candidate_moves.sort(key=lambda x: x[1]['duration'])

        # Split into primary and alternates
        primary = [name for name, data in candidate_moves[:max_results]]
        alternates = [name for name, data in candidate_moves[max_results:max_results*2]]

        return {
            'context': {
                'bpm': bpm,
                'bpm_range': bpm_range,
                'energy': energy,
                'energy_range': energy_range,
                'move_type': move_type
            },
            'recommendations': {
                'primary': primary,
                'alternates': alternates
            },
            'rationale': rationale,
            'total_available': len(candidate_moves)
        }

    def validate_duration(self, sequence: List[Dict[str, Any]],
                         target_duration: float,
                         threshold: float = 1.5) -> Dict[str, Any]:
        """
        Validate if a choreography sequence matches the target duration.

        Args:
            sequence: List of moves
            target_duration: Expected duration in seconds
            threshold: Acceptable error in seconds (default 0.5s)

        Returns:
            {
                'valid': bool,
                'actual_duration': float,
                'target_duration': float,
                'difference': float,
                'too_long': bool,
                'too_short': bool
            }
        """
        actual = self.calculate_sequence_duration(sequence)
        diff = actual - target_duration
        abs_diff = abs(diff)

        return {
            'valid': abs_diff <= threshold,
            'actual_duration': actual,
            'target_duration': target_duration,
            'difference': diff,
            'too_long': diff > threshold,
            'too_short': diff < -threshold,
            'percent_off': (abs_diff / target_duration * 100) if target_duration > 0 else 0
        }

    def get_vocal_sections(self) -> List[Dict[str, Any]]:
        """
        Identify sections of the audio with strong vocal content.

        Returns:
            List of sections: [{start: float, end: float, vocal_prob: float, characteristics: dict}]
        """
        vocal_prob = self.audio_analysis.get('vocal_instrumental', {}).get('vocal_probability', 0.5)

        # For MVP, treat entire audio as one section
        # TODO: Implement actual segmentation based on time-series vocal detection
        duration = self.audio_analysis['duration']

        if vocal_prob > 0.7:
            # Strong vocals throughout
            return [{
                'start': 0.0,
                'end': duration,
                'vocal_prob': vocal_prob,
                'characteristics': {
                    'harmonic_ratio': self.audio_analysis.get('harmonic_percussive', {}).get('harmonic_ratio', 0.5),
                    'melodic_content': self.audio_analysis.get('pitch_content', {}).get('melodic_content', 0.5)
                }
            }]
        else:
            return []

    def get_instrumental_sections(self) -> List[Dict[str, Any]]:
        """
        Identify instrumental/non-vocal sections of the audio.

        Returns:
            List of sections: [{start: float, end: float, energy: float, percussive_ratio: float}]
        """
        vocal_prob = self.audio_analysis.get('vocal_instrumental', {}).get('vocal_probability', 0.5)
        duration = self.audio_analysis['duration']

        if vocal_prob < 0.3:
            # Strongly instrumental
            return [{
                'start': 0.0,
                'end': duration,
                'energy': self.audio_analysis['energy'],
                'percussive_ratio': self.audio_analysis.get('harmonic_percussive', {}).get('percussive_ratio', 0.5),
                'bpm': self.audio_analysis['bpm']
            }]
        else:
            return []

    def get_audio_duration(self) -> float:
        """Get the actual content duration of the audio (excluding silent padding)."""
        return float(self.audio_analysis['duration'])

    def get_audio_bpm(self) -> float:
        """Get the detected BPM of the audio."""
        return float(self.audio_analysis['bpm'])

    def get_audio_energy(self) -> float:
        """Get the overall energy level of the audio."""
        return float(self.audio_analysis['energy'])

    def is_vocal_content(self) -> bool:
        """Determine if the audio has significant vocal content."""
        vocal_prob = self.audio_analysis.get('vocal_instrumental', {}).get('vocal_probability', 0.5)
        return vocal_prob > 0.7

    def is_instrumental_content(self) -> bool:
        """Determine if the audio is primarily instrumental."""
        vocal_prob = self.audio_analysis.get('vocal_instrumental', {}).get('vocal_probability', 0.5)
        return vocal_prob < 0.3

    def solve_duration_constraint(
        self,
        target_duration: float,
        move_type: str | None = None,
        tolerance: float = 1.5,
        num_solutions: int = 3
    ) -> Dict[str, Any]:
        """
        Solve the duration constraint by finding valid move combinations.

        This tool handles the combinatorial optimization so the LLM can focus on
        creative decisions. It generates multiple valid sequences that fit the
        duration constraint.

        Args:
            target_duration: Target duration in seconds
            move_type: Optional filter - "dance" or "emotion" or None for both
            tolerance: Acceptable error in seconds (default 1.5s)
            num_solutions: Number of different solutions to return (default 3)

        Returns:
            {
                'solutions': [
                    {
                        'sequence': [{'move': 'name', 'cycles': 1}, ...],
                        'duration': float,
                        'move_count': int,
                        'variety_score': float
                    },
                    ...
                ],
                'target_duration': float,
                'tolerance': float
            }
        """
        import random

        # Filter moves by type if requested
        available_moves = []
        for name, data in self.move_metadata.items():
            if move_type is None or data['type'] == move_type:
                available_moves.append((name, data['duration']))

        if not available_moves:
            return {'error': f'No moves found for type={move_type}'}

        # Sort by duration for greedy algorithm
        available_moves.sort(key=lambda x: x[1])

        solutions = []

        # Generate multiple solutions using randomized greedy approach
        for attempt in range(num_solutions * 3):  # Try 3x more to get variety
            sequence = []
            current_duration = 0.0
            used_moves = set()

            # Greedy fill with randomization
            while current_duration < target_duration - tolerance:
                remaining = target_duration - current_duration

                # Find moves that could fit in remaining time
                candidates = [(name, dur) for name, dur in available_moves if dur <= remaining + tolerance]

                if not candidates:
                    break

                # Add randomization to get variety across solutions
                if random.random() < 0.7:  # 70% pick from best fits
                    # Pick moves close to remaining duration
                    candidates.sort(key=lambda x: abs(x[1] - remaining))
                    move_name, move_dur = random.choice(candidates[:min(5, len(candidates))])
                else:  # 30% pick random
                    move_name, move_dur = random.choice(candidates)

                # Determine cycles
                max_cycles = int((remaining + tolerance) / move_dur)
                cycles = random.randint(1, min(3, max_cycles)) if max_cycles > 1 else 1

                sequence.append({'move': move_name, 'cycles': cycles})
                current_duration += move_dur * cycles
                used_moves.add(move_name)

            # Validate solution
            actual_duration = self.calculate_sequence_duration(sequence)
            if abs(actual_duration - target_duration) <= tolerance and len(sequence) > 0:
                # Calculate variety score (higher = more different moves)
                variety_score = len(used_moves) / len(sequence) if sequence else 0

                solutions.append({
                    'sequence': sequence,
                    'duration': actual_duration,
                    'move_count': len(sequence),
                    'variety_score': variety_score,
                    'error': abs(actual_duration - target_duration)
                })

            if len(solutions) >= num_solutions:
                break

        # Sort by error (best fit first)
        solutions.sort(key=lambda x: x['error'])

        return {
            'solutions': solutions[:num_solutions],
            'target_duration': target_duration,
            'tolerance': tolerance,
            'found': len(solutions)
        }

    def submit_choreography(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Submit the final choreography sequence.
        Call this when you're confident the sequence is complete and valid.

        Args:
            sequence: Final choreography sequence

        Returns:
            Validation results and confirmation
        """
        validation = self.validate_duration(sequence, self.audio_analysis['duration'])

        return {
            'submitted': True,
            'sequence': sequence,
            'validation': validation,
            'message': 'Choreography submitted successfully' if validation['valid'] else 'Warning: Submitted choreography does not pass validation'
        }

    def get_tool_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get tool descriptions for LLM function calling.

        Returns:
            List of tool definitions in Anthropic function calling format
        """
        return [
            {
                "name": "get_move_duration",
                "description": "Get the actual SDK duration for a specific move in seconds. Use this to know exactly how long a move will take.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "move_name": {
                            "type": "string",
                            "description": "Name of the move (e.g., 'groovy_sway_and_roll')"
                        }
                    },
                    "required": ["move_name"]
                }
            },
            {
                "name": "calculate_sequence_duration",
                "description": "Calculate the total duration of a choreography sequence using actual SDK move durations. Returns precise timing.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sequence": {
                            "type": "array",
                            "description": "List of moves, each with 'move' (name) and optional 'cycles' keys",
                            "items": {"type": "object"}
                        }
                    },
                    "required": ["sequence"]
                }
            },
            {
                "name": "solve_duration_constraint",
                "description": "**PRIMARY TOOL** Solves the duration constraint mathematically and returns multiple valid move sequences. Use this to get choreography options that fit the target duration, then evaluate them for artistic merit. This handles all the math - you focus on creative decisions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_duration": {"type": "number", "description": "Target duration in seconds"},
                        "move_type": {"type": "string", "enum": ["dance", "emotion", None], "description": "Optional filter: 'dance', 'emotion', or null for both"},
                        "tolerance": {"type": "number", "description": "Acceptable error in seconds (default 1.5)"},
                        "num_solutions": {"type": "number", "description": "Number of different solutions to generate (default 3)"}
                    },
                    "required": ["target_duration"]
                }
            },
            {
                "name": "get_moves_by_duration",
                "description": "Find all moves within a specific duration range. Useful for finding moves that fit a time gap.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "min_dur": {"type": "number", "description": "Minimum duration in seconds"},
                        "max_dur": {"type": "number", "description": "Maximum duration in seconds"}
                    },
                    "required": ["min_dur", "max_dur"]
                }
            },
            {
                "name": "get_moves_by_type",
                "description": "Get all moves of a specific type. Use 'dance' for rhythmic moves, 'emotion' for expressive moves.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "move_type": {
                            "type": "string",
                            "enum": ["dance", "emotion"],
                            "description": "Type of moves to retrieve"
                        }
                    },
                    "required": ["move_type"]
                }
            },
            {
                "name": "get_move_info",
                "description": "Get comprehensive information about a specific move (duration, type, library, category). Use when you need details about a particular move.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "move_name": {
                            "type": "string",
                            "description": "Name of the move to query"
                        }
                    },
                    "required": ["move_name"]
                }
            },
            {
                "name": "get_music_structure",
                "description": "**CALL THIS FIRST** Get the music structure with labeled segments (intro/verse/chorus/bridge/outro) and per-segment features (energy, spectral characteristics, beats). Use this to understand the song's structure before creating choreography. This enables section-by-section planning instead of whole-track solving.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "suggest_moves_for_context",
                "description": "Get move recommendations based on musical context (BPM, energy, type). Use when you want ideas for what moves would fit the music well.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "bpm_range": {
                            "type": "string",
                            "enum": ["slow", "medium", "fast"],
                            "description": "Tempo range: 'slow' (<100), 'medium' (100-140), 'fast' (>140)"
                        },
                        "energy_range": {
                            "type": "string",
                            "enum": ["low", "moderate", "high"],
                            "description": "Energy level: 'low' (0-0.4), 'moderate' (0.4-0.7), 'high' (>0.7)"
                        },
                        "move_type": {
                            "type": "string",
                            "enum": ["dance", "emotion"],
                            "description": "Type of moves: 'dance' or 'emotion'"
                        },
                        "max_results": {
                            "type": "number",
                            "description": "Maximum number of moves to return (default 10)"
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "validate_duration",
                "description": "Check if a choreography sequence matches the target audio duration. Returns whether it's valid, too long, or too short.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sequence": {
                            "type": "array",
                            "description": "List of moves to validate",
                            "items": {"type": "object"}
                        },
                        "target_duration": {"type": "number", "description": "Target duration in seconds"},
                        "threshold": {"type": "number", "description": "Acceptable error in seconds (default 1.5)"}
                    },
                    "required": ["sequence", "target_duration"]
                }
            },
            {
                "name": "get_vocal_sections",
                "description": "Identify sections of the audio with strong vocal content. Returns timing and characteristics.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_instrumental_sections",
                "description": "Identify instrumental/non-vocal sections. Returns timing, energy, and percussive characteristics.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_audio_duration",
                "description": "Get the actual content duration of the audio file (excluding silent padding).",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_audio_bpm",
                "description": "Get the detected BPM (tempo) of the audio.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "is_vocal_content",
                "description": "Check if the audio has significant vocal content (singing). Returns true if vocals are present.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "is_instrumental_content",
                "description": "Check if the audio is primarily instrumental (no singing). Returns true if mostly instrumental.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "submit_choreography",
                "description": "Submit the final choreography sequence when you're confident it's complete and validated. Call this to finish generation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sequence": {
                            "type": "array",
                            "description": "Final choreography sequence to submit",
                            "items": {"type": "object"}
                        }
                    },
                    "required": ["sequence"]
                },
                "cache_control": {"type": "ephemeral"}
            }
        ]
