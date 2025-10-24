"""Audio analysis module using Essentia for choreography generation."""

import essentia.standard as es
import numpy as np
from .segment_analyzer import analyze_segments


class AudioAnalyzer:
    """Analyzes audio files to extract features for choreography generation."""

    def __init__(self):
        """Initialize the audio analyzer with Essentia algorithms."""
        self.mono_loader = None
        self.rhythm_extractor = None

    def analyze(self, audio_path):
        """
        Analyze an audio file and extract comprehensive features.

        Args:
            audio_path: Path to audio file (.mp3, .wav, .flac, etc.)

        Returns:
            dict: Audio analysis containing BPM, beats, segments, mood, etc.
        """
        try:
            # Load audio
            audio = es.MonoLoader(filename=audio_path)()

            # Detect and trim silence at the end
            actual_audio_end = self._detect_audio_end(audio)

            # Get actual content duration (excluding silent tail)
            duration = actual_audio_end
            total_duration = len(audio) / 44100.0  # Full file duration including silence

            # Extract rhythm features
            bpm, beats, beats_confidence, _, beats_intervals = self._extract_rhythm(audio)

            # Extract segments (music structure)
            segments = self._extract_segments(audio_path)

            # Extract danceability (PRIMARY energy metric for choreography!)
            danceability = self._extract_danceability(audio)

            # Derive choreography energy from danceability + BPM
            # Danceability already measures rhythmic strength, tempo appropriateness
            # This is what matters for choosing moves, NOT acoustic loudness
            energy = self._calculate_choreography_energy(danceability, bpm)

            # Extract acoustic dynamics (for reference only, not used for move selection)
            loudness, dynamic_complexity = self._extract_acoustic_dynamics(audio)

            # Extract key and scale
            key, scale, key_strength = self._extract_key(audio)

            # Extract spectral features
            spectral = self._extract_spectral_features(audio)

            # Extract onset density
            onset_rate = self._extract_onset_rate(audio)

            # Extract vocal/instrumental characteristics
            vocal_instrumental = self._extract_vocal_instrumental(audio)

            # Extract harmonic/percussive components
            harmonic_percussive = self._extract_harmonic_percussive(audio)

            # Extract pitch/melody content
            pitch_content = self._extract_pitch_content(audio)

            # Extract timbre characteristics
            timbre = self._extract_timbre(audio)

            # Extract rhythm patterns
            rhythm_patterns = self._extract_rhythm_patterns(audio)

            # Extract dissonance
            dissonance = self._extract_dissonance(audio)

            # Extract tempo stability
            tempo_stability = self._extract_tempo_stability(beats_intervals)

            # Build analysis result
            analysis = {
                'audio_file': audio_path,
                'duration': float(duration),
                'sample_rate': 44100,

                # Rhythm
                'bpm': float(bpm),
                'beats': beats.tolist() if isinstance(beats, np.ndarray) else beats,
                'beats_confidence': beats_confidence.tolist() if isinstance(beats_confidence, np.ndarray) else beats_confidence,
                'beats_intervals': beats_intervals.tolist() if isinstance(beats_intervals, np.ndarray) else beats_intervals,
                'beat_count': len(beats) if beats is not None else 0,
                'onset_rate': float(onset_rate) if np.isscalar(onset_rate) else (float(onset_rate[0]) if len(onset_rate) > 0 else 0.0),

                # Structure
                'segments': segments,
                'segment_count': len(segments),

                # Energy
                'energy': float(energy),
                'loudness': float(loudness),
                'dynamic_complexity': float(dynamic_complexity),

                # Key
                'key': key,
                'scale': scale,
                'key_strength': float(key_strength),

                # Spectral
                'spectral': spectral,

                # Danceability
                'danceability': float(danceability),

                # Vocal/Instrumental
                'vocal_instrumental': vocal_instrumental,

                # Harmonic/Percussive
                'harmonic_percussive': harmonic_percussive,

                # Pitch/Melody
                'pitch_content': pitch_content,

                # Timbre
                'timbre': timbre,

                # Rhythm patterns
                'rhythm_patterns': rhythm_patterns,

                # Dissonance
                'dissonance': float(dissonance),

                # Tempo stability
                'tempo_stability': float(tempo_stability),
            }

            # Print comprehensive debug summary
            print("\n" + "="*60)
            print("AUDIO ANALYSIS SUMMARY")
            print("="*60)
            print(f"File: {audio_path}")
            print(f"Content Duration: {duration:.1f}s (Total file: {total_duration:.1f}s)")

            print(f"\n🎵 CHOREOGRAPHY ENERGY:")
            print(f"  Energy: {energy:.3f} ← PRIMARY metric")
            print(f"  Danceability: {danceability:.3f}")

            print(f"\n📊 RHYTHM:")
            print(f"  BPM: {bpm:.1f}")
            print(f"  Tempo Stability: {tempo_stability:.3f}")
            print(f"  Beat count: {len(beats) if beats is not None else 0}")
            print(f"  Onset rate: {analysis['onset_rate']:.2f} events/s")
            print(f"  Beat loudness ratio: {rhythm_patterns['beat_loudness_ratio']:.3f}")

            print(f"\n🎤 VOCAL/INSTRUMENTAL:")
            print(f"  Vocal probability: {vocal_instrumental['vocal_probability']:.3f}")
            print(f"  Pitch salience: {vocal_instrumental['pitch_salience']:.3f}")
            print(f"  Spectral complexity: {vocal_instrumental['spectral_complexity']:.3f}")

            print(f"\n🎼 HARMONIC/PERCUSSIVE:")
            print(f"  Harmonic ratio: {harmonic_percussive['harmonic_ratio']:.3f}")
            print(f"  Percussive ratio: {harmonic_percussive['percussive_ratio']:.3f}")

            print(f"\n🎹 PITCH/MELODY:")
            print(f"  Average pitch: {pitch_content['average_pitch']:.1f} Hz")
            print(f"  Pitch range: {pitch_content['pitch_range']:.1f} Hz")
            print(f"  Melodic content: {pitch_content['melodic_content']:.3f}")

            print(f"\n🎨 TIMBRE:")
            print(f"  Spectral contrast: {timbre['spectral_contrast']:.3f}")
            print(f"  Inharmonicity: {timbre['inharmonicity']:.3f}")
            print(f"  Entropy: {timbre['entropy']:.3f}")

            print(f"\n🎹 MUSICAL KEY:")
            print(f"  Key: {key} {scale}")
            print(f"  Strength: {key_strength:.3f}")
            print(f"  Dissonance: {dissonance:.3f}")

            print(f"\n🔊 ACOUSTIC DYNAMICS:")
            print(f"  Loudness: {loudness:.1f} dB")
            print(f"  Dynamic complexity: {dynamic_complexity:.3f}")

            print(f"\n📈 SPECTRAL:")
            print(f"  Centroid: {spectral['centroid']:.0f} Hz")
            print(f"  Rolloff: {spectral['rolloff']:.0f} Hz")
            print(f"  Flatness: {spectral['flatness']:.3f}")

            print("="*60 + "\n")

            return analysis

        except Exception as e:
            print(f"Error analyzing audio: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_rhythm(self, audio):
        """Extract BPM and beat positions."""
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        return bpm, beats, beats_confidence, _, beats_intervals

    def _extract_segments(self, audio_path):
        """
        Extract music structure segments using real spectral clustering + Essentia features.

        Returns segments with:
        - Boundaries detected via agglomerative clustering on MFCCs
        - Labels inferred from position and energy patterns
        - Per-segment energy, spectral centroid, spectral rolloff, beat count
        """
        try:
            # Use the new segment_analyzer module
            segments = analyze_segments(audio_path, target_segments=None)
            return segments

        except Exception as e:
            print(f"Error extracting segments: {e}")
            # Fallback to single-segment
            audio = es.MonoLoader(filename=audio_path)()
            duration = len(audio) / 44100.0
            return [{'start': 0.0, 'end': duration, 'label': 'full', 'energy': 0.5,
                     'spectral_centroid': 2000.0, 'spectral_rolloff': 1000.0,
                     'beats_count': 0, 'duration': duration}]

    def _calculate_choreography_energy(self, danceability, bpm):
        """
        Calculate energy metric for choreography based on danceability and BPM.

        This is NOT acoustic energy (loudness/compression).
        This represents how energetic the DANCE should be.

        Args:
            danceability: Essentia danceability score (can be >1.0)
            bpm: Beats per minute

        Returns:
            Energy score 0-1+ (can exceed 1.0 for extremely high energy tracks)
        """
        # Danceability is the primary metric (measures rhythmic strength)
        energy = danceability

        # Boost for high BPM (fast tempo = more energy)
        if bpm > 120:
            # Add up to +0.2 for very fast tempos (160+ BPM)
            bpm_boost = min(0.2, (bpm - 120) / 200)
            energy += bpm_boost
            print(f"[AudioAnalyzer] BPM boost: +{bpm_boost:.3f} (BPM={bpm:.0f})")

        # Reduce for very slow tempos
        elif bpm < 90:
            # Reduce by up to -0.2 for very slow tempos (60 BPM)
            bpm_penalty = max(-0.2, (90 - bpm) / 150)
            energy += bpm_penalty  # bpm_penalty is negative
            print(f"[AudioAnalyzer] BPM penalty: {bpm_penalty:.3f} (BPM={bpm:.0f})")

        print(f"[AudioAnalyzer] Choreography energy: {energy:.3f} (danceability={danceability:.3f}, bpm={bpm:.0f})")

        return energy

    def _extract_acoustic_dynamics(self, audio):
        """
        Extract acoustic dynamics (loudness, dynamic range).

        These are for reference only - NOT used for choreography move selection.
        Use danceability-based energy for that.
        """
        try:
            # Loudness (Stevens power law)
            loudness_extractor = es.Loudness()
            loudness = loudness_extractor(audio)

            # Dynamic complexity (variance in frame energy)
            energy_extractor = es.Energy()
            w = es.Windowing(type='hann')
            energies = []
            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
                energies.append(energy_extractor(w(frame)))

            dynamic_complexity = np.std(energies) / (np.mean(energies) + 1e-6)

            print(f"[AudioAnalyzer] Acoustic dynamics - Loudness: {loudness:.1f}, Dynamic complexity: {dynamic_complexity:.3f}")

            return loudness, dynamic_complexity

        except Exception as e:
            print(f"Error extracting acoustic dynamics: {e}")
            import traceback
            traceback.print_exc()
            return 0.0, 0.0


    def _extract_danceability(self, audio):
        """Extract danceability score."""
        try:
            danceability_extractor = es.Danceability()
            result = danceability_extractor(audio)

            # Handle different return formats
            if isinstance(result, tuple):
                danceability = result[0]
            else:
                danceability = result

            print(f"[AudioAnalyzer] Raw danceability: {danceability:.3f}")

            # Danceability should already be 0-1, but clip to be safe
            danceability = np.clip(danceability, 0.0, 1.0)

            return danceability
        except Exception as e:
            print(f"Error extracting danceability: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: estimate from BPM and energy
            return 0.7  # Default moderate danceability

    def _extract_key(self, audio):
        """Extract musical key and scale."""
        try:
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio)
            return key, scale, strength
        except Exception as e:
            print(f"Error extracting key: {e}")
            return "Unknown", "Unknown", 0.0

    def _extract_spectral_features(self, audio):
        """Extract spectral characteristics."""
        try:
            w = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            centroid = es.Centroid()
            rolloff = es.RollOff()
            flatness = es.Flatness()

            centroids = []
            rolloffs = []
            flatnesses = []

            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
                spec = spectrum(w(frame))
                cent_val = centroid(spec)
                roll_val = rolloff(spec)
                flat_val = flatness(spec)

                # Handle array vs scalar returns
                centroids.append(float(cent_val) if np.isscalar(cent_val) else float(cent_val[0]))
                rolloffs.append(float(roll_val) if np.isscalar(roll_val) else float(roll_val[0]))
                flatnesses.append(float(flat_val) if np.isscalar(flat_val) else float(flat_val[0]))

            return {
                'centroid': float(np.mean(centroids)),  # Brightness
                'rolloff': float(np.mean(rolloffs)),    # Spectral rolloff
                'flatness': float(np.mean(flatnesses))  # Noisiness
            }
        except Exception as e:
            print(f"Error extracting spectral features: {e}")
            import traceback
            traceback.print_exc()
            return {'centroid': 0.0, 'rolloff': 0.0, 'flatness': 0.0}

    def _extract_onset_rate(self, audio):
        """Extract onset density (musical events per second)."""
        try:
            onset_detector = es.OnsetRate()
            result = onset_detector(audio)
            # OnsetRate returns a tuple or single value depending on Essentia version
            if isinstance(result, tuple):
                onset_rate = result[0]
            else:
                onset_rate = result
            return onset_rate
        except Exception as e:
            print(f"Error extracting onset rate: {e}")
            return 0.0

    def _detect_audio_end(self, audio, sample_rate=44100):
        """
        Detect where actual audio content ends (trim silent tail).

        Args:
            audio: Audio samples array
            sample_rate: Sample rate in Hz (default 44100)

        Returns:
            float: Time in seconds where actual content ends
        """
        try:
            # Calculate RMS energy in windows
            window_size = int(sample_rate * 0.1)  # 100ms windows
            hop_size = int(sample_rate * 0.05)    # 50ms hop

            # Calculate RMS for each window
            rms_values = []
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_values.append(rms)

            if not rms_values:
                # Audio too short, return full duration
                return len(audio) / sample_rate

            rms_array = np.array(rms_values)

            # Define silence threshold as 5% of max RMS
            max_rms = np.max(rms_array)
            silence_threshold = max_rms * 0.05

            # Find last window above threshold
            # Scan from end backwards
            for i in range(len(rms_array) - 1, -1, -1):
                if rms_array[i] > silence_threshold:
                    # Found last non-silent window
                    # Convert window index to time
                    end_time = (i * hop_size + window_size) / sample_rate
                    print(f"[AudioAnalyzer] Detected audio content ends at {end_time:.1f}s (file total: {len(audio)/sample_rate:.1f}s)")
                    return end_time

            # If all windows are below threshold, return full duration
            return len(audio) / sample_rate

        except Exception as e:
            print(f"Error detecting audio end: {e}")
            # Fallback to full duration
            return len(audio) / sample_rate

    def _extract_vocal_instrumental(self, audio):
        """Detect vocal vs instrumental content."""
        try:
            # Use pitch salience to detect vocals
            pitch_salience_func = es.PitchSalienceFunction()
            pitch_salience_values = pitch_salience_func(audio)

            # Average pitch salience as vocal indicator
            avg_salience = float(np.mean(pitch_salience_values))

            # Spectral complexity (higher = more instrumental/complex)
            spec_complex = es.SpectralComplexity()
            complexity = float(spec_complex(audio))

            # Zero crossing rate (higher = more noisy/percussive)
            zcr = es.ZeroCrossingRate()
            zcr_value = float(zcr(audio))

            return {
                'pitch_salience': avg_salience,
                'spectral_complexity': complexity,
                'zero_crossing_rate': zcr_value,
                'vocal_probability': min(1.0, avg_salience * 2)  # Heuristic
            }
        except Exception as e:
            print(f"Error extracting vocal/instrumental: {e}")
            return {'pitch_salience': 0.0, 'spectral_complexity': 0.0, 'zero_crossing_rate': 0.0, 'vocal_probability': 0.5}

    def _extract_harmonic_percussive(self, audio):
        """Separate harmonic and percussive components."""
        try:
            # HPSS = Harmonic-Percussive Source Separation
            hpss = es.HPSS()
            harmonic, percussive = hpss(audio)

            # Calculate energy in each component
            harmonic_energy = float(np.sqrt(np.mean(harmonic ** 2)))
            percussive_energy = float(np.sqrt(np.mean(percussive ** 2)))

            total_energy = harmonic_energy + percussive_energy
            if total_energy > 0:
                harmonic_ratio = harmonic_energy / total_energy
                percussive_ratio = percussive_energy / total_energy
            else:
                harmonic_ratio = 0.5
                percussive_ratio = 0.5

            return {
                'harmonic_energy': harmonic_energy,
                'percussive_energy': percussive_energy,
                'harmonic_ratio': harmonic_ratio,
                'percussive_ratio': percussive_ratio
            }
        except Exception as e:
            print(f"Error extracting harmonic/percussive: {e}")
            return {'harmonic_energy': 0.0, 'percussive_energy': 0.0, 'harmonic_ratio': 0.5, 'percussive_ratio': 0.5}

    def _extract_pitch_content(self, audio):
        """Extract pitch and melody characteristics."""
        try:
            # Predominant pitch detection
            pitch_detect = es.PredominantPitchMelodia()
            pitch, pitch_confidence = pitch_detect(audio)

            # Filter out non-pitched regions (pitch = 0)
            valid_pitches = pitch[pitch > 0]

            if len(valid_pitches) > 0:
                avg_pitch = float(np.mean(valid_pitches))
                pitch_range = float(np.max(valid_pitches) - np.min(valid_pitches))
                pitch_variance = float(np.var(valid_pitches))
            else:
                avg_pitch = 0.0
                pitch_range = 0.0
                pitch_variance = 0.0

            avg_confidence = float(np.mean(pitch_confidence))

            return {
                'average_pitch': avg_pitch,
                'pitch_range': pitch_range,
                'pitch_variance': pitch_variance,
                'pitch_confidence': avg_confidence,
                'melodic_content': avg_confidence  # How melodic vs non-pitched
            }
        except Exception as e:
            print(f"Error extracting pitch content: {e}")
            return {'average_pitch': 0.0, 'pitch_range': 0.0, 'pitch_variance': 0.0, 'pitch_confidence': 0.0, 'melodic_content': 0.0}

    def _extract_timbre(self, audio):
        """Extract timbre characteristics (texture/color of sound)."""
        try:
            # Spectral contrast (bright vs dark timbre)
            spec_contrast = es.SpectralContrast()
            contrast = spec_contrast(audio)
            avg_contrast = float(np.mean(contrast))

            # Inharmonicity
            inharmonicity = es.Inharmonicity()
            inharm_value = float(inharmonicity(audio))

            # Odd to even harmonic energy ratio
            odd_even = es.OddToEvenHarmonicEnergyRatio()
            odd_even_ratio = float(odd_even(audio))

            # Spectral entropy (measure of noisiness)
            entropy = es.Entropy()
            entropy_value = float(entropy(audio))

            return {
                'spectral_contrast': avg_contrast,
                'inharmonicity': inharm_value,
                'odd_even_ratio': odd_even_ratio,
                'entropy': entropy_value
            }
        except Exception as e:
            print(f"Error extracting timbre: {e}")
            return {'spectral_contrast': 0.0, 'inharmonicity': 0.0, 'odd_even_ratio': 1.0, 'entropy': 0.0}

    def _extract_rhythm_patterns(self, audio):
        """Extract detailed rhythm pattern characteristics."""
        try:
            # Beat loudness
            beat_loud = es.BeatsLoudness()
            loudness_band_ratio = beat_loud(audio)

            # BPM histogram (tempo distribution)
            bpm_histogram = es.BpmHistogramDescriptors()
            bpm_desc = bpm_histogram(audio)

            return {
                'beat_loudness_ratio': float(loudness_band_ratio) if np.isscalar(loudness_band_ratio) else float(loudness_band_ratio[0]),
                'bpm_histogram_first_peak': float(bpm_desc[0]) if len(bpm_desc) > 0 else 0.0,
                'bpm_histogram_second_peak': float(bpm_desc[1]) if len(bpm_desc) > 1 else 0.0
            }
        except Exception as e:
            print(f"Error extracting rhythm patterns: {e}")
            return {'beat_loudness_ratio': 0.0, 'bpm_histogram_first_peak': 0.0, 'bpm_histogram_second_peak': 0.0}

    def _extract_dissonance(self, audio):
        """Extract dissonance level (consonance vs dissonance)."""
        try:
            diss = es.Dissonance()
            dissonance_value = diss(audio)
            return float(dissonance_value) if np.isscalar(dissonance_value) else float(dissonance_value[0])
        except Exception as e:
            print(f"Error extracting dissonance: {e}")
            return 0.5

    def _extract_tempo_stability(self, beats_intervals):
        """Measure how stable the tempo is (low variance = steady tempo)."""
        try:
            if len(beats_intervals) > 1:
                # Calculate variance of beat intervals
                variance = np.var(beats_intervals)
                # Convert to stability metric (inverse of variance, normalized)
                stability = 1.0 / (1.0 + variance)
                return float(stability)
            return 1.0  # Perfect stability if not enough beats
        except Exception as e:
            print(f"Error extracting tempo stability: {e}")
            return 0.5
