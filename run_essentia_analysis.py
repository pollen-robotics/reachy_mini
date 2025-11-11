
import sys
import json
import essentia.standard as es
import numpy as np

# This class is a direct copy of the one from choreography/audio_analyzer.py
# It is self-contained here to avoid import conflicts in the main application.
class AudioAnalyzer:
    """Analyzes audio files to extract features for choreography generation."""

    def __init__(self):
        """Initialize the audio analyzer with Essentia algorithms."""
        pass

    def analyze(self, audio_path):
        """
        Analyze an audio file and extract comprehensive features.
        """
        try:
            audio = es.MonoLoader(filename=audio_path)()
            duration = len(audio) / 44100.0

            bpm, beats, beats_confidence, _, beats_intervals = self._extract_rhythm(audio)
            segments = self._extract_segments(audio, duration)
            danceability = self._extract_danceability(audio)
            energy = self._calculate_choreography_energy(danceability, bpm)
            loudness, dynamic_complexity = self._extract_acoustic_dynamics(audio)
            key, scale, key_strength = self._extract_key(audio)
            spectral = self._extract_spectral_features(audio)
            onset_rate = self._extract_onset_rate(audio)

            analysis = {
                'audio_file': audio_path,
                'duration': float(duration),
                'sample_rate': 44100,
                'bpm': float(bpm),
                'beats': beats.tolist() if isinstance(beats, np.ndarray) else beats,
                'beats_confidence': beats_confidence.tolist() if isinstance(beats_confidence, np.ndarray) else beats_confidence,
                'beats_intervals': beats_intervals.tolist() if isinstance(beats_intervals, np.ndarray) else beats_intervals,
                'beat_count': len(beats) if beats is not None else 0,
                'onset_rate': float(onset_rate) if np.isscalar(onset_rate) else (float(onset_rate[0]) if hasattr(onset_rate, '__len__') and len(onset_rate) > 0 else 0.0),
                'segments': segments,
                'segment_count': len(segments),
                'energy': float(energy),
                'loudness': float(loudness),
                'dynamic_complexity': float(dynamic_complexity),
                'key': key,
                'scale': scale,
                'key_strength': float(key_strength),
                'spectral': spectral,
                'danceability': float(danceability),
            }
            return analysis
        except Exception as e:
            print(f"Error analyzing audio in subprocess: {e}", file=sys.stderr)
            return None

    def _extract_rhythm(self, audio):
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, beats, beats_confidence, _, beats_intervals = rhythm_extractor(audio)
        return bpm, beats, beats_confidence, _, beats_intervals

    def _extract_segments(self, audio, duration):
        try:
            w = es.Windowing(type='hann')
            spectrum = es.Spectrum()
            mfcc = es.MFCC()
            frame_size = 2048
            hop_size = 1024
            mfccs = []
            for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
                spec = spectrum(w(frame))
                mfcc_bands, mfcc_coeffs = mfcc(spec)
                mfccs.append(mfcc_coeffs)
            
            segments = []
            if duration < 60:
                segments = [
                    {'start': 0.0, 'end': duration * 0.2, 'label': 'intro'},
                    {'start': duration * 0.2, 'end': duration, 'label': 'main'}
                ]
            else:
                segments = [
                    {'start': 0.0, 'end': duration * 0.08, 'label': 'intro'},
                    {'start': duration * 0.08, 'end': duration * 0.25, 'label': 'verse'},
                    {'start': duration * 0.25, 'end': duration * 0.45, 'label': 'chorus'},
                    {'start': duration * 0.45, 'end': duration * 0.60, 'label': 'verse'},
                    {'start': duration * 0.60, 'end': duration * 0.80, 'label': 'chorus'},
                    {'start': duration * 0.80, 'end': duration * 0.90, 'label': 'bridge'},
                    {'start': duration * 0.90, 'end': duration, 'label': 'outro'}
                ]
            return segments
        except Exception:
            return [{'start': 0.0, 'end': duration, 'label': 'full'}]

    def _calculate_choreography_energy(self, danceability, bpm):
        energy = danceability
        if bpm > 120:
            energy += min(0.2, (bpm - 120) / 200)
        elif bpm < 90:
            energy += max(-0.2, (90 - bpm) / 150)
        return energy

    def _extract_acoustic_dynamics(self, audio):
        try:
            loudness = es.Loudness()(audio)
            energy_extractor = es.Energy()
            w = es.Windowing(type='hann')
            energies = [energy_extractor(w(frame)) for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024)]
            dynamic_complexity = np.std(energies) / (np.mean(energies) + 1e-6)
            return loudness, dynamic_complexity
        except Exception:
            return 0.0, 0.0

    def _extract_danceability(self, audio):
        try:
            result = es.Danceability()(audio)
            danceability = result[0] if isinstance(result, tuple) else result
            return np.clip(danceability, 0.0, 1.0)
        except Exception:
            return 0.7

    def _extract_key(self, audio):
        try:
            return es.KeyExtractor()(audio)
        except Exception:
            return "Unknown", "Unknown", 0.0

    def _extract_spectral_features(self, audio):
        try:
            w, spectrum, centroid, rolloff, flatness = es.Windowing(type='hann'), es.Spectrum(), es.Centroid(), es.RollOff(), es.Flatness()
            centroids, rolloffs, flatnesses = [], [], []
            for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
                spec = spectrum(w(frame))
                centroids.append(float(centroid(spec)))
                rolloffs.append(float(rolloff(spec)))
                flatnesses.append(float(flatness(spec)))
            return {'centroid': np.mean(centroids), 'rolloff': np.mean(rolloffs), 'flatness': np.mean(flatnesses)}
        except Exception:
            return {'centroid': 0.0, 'rolloff': 0.0, 'flatness': 0.0}

    def _extract_onset_rate(self, audio):
        try:
            result = es.OnsetRate()(audio)
            return result[0] if isinstance(result, tuple) else result
        except Exception:
            return 0.0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_essentia_analysis.py <audio_file_path>", file=sys.stderr)
        sys.exit(1)

    audio_file = sys.argv[1]
    analyzer = AudioAnalyzer()
    analysis_result = analyzer.analyze(audio_file)

    if analysis_result:
        print(json.dumps(analysis_result))
        sys.exit(0)
    else:
        sys.exit(1)
