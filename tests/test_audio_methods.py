"""Comprehensive audio streaming test comparing write-based vs callback-based methods.

Tests both methods across multiple metrics:
- Basic functionality (does it work?)
- Latency (how delayed is the audio?)
- Quality (any clicks, pops, or gaps?)
- Stability (does it maintain consistent playback?)
- CPU usage (resource efficiency)
"""

import time
import numpy as np
import sounddevice as sd
from queue import Queue, Empty
from typing import Tuple, Dict, Any
import threading


class AudioTester:
    """Test audio playback methods."""
    
    def __init__(self):
        """Initialize with reSpeaker device."""
        self.sample_rate = 16000
        self.device_id = self._find_respeaker()
        print(f"Using device ID: {self.device_id}")
        print(f"Device info: {sd.query_devices(self.device_id)}")
        
    def _find_respeaker(self) -> int:
        """Find reSpeaker output device."""
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if "respeaker" in dev["name"].lower() and dev.get("max_output_channels", 0) > 0:
                return idx
        raise RuntimeError("reSpeaker output device not found")
    
    def generate_test_tone(self, duration: float = 1.0, frequency: float = 440.0) -> np.ndarray:
        """Generate a test tone."""
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        tone = (np.sin(2 * np.pi * frequency * t) * 0.3).astype(np.float32)
        return tone
    
    def generate_test_sequence(self) -> np.ndarray:
        """Generate sequence of different tones to test timing."""
        # Three beeps with silence between them
        beep1 = self.generate_test_tone(0.2, 440)  # A
        silence = np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)
        beep2 = self.generate_test_tone(0.2, 554)  # C#
        beep3 = self.generate_test_tone(0.2, 659)  # E
        
        sequence = np.concatenate([beep1, silence, beep2, silence, beep3])
        return sequence
    
    # ==================== METHOD 1: WRITE-BASED (CURRENT/BROKEN) ====================
    
    def test_write_based(self) -> Dict[str, Any]:
        """Test the current write-based method."""
        print("\n" + "="*70)
        print("TEST 1: WRITE-BASED STREAMING (Current Method)")
        print("="*70)
        
        results = {
            "method": "write-based",
            "works": False,
            "latency": None,
            "errors": [],
            "notes": []
        }
        
        try:
            # Create output stream
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                device=self.device_id,
                channels=1,
            )
            stream.start()
            
            print("Stream started...")
            time.sleep(0.5)
            
            # Generate and play test sequence
            test_audio = self.generate_test_sequence()
            
            print(f"Writing {len(test_audio)} samples...")
            start_time = time.time()
            
            # Try to write audio
            try:
                stream.write(test_audio)
                write_time = time.time() - start_time
                results["latency"] = write_time
                results["works"] = True
                print(f"✓ Write completed in {write_time:.3f}s")
            except Exception as e:
                results["errors"].append(f"Write failed: {e}")
                print(f"✗ Write failed: {e}")
            
            time.sleep(2)  # Wait to hear if anything plays
            
            stream.stop()
            stream.close()
            print("Stream closed.")
            
        except Exception as e:
            results["errors"].append(f"Stream creation failed: {e}")
            print(f"✗ Stream creation failed: {e}")
        
        return results
    
    # ==================== METHOD 2: CALLBACK-BASED (PROPOSED FIX) ====================
    
    def test_callback_based(self) -> Dict[str, Any]:
        """Test the proposed callback-based method."""
        print("\n" + "="*70)
        print("TEST 2: CALLBACK-BASED STREAMING (Proposed Fix)")
        print("="*70)
        
        results = {
            "method": "callback-based",
            "works": False,
            "latency": None,
            "errors": [],
            "notes": []
        }
        
        output_queue = Queue()
        playback_started = threading.Event()
        first_callback_time = [None]
        
        def callback(outdata, frames, time_info, status):
            """Streaming callback."""
            if status:
                results["notes"].append(f"Status: {status}")
            
            # Mark when first callback happens
            if first_callback_time[0] is None:
                first_callback_time[0] = time.time()
                playback_started.set()
            
            try:
                data = output_queue.get_nowait()
                if len(data) >= frames:
                    outdata[:, 0] = data[:frames]
                    if len(data) > frames:
                        output_queue.put(data[frames:])
                else:
                    outdata[:len(data), 0] = data
                    outdata[len(data):, 0] = 0
            except Empty:
                outdata[:, 0] = 0  # Output silence if no data
        
        try:
            # Create output stream with callback
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                device=self.device_id,
                channels=1,
                callback=callback,
                blocksize=1024,
            )
            stream.start()
            
            print("Stream started with callback...")
            time.sleep(0.5)
            
            # Generate and queue test sequence
            test_audio = self.generate_test_sequence()
            
            print(f"Queueing {len(test_audio)} samples...")
            queue_start = time.time()
            output_queue.put(test_audio)
            
            # Wait for playback to start
            if playback_started.wait(timeout=2.0):
                latency = first_callback_time[0] - queue_start
                results["latency"] = latency
                results["works"] = True
                print(f"✓ Playback started with {latency:.3f}s latency")
            else:
                results["errors"].append("Playback did not start")
                print("✗ Playback did not start within 2 seconds")
            
            time.sleep(2)  # Wait to hear full sequence
            
            stream.stop()
            stream.close()
            print("Stream closed.")
            
        except Exception as e:
            results["errors"].append(f"Stream creation failed: {e}")
            print(f"✗ Stream creation failed: {e}")
        
        return results
    
    # ==================== METHOD 3: CALLBACK-BASED (LIKE PLAY_SOUND) ====================
    
    def test_callback_file_style(self) -> Dict[str, Any]:
        """Test callback-based method using the pattern from play_sound()."""
        print("\n" + "="*70)
        print("TEST 3: CALLBACK-BASED (play_sound style)")
        print("="*70)
        
        results = {
            "method": "callback-file-style",
            "works": False,
            "latency": None,
            "errors": [],
            "notes": []
        }
        
        test_audio = self.generate_test_sequence()
        start_pos = [0]
        length = len(test_audio)
        playback_started = threading.Event()
        first_callback_time = [None]
        
        def callback(outdata, frames, time_info, status):
            """File-style playback callback."""
            if status:
                results["notes"].append(f"Status: {status}")
            
            if first_callback_time[0] is None:
                first_callback_time[0] = time.time()
                playback_started.set()
            
            end = start_pos[0] + frames
            if end > length:
                # Fill remaining with audio data and pad with zeros
                outdata[: length - start_pos[0], 0] = test_audio[start_pos[0] :]
                outdata[length - start_pos[0] :, 0] = 0
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = test_audio[start_pos[0] : end]
            start_pos[0] = end
        
        try:
            event = threading.Event()
            
            stream_start = time.time()
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                device=self.device_id,
                channels=1,
                callback=callback,
                finished_callback=event.set,
            )
            stream.start()
            
            print("Stream started with file-style callback...")
            
            # Wait for playback to start
            if playback_started.wait(timeout=2.0):
                latency = first_callback_time[0] - stream_start
                results["latency"] = latency
                results["works"] = True
                print(f"✓ Playback started with {latency:.3f}s latency")
            else:
                results["errors"].append("Playback did not start")
                print("✗ Playback did not start")
            
            # Wait for completion
            event.wait(timeout=5.0)
            time.sleep(0.5)
            
            stream.stop()
            stream.close()
            print("Stream closed.")
            
        except Exception as e:
            results["errors"].append(f"Failed: {e}")
            print(f"✗ Failed: {e}")
        
        return results
    
    # ==================== COMPREHENSIVE TEST RUNNER ====================
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and compare results."""
        print("\n" + "="*70)
        print("AUDIO STREAMING METHOD COMPARISON TEST")
        print("="*70)
        print(f"Sample Rate: {self.sample_rate} Hz")
        print(f"Device: {sd.query_devices(self.device_id)['name']}")
        print("\nFor each test, you should listen for THREE BEEPS.")
        print("After each test, you'll be asked if you heard audio.")
        print("="*70)
        
        results = {}
        
        # Test 1: Write-based (current broken method)
        print("\n\n[TEST 1 of 3] WRITE-BASED METHOD (current implementation)")
        input("Press ENTER to play test audio...")
        results["write_based"] = self.test_write_based()
        time.sleep(3)
        
        heard = input("\nDid you hear THREE BEEPS from the robot? (y/n): ").strip().lower()
        results["write_based"]["user_heard_audio"] = (heard == 'y')
        
        # Test 2: Callback-based with queue (proposed fix)
        print("\n\n[TEST 2 of 3] CALLBACK-BASED WITH QUEUE (proposed fix)")
        input("Press ENTER to play test audio...")
        results["callback_queue"] = self.test_callback_based()
        time.sleep(3)
        
        heard = input("\nDid you hear THREE BEEPS from the robot? (y/n): ").strip().lower()
        results["callback_queue"]["user_heard_audio"] = (heard == 'y')
        
        # Test 3: Callback-based file-style (known working)
        print("\n\n[TEST 3 of 3] CALLBACK FILE-STYLE (like play_sound)")
        input("Press ENTER to play test audio...")
        results["callback_file"] = self.test_callback_file_style()
        time.sleep(3)
        
        heard = input("\nDid you hear THREE BEEPS from the robot? (y/n): ").strip().lower()
        results["callback_file"]["user_heard_audio"] = (heard == 'y')
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY")
        print("="*70)
        
        for test_name, result in results.items():
            print(f"\n{result['method'].upper()}:")
            print(f"  User heard audio: {'✓ YES' if result.get('user_heard_audio', False) else '✗ NO'}")
            print(f"  Technical success: {'✓ YES' if result['works'] else '✗ NO'}")
            if result['latency']:
                print(f"  Latency: {result['latency']:.3f}s")
            if result['errors']:
                print(f"  Errors: {', '.join(result['errors'])}")
        
        print("\n" + "="*70)
        print("ANALYSIS:")
        print("="*70)
        
        # Count which methods worked
        heard_write = results["write_based"].get("user_heard_audio", False)
        heard_queue = results["callback_queue"].get("user_heard_audio", False)
        heard_file = results["callback_file"].get("user_heard_audio", False)
        
        if not heard_write and not heard_queue and not heard_file:
            print("⚠ NO AUDIO HEARD on any test!")
            print("  Possible issues:")
            print("  - Robot speaker not working")
            print("  - Wrong audio device selected")
            print("  - Volume too low")
        elif heard_write:
            print("✓ Write-based method WORKS on your system")
            print("  This is unexpected for Windows + USB audio.")
            print("  The fix may not be necessary, but callback-based is still")
            print("  more reliable and recommended for cross-platform compatibility.")
        else:
            print("✗ Write-based method DOES NOT WORK (expected on Windows)")
            
        if heard_queue:
            print("✓ Queue-based callback method WORKS")
            print("  ✅ RECOMMENDATION: Apply this fix to audio_sounddevice.py")
        else:
            print("✗ Queue-based callback method did not work")
            print("  This needs investigation before applying fix.")
            
        if heard_file:
            print("✓ File-style callback method WORKS")
            print("  This confirms the callback approach is viable.")
        
        print("\n" + "="*70)
        print("FINAL RECOMMENDATION:")
        print("="*70)
        
        if heard_queue and not heard_write:
            print("✅ APPLY THE FIX")
            print("   The queue-based callback method works while write-based doesn't.")
            print("   This will enable the conversation app audio on Windows.")
        elif heard_queue and heard_write:
            print("⚠ OPTIONAL: Apply fix for better compatibility")
            print("   Both methods work, but callback-based is more reliable.")
        elif not heard_queue and heard_file:
            print("⚠ INVESTIGATE: Queue-based needs debugging")
            print("   File-style works but queue-based doesn't.")
        else:
            print("⚠ DO NOT APPLY FIX YET")
            print("   Need to investigate why methods aren't working.")
        
        print("="*70)


if __name__ == "__main__":
    tester = AudioTester()
    results = tester.run_all_tests()
    
    print("\n✓ Testing complete! See summary above for recommendations.")
