import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import librosa

class AudioRecorder:
    """Handles audio recording from microphone."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate: Sampling rate in Hz (16kHz recommended for speech)
        """
        self.sample_rate = sample_rate
        self.recording = None
    
    def record(self, duration: int = 10) -> np.ndarray:
        """
        Record audio from microphone.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio as numpy array
        """
        print(f"Recording for {duration} seconds... Speak now!")
        self.recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        print("Recording completed!")
        return self.recording.flatten()
    
    def save_audio(self, audio: np.ndarray, filepath: str) -> None:
        """
        Save audio to file.
        
        Args:
            audio: Audio data as numpy array
            filepath: Path to save the audio file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        sf.write(filepath, audio, self.sample_rate)
        print(f"Audio saved to {filepath}")
    
    def record_and_save(self, duration: int = 10, filepath: str = "temp_audio.wav") -> Tuple[np.ndarray, str]:
        """
        Record audio and save to file.
        
        Args:
            duration: Recording duration in seconds
            filepath: Path to save the audio file
            
        Returns:
            Tuple of (audio array, filepath)
        """
        audio = self.record(duration)
        self.save_audio(audio, filepath)
        return audio, filepath
    
    def detect_speech_segments(self, audio: np.ndarray, 
                              top_db: int = 20,
                              frame_length: int = 2048,
                              hop_length: int = 512) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect speech segments and pauses in audio.
        
        Args:
            audio: Audio signal
            top_db: Threshold in dB below reference to consider as silence
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            
        Returns:
            Tuple of (intervals array, segment statistics)
        """
        # Detect non-silent intervals
        intervals = librosa.effects.split(
            audio, 
            top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Calculate statistics
        segment_stats = []
        for i, (start, end) in enumerate(intervals):
            duration = (end - start) / self.sample_rate
            
            # Calculate pause before this segment
            if i > 0:
                prev_end = intervals[i-1][1]
                pause_duration = (start - prev_end) / self.sample_rate
            else:
                pause_duration = 0.0
            
            segment_stats.append({
                'segment_index': i,
                'start_time': start / self.sample_rate,
                'end_time': end / self.sample_rate,
                'duration': duration,
                'pause_before': pause_duration
            })
        
        return intervals, segment_stats
    
    def calculate_temporal_features(self, audio: np.ndarray) -> Dict:
        """
        Calculate temporal features related to speech fluency.
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of temporal features
        """
        intervals, segment_stats = self.detect_speech_segments(audio)
        
        if len(segment_stats) == 0:
            return {
                'num_segments': 0,
                'avg_segment_duration': 0.0,
                'avg_pause_duration': 0.0,
                'speech_rate': 0.0,
                'pause_ratio': 1.0,
                'max_pause': 0.0,
                'segment_variability': 0.0
            }
        
        # Extract durations
        segment_durations = [s['duration'] for s in segment_stats]
        pause_durations = [s['pause_before'] for s in segment_stats if s['pause_before'] > 0]
        
        # Calculate features
        total_speech_time = sum(segment_durations)
        total_audio_time = len(audio) / self.sample_rate
        
        features = {
            'num_segments': len(segment_stats),
            'avg_segment_duration': np.mean(segment_durations),
            'avg_pause_duration': np.mean(pause_durations) if pause_durations else 0.0,
            'speech_rate': len(segment_stats) / total_audio_time if total_audio_time > 0 else 0.0,
            'pause_ratio': (total_audio_time - total_speech_time) / total_audio_time if total_audio_time > 0 else 0.0,
            'max_pause': max(pause_durations) if pause_durations else 0.0,
            'segment_variability': np.std(segment_durations) if len(segment_durations) > 1 else 0.0,
            'speaking_time': total_speech_time,
            'total_time': total_audio_time
        }
        
        return features
