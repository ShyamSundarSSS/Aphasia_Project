import whisper
import torch
import numpy as np
import librosa
from typing import Optional, Dict, List
import warnings

class SpeechToText:
    """Converts speech to text using Whisper model."""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize speech-to-text transcriber.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        print(f"Loading Whisper {model_size} model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=self.device)
        
        # Common filler words/sounds in speech
        self.filler_patterns = {
            'uh', 'uhh', 'uhhh', 'um', 'umm', 'ummm', 
            'er', 'err', 'errr', 'ah', 'ahh', 'ahhh',
            'eh', 'ehh', 'ehhh', 'mm', 'mmm', 'mmmm',
            'hm', 'hmm', 'hmmm', 'oh', 'ohh', 'ohhh'
        }
        
        print(f"Model loaded on {self.device}")
    
    def transcribe(self, audio_path: str, language: Optional[str] = "en") -> dict:
        """
        Transcribe audio file to text with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en' for English)
            
        Returns:
            Dictionary containing transcription results with timestamps
        """
        print("Transcribing audio...")
        try:
            # Try default Whisper transcription with word timestamps
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=False if self.device == "cpu" else True,
                word_timestamps=True,  # Enable word-level timestamps
                verbose=False
            )
        except FileNotFoundError as e:
            # Fallback: Load audio with librosa (doesn't require FFmpeg)
            print("FFmpeg not found. Using librosa fallback...")
            audio, sr = librosa.load(audio_path, sr=16000)
            result = self.model.transcribe(
                audio,
                language=language,
                fp16=False if self.device == "cpu" else True,
                word_timestamps=True,
                verbose=False
            )
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            print("\nNote: If you see FFmpeg errors, please install FFmpeg:")
            print("  Windows: Download from https://www.gyan.dev/ffmpeg/builds/")
            print("  Or use: winget install ffmpeg")
            raise
        
        print(f"Transcription: {result['text']}")
        return result
    
    def get_text(self, audio_path: str) -> str:
        """
        Get transcription text only.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        result = self.transcribe(audio_path)
        return result['text'].strip()
    
    def get_annotated_transcript(self, audio_path: str, pause_threshold: float = 0.5) -> Dict:
        """
        Get transcript with pause annotations, showing fillers as spoken and silent pauses with dots.
        
        Args:
            audio_path: Path to audio file
            pause_threshold: Minimum pause duration in seconds to annotate
            
        Returns:
            Dictionary with annotated transcript and timing information
        """
        result = self.transcribe(audio_path)
        
        annotated_transcript = ""
        pause_info = []
        word_timings = []
        filler_count = 0
        
        # Check if word-level timestamps are available
        if 'segments' in result:
            prev_end = 0.0
            
            for segment in result['segments']:
                if 'words' in segment:
                    for word_info in segment['words']:
                        word = word_info.get('word', '').strip()
                        word_lower = word.lower().replace(',', '').replace('.', '')
                        start = word_info.get('start', 0.0)
                        end = word_info.get('end', 0.0)
                        
                        # Calculate pause before this word
                        pause_duration = start - prev_end
                        
                        # Add pause annotation if significant (SILENT pause - no sound)
                        if pause_duration >= pause_threshold and prev_end > 0:
                            # Use dots for silent pauses (thinking time)
                            if pause_duration >= 3.0:
                                pause_marker = " ...... "  # Very long pause
                            elif pause_duration >= 2.0:
                                pause_marker = " ..... "   # Long pause
                            elif pause_duration >= 1.5:
                                pause_marker = " .... "    # Medium-long pause
                            elif pause_duration >= 1.0:
                                pause_marker = " ... "     # Medium pause
                            elif pause_duration >= 0.7:
                                pause_marker = " .. "      # Short pause
                            else:
                                pause_marker = " . "       # Very short pause
                            
                            annotated_transcript += pause_marker
                            pause_info.append({
                                'position': len(annotated_transcript),
                                'duration': pause_duration,
                                'before_word': word,
                                'type': 'silent'
                            })
                        
                        # Check if word is a filler (spoken hesitation)
                        is_filler = word_lower in self.filler_patterns
                        
                        # Add the word - keep fillers as-is since they were actually spoken
                        if is_filler:
                            # Extend filler representation based on duration
                            word_duration = end - start
                            if word_duration > 0.5:
                                # Elongated filler (uhhhhh, ummmm, etc.)
                                if word_lower.startswith('uh'):
                                    word = 'uhhhh' if word_duration > 0.8 else 'uhhh'
                                elif word_lower.startswith('um'):
                                    word = 'ummmm' if word_duration > 0.8 else 'ummm'
                                elif word_lower.startswith('er'):
                                    word = 'errrr' if word_duration > 0.8 else 'errr'
                                elif word_lower.startswith('ah'):
                                    word = 'ahhhh' if word_duration > 0.8 else 'ahhh'
                            
                            filler_count += 1
                            word_type = 'filler'
                        else:
                            word_type = 'content'
                        
                        annotated_transcript += word + " "
                        
                        # Track word timing
                        word_timings.append({
                            'word': word,
                            'start': start,
                            'end': end,
                            'duration': end - start,
                            'pause_before': pause_duration if prev_end > 0 else 0.0,
                            'type': word_type
                        })
                        
                        prev_end = end
                else:
                    # Fallback: segment-level only
                    text = segment.get('text', '').strip()
                    start = segment.get('start', 0.0)
                    end = segment.get('end', 0.0)
                    
                    pause_duration = start - prev_end
                    if pause_duration >= pause_threshold and prev_end > 0:
                        if pause_duration >= 2.0:
                            pause_marker = " ..... "
                        elif pause_duration >= 1.0:
                            pause_marker = " ... "
                        else:
                            pause_marker = " .. "
                        annotated_transcript += pause_marker
                    
                    annotated_transcript += text + " "
                    prev_end = end
        else:
            # No segments available, return plain text
            annotated_transcript = result.get('text', '')
        
        # Calculate statistics
        total_pauses = [p for p in pause_info if p['type'] == 'silent']
        avg_pause = np.mean([p['duration'] for p in total_pauses]) if total_pauses else 0.0
        
        return {
            'plain_text': result.get('text', '').strip(),
            'annotated_transcript': annotated_transcript.strip(),
            'word_timings': word_timings,
            'pause_info': pause_info,
            'total_pauses': len(total_pauses),
            'avg_pause_duration': avg_pause,
            'filler_count': filler_count,
            'filler_words': [w for w in word_timings if w['type'] == 'filler']
        }
