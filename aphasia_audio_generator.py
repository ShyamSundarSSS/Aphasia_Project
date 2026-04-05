"""
Aphasia Audio Sample Generator
Generates realistic simulated audio samples for different types of aphasia
using text-to-speech with customizable parameters.
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import wave
import tempfile

try:
    import pyttsx3
    TTS_AVAILABLE = "pyttsx3"
except ImportError:
    print("Warning: pyttsx3 not available. Install with: pip install pyttsx3")
    TTS_AVAILABLE = None

try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    print("Warning: soundfile/sounddevice not available for audio recording")
    AUDIO_AVAILABLE = False


class AphasiaAudioGenerator:
    """Generates simulated audio samples for different aphasia types."""
    
    # Sample text templates for different aphasia types
    TEMPLATES = {
        'broca': {
            'name': "Broca's Aphasia (Nonfluent)",
            'description': 'Slow, effortful speech with broken phrases and many pauses',
            'samples': [
                "Boy... uh... go... store... yesterday...",
                "I... want... uh... coffee... please...",
                "Mother... come... home... tomorrow... maybe...",
                "Book... on... table... uh... red... book...",
                "Dog... run... park... uh... big... dog...",
                "Father... work... Monday... uh... always...",
                "Car... broken... need... fix... soon...",
                "Phone... where... uh... my... phone...",
            ],
            'speed_range': (50, 80),  # Very slow WPM
            'pause_range': (0.8, 2.0),  # Long pauses
            'filler_probability': 0.4,  # 40% chance of filler words
            'repetition_probability': 0.2  # 20% chance of word repetition
        },
        
        'wernicke': {
            'name': "Wernicke's Aphasia (Fluent)",
            'description': 'Fluent but meaningless speech with jargon and neologisms',
            'samples': [
                "The flanter bloomed the raveling tree quite wonderfully and everyone.",
                "I was thinking about the plixation in the morning strangle yesterday.",
                "My brother went to the sliving place for the grabble situation.",
                "The beautiful glimson was very interesting in the parkle today.",
                "We should consider the mantelation before the frizzle happens tomorrow.",
                "The children were playing with the blendrick in the garden situation.",
                "I believe the transportation will arrive at the skliven station soon.",
                "She told me about the fascinating plendor in the wonderful area.",
            ],
            'speed_range': (130, 160),  # Normal to fast WPM
            'pause_range': (0.1, 0.3),  # Short pauses
            'filler_probability': 0.05,  # Few filler words
            'repetition_probability': 0.0  # No repetitions
        },
        
        'severe': {
            'name': 'Severe Aphasia',
            'description': 'Very limited speech with mostly single words and long pauses',
            'samples': [
                "Uh... yes... uh... water... no... uh...",
                "Home... uh... go... yes... uh... tomorrow...",
                "Mother... uh... no... uh... yes... phone...",
                "Good... uh... food... yes... uh... no...",
                "Boy... uh... uh... yes... go... uh...",
                "Yes... uh... no... water... uh... yes...",
                "Work... uh... tomorrow... no... uh... yes...",
                "Car... uh... yes... uh... broken... no...",
            ],
            'speed_range': (40, 60),  # Extremely slow WPM
            'pause_range': (1.5, 3.5),  # Very long pauses
            'filler_probability': 0.6,  # Frequent filler words
            'repetition_probability': 0.3  # Frequent repetitions
        },
        
        'very_severe': {
            'name': 'Very Severe Aphasia',
            'description': 'Mostly unintelligible with occasional real words',
            'samples': [
                "Na... na... feru... la... uh... home...",
                "Bla... uh... skree... uh... yes... na...",
                "Fru... fru... uh... water... na... bla...",
                "Ske... na... uh... uh... mother... fru...",
                "La... la... uh... home... ske... na...",
                "Bree... uh... na... yes... fru... uh...",
                "Na... skree... uh... go... la... uh...",
                "Fru... bla... uh... no... na... ske...",
            ],
            'speed_range': (30, 50),  # Extremely slow WPM
            'pause_range': (2.0, 4.5),  # Extremely long pauses
            'filler_probability': 0.5,  # Frequent filler words
            'repetition_probability': 0.4  # Very frequent repetitions
        },
        
        'mild': {
            'name': 'Mild Aphasia',
            'description': 'Slight hesitations and word-finding difficulties',
            'samples': [
                "Yesterday I... uh... went to the place, the store, and bought some things.",
                "My family and I had dinner, um, together last night at home.",
                "I was thinking about... uh... going to the park tomorrow maybe.",
                "The weather is nice today and we should... um... go outside later.",
                "I need to call my brother about... uh... the meeting next week.",
                "We were talking about, um, the vacation plans for summer time.",
                "She told me that... uh... the appointment is on Friday morning.",
                "I want to read that book about... um... the interesting history topic.",
            ],
            'speed_range': (110, 130),  # Slightly slow WPM
            'pause_range': (0.3, 0.8),  # Moderate pauses
            'filler_probability': 0.15,  # Some filler words
            'repetition_probability': 0.05  # Rare repetitions
        },
        
        'normal': {
            'name': 'Normal Speech (Control)',
            'description': 'Typical fluent speech without impairment',
            'samples': [
                "Yesterday I went to the store and bought vegetables for dinner.",
                "My family and I had a wonderful time at the beach last weekend.",
                "I enjoy reading books about history and science in my free time.",
                "The weather has been beautiful lately with clear skies and sunshine.",
                "We are planning to visit my parents next month for the holidays.",
                "She completed her project ahead of schedule and received great feedback.",
                "The children were playing happily in the park this afternoon.",
                "I need to schedule an appointment with the doctor next week.",
            ],
            'speed_range': (140, 160),  # Normal WPM
            'pause_range': (0.1, 0.2),  # Natural pauses
            'filler_probability': 0.02,  # Minimal filler words
            'repetition_probability': 0.0  # No repetitions
        }
    }
    
    FILLER_WORDS = ['uh', 'um', 'er', 'ah', 'uhh', 'umm']
    
    def __init__(self, output_dir: str = "generated_aphasia_samples"):
        """
        Initialize the generator.
        
        Args:
            output_dir: Directory to save generated audio files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if TTS_AVAILABLE:
            self.engine = pyttsx3.init()
            # Get available voices
            self.voices = self.engine.getProperty('voices')
        else:
            self.engine = None
            self.voices = []
        
        self.recording = []
        self.sample_rate = 22050  # Standard sample rate for TTS
        
        print(f"Aphasia Audio Generator initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Available aphasia types: {', '.join(self.TEMPLATES.keys())}")
    
    def set_voice(self, gender: str = 'female'):
        """
        Set TTS voice gender.
        
        Args:
            gender: 'male' or 'female'
        """
        if not self.engine or not self.voices:
            return
        
        for voice in self.voices:
            if gender.lower() in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                print(f"Voice set to: {voice.name}")
                return
        
        # Default to first voice
        self.engine.setProperty('voice', self.voices[0].id)
        print(f"Voice set to default: {self.voices[0].name}")
    
    def add_pauses_and_fillers(self, text: str, aphasia_type: str) -> List[Dict]:
        """
        Parse text and add pauses and filler words based on aphasia type.
        
        Args:
            text: Original text
            aphasia_type: Type of aphasia
            
        Returns:
            List of speech segments with timing information
        """
        config = self.TEMPLATES[aphasia_type]
        words = text.split()
        segments = []
        
        for i, word in enumerate(words):
            # Add filler word before current word sometimes
            if random.random() < config['filler_probability']:
                filler = random.choice(self.FILLER_WORDS)
                segments.append({
                    'type': 'text',
                    'content': filler
                })
                # Pause after filler
                pause_duration = random.uniform(config['pause_range'][0] * 0.5, config['pause_range'][1] * 0.5)
                segments.append({
                    'type': 'pause',
                    'duration': pause_duration
                })
            
            # Add word repetition sometimes
            if random.random() < config['repetition_probability']:
                # Repeat word 2-3 times
                repeat_count = random.randint(2, 3)
                for _ in range(repeat_count):
                    segments.append({
                        'type': 'text',
                        'content': word
                    })
                    segments.append({
                        'type': 'pause',
                        'duration': random.uniform(0.2, 0.4)
                    })
            else:
                # Add the word normally
                segments.append({
                    'type': 'text',
                    'content': word
                })
            
            # Add pause after word (except last word)
            if i < len(words) - 1:
                # Pause duration based on aphasia type
                if '...' in text:  # Check for ellipsis in original text
                    pause_duration = random.uniform(config['pause_range'][0], config['pause_range'][1])
                else:
                    pause_duration = random.uniform(
                        config['pause_range'][0] * 0.3,
                        config['pause_range'][1] * 0.5
                    )
                
                segments.append({
                    'type': 'pause',
                    'duration': pause_duration
                })
        
        return segments
    
    def generate_sample(self, aphasia_type: str, sample_index: int = None,
                       custom_text: str = None, voice_gender: str = 'female') -> str:
        """
        Generate a single audio sample with proper file saving.
        
        Args:
            aphasia_type: Type of aphasia
            sample_index: Index of sample text to use
            custom_text: Custom text instead of template
            voice_gender: 'male' or 'female'
            
        Returns:
            Path to generated audio file
        """
        if aphasia_type not in self.TEMPLATES:
            raise ValueError(f"Unknown aphasia type: {aphasia_type}")
        
        if not self.engine:
            raise RuntimeError("TTS engine not available")
        
        config = self.TEMPLATES[aphasia_type]
        
        # Select text
        if custom_text:
            text = custom_text
        else:
            if sample_index is None:
                sample_index = random.randint(0, len(config['samples']) - 1)
            text = config['samples'][sample_index]
        
        # Set voice
        self.set_voice(voice_gender)
        
        # Set speech rate
        speed = random.randint(config['speed_range'][0], config['speed_range'][1])
        self.engine.setProperty('rate', speed)
        self.engine.setProperty('volume', 0.9)
        
        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_index = sample_index if sample_index is not None else 'custom'
        filename = f"{aphasia_type}_{timestamp}_sample{safe_index}.wav"
        filepath = self.output_dir / filename
        
        print(f"\nGenerating: {config['name']}")
        print(f"Text: {text[:100]}...")
        print(f"Speed: {speed} WPM")
        print(f"Output: {filepath}")
        
        # FIXED: Save directly to file using pyttsx3's save_to_file
        self.engine.save_to_file(text, str(filepath))
        self.engine.runAndWait()
        
        # If direct save failed, use alternative method with pauses
        if not filepath.exists() or filepath.stat().st_size < 1000:
            print("Using alternative generation method with simulation...")
            self._generate_with_simulation(text, aphasia_type, str(filepath))
        
        if filepath.exists():
            print(f"✓ Generated: {filepath.name} ({filepath.stat().st_size / 1024:.1f} KB)")
            return str(filepath)
        else:
            print(f"✗ Failed to generate file")
            return ""
    
    def _generate_with_simulation(self, text: str, aphasia_type: str, filepath: str):
        """
        Generate audio with aphasia simulation using system audio recording.
        This speaks the text and you can record it with the training GUI.
        
        Args:
            text: Text to speak
            aphasia_type: Type of aphasia
            filepath: Output file path
        """
        print("\n⚠️  Direct file save not supported by this TTS engine.")
        print("📢 ALTERNATIVE METHOD:")
        print(f"   1. The text will be spoken aloud now")
        print(f"   2. Use your training GUI to record it")
        print(f"   3. Or use system audio recorder to capture output")
        print(f"\n   Press Enter when ready to hear the speech...")
        
        input()  # Wait for user
        
        config = self.TEMPLATES[aphasia_type]
        segments = self.add_pauses_and_fillers(text, aphasia_type)
        
        print("\n🔊 Speaking now...")
        
        # Speak with pauses
        for segment in segments:
            if segment['type'] == 'text':
                self.engine.say(segment['content'])
                self.engine.runAndWait()
            elif segment['type'] == 'pause':
                time.sleep(segment['duration'])
        
        print("✓ Speech completed!")
        print(f"\nTo save this audio:")
        print(f"1. Run the training GUI: python gui_trainer.py")
        print(f"2. Click 'Start Recording'")
        print(f"3. Run this generator again and let it speak")
        print(f"4. The training GUI will capture and save the audio\n")
    
    def generate_batch(self, aphasia_types: List[str] = None,
                      samples_per_type: int = 3,
                      voice_gender: str = 'female') -> Dict[str, List[str]]:
        """Generate multiple samples for specified aphasia types."""
        if aphasia_types is None:
            aphasia_types = list(self.TEMPLATES.keys())
        
        results = {}
        
        print(f"\n{'='*60}")
        print(f"BATCH GENERATION")
        print(f"{'='*60}")
        print(f"Types: {', '.join(aphasia_types)}")
        print(f"Samples per type: {samples_per_type}")
        print(f"Voice: {voice_gender}")
        print(f"{'='*60}\n")
        
        for aphasia_type in aphasia_types:
            print(f"\n--- Generating {self.TEMPLATES[aphasia_type]['name']} ---")
            results[aphasia_type] = []
            
            for i in range(samples_per_type):
                try:
                    # Get the full text, don't truncate
                    config = self.TEMPLATES[aphasia_type]
                    sample_idx = i % len(config['samples'])
                    full_text = config['samples'][sample_idx]
                    
                    print(f"\nSample {i+1}/{samples_per_type}")
                    print(f"Full text: {full_text}")
                    
                    filepath = self.generate_sample(
                        aphasia_type,
                        sample_index=sample_idx,
                        voice_gender=voice_gender
                    )
                    
                    if filepath:
                        results[aphasia_type].append(filepath)
                    
                    time.sleep(1.0)  # Delay between generations
                
                except Exception as e:
                    print(f"✗ Error generating sample: {e}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\n{'='*60}")
        print(f"BATCH GENERATION COMPLETE")
        print(f"Total samples generated: {sum(len(files) for files in results.values())}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return results
    
    def list_templates(self):
        """Display all available aphasia templates with descriptions."""
        print(f"\n{'='*60}")
        print("AVAILABLE APHASIA TEMPLATES")
        print(f"{'='*60}\n")
        
        for key, config in self.TEMPLATES.items():
            print(f"Type: {key}")
            print(f"Name: {config['name']}")
            print(f"Description: {config['description']}")
            print(f"Speed Range: {config['speed_range'][0]}-{config['speed_range'][1]} WPM")
            print(f"Sample: \"{config['samples'][0]}\"")
            print()
        
        print(f"{'='*60}\n")


def main():
    """Main function demonstrating usage."""
    print("Aphasia Audio Sample Generator")
    print("=" * 60)
    
    # Initialize generator
    generator = AphasiaAudioGenerator()
    
    # List available templates
    generator.list_templates()
    
    # Example 1: Generate single sample
    print("\nExample 1: Generating single Broca's aphasia sample...")
    generator.generate_sample('broca', sample_index=0, voice_gender='female')
    
    # Example 2: Generate batch for specific types
    print("\nExample 2: Generating batch samples...")
    results = generator.generate_batch(
        aphasia_types=['broca', 'wernicke', 'severe'],
        samples_per_type=2,
        voice_gender='female'
    )
    
    # Example 3: Generate with custom text
    print("\nExample 3: Generating with custom text...")
    custom_text = "I... uh... want... go... home... please..."
    generator.generate_sample('severe', custom_text=custom_text, voice_gender='female')
    
    print("\n" + "="*60)
    print("Generation complete!")
    print(f"Check the '{generator.output_dir}' directory for generated files.")
    print("="*60)


if __name__ == "__main__":
    main()
