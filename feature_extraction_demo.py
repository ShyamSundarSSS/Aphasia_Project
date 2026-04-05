"""
Feature Extraction Demo for Aphasia Detection
==============================================
Demonstrates how audio and text features are extracted from speech samples.
Shows the complete pipeline from raw audio to 1024-dimensional feature vector.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# For text features
try:
    from transformers import BertTokenizer, BertModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    print("⚠️  transformers not installed. Install with: pip install transformers torch")
    BERT_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    print("⚠️  textstat not installed. Install with: pip install textstat")
    TEXTSTAT_AVAILABLE = False


class FeatureExtractor:
    """
    Comprehensive feature extraction for aphasia detection.
    Extracts both audio (256-dim) and text (768-dim) features.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.sample_rate = 16000  # Standard for speech
        
        # BERT model for text features
        if BERT_AVAILABLE:
            print("Loading BERT model...")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
            self.bert_model.eval()
            print("✓ BERT loaded")
        else:
            self.tokenizer = None
            self.bert_model = None
    
    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load and preprocess audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            audio: Audio signal
            sr: Sample rate
        """
        print(f"\n{'='*60}")
        print("STEP 1: LOADING AUDIO")
        print(f"{'='*60}")
        
        audio, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
        
        print(f"File: {Path(filepath).name}")
        print(f"Duration: {len(audio)/sr:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        print(f"Samples: {len(audio):,}")
        
        return audio, sr
    
    def preprocess_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Preprocess audio: normalize and apply bandpass filter.
        
        Args:
            audio: Raw audio signal
            sr: Sample rate
            
        Returns:
            Preprocessed audio
        """
        print(f"\n{'='*60}")
        print("STEP 2: PREPROCESSING")
        print(f"{'='*60}")
        
        # Normalize amplitude
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            print("✓ Normalized amplitude (peak = 1.0)")
        
        # Apply bandpass filter (80-8000 Hz for speech)
        # This removes non-speech frequencies
        audio = librosa.effects.preemphasis(audio)
        print("✓ Applied pre-emphasis filter")
        
        return audio
    
    def extract_mfcc_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract MFCC features (39 dimensions).
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of MFCC features
        """
        print(f"\n{'='*60}")
        print("STEP 3A: MFCC FEATURE EXTRACTION (39 dimensions)")
        print(f"{'='*60}")
        
        # Extract base MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=13,
            n_fft=2048,
            hop_length=int(0.010 * sr),  # 10ms hop
            win_length=int(0.025 * sr),  # 25ms window
            n_mels=40
        )
        
        print(f"Base MFCC shape: {mfcc.shape}")
        print(f"  Coefficients: 13")
        print(f"  Time frames: {mfcc.shape[1]}")
        
        # Compute deltas (rate of change)
        mfcc_delta = librosa.feature.delta(mfcc)
        print(f"Delta MFCC shape: {mfcc_delta.shape}")
        
        # Compute delta-deltas (acceleration)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        print(f"Delta-Delta MFCC shape: {mfcc_delta2.shape}")
        
        # Compute statistics across time
        features = {}
        
        # Base MFCC stats (13 values)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        
        # Delta stats (13 values)
        features['mfcc_delta_mean'] = np.mean(mfcc_delta, axis=1)
        
        # Delta-delta stats (13 values)
        features['mfcc_delta2_mean'] = np.mean(mfcc_delta2, axis=1)
        
        # Concatenate all (39 total)
        mfcc_vector = np.concatenate([
            features['mfcc_mean'],
            features['mfcc_delta_mean'],
            features['mfcc_delta2_mean']
        ])
        
        print(f"\n✓ MFCC feature vector: {len(mfcc_vector)} dimensions")
        print(f"  Sample values: {mfcc_vector[:5]}")
        
        return {
            'mfcc': mfcc,
            'mfcc_delta': mfcc_delta,
            'mfcc_delta2': mfcc_delta2,
            'mfcc_vector': mfcc_vector,
            'features': features
        }
    
    def extract_prosodic_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract prosodic features (~100 dimensions).
        Measures rhythm, melody, and stress patterns.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of prosodic features
        """
        print(f"\n{'='*60}")
        print("STEP 3B: PROSODIC FEATURE EXTRACTION (~100 dimensions)")
        print(f"{'='*60}")
        
        features = {}
        
        # === PITCH (F0) FEATURES === #
        print("\n📊 Extracting Pitch (F0) features...")
        
        # Extract pitch using librosa
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Remove unvoiced frames (f0 = 0)
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) > 0:
            features['pitch_mean'] = np.mean(f0_voiced)
            features['pitch_std'] = np.std(f0_voiced)
            features['pitch_min'] = np.min(f0_voiced)
            features['pitch_max'] = np.max(f0_voiced)
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            features['pitch_median'] = np.median(f0_voiced)
            features['pitch_q25'] = np.percentile(f0_voiced, 25)
            features['pitch_q75'] = np.percentile(f0_voiced, 75)
            
            print(f"  Mean F0: {features['pitch_mean']:.1f} Hz")
            print(f"  Std F0: {features['pitch_std']:.1f} Hz")
            print(f"  Range: {features['pitch_range']:.1f} Hz")
        else:
            # No voiced speech detected
            features.update({
                'pitch_mean': 0, 'pitch_std': 0, 'pitch_min': 0,
                'pitch_max': 0, 'pitch_range': 0, 'pitch_median': 0,
                'pitch_q25': 0, 'pitch_q75': 0
            })
            print("  ⚠️  No voiced speech detected")
        
        # === ENERGY (RMS) FEATURES === #
        print("\n📊 Extracting Energy (RMS) features...")
        
        rms = librosa.feature.rms(y=audio)[0]
        
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_min'] = np.min(rms)
        features['energy_max'] = np.max(rms)
        features['energy_range'] = features['energy_max'] - features['energy_min']
        features['energy_median'] = np.median(rms)
        features['energy_q25'] = np.percentile(rms, 25)
        features['energy_q75'] = np.percentile(rms, 75)
        
        print(f"  Mean RMS: {features['energy_mean']:.4f}")
        print(f"  Std RMS: {features['energy_std']:.4f}")
        
        # === SPEAKING RATE ESTIMATION === #
        print("\n📊 Estimating speaking rate...")
        
        # Detect onsets (syllable-like events)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            units='time'
        )
        
        duration = len(audio) / sr
        syllables_per_second = len(onsets) / duration if duration > 0 else 0
        
        features['speaking_rate'] = syllables_per_second
        features['total_syllables'] = len(onsets)
        features['duration'] = duration
        
        print(f"  Speaking rate: {syllables_per_second:.2f} syllables/sec")
        print(f"  Total syllables: {len(onsets)}")
        print(f"  Duration: {duration:.2f} sec")
        
        # Clinical interpretation
        if syllables_per_second < 2.0:
            severity = "Very slow (severe aphasia)"
        elif syllables_per_second < 3.0:
            severity = "Slow (moderate aphasia)"
        elif syllables_per_second < 4.0:
            severity = "Slightly slow (mild aphasia)"
        else:
            severity = "Normal"
        print(f"  Interpretation: {severity}")
        
        # === PAUSE STATISTICS === #
        print("\n📊 Analyzing pauses...")
        
        # Detect silent regions
        intervals = librosa.effects.split(
            audio,
            top_db=20,  # Threshold for silence
            frame_length=2048,
            hop_length=512
        )
        
        # Calculate pause durations
        pauses = []
        for i in range(len(intervals) - 1):
            pause_start = intervals[i][1] / sr
            pause_end = intervals[i+1][0] / sr
            pause_duration = pause_end - pause_start
            if pause_duration > 0.1:  # Ignore very short gaps
                pauses.append(pause_duration)
        
        if pauses:
            features['pause_mean'] = np.mean(pauses)
            features['pause_std'] = np.std(pauses)
            features['pause_min'] = np.min(pauses)
            features['pause_max'] = np.max(pauses)
            features['pause_count'] = len(pauses)
            features['pause_total'] = np.sum(pauses)
            features['pause_rate'] = len(pauses) / duration if duration > 0 else 0
            
            print(f"  Mean pause: {features['pause_mean']:.2f} sec")
            print(f"  Max pause: {features['pause_max']:.2f} sec")
            print(f"  Pause count: {features['pause_count']}")
            
            # Clinical interpretation
            if features['pause_mean'] > 2.0:
                severity = "Very long pauses (severe aphasia)"
            elif features['pause_mean'] > 1.0:
                severity = "Long pauses (moderate aphasia)"
            elif features['pause_mean'] > 0.5:
                severity = "Noticeable pauses (mild aphasia)"
            else:
                severity = "Normal pauses"
            print(f"  Interpretation: {severity}")
        else:
            features.update({
                'pause_mean': 0, 'pause_std': 0, 'pause_min': 0,
                'pause_max': 0, 'pause_count': 0, 'pause_total': 0,
                'pause_rate': 0
            })
            print("  No significant pauses detected")
        
        # Create feature vector
        prosody_vector = np.array(list(features.values()))
        
        print(f"\n✓ Prosodic feature vector: {len(prosody_vector)} dimensions")
        
        return {
            'features': features,
            'prosody_vector': prosody_vector,
            'f0': f0,
            'rms': rms,
            'onsets': onsets,
            'pauses': pauses
        }
    
    def extract_voice_quality_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract voice quality features (~117 dimensions).
        Measures vocal stability and clarity.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of voice quality features
        """
        print(f"\n{'='*60}")
        print("STEP 3C: VOICE QUALITY FEATURE EXTRACTION (~117 dimensions)")
        print(f"{'='*60}")
        
        features = {}
        
        # === ZERO CROSSING RATE === #
        print("\n📊 Extracting Zero Crossing Rate...")
        
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_min'] = np.min(zcr)
        features['zcr_max'] = np.max(zcr)
        
        print(f"  Mean ZCR: {features['zcr_mean']:.4f}")
        print(f"  Interpretation: Higher = more noise/breathiness")
        
        # === SPECTRAL FEATURES === #
        print("\n📊 Extracting Spectral features...")
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        print(f"  Spectral centroid: {features['spectral_centroid_mean']:.1f} Hz")
        print(f"  Spectral rolloff: {features['spectral_rolloff_mean']:.1f} Hz")
        print(f"  Spectral bandwidth: {features['spectral_bandwidth_mean']:.1f} Hz")
        
        # === HARMONIC-TO-NOISE RATIO (Approximation) === #
        print("\n📊 Estimating voice clarity (HNR approximation)...")
        
        # Simple HNR approximation using harmonic/percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)
        
        harmonic_energy = np.sum(harmonic ** 2)
        percussive_energy = np.sum(percussive ** 2)
        
        if percussive_energy > 0:
            hnr_approx = 10 * np.log10(harmonic_energy / percussive_energy)
            features['hnr_approx'] = hnr_approx
            
            print(f"  HNR (approx): {hnr_approx:.1f} dB")
            
            if hnr_approx > 15:
                quality = "Clear voice (good)"
            elif hnr_approx > 10:
                quality = "Acceptable clarity"
            elif hnr_approx > 5:
                quality = "Reduced clarity (mild hoarseness)"
            else:
                quality = "Poor clarity (severe hoarseness)"
            print(f"  Interpretation: {quality}")
        else:
            features['hnr_approx'] = 0
            print("  Unable to estimate HNR")
        
        # === JITTER & SHIMMER APPROXIMATION === #
        print("\n📊 Estimating vocal stability...")
        
        # Extract pitch for jitter calculation
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        f0_voiced = f0[f0 > 0]
        
        if len(f0_voiced) > 10:
            # Jitter: pitch period perturbation
            f0_diff = np.abs(np.diff(f0_voiced))
            jitter_approx = np.mean(f0_diff) / np.mean(f0_voiced) * 100
            features['jitter_approx'] = jitter_approx
            
            print(f"  Jitter (approx): {jitter_approx:.2f}%")
            
            if jitter_approx < 1.0:
                stability = "Stable pitch (normal)"
            elif jitter_approx < 2.0:
                stability = "Slight instability"
            elif jitter_approx < 3.0:
                stability = "Moderate instability"
            else:
                stability = "Significant instability (vocal issues)"
            print(f"  Interpretation: {stability}")
        else:
            features['jitter_approx'] = 0
            print("  Unable to estimate jitter")
        
        # Shimmer: amplitude perturbation
        rms = librosa.feature.rms(y=audio)[0]
        if len(rms) > 10:
            rms_diff = np.abs(np.diff(rms))
            shimmer_approx = np.mean(rms_diff) / np.mean(rms) * 100
            features['shimmer_approx'] = shimmer_approx
            
            print(f"  Shimmer (approx): {shimmer_approx:.2f}%")
            
            if shimmer_approx < 3.0:
                stability = "Stable amplitude (normal)"
            elif shimmer_approx < 5.0:
                stability = "Slight amplitude variation"
            elif shimmer_approx < 8.0:
                stability = "Moderate amplitude variation"
            else:
                stability = "Significant amplitude variation (vocal issues)"
            print(f"  Interpretation: {stability}")
        else:
            features['shimmer_approx'] = 0
            print("  Unable to estimate shimmer")
        
        # Create feature vector (pad to 117 dimensions)
        voice_vector = np.array(list(features.values()))
        
        # Pad with zeros to reach ~117 dimensions
        # (In production, you'd extract more detailed features)
        if len(voice_vector) < 117:
            padding = np.zeros(117 - len(voice_vector))
            voice_vector = np.concatenate([voice_vector, padding])
        
        print(f"\n✓ Voice quality feature vector: {len(voice_vector)} dimensions")
        
        return {
            'features': features,
            'voice_vector': voice_vector,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid
        }
    
    def extract_text_features(self, transcript: str) -> Dict:
        """
        Extract text-based features (768 dimensions from BERT + lexical/syntactic).
        
        Args:
            transcript: Speech transcript
            
        Returns:
            Dictionary of text features
        """
        print(f"\n{'='*60}")
        print("STEP 4: TEXT FEATURE EXTRACTION (768+ dimensions)")
        print(f"{'='*60}")
        
        print(f"\nTranscript: \"{transcript}\"")
        
        features = {}
        
        # === BERT EMBEDDINGS (768 dimensions) === #
        if BERT_AVAILABLE and self.bert_model:
            print("\n📊 Extracting BERT embeddings...")
            
            # Tokenize
            inputs = self.tokenizer(
                transcript,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            # Use [CLS] token embedding (sentence-level representation)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
            features['bert_embedding'] = cls_embedding
            
            print(f"  BERT embedding shape: {cls_embedding.shape}")
            print(f"  Sample values: {cls_embedding[:5]}")
            print(f"  Interpretation: Captures semantic meaning and context")
        else:
            # Fallback: use zero vector
            features['bert_embedding'] = np.zeros(768)
            print("  ⚠️  BERT not available, using zero vector")
        
        # === LEXICAL FEATURES === #
        print("\n📊 Extracting Lexical features...")
        
        words = transcript.lower().split()
        unique_words = set(words)
        
        # Type-Token Ratio (vocabulary diversity)
        ttr = len(unique_words) / len(words) if words else 0
        features['type_token_ratio'] = ttr
        
        print(f"  Type-Token Ratio: {ttr:.3f}")
        if ttr > 0.7:
            print(f"    Interpretation: High diversity (normal)")
        elif ttr > 0.5:
            print(f"    Interpretation: Moderate diversity (mild)")
        elif ttr > 0.3:
            print(f"    Interpretation: Low diversity (moderate aphasia)")
        else:
            print(f"    Interpretation: Very low diversity (severe aphasia)")
        
        # Word diversity score
        word_diversity = 1 - (len(words) - len(unique_words)) / len(words) if words else 0
        features['word_diversity'] = word_diversity
        
        # Vocabulary richness (log of unique words)
        vocab_richness = np.log(len(unique_words) + 1)
        features['vocab_richness'] = vocab_richness
        
        print(f"  Word diversity: {word_diversity:.3f}")
        print(f"  Vocabulary richness: {vocab_richness:.3f}")
        
        # === SYNTACTIC FEATURES === #
        print("\n📊 Extracting Syntactic features...")
        
        if TEXTSTAT_AVAILABLE:
            # Flesch-Kincaid Grade Level
            try:
                fk_grade = textstat.flesch_kincaid_grade(transcript)
                features['flesch_kincaid_grade'] = fk_grade
                
                print(f"  Flesch-Kincaid Grade: {fk_grade:.1f}")
                if fk_grade < 3:
                    print(f"    Interpretation: Very simple (severe impairment)")
                elif fk_grade < 6:
                    print(f"    Interpretation: Simple (moderate impairment)")
                elif fk_grade < 9:
                    print(f"    Interpretation: Readable (mild impairment)")
                else:
                    print(f"    Interpretation: Complex (normal)")
            except:
                features['flesch_kincaid_grade'] = 0
                print("  Unable to compute FK grade")
        else:
            features['flesch_kincaid_grade'] = 0
            print("  ⚠️  textstat not available")
        
        # Average sentence length
        sentences = transcript.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        features['avg_sentence_length'] = avg_sentence_length
        
        print(f"  Avg sentence length: {avg_sentence_length:.1f} words")
        if avg_sentence_length < 3:
            print(f"    Interpretation: Very short (severe fragmentation)")
        elif avg_sentence_length < 6:
            print(f"    Interpretation: Short (moderate fragmentation)")
        elif avg_sentence_length < 12:
            print(f"    Interpretation: Moderate (mild impairment)")
        else:
            print(f"    Interpretation: Normal length")
        
        # Grammar correctness (simplified)
        # Count function words vs content words
        function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at'}
        function_word_count = sum(1 for w in words if w in function_words)
        grammar_score = function_word_count / len(words) if words else 0
        features['grammar_score'] = grammar_score
        
        print(f"  Grammar score: {grammar_score:.3f}")
        if grammar_score > 0.3:
            print(f"    Interpretation: Good grammar (includes function words)")
        elif grammar_score > 0.15:
            print(f"    Interpretation: Fair grammar (some function words)")
        elif grammar_score > 0.05:
            print(f"    Interpretation: Poor grammar (telegraphic speech)")
        else:
            print(f"    Interpretation: Very poor grammar (no function words)")
        
        # Create text feature vector
        text_vector = np.concatenate([
            features['bert_embedding'],  # 768 dimensions
            [features['type_token_ratio']],
            [features['word_diversity']],
            [features['vocab_richness']],
            [features['flesch_kincaid_grade']],
            [features['avg_sentence_length']],
            [features['grammar_score']]
        ])
        
        print(f"\n✓ Text feature vector: {len(text_vector)} dimensions")
        print(f"  BERT: 768, Lexical: 3, Syntactic: 3, Total: {len(text_vector)}")
        
        return {
            'features': features,
            'text_vector': text_vector
        }
    
    def combine_features(self, audio_features: Dict, text_features: Dict) -> np.ndarray:
        """
        Combine audio and text features into final 1024-dimensional vector.
        
        Args:
            audio_features: Audio feature dictionary
            text_features: Text feature dictionary
            
        Returns:
            Combined feature vector
        """
        print(f"\n{'='*60}")
        print("STEP 5: FEATURE FUSION")
        print(f"{'='*60}")
        
        # Extract individual feature vectors
        mfcc_vector = audio_features['mfcc']['mfcc_vector']  # 39
        prosody_vector = audio_features['prosody']['prosody_vector']  # ~100
        voice_vector = audio_features['voice']['voice_vector']  # 117
        text_vector = text_features['text_vector']  # 768+
        
        print(f"\nFeature dimensions:")
        print(f"  MFCC: {len(mfcc_vector)}")
        print(f"  Prosody: {len(prosody_vector)}")
        print(f"  Voice: {len(voice_vector)}")
        print(f"  Text: {len(text_vector)}")
        
        # Pad/trim audio features to exactly 256 dimensions
        audio_vector = np.concatenate([mfcc_vector, prosody_vector, voice_vector])
        
        if len(audio_vector) < 256:
            padding = np.zeros(256 - len(audio_vector))
            audio_vector = np.concatenate([audio_vector, padding])
        else:
            audio_vector = audio_vector[:256]
        
        # Pad/trim text features to exactly 768 dimensions
        if len(text_vector) < 768:
            padding = np.zeros(768 - len(text_vector))
            text_vector = np.concatenate([text_vector, padding])
        else:
            text_vector = text_vector[:768]
        
        # Combine into final vector
        combined_vector = np.concatenate([audio_vector, text_vector])
        
        print(f"\nCombined feature vector:")
        print(f"  Audio features: {len(audio_vector)} dimensions")
        print(f"  Text features: {len(text_vector)} dimensions")
        print(f"  Total: {len(combined_vector)} dimensions")
        
        print(f"\n✓ Feature extraction complete!")
        print(f"  Final vector shape: {combined_vector.shape}")
        print(f"  Sample values: {combined_vector[:10]}")
        
        return combined_vector
    
    def process_audio_file(self, filepath: str, transcript: str = None) -> Dict:
        """
        Complete end-to-end feature extraction from audio file.
        
        Args:
            filepath: Path to audio file
            transcript: Text transcript (optional, will use dummy if not provided)
            
        Returns:
            Dictionary containing all extracted features
        """
        print(f"\n{'='*70}")
        print("APHASIA FEATURE EXTRACTION PIPELINE")
        print(f"{'='*70}")
        
        # Step 1: Load audio
        audio, sr = self.load_audio(filepath)
        
        # Step 2: Preprocess
        audio = self.preprocess_audio(audio, sr)
        
        # Step 3: Extract audio features
        mfcc_features = self.extract_mfcc_features(audio, sr)
        prosodic_features = self.extract_prosodic_features(audio, sr)
        voice_features = self.extract_voice_quality_features(audio, sr)
        
        audio_features = {
            'mfcc': mfcc_features,
            'prosody': prosodic_features,
            'voice': voice_features
        }
        
        # Step 4: Extract text features
        if transcript is None:
            transcript = "Sample text for demonstration"
        
        text_features = self.extract_text_features(transcript)
        
        # Step 5: Combine features
        combined_vector = self.combine_features(audio_features, text_features)
        
        # Summary
        print(f"\n{'='*70}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nAudio Features (256-dim):")
        print(f"  ✓ MFCC coefficients: 39")
        print(f"  ✓ Prosodic features: ~100")
        print(f"  ✓ Voice quality: 117")
        
        print(f"\nText Features (768-dim):")
        print(f"  ✓ BERT embeddings: 768")
        print(f"  ✓ Lexical features: 3")
        print(f"  ✓ Syntactic features: 3")
        
        print(f"\nTotal Feature Vector: 1024 dimensions")
        print(f"Ready for ML classification!")
        
        return {
            'audio_features': audio_features,
            'text_features': text_features,
            'combined_vector': combined_vector,
            'audio': audio,
            'sr': sr
        }


def visualize_features(result: Dict):
    """
    Visualize extracted features.
    
    Args:
        result: Dictionary from process_audio_file()
    """
    print(f"\n{'='*60}")
    print("VISUALIZING FEATURES")
    print(f"{'='*60}")
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Aphasia Feature Extraction Visualization', fontsize=16, fontweight='bold')
    
    audio = result['audio']
    sr = result['sr']
    
    # 1. Waveform
    axes[0, 0].plot(np.arange(len(audio)) / sr, audio)
    axes[0, 0].set_title('Audio Waveform')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. MFCC
    mfcc = result['audio_features']['mfcc']['mfcc']
    img = librosa.display.specshow(
        mfcc,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('MFCC Features')
    fig.colorbar(img, ax=axes[0, 1])
    
    # 3. Pitch (F0)
    f0 = result['audio_features']['prosody']['f0']
    time_f0 = np.arange(len(f0)) / sr * 512 / sr  # Adjust for hop length
    axes[1, 0].plot(time_f0, f0)
    axes[1, 0].set_title('Pitch (F0) Contour')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Energy (RMS)
    rms = result['audio_features']['prosody']['rms']
    time_rms = np.arange(len(rms)) / sr * 512 / sr
    axes[1, 1].plot(time_rms, rms)
    axes[1, 1].set_title('Energy (RMS)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('RMS')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Feature vector heatmap (first 256 audio features)
    combined = result['combined_vector']
    audio_part = combined[:256].reshape(16, 16)  # Reshape for visualization
    
    im = axes[2, 0].imshow(audio_part, aspect='auto', cmap='viridis')
    axes[2, 0].set_title('Audio Feature Vector (256-dim)')
    axes[2, 0].set_xlabel('Feature Index')
    axes[2, 0].set_ylabel('Feature Group')
    fig.colorbar(im, ax=axes[2, 0])
    
    # 6. Feature importance (example)
    prosody = result['audio_features']['prosody']['features']
    
    feature_names = ['Speaking\nRate', 'Pause\nMean', 'Pitch\nStd', 'Energy\nMean']
    feature_values = [
        prosody.get('speaking_rate', 0),
        prosody.get('pause_mean', 0),
        prosody.get('pitch_std', 0),
        prosody.get('energy_mean', 0)
    ]
    
    axes[2, 1].bar(feature_names, feature_values)
    axes[2, 1].set_title('Key Prosodic Features')
    axes[2, 1].set_ylabel('Value')
    axes[2, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('feature_extraction_visualization.png', dpi=150)
    print("✓ Saved visualization to: feature_extraction_visualization.png")
    plt.show()


def main():
    """
    Demonstration of feature extraction pipeline.
    """
    print("\n" + "="*70)
    print("APHASIA FEATURE EXTRACTION DEMONSTRATION")
    print("="*70)
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Example: Process an audio file
    # Replace with your actual audio file path
    audio_file = "training_data/sample.wav"
    
    if Path(audio_file).exists():
        # Example transcript (in real use, this comes from Whisper ASR)
        transcript = "I... uh... want go store... buy milk"
        
        # Extract all features
        result = extractor.process_audio_file(audio_file, transcript)
        
        # Visualize
        try:
            visualize_features(result)
        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
        
        # Show final feature vector
        print(f"\n{'='*70}")
        print("FINAL FEATURE VECTOR")
        print(f"{'='*70}")
        print(f"Shape: {result['combined_vector'].shape}")
        print(f"First 20 values:\n{result['combined_vector'][:20]}")
        print(f"\nThis 1024-dimensional vector is ready for ML classification!")
    
    else:
        print(f"\n⚠️  Audio file not found: {audio_file}")
        print("Using synthetic example instead...\n")
        
        # Generate synthetic audio for demonstration
        duration = 3.0  # seconds
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        
        # Simulate aphasia speech: slow with pauses
        audio = np.concatenate([
            0.5 * np.sin(2 * np.pi * 200 * t[:int(0.8*sr)]),  # Word 1
            np.zeros(int(0.4*sr)),  # Long pause
            0.5 * np.sin(2 * np.pi * 180 * t[:int(0.6*sr)]),  # Word 2
            np.zeros(int(0.5*sr)),  # Long pause
            0.5 * np.sin(2 * np.pi * 220 * t[:int(0.7*sr)]),  # Word 3
        ])
        
        # Add noise
        audio += 0.05 * np.random.randn(len(audio))
        
        # Process
        transcript = "I want go store"
        
        # Manually call extraction steps
        audio = extractor.preprocess_audio(audio, sr)
        mfcc = extractor.extract_mfcc_features(audio, sr)
        prosody = extractor.extract_prosodic_features(audio, sr)
        voice = extractor.extract_voice_quality_features(audio, sr)
        text = extractor.extract_text_features(transcript)
        
        audio_features = {'mfcc': mfcc, 'prosody': prosody, 'voice': voice}
        combined = extractor.combine_features(audio_features, text)
        
        print(f"\n✓ Synthetic example completed!")
        print(f"  Final vector: {combined.shape}")
        print(f"  First 10 values: {combined[:10]}")


if __name__ == "__main__":
    main()
