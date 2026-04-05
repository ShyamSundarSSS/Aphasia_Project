from .audio_recorder import AudioRecorder
from .speech_to_text import SpeechToText
from .feature_extractor import FeatureExtractor
from .model import MultimodalClassifier
from .scenario_generator import ScenarioGenerator
from typing import Dict
import os
import numpy as np

class AphasiaPipeline:
    """End-to-end aphasia detection pipeline."""
    
    def __init__(self, whisper_model: str = "base", bert_model: str = "bert-base-uncased"):
        """
        Initialize the pipeline.
        
        Args:
            whisper_model: Whisper model size
            bert_model: BERT model name
        """
        print("Initializing Aphasia Detection Pipeline...")
        self.recorder = AudioRecorder()
        self.transcriber = SpeechToText(model_size=whisper_model)
        self.feature_extractor = FeatureExtractor(bert_model=bert_model)
        self.classifier = MultimodalClassifier()
        self.scenario_generator = ScenarioGenerator()
        self.classifier.build_models()
        print("Pipeline initialized successfully!\n")
    
    def process_audio_file(self, audio_path: str, scenario: Dict = None) -> Dict:
        """
        Process existing audio file.
        
        Args:
            audio_path: Path to audio file
            scenario: Scenario dictionary (optional)
            
        Returns:
            Dictionary with results
        """
        # Step 1: Transcribe with annotations
        print("Transcribing audio with timing information...")
        transcript_data = self.transcriber.get_annotated_transcript(audio_path, pause_threshold=0.3)
        transcript = transcript_data['plain_text']
        
        # Step 2: Extract temporal features from audio
        print("Analyzing speech patterns...")
        import librosa
        audio_array, sr = librosa.load(audio_path, sr=16000)
        temporal_features = self.recorder.calculate_temporal_features(audio_array)
        
        # Step 3: Extract all features (including scenario relevance if provided)
        scenario_keywords = scenario.get('keywords', []) if scenario else None
        scenario_prompt = scenario.get('prompt', '') if scenario else None
        
        all_features = self.feature_extractor.extract_all_features(
            audio_path, transcript, temporal_features,
            scenario_keywords=scenario_keywords,
            scenario_prompt=scenario_prompt
        )
        
        # Step 4: Predict severity level
        severity_label, severity_level, confidence, detailed_scores = self.classifier.predict(all_features)
        
        return {
            "transcript": transcript,
            "annotated_transcript": transcript_data['annotated_transcript'],
            "word_timings": transcript_data['word_timings'],
            "pause_info": transcript_data['pause_info'],
            "filler_count": transcript_data.get('filler_count', 0),
            "filler_words": transcript_data.get('filler_words', []),
            "severity_label": severity_label,
            "severity_level": severity_level,
            "confidence": confidence,
            "detailed_scores": detailed_scores,
            "temporal_features": temporal_features,
            "linguistic_features": all_features['linguistic'],
            "grammar_features": all_features['grammar'],
            "prosodic_features": all_features['prosodic'],
            "relevance_features": all_features.get('relevance'),
            "audio_path": audio_path
        }
    
    def record_and_predict(self, duration: int = 30, save_path: str = "temp_audio.wav",
                          scenario: Dict = None) -> Dict:
        """
        Record audio and make prediction.
        
        Args:
            duration: Recording duration in seconds (default: 30)
            save_path: Path to save temporary audio
            scenario: Scenario dictionary (optional)
            
        Returns:
            Dictionary with results
        """
        # Display scenario if provided
        if scenario:
            self.scenario_generator.display_scenario(scenario, duration)
            input("Press Enter when you're ready to start recording...")
        
        # Step 1: Record audio
        print("\n=== STEP 1: Recording Audio ===")
        _, audio_path = self.recorder.record_and_save(duration, save_path)
        
        # Step 2: Process the recorded audio
        print("\n=== STEP 2: Processing Audio ===")
        results = self.process_audio_file(audio_path, scenario=scenario)
        
        # Add scenario info if provided
        if scenario:
            results['scenario'] = scenario['title']
        
        return results
    
    def display_results(self, results: Dict):
        """
        Display prediction results with detailed analysis.
        
        Args:
            results: Results dictionary
        """
        print("\n" + "="*60)
        print("APHASIA SEVERITY ASSESSMENT RESULTS")
        print("="*60)
        
        # Scenario
        if 'scenario' in results:
            print(f"\n📋 Scenario: {results['scenario']}")
        
        # Plain Transcript
        print(f"\n📝 Plain Transcript:")
        print(f"   {results['transcript']}")
        
        # Annotated Transcript with Pauses
        if 'annotated_transcript' in results and results['annotated_transcript']:
            print(f"\n🎤 Annotated Transcript:")
            print(f"   Legend: 'uhh/umm/err' = spoken fillers | '. .. ...' = silent pauses")
            print(f"\n   {results['annotated_transcript']}")
            
            # Pause and Filler Statistics
            print(f"\n⏸️  Speech Pattern Analysis:")
            
            # Silent pauses
            if 'pause_info' in results and results['pause_info']:
                silent_pauses = [p for p in results['pause_info'] if p['type'] == 'silent']
                if silent_pauses:
                    print(f"\n   Silent Pauses (thinking time):")
                    print(f"   • Total silent pauses: {len(silent_pauses)}")
                    pause_durations = [p['duration'] for p in silent_pauses]
                    print(f"   • Average pause: {np.mean(pause_durations):.2f}s")
                    print(f"   • Longest pause: {max(pause_durations):.2f}s")
                    
                    # Show significant pauses
                    long_pauses = [p for p in silent_pauses if p['duration'] >= 1.0]
                    if long_pauses:
                        print(f"\n   ⚠️  Significant silent pauses (≥1s):")
                        for p in long_pauses[:5]:  # Show first 5
                            print(f"      • {p['duration']:.2f}s before '{p['before_word']}'")
            
            # Filler words
            if 'filler_count' in results and results['filler_count'] > 0:
                print(f"\n   Spoken Fillers (uhh, umm, err):")
                print(f"   • Total fillers: {results['filler_count']}")
                
                if 'filler_words' in results and results['filler_words']:
                    filler_list = [w['word'] for w in results['filler_words'][:10]]
                    print(f"   • Examples: {', '.join(filler_list)}")
                    
                    # Elongated fillers (word-finding difficulty)
                    elongated = [w for w in results['filler_words'] if w['duration'] > 0.5]
                    if elongated:
                        print(f"   • Elongated fillers (word-finding): {len(elongated)}")
        
        # Word Timing Details
        if 'word_timings' in results and results['word_timings']:
            word_timings = results['word_timings']
            content_words = [w for w in word_timings if w.get('type') != 'filler']
            
            if len(content_words) > 0:
                word_durations = [w['duration'] for w in content_words]
                print(f"\n🗣️  Word Production (content words):")
                print(f"   • Total content words: {len(content_words)}")
                print(f"   • Average word duration: {np.mean(word_durations):.2f}s")
                
                # Detect elongated words (potential hesitations)
                avg_duration = np.mean(word_durations)
                std_duration = np.std(word_durations)
                elongated_words = [w for w in content_words 
                                  if w['duration'] > avg_duration + std_duration]
                
                if elongated_words:
                    print(f"   • Elongated words (hesitation): {len(elongated_words)}")
                    examples = ', '.join([f"{w['word']}({w['duration']:.1f}s)" 
                                        for w in elongated_words[:3]])
                    print(f"     Examples: {examples}")
        
        # Main classification
        print(f"\n{'='*60}")
        print(f"🔍 SEVERITY LEVEL: {results['severity_label']}")
        print(f"📊 CONFIDENCE: {results['confidence']:.2%}")
        
        # Model ensemble information
        if 'detailed_scores' in results:
            detailed = results['detailed_scores']
            
            # Show model type
            model_type = detailed.get('model_type', 'Ensemble (LSTM + LightGBM + XGBoost)')
            print(f"🤖 MODEL: {model_type}")
            
            # Show individual model predictions if available
            if 'lstm_prediction' in detailed:
                print(f"\n📈 Model Ensemble Breakdown:")
                print(f"   • LSTM Prediction: Level {detailed['lstm_prediction']} ({detailed.get('lstm_confidence', 0):.2%})")
                print(f"   • LightGBM Prediction: Level {detailed['lgb_prediction']} ({detailed.get('lgb_confidence', 0):.2%})")
                print(f"   • XGBoost Prediction: Level {detailed['xgb_prediction']} ({detailed.get('xgb_confidence', 0):.2%})")
                print(f"   • Model Agreement: {detailed.get('model_agreement', 0):.2%}")
            
            # Aphasia Type and Brain Region
            if 'aphasia_type' in detailed:
                aphasia_type = detailed['aphasia_type']
                brain_region = detailed['brain_region']
                
                print(f"\n🧠 APHASIA TYPE: {aphasia_type}")
                print(f"📍 AFFECTED BRAIN REGION: {brain_region['primary']}")
                print(f"   Location: {brain_region['location']}")
                if 'characteristics' in brain_region:
                    print(f"   Characteristics: {brain_region['characteristics']}")
        
        print(f"{'='*60}")
        
        # Detailed Analysis
        print("\nDETAILED ANALYSIS:")
        print("-" * 60)
        
        # Temporal/Fluency features
        temporal = results['temporal_features']
        print("\n⏱️  Speech Fluency:")
        print(f"  • Number of speech segments: {temporal['num_segments']}")
        print(f"  • Average pause duration: {temporal['avg_pause_duration']:.2f}s")
        print(f"  • Maximum pause: {temporal['max_pause']:.2f}s")
        print(f"  • Speech rate: {temporal['speech_rate']:.2f} segments/sec")
        print(f"  • Pause ratio: {temporal['pause_ratio']:.2%}")
        print(f"  • Speaking time: {temporal['speaking_time']:.1f}s / {temporal['total_time']:.1f}s")
        
        # Linguistic features
        linguistic = results['linguistic_features']
        print("\n💬 Linguistic Content:")
        print(f"  • Total words: {linguistic['word_count']}")
        print(f"  • Unique word ratio: {linguistic['unique_word_ratio']:.2%}")
        print(f"  • Average word length: {linguistic['avg_word_length']:.1f} characters")
        print(f"  • Filler words ratio: {linguistic['filler_word_ratio']:.2%}")
        print(f"  • Repetition ratio: {linguistic['repetition_ratio']:.2%}")
        print(f"  • Meaningfulness score: {linguistic['has_meaning']:.2%}")
        print(f"  • Sentences: {linguistic['sentence_count']}")
        
        # Grammar Analysis
        if 'grammar_features' in results:
            grammar = results['grammar_features']
            print("\n📖 Grammar & Structure:")
            print(f"  • Grammar errors detected: {grammar['grammar_errors']}")
            print(f"  • Errors per word: {grammar['errors_per_word']:.3f}")
            print(f"  • Complete sentences: {'Yes' if grammar['has_complete_sentences'] else 'No'}")
            print(f"  • Function word usage: {grammar['function_word_ratio']:.2%} (normal: ~30%)")
            print(f"  • Sentence completeness: {grammar['complete_sentence_ratio']:.2%}")
            
            if grammar['grammar_errors'] > 0 and grammar.get('error_types'):
                print(f"  • Error types: {', '.join(grammar['error_types'][:5])}")
        
        # Aphasia indicators
        if 'detailed_scores' in results and results['detailed_scores']:
            scores = results['detailed_scores']
            print("\n🎯 Aphasia Indicators:")
            print(f"  • Fluency impairment: {scores['fluency_score']:.2%}")
            print(f"  • Content impairment: {scores['content_score']:.2%}")
            print(f"  • Word production impairment: {scores['word_production_score']:.2%}")
            print(f"  • Grammar impairment: {scores['grammar_score']:.2%}")
            print(f"  • Overall severity score: {scores['overall_score']:.2%}")
            
            print(f"\n📏 Severity Classification Scale:")
            for level, threshold in scores['severity_thresholds'].items():
                marker = "👉" if level in results['severity_label'] else "  "
                print(f"  {marker} {level}: {threshold}")
        
        print("="*60)
        
        # Clinical Interpretation
        self._display_interpretation(results)
        
        print("="*60)
    
    def _display_interpretation(self, results: Dict):
        """Display clinical interpretation based on severity and type."""
        print("\n🏥 CLINICAL INTERPRETATION:")
        
        severity = results['severity_label']
        detailed = results.get('detailed_scores', {})
        aphasia_type_key = detailed.get('aphasia_type_key', 'normal')
        temporal = results['temporal_features']
        linguistic = results['linguistic_features']
        grammar = results.get('grammar_features', {})
        
        if aphasia_type_key == 'normal':
            print("  ✅ Speech patterns appear typical with:")
            print("     • Normal fluency and pauses")
            print("     • Meaningful and grammatically correct content")
            print("     • Adequate word production")
        
        elif aphasia_type_key == 'broca':
            print("  🔴 BROCA'S APHASIA (Nonfluent) detected:")
            print("     • Anterior brain lesion affecting speech production")
            print("     • Effortful, slow, halting speech")
            print("     • Good comprehension abilities")
            print("     • Telegraphic speech (missing function words)")
            if grammar.get('missing_function_words', 0) > 0.15:
                print(f"     ⚠️  Significant omission of function words")
            if temporal['speech_rate'] < 0.8:
                print(f"     ⚠️  Reduced speech rate")
        
        elif aphasia_type_key == 'wernicke':
            print("  🔵 WERNICKE'S APHASIA (Fluent) detected:")
            print("     • Posterior brain lesion affecting comprehension")
            print("     • Fluent but often meaningless speech")
            print("     • Poor comprehension")
            print("     • May produce jargon or neologisms")
            if linguistic['has_meaning'] < 0.4:
                print(f"     ⚠️  Low meaningful content in speech")
            if grammar.get('grammar_errors', 0) > 5:
                print(f"     ⚠️  Frequent grammatical errors ({grammar['grammar_errors']} detected)")
        
        elif aphasia_type_key == 'global':
            print("  🚨 GLOBAL APHASIA detected:")
            print("     • Extensive brain damage (both anterior and posterior)")
            print("     • Severe impairment in all language modalities")
            print("     • Minimal speech output")
            print("     • Poor comprehension and repetition")
            if linguistic['word_count'] < 5:
                print(f"     ⚠️  Very limited verbal output ({linguistic['word_count']} words)")
        
        elif aphasia_type_key == 'conduction':
            print("  🟡 CONDUCTION APHASIA detected:")
            print("     • Damage to arcuate fasciculus pathway")
            print("     • Fluent speech with good comprehension")
            print("     • Difficulty with repetition")
            print("     • Frequent phonemic paraphasias")
            if linguistic.get('repetition_ratio', 0) > 0.2:
                print(f"     ⚠️  Word repetition difficulties present")
        
        elif aphasia_type_key == 'transcortical_motor':
            print("  🟠 TRANSCORTICAL MOTOR APHASIA detected:")
            print("     • Lesion near Broca's area")
            print("     • Nonfluent speech but preserved repetition")
            print("     • Good comprehension")
            if temporal['num_segments'] < 10:
                print(f"     ⚠️  Reduced speech segments ({temporal['num_segments']})")
        
        elif aphasia_type_key == 'transcortical_sensory':
            print("  🟣 TRANSCORTICAL SENSORY APHASIA detected:")
            print("     • Lesion near Wernicke's area")
            print("     • Fluent speech but poor comprehension")
            print("     • Preserved repetition ability")
        
        elif aphasia_type_key == 'mixed':
            print("  🟤 MIXED TRANSCORTICAL APHASIA detected:")
            print("     • Watershed area damage")
            print("     • Nonfluent with poor comprehension")
            print("     • Preserved repetition (distinguishes from global)")
        
        elif aphasia_type_key == 'anomic':
            print("  🟢 ANOMIC APHASIA detected:")
            print("     • Mildest form of aphasia")
            print("     • Primary difficulty: word-finding")
            print("     • Fluent speech with good comprehension")
            print("     • Circumlocution common")
            if results.get('filler_count', 0) > 3:
                print(f"     ⚠️  Frequent fillers indicating word-finding difficulty")
        
        # Additional interpretation based on relevance (if available)
        if 'relevance_features' in results and results['relevance_features']:
            relevance = results['relevance_features']
            if not relevance['is_relevant']:
                print("\n  ⚠️  ADDITIONAL CONCERN:")
                print(f"     • Speech not clearly related to the given scenario")
                print(f"     • Relevance score: {relevance['relevance_score']:.2%}")
                if relevance['relevance_score'] < 0.15:
                    print("     • May indicate significant comprehension deficit")
                    print("     • Consider Wernicke's or Global Aphasia")
    
    def cleanup(self, audio_path: str):
        """Remove temporary audio file."""
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"\nCleaned up temporary file: {audio_path}")
