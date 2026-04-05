import librosa
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from typing import Tuple, Dict, List
import re
from collections import Counter
import language_tool_python
from sklearn.metrics.pairwise import cosine_similarity

class FeatureExtractor:
    """Extracts acoustic (MFCC) and text (BERT) features."""
    
    def __init__(self, bert_model: str = "bert-base-uncased"):
        """
        Initialize feature extractor.
        
        Args:
            bert_model: Pre-trained BERT model name
        """
        print("Loading BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.bert_model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bert_model.to(self.device)
        
        # Initialize grammar checker
        print("Loading grammar checker...")
        try:
            self.grammar_tool = language_tool_python.LanguageTool('en-US')
        except:
            print("Warning: Grammar checker failed to load. Using basic analysis.")
            self.grammar_tool = None
        
        # Common filler words and repetition indicators
        self.filler_words = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'well', 
                            'umm', 'uhh', 'err', 'ahh', 'hmm', 'hm', 'oh', 'ehh'}
        
        # Common function words (articles, prepositions, conjunctions)
        self.function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 
                              'in', 'on', 'at', 'by', 'for', 'with', 'and', 'or', 'but'}
        
        print(f"BERT model loaded on {self.device}")
    
    def extract_mfcc(self, audio_path: str, n_mfcc: int = 40, max_len: int = 100) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio_path: Path to audio file
            n_mfcc: Number of MFCC coefficients
            max_len: Maximum time frames (pad/truncate)
            
        Returns:
            MFCC features as numpy array (n_mfcc, max_len)
        """
        print("Extracting MFCC features...")
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Pad or truncate to fixed length
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :max_len]
        
        return mfcc
    
    def extract_prosodic_features(self, audio_path: str) -> np.ndarray:
        """
        Extract prosodic features (pitch, energy, tempo).
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Prosodic features as numpy array
        """
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Pitch (F0) using piptrack
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        # Calculate pitch statistics safely
        if len(pitch_values) > 0:
            pitch_mean = float(np.mean(pitch_values))
            pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 1 else 0.0
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
        
        # Energy (RMS)
        rms = librosa.feature.rms(y=audio)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        
        # Tempo
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0]) if len(tempo) > 0 else 0.0
        except:
            tempo = 0.0
        
        # Zero crossing rate (voice quality indicator)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))
        
        # Aggregate features - ensure all are scalars
        features = np.array([
            pitch_mean,
            pitch_std,
            rms_mean,
            rms_std,
            tempo,
            zcr_mean,
            zcr_std
        ], dtype=np.float64)
        
        return features
    
    def analyze_grammar(self, text: str) -> Dict:
        """
        Analyze grammatical correctness and structure.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of grammar features
        """
        if not text.strip():
            return {
                'grammar_errors': 0,
                'errors_per_word': 0.0,
                'has_complete_sentences': False,
                'missing_function_words': 0.0,
                'error_types': []
            }
        
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        
        # Check for grammar errors using LanguageTool
        grammar_errors = 0
        error_types = []
        
        if self.grammar_tool:
            try:
                matches = self.grammar_tool.check(text)
                grammar_errors = len(matches)
                error_types = [m.ruleId for m in matches[:10]]  # Get first 10 error types
            except:
                grammar_errors = 0
        
        # Check for complete sentences (basic heuristic)
        sentences = re.split(r'[.!?]+', text)
        complete_sentences = 0
        for sent in sentences:
            sent = sent.strip()
            if sent:
                # Check if sentence has subject and verb (basic check)
                sent_words = sent.lower().split()
                has_verb = any(w in ['is', 'are', 'was', 'were', 'do', 'does', 'did', 
                                    'has', 'have', 'had', 'go', 'went', 'went', 'said', 
                                    'make', 'get', 'come', 'take'] for w in sent_words)
                if has_verb and len(sent_words) >= 3:
                    complete_sentences += 1
        
        has_complete_sentences = complete_sentences > 0
        
        # Check for missing function words (telegraphic speech indicator)
        function_word_count = sum(1 for w in words if w in self.function_words)
        expected_function_ratio = 0.3  # Normal speech has ~30% function words
        actual_function_ratio = function_word_count / word_count if word_count > 0 else 0
        missing_function_words = max(0, expected_function_ratio - actual_function_ratio)
        
        return {
            'grammar_errors': grammar_errors,
            'errors_per_word': grammar_errors / word_count if word_count > 0 else 0,
            'has_complete_sentences': has_complete_sentences,
            'missing_function_words': missing_function_words,
            'function_word_ratio': actual_function_ratio,
            'complete_sentence_ratio': complete_sentences / max(len(sentences), 1),
            'error_types': error_types
        }
    
    def analyze_linguistic_content(self, text: str) -> Dict:
        """
        Analyze linguistic content for coherence and meaningfulness.
        
        Args:
            text: Transcribed text
            
        Returns:
            Dictionary of linguistic features
        """
        text_lower = text.lower().strip()
        
        if not text_lower:
            return {
                'word_count': 0,
                'unique_word_ratio': 0.0,
                'avg_word_length': 0.0,
                'filler_word_ratio': 0.0,
                'repetition_ratio': 0.0,
                'sentence_count': 0,
                'has_meaning': 0.0,
                'content_word_ratio': 0.0,
                'semantic_coherence': 0.0
            }
        
        # Tokenize words
        words = re.findall(r'\b\w+\b', text_lower)
        word_count = len(words)
        
        if word_count == 0:
            return {
                'word_count': 0,
                'unique_word_ratio': 0.0,
                'avg_word_length': 0.0,
                'filler_word_ratio': 0.0,
                'repetition_ratio': 0.0,
                'sentence_count': 0,
                'has_meaning': 0.0,
                'content_word_ratio': 0.0,
                'semantic_coherence': 0.0
            }
        
        # Unique words
        unique_words = set(words)
        unique_ratio = len(unique_words) / word_count
        
        # Average word length
        avg_word_length = np.mean([len(w) for w in words])
        
        # Filler words
        filler_count = sum(1 for w in words if w in self.filler_words)
        filler_ratio = filler_count / word_count
        
        # Content words (non-fillers, non-function words)
        content_word_count = sum(1 for w in words 
                                if w not in self.filler_words and w not in self.function_words)
        content_word_ratio = content_word_count / word_count if word_count > 0 else 0
        
        # Repetition detection
        word_counts = Counter(words)
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        repetition_ratio = repeated_words / word_count if word_count > 0 else 0.0
        
        # Immediate repetition (stuttering, perseveration)
        immediate_repetitions = 0
        for i in range(len(words) - 1):
            if words[i] == words[i+1]:
                immediate_repetitions += 1
        immediate_repetition_ratio = immediate_repetitions / max(word_count - 1, 1)
        
        # Sentence count
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Semantic coherence (basic heuristic)
        # Check if words relate to common topics
        semantic_coherence = min(1.0, content_word_ratio * 1.5)  # Higher content = more coherent
        
        # Meaningfulness heuristic (enhanced)
        meaningfulness_score = (
            unique_ratio * 0.25 +
            min(avg_word_length / 6.0, 1.0) * 0.20 +
            (1.0 - filler_ratio) * 0.20 +
            (1.0 - repetition_ratio) * 0.15 +
            content_word_ratio * 0.20
        )
        
        # Penalize if almost all words are meaningless
        if content_word_count < 2 and word_count > 5:
            meaningfulness_score *= 0.3  # Severe penalty for jargon
        
        return {
            'word_count': word_count,
            'unique_word_ratio': unique_ratio,
            'avg_word_length': avg_word_length,
            'filler_word_ratio': filler_ratio,
            'repetition_ratio': repetition_ratio,
            'immediate_repetition_ratio': immediate_repetition_ratio,
            'sentence_count': sentence_count,
            'has_meaning': meaningfulness_score,
            'content_word_ratio': content_word_ratio,
            'semantic_coherence': semantic_coherence,
            'content_word_count': content_word_count
        }
    
    def extract_bert_embeddings(self, text: str) -> np.ndarray:
        """
        Extract BERT embeddings from text.
        
        Args:
            text: Input text
            
        Returns:
            BERT embeddings as numpy array (768,)
        """
        print("Extracting BERT embeddings...")
        
        if not text.strip():
            return np.zeros(768)
        
        # Tokenize and encode
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings.flatten()
    
    def calculate_semantic_relevance(self, text: str, scenario_keywords: List[str],
                                     scenario_prompt: str) -> Dict:
        """
        Calculate semantic relevance between response and expected scenario.
        
        Args:
            text: User's response text
            scenario_keywords: Expected keywords for the scenario
            scenario_prompt: The original prompt/question
            
        Returns:
            Dictionary with relevance scores
        """
        if not text.strip():
            return {
                'relevance_score': 0.0,
                'keyword_match_ratio': 0.0,
                'semantic_similarity': 0.0,
                'is_relevant': False,
                'matched_keywords': []
            }
        
        text_lower = text.lower()
        
        # 1. Keyword matching (simple but effective)
        matched_keywords = []
        for keyword in scenario_keywords:
            # Check for keyword or its variations
            if keyword in text_lower or any(word.startswith(keyword[:4]) for word in text_lower.split()):
                matched_keywords.append(keyword)
        
        keyword_match_ratio = len(matched_keywords) / len(scenario_keywords) if scenario_keywords else 0.0
        
        # 2. Semantic similarity using BERT embeddings
        prompt_embedding = self.extract_bert_embeddings(scenario_prompt)
        response_embedding = self.extract_bert_embeddings(text)
        
        # Calculate cosine similarity
        prompt_embedding = prompt_embedding.reshape(1, -1)
        response_embedding = response_embedding.reshape(1, -1)
        semantic_similarity = cosine_similarity(prompt_embedding, response_embedding)[0][0]
        
        # 3. Topic coherence check (are words related to each other?)
        words = re.findall(r'\b\w+\b', text_lower)
        content_words = [w for w in words if w not in self.filler_words and w not in self.function_words]
        
        topic_coherence = 0.5  # Default
        if len(content_words) >= 3:
            # Check if content words form coherent topics
            # Higher repetition of similar topics = more coherent
            word_categories = self._categorize_words(content_words)
            max_category_size = max(len(words) for words in word_categories.values()) if word_categories else 0
            topic_coherence = min(1.0, max_category_size / len(content_words))
        
        # 4. Overall relevance score (weighted combination)
        relevance_score = (
            keyword_match_ratio * 0.40 +
            semantic_similarity * 0.40 +
            topic_coherence * 0.20
        )
        
        # 5. Threshold-based classification
        # Normal conversations should score > 0.3 on relevance
        is_relevant = relevance_score > 0.25
        
        return {
            'relevance_score': float(relevance_score),
            'keyword_match_ratio': float(keyword_match_ratio),
            'semantic_similarity': float(semantic_similarity),
            'topic_coherence': float(topic_coherence),
            'is_relevant': is_relevant,
            'matched_keywords': matched_keywords,
            'expected_keywords': scenario_keywords
        }
    
    def _categorize_words(self, words: List[str]) -> Dict[str, List[str]]:
        """
        Categorize words into semantic groups (basic implementation).
        
        Args:
            words: List of words
            
        Returns:
            Dictionary of categories
        """
        categories = {
            'time': ['yesterday', 'today', 'tomorrow', 'morning', 'evening', 'night', 'week', 'month', 'year'],
            'family': ['mother', 'father', 'sister', 'brother', 'family', 'parent', 'child', 'son', 'daughter'],
            'food': ['eat', 'food', 'dinner', 'lunch', 'breakfast', 'meal', 'cook', 'restaurant'],
            'place': ['home', 'house', 'market', 'store', 'shop', 'city', 'place', 'location'],
            'activity': ['go', 'went', 'do', 'did', 'make', 'watch', 'play', 'work', 'visit'],
            'people': ['friend', 'person', 'people', 'someone', 'everyone', 'we', 'they']
        }
        
        categorized = {cat: [] for cat in categories}
        
        for word in words:
            for category, category_words in categories.items():
                if word in category_words or any(word.startswith(cw[:4]) for cw in category_words):
                    categorized[category].append(word)
                    break
        
        return {k: v for k, v in categorized.items() if v}  # Return only non-empty categories
    
    def extract_all_features(self, audio_path: str, text: str, 
                           temporal_features: Dict, 
                           scenario_keywords: List[str] = None,
                           scenario_prompt: str = None) -> Dict:
        """
        Extract all features including acoustic, prosodic, linguistic, grammar, temporal, and relevance.
        
        Args:
            audio_path: Path to audio file
            text: Transcribed text
            temporal_features: Temporal features from audio recorder
            scenario_keywords: Expected keywords for scenario (optional)
            scenario_prompt: Original prompt text (optional)
            
        Returns:
            Dictionary containing all features
        """
        mfcc = self.extract_mfcc(audio_path)
        prosodic = self.extract_prosodic_features(audio_path)
        bert_emb = self.extract_bert_embeddings(text)
        linguistic = self.analyze_linguistic_content(text)
        grammar = self.analyze_grammar(text)
        
        # Calculate semantic relevance if scenario info provided
        relevance = None
        if scenario_keywords and scenario_prompt:
            relevance = self.calculate_semantic_relevance(text, scenario_keywords, scenario_prompt)
        
        return {
            'mfcc': mfcc,
            'prosodic': prosodic,
            'bert_embeddings': bert_emb,
            'linguistic': linguistic,
            'grammar': grammar,
            'temporal': temporal_features,
            'relevance': relevance
        }
