import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict
import pickle
from pathlib import Path
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
import joblib

class LSTMModel(nn.Module):
    """LSTM model for MFCC features."""
    
    def __init__(self, input_size: int = 40, hidden_size: int = 128, num_layers: int = 2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, n_mfcc, time_steps) -> (batch, time_steps, n_mfcc)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        # Take last output
        out = self.fc(lstm_out[:, -1, :])
        return self.relu(out)

class MultimodalClassifier:
    """Multimodal classifier for aphasia severity levels."""
    
    # Aphasia severity levels
    SEVERITY_LEVELS = {
        0: "Normal Speech",
        1: "Mild Aphasia",
        2: "Moderate Aphasia",
        3: "Severe Aphasia",
        4: "Very Severe Aphasia"
    }
    
    # Aphasia types based on brain regions
    APHASIA_TYPES = {
        'broca': "Broca's Aphasia (Nonfluent)",
        'wernicke': "Wernicke's Aphasia (Fluent)",
        'global': "Global Aphasia",
        'conduction': "Conduction Aphasia",
        'transcortical_motor': "Transcortical Motor Aphasia",
        'transcortical_sensory': "Transcortical Sensory Aphasia",
        'mixed': "Mixed Transcortical Aphasia",
        'anomic': "Anomic Aphasia",
        'normal': "No Aphasia Detected"
    }
    
    def __init__(self):
        """Initialize multimodal classifier."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lstm_model = None
        self.text_classifier = None
        self.lgb_model = None
        self.xgb_model = None
        self.ensemble_weights = {'lstm': 0.30, 'lgb': 0.35, 'xgb': 0.35}
        self.is_trained = False
    
    def build_models(self):
        """Build LSTM, LightGBM, and XGBoost models."""
        # LSTM for MFCC
        self.lstm_model = LSTMModel().to(self.device)
        
        # Neural network for combined features (5 severity levels)
        self.text_classifier = nn.Sequential(
            nn.Linear(64 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)  # 5-class classification
        ).to(self.device)
        
        # LightGBM classifier
        self.lgb_model = lgb.LGBMClassifier(
            num_leaves=31,
            max_depth=10,
            learning_rate=0.05,
            n_estimators=100,
            objective='multiclass',
            num_class=5,
            random_state=42,
            verbose=-1
        )
        
        # XGBoost classifier
        self.xgb_model = xgb.XGBClassifier(
            max_depth=8,
            learning_rate=0.05,
            n_estimators=100,
            objective='multi:softmax',
            num_class=5,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        print("Models built successfully (LSTM + LightGBM + XGBoost)")
    
    def _extract_combined_features(self, all_features: Dict) -> np.ndarray:
        """
        Extract and combine all features into a single feature vector for tree-based models.
        
        Args:
            all_features: Dictionary containing all extracted features
            
        Returns:
            Combined feature vector as numpy array
        """
        linguistic = all_features['linguistic']
        temporal = all_features['temporal']
        prosodic = all_features['prosodic']
        grammar = all_features['grammar']
        
        # Compile all numerical features into a single vector
        feature_vector = [
            # Temporal features (9)
            temporal['num_segments'],
            temporal['avg_segment_duration'],
            temporal['avg_pause_duration'],
            temporal['speech_rate'],
            temporal['pause_ratio'],
            temporal['max_pause'],
            temporal['segment_variability'],
            temporal['speaking_time'],
            temporal['total_time'],
            
            # Linguistic features (10)
            linguistic['word_count'],
            linguistic['unique_word_ratio'],
            linguistic['avg_word_length'],
            linguistic['filler_word_ratio'],
            linguistic['repetition_ratio'],
            linguistic['immediate_repetition_ratio'],
            linguistic['sentence_count'],
            linguistic['has_meaning'],
            linguistic['content_word_ratio'],
            linguistic['semantic_coherence'],
            
            # Grammar features (5)
            grammar['grammar_errors'],
            grammar['errors_per_word'],
            1.0 if grammar['has_complete_sentences'] else 0.0,
            grammar['missing_function_words'],
            grammar['function_word_ratio'],
            
            # Prosodic features (7)
            prosodic[0],  # pitch_mean
            prosodic[1],  # pitch_std
            prosodic[2],  # rms_mean
            prosodic[3],  # rms_std
            prosodic[4],  # tempo
            prosodic[5],  # zcr_mean
            prosodic[6],  # zcr_std
            
            # Relevance features (if available) (4)
            all_features.get('relevance', {}).get('relevance_score', 0.5),
            all_features.get('relevance', {}).get('keyword_match_ratio', 0.5),
            all_features.get('relevance', {}).get('semantic_similarity', 0.5),
            all_features.get('relevance', {}).get('topic_coherence', 0.5),
        ]
        
        return np.array(feature_vector, dtype=np.float32).reshape(1, -1)
    
    def predict(self, all_features: Dict) -> Tuple[str, int, float, Dict]:
        """
        Predict aphasia severity level using ensemble of models.
        
        Args:
            all_features: Dictionary containing all extracted features
            
        Returns:
            Tuple of (severity_label, severity_level, confidence, detailed_scores)
        """
        if not self.is_trained:
            # Enhanced demo prediction with severity levels
            return self._enhanced_predict_with_severity(all_features)
        
        # Extract combined feature vector for tree-based models
        combined_features = self._extract_combined_features(all_features)
        
        # 1. LSTM + Neural Network Prediction
        self.lstm_model.eval()
        self.text_classifier.eval()
        
        with torch.no_grad():
            # Process MFCC
            mfcc_tensor = torch.FloatTensor(all_features['mfcc']).unsqueeze(0).to(self.device)
            acoustic_features = self.lstm_model(mfcc_tensor)
            
            # Process BERT
            bert_tensor = torch.FloatTensor(all_features['bert_embeddings']).unsqueeze(0).to(self.device)
            
            # Combine features
            combined = torch.cat([acoustic_features, bert_tensor], dim=1)
            
            # Predict
            output = self.text_classifier(combined)
            lstm_probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        
        # 2. LightGBM Prediction
        lgb_probs = self.lgb_model.predict_proba(combined_features)[0]
        
        # 3. XGBoost Prediction
        xgb_probs = self.xgb_model.predict_proba(combined_features)[0]
        
        # 4. Ensemble - Weighted Average
        ensemble_probs = (
            lstm_probs * self.ensemble_weights['lstm'] +
            lgb_probs * self.ensemble_weights['lgb'] +
            xgb_probs * self.ensemble_weights['xgb']
        )
        
        severity_level = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[severity_level])
        
        severity_label = self.SEVERITY_LEVELS[severity_level]
        
        # Add model agreement info
        detailed_scores = {
            'lstm_prediction': int(np.argmax(lstm_probs)),
            'lgb_prediction': int(np.argmax(lgb_probs)),
            'xgb_prediction': int(np.argmax(xgb_probs)),
            'ensemble_prediction': severity_level,
            'model_agreement': self._calculate_agreement(lstm_probs, lgb_probs, xgb_probs),
            'lstm_confidence': float(lstm_probs[severity_level]),
            'lgb_confidence': float(lgb_probs[severity_level]),
            'xgb_confidence': float(xgb_probs[severity_level])
        }
        
        return severity_label, severity_level, confidence, detailed_scores
    
    def _calculate_agreement(self, lstm_probs: np.ndarray, 
                            lgb_probs: np.ndarray, 
                            xgb_probs: np.ndarray) -> float:
        """
        Calculate agreement score between models.
        
        Args:
            lstm_probs: LSTM probability distribution
            lgb_probs: LightGBM probability distribution
            xgb_probs: XGBoost probability distribution
            
        Returns:
            Agreement score (0-1, higher = more agreement)
        """
        lstm_pred = np.argmax(lstm_probs)
        lgb_pred = np.argmax(lgb_probs)
        xgb_pred = np.argmax(xgb_probs)
        
        # Count how many models agree
        predictions = [lstm_pred, lgb_pred, xgb_pred]
        most_common = max(set(predictions), key=predictions.count)
        agreement_count = predictions.count(most_common)
        
        # Agreement score: 1.0 if all agree, 0.67 if 2 agree, 0.33 if none agree
        return agreement_count / 3.0
    
    def _classify_aphasia_type(self, linguistic: Dict, temporal: Dict, 
                               grammar: Dict, overall_score: float) -> str:
        """
        Classify the specific type of aphasia based on symptom patterns.
        
        Based on the clinical decision tree:
        - Fluency (speech rate, pauses)
        - Comprehension (grammar, content)
        - Repetition (immediate repetitions)
        
        Args:
            linguistic: Linguistic features
            temporal: Temporal features
            grammar: Grammar features
            overall_score: Overall aphasia severity score
            
        Returns:
            Aphasia type classification
        """
        # If no significant aphasia, return normal
        if overall_score < 0.20:
            return 'normal'
        
        # 1. ASSESS FLUENCY (primary branching criterion)
        # Nonfluent: slow rate, many pauses, effortful speech
        is_nonfluent = (
            temporal['speech_rate'] < 0.8 or
            temporal['pause_ratio'] > 0.45 or
            temporal['avg_pause_duration'] > 1.2
        )
        
        # 2. ASSESS REPETITION
        has_repetition_difficulty = (
            linguistic['repetition_ratio'] > 0.25 or
            linguistic['immediate_repetition_ratio'] > 0.15
        )
        
        # 3. ASSESS COMPREHENSION (grammar and content)
        has_comprehension_deficit = (
            grammar['errors_per_word'] > 0.3 or
            grammar['missing_function_words'] > 0.15 or
            linguistic['has_meaning'] < 0.4 or
            linguistic['content_word_ratio'] < 0.3
        )
        
        # CLASSIFICATION LOGIC based on clinical patterns
        
        if is_nonfluent:
            # NONFLUENT APHASIAS (Anterior lesions - Broca's area)
            
            if has_comprehension_deficit:
                if has_repetition_difficulty:
                    # Poor fluency + Poor comprehension + Poor repetition
                    return 'global'  # Global Aphasia
                else:
                    # Poor fluency + Poor comprehension + Good repetition
                    return 'mixed'  # Mixed Transcortical
            else:
                # Good comprehension
                if has_repetition_difficulty:
                    # Poor fluency + Good comprehension + Poor repetition
                    return 'broca'  # Broca's Aphasia
                else:
                    # Poor fluency + Good comprehension + Good repetition
                    return 'transcortical_motor'  # Transcortical Motor
        
        else:
            # FLUENT APHASIAS (Posterior lesions - Wernicke's area)
            
            if has_comprehension_deficit:
                if has_repetition_difficulty:
                    # Good fluency + Poor comprehension + Poor repetition
                    # Check if speech is jargon-like (Wernicke's)
                    if linguistic['content_word_ratio'] < 0.35:
                        return 'wernicke'  # Wernicke's Aphasia
                    else:
                        return 'wernicke'  # Still Wernicke's
                else:
                    # Good fluency + Poor comprehension + Good repetition
                    return 'transcortical_sensory'  # Transcortical Sensory
            else:
                # Good comprehension
                if has_repetition_difficulty:
                    # Good fluency + Good comprehension + Poor repetition
                    return 'conduction'  # Conduction Aphasia
                else:
                    # Good fluency + Good comprehension + Good repetition
                    # Mild word-finding difficulty only
                    return 'anomic'  # Anomic Aphasia
    
    def _enhanced_predict_with_severity(self, all_features: Dict) -> Tuple[str, int, float, Dict]:
        """
        Enhanced prediction with severity classification and aphasia type.
        
        Args:
            all_features: Dictionary containing all extracted features
            
        Returns:
            Tuple of (severity_label, severity_level, confidence, detailed_scores)
        """
        linguistic = all_features['linguistic']
        temporal = all_features['temporal']
        prosodic = all_features['prosodic']
        grammar = all_features['grammar']
        relevance = all_features.get('relevance')  # FIX: Properly extract relevance
        
        # Initialize scores
        aphasia_indicators = {}
        
        # 1. Fluency Score (0-1, higher = more impaired)
        pause_score = min(temporal['pause_ratio'] / 0.6, 1.0)
        speech_rate_score = 1.0 - min(temporal['speech_rate'] / 1.5, 1.0)
        segment_score = 1.0 - min(temporal['num_segments'] / 15.0, 1.0)
        max_pause_score = min(temporal['max_pause'] / 3.0, 1.0)
        
        fluency_score = (
            pause_score * 0.3 + 
            speech_rate_score * 0.3 + 
            segment_score * 0.2 +
            max_pause_score * 0.2
        )
        aphasia_indicators['fluency_score'] = fluency_score
        
        # 2. Grammar/Comprehension Score (0-1, higher = more impaired)
        grammar_error_score = min(grammar['errors_per_word'] * 2, 1.0)
        function_word_score = grammar['missing_function_words'] / 0.3
        sentence_structure_score = 1.0 - grammar['complete_sentence_ratio']
        
        grammar_score = (
            grammar_error_score * 0.4 +
            function_word_score * 0.3 +
            sentence_structure_score * 0.3
        )
        aphasia_indicators['grammar_score'] = grammar_score
        
        # 3. Content/Meaningfulness Score (0-1, higher = more impaired)
        meaning_score = 1.0 - linguistic['has_meaning']
        repetition_score = min(linguistic['repetition_ratio'], 1.0)
        filler_score = min(linguistic['filler_word_ratio'] / 0.3, 1.0)
        content_score_component = 1.0 - linguistic['content_word_ratio']
        
        content_score = (
            meaning_score * 0.35 + 
            repetition_score * 0.25 + 
            filler_score * 0.20 +
            content_score_component * 0.20
        )
        aphasia_indicators['content_score'] = content_score
        
        # 4. Semantic Relevance Score (NEW - indicates comprehension/confusion)
        relevance_impairment = 0.5  # Default
        if relevance:
            # Low relevance = high impairment (speaking off-topic = comprehension issue)
            relevance_impairment = 1.0 - relevance['relevance_score']
            
            # Severe penalty for completely irrelevant speech (Wernicke's indicator)
            if not relevance['is_relevant']:
                relevance_impairment = min(1.0, relevance_impairment + 0.3)
            
            # Store relevance details
            aphasia_indicators['relevance_score'] = relevance['relevance_score']
            aphasia_indicators['is_relevant'] = relevance['is_relevant']
            aphasia_indicators['keyword_matches'] = len(relevance['matched_keywords'])
        
        aphasia_indicators['relevance_impairment'] = relevance_impairment
        
        # 5. Word Production Score (0-1, higher = more impaired)
        if linguistic['word_count'] == 0:
            word_production_score = 1.0
        elif linguistic['word_count'] < 3:
            word_production_score = 0.9
        elif linguistic['word_count'] < 8:
            word_production_score = 0.7
        elif linguistic['word_count'] < 15:
            word_production_score = 0.4
        else:
            word_production_score = max(0.0, 1.0 - (linguistic['word_count'] / 50.0))
        
        # Factor in word length
        if linguistic['avg_word_length'] < 2.5:
            word_production_score = min(1.0, word_production_score + 0.3)
        elif linguistic['avg_word_length'] < 3.5:
            word_production_score = min(1.0, word_production_score + 0.15)
        
        aphasia_indicators['word_production_score'] = word_production_score
        
        # 6. Overall Aphasia Severity Score (0-1) - NOW INCLUDES RELEVANCE
        overall_score = (
            fluency_score * 0.22 +
            grammar_score * 0.25 +
            content_score * 0.25 +
            word_production_score * 0.13 +
            relevance_impairment * 0.15  # NEW: Semantic relevance weight
        )
        
        aphasia_indicators['overall_score'] = overall_score
        
        # Classify severity level
        if overall_score < 0.20:
            severity_level = 0  # Normal Speech
            confidence = 0.75 + (0.20 - overall_score) * 0.8
        elif overall_score < 0.40:
            severity_level = 1  # Mild Aphasia
            confidence = 0.70 + abs(overall_score - 0.30) * 0.5
        elif overall_score < 0.60:
            severity_level = 2  # Moderate Aphasia
            confidence = 0.70 + abs(overall_score - 0.50) * 0.5
        elif overall_score < 0.80:
            severity_level = 3  # Severe Aphasia
            confidence = 0.70 + abs(overall_score - 0.70) * 0.5
        else:
            severity_level = 4  # Very Severe Aphasia
            confidence = 0.75 + (overall_score - 0.80) * 0.8
        
        confidence = min(confidence, 0.92)
        
        severity_label = self.SEVERITY_LEVELS[severity_level]
        
        # Classify aphasia type (now considering relevance)
        aphasia_type_key = self._classify_aphasia_type(linguistic, temporal, grammar, 
                                                       overall_score, relevance)
        aphasia_type = self.APHASIA_TYPES[aphasia_type_key]
        
        # Add brain region information
        brain_region = self._get_brain_region(aphasia_type_key)
        
        # Add detailed breakdown
        aphasia_indicators['aphasia_type'] = aphasia_type
        aphasia_indicators['aphasia_type_key'] = aphasia_type_key
        aphasia_indicators['brain_region'] = brain_region
        aphasia_indicators['severity_thresholds'] = {
            'Normal': '< 0.20',
            'Mild': '0.20 - 0.40',
            'Moderate': '0.40 - 0.60',
            'Severe': '0.60 - 0.80',
            'Very Severe': '≥ 0.80'
        }
        aphasia_indicators['model_type'] = 'Rule-based (Demo Mode)'
        
        return severity_label, severity_level, confidence, aphasia_indicators
    
    def _classify_aphasia_type(self, linguistic: Dict, temporal: Dict, 
                               grammar: Dict, overall_score: float,
                               relevance: Dict = None) -> str:
        """
        Classify the specific type of aphasia based on symptom patterns.
        
        Args:
            linguistic: Linguistic features
            temporal: Temporal features
            grammar: Grammar features
            overall_score: Overall aphasia severity score
            relevance: Semantic relevance features (NEW)
            
        Returns:
            Aphasia type classification
        """
        # If no significant aphasia, return normal
        if overall_score < 0.20:
            return 'normal'
        
        # 1. ASSESS FLUENCY
        is_nonfluent = (
            temporal['speech_rate'] < 0.8 or
            temporal['pause_ratio'] > 0.45 or
            temporal['avg_pause_duration'] > 1.2
        )
        
        # 2. ASSESS REPETITION
        has_repetition_difficulty = (
            linguistic['repetition_ratio'] > 0.25 or
            linguistic.get('immediate_repetition_ratio', 0) > 0.15
        )
        
        # 3. ASSESS COMPREHENSION (grammar, content, AND relevance)
        has_comprehension_deficit = (
            grammar['errors_per_word'] > 0.3 or
            grammar['missing_function_words'] > 0.15 or
            linguistic['has_meaning'] < 0.4 or
            linguistic['content_word_ratio'] < 0.3
        )
        
        # NEW: Factor in semantic relevance (strong indicator of comprehension)
        if relevance:
            # Speaking completely off-topic = comprehension deficit
            if not relevance['is_relevant'] or relevance['relevance_score'] < 0.25:
                has_comprehension_deficit = True
        
        # CLASSIFICATION LOGIC
        if is_nonfluent:
            # NONFLUENT APHASIAS
            if has_comprehension_deficit:
                if has_repetition_difficulty:
                    return 'global'
                else:
                    return 'mixed'
            else:
                if has_repetition_difficulty:
                    return 'broca'
                else:
                    return 'transcortical_motor'
        else:
            # FLUENT APHASIAS
            if has_comprehension_deficit:
                if has_repetition_difficulty:
                    # Check for Wernicke's specific patterns
                    if linguistic['content_word_ratio'] < 0.35 or (relevance and not relevance['is_relevant']):
                        return 'wernicke'  # Fluent jargon, off-topic
                    else:
                        return 'wernicke'
                else:
                    return 'transcortical_sensory'
            else:
                if has_repetition_difficulty:
                    return 'conduction'
                else:
                    return 'anomic'
    
    def _get_brain_region(self, aphasia_type_key: str) -> Dict:
        """
        Get affected brain region based on aphasia type.
        
        Args:
            aphasia_type_key: Key for aphasia type
            
        Returns:
            Dictionary with brain region information
        """
        brain_regions = {
            'normal': {
                'primary': 'None',
                'description': 'No significant brain damage affecting language',
                'location': 'Normal'
            },
            'broca': {
                'primary': "Broca's Area (Frontal Lobe)",
                'description': 'Left inferior frontal gyrus - speech production area',
                'location': 'Anterior (Front)',
                'characteristics': 'Nonfluent, effortful speech, good comprehension'
            },
            'wernicke': {
                'primary': "Wernicke's Area (Temporal Lobe)",
                'description': 'Left superior temporal gyrus - language comprehension area',
                'location': 'Posterior (Back)',
                'characteristics': 'Fluent but meaningless speech, poor comprehension'
            },
            'global': {
                'primary': 'Extensive Perisylvian Region',
                'description': "Both Broca's and Wernicke's areas severely damaged",
                'location': 'Anterior + Posterior',
                'characteristics': 'Severe impairment in all language functions'
            },
            'conduction': {
                'primary': 'Arcuate Fasciculus',
                'description': 'Fiber tract connecting Broca\'s and Wernicke\'s areas',
                'location': 'Posterior (Parietal)',
                'characteristics': 'Fluent speech, good comprehension, poor repetition'
            },
            'transcortical_motor': {
                'primary': 'Area near Broca\'s (Supplementary Motor Area)',
                'description': 'Frontal lobe anterior or superior to Broca\'s area',
                'location': 'Anterior (Front)',
                'characteristics': 'Nonfluent, preserved repetition'
            },
            'transcortical_sensory': {
                'primary': "Area near Wernicke's (Posterior)",
                'description': 'Temporal-parietal junction, sparing Wernicke\'s core',
                'location': 'Posterior (Back)',
                'characteristics': 'Fluent, poor comprehension, preserved repetition'
            },
            'mixed': {
                'primary': 'Watershed Areas',
                'description': 'Border zones between major vascular territories',
                'location': 'Multiple regions',
                'characteristics': 'Nonfluent, poor comprehension, preserved repetition'
            },
            'anomic': {
                'primary': 'Angular Gyrus or Scattered',
                'description': 'Often multiple small lesions or mild damage',
                'location': 'Various (often posterior)',
                'characteristics': 'Fluent speech with word-finding difficulties only'
            }
        }
        
        return brain_regions.get(aphasia_type_key, brain_regions['normal'])
    
    def train_tree_models(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train LightGBM and XGBoost models.
        
        Args:
            X_train: Training features (combined feature vectors)
            y_train: Training labels (severity levels 0-4)
        """
        print("Training LightGBM model...")
        self.lgb_model.fit(X_train, y_train)
        
        print("Training XGBoost model...")
        self.xgb_model.fit(X_train, y_train)
        
        print("Tree-based models trained successfully!")
        self.is_trained = True
    
    def save_models(self, path: str):
        """Save all trained models."""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch models
        torch.save(self.lstm_model.state_dict(), f"{path}/lstm_model.pth")
        torch.save(self.text_classifier.state_dict(), f"{path}/text_classifier.pth")
        
        # Save LightGBM model
        joblib.dump(self.lgb_model, f"{path}/lgb_model.pkl")
        
        # Save XGBoost model
        joblib.dump(self.xgb_model, f"{path}/xgb_model.pkl")
        
        # Save ensemble weights
        joblib.dump(self.ensemble_weights, f"{path}/ensemble_weights.pkl")
        
        print(f"All models saved to {path}")
    
    def load_models(self, path: str):
        """Load all trained models."""
        self.build_models()
        
        # Load PyTorch models
        self.lstm_model.load_state_dict(torch.load(f"{path}/lstm_model.pth", map_location=self.device))
        self.text_classifier.load_state_dict(torch.load(f"{path}/text_classifier.pth", map_location=self.device))
        
        # Load LightGBM model
        self.lgb_model = joblib.load(f"{path}/lgb_model.pkl")
        
        # Load XGBoost model
        self.xgb_model = joblib.load(f"{path}/xgb_model.pkl")
        
        # Load ensemble weights
        self.ensemble_weights = joblib.load(f"{path}/ensemble_weights.pkl")
        
        self.is_trained = True
        print(f"All models loaded from {path}")
