from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve, train_test_split

from .model import MultimodalClassifier
from training_visualizer import TrainingVisualizer

SEVERITY_MAP = {
    "Normal Speech": 0,
    "Mild Aphasia": 1,
    "Moderate Aphasia": 2,
    "Severe Aphasia": 3,
    "Very Severe Aphasia": 4
}

SEVERITY_LABELS = [
    "Normal Speech",
    "Mild Aphasia",
    "Moderate Aphasia",
    "Severe Aphasia",
    "Very Severe Aphasia"
]

FEATURE_NAMES = [
    "num_segments",
    "avg_segment_duration",
    "avg_pause_duration",
    "speech_rate",
    "pause_ratio",
    "max_pause",
    "segment_variability",
    "speaking_time",
    "total_time",
    "word_count",
    "unique_word_ratio",
    "avg_word_length",
    "filler_word_ratio",
    "repetition_ratio",
    "immediate_repetition_ratio",
    "sentence_count",
    "has_meaning",
    "content_word_ratio",
    "semantic_coherence",
    "grammar_errors",
    "errors_per_word",
    "has_complete_sentences",
    "missing_function_words",
    "function_word_ratio",
    "pitch_mean",
    "pitch_std",
    "rms_mean",
    "rms_std",
    "tempo",
    "zcr_mean",
    "zcr_std",
    "relevance_score",
    "keyword_match_ratio",
    "semantic_similarity",
    "topic_coherence"
]


def _extract_feature_vector(features: Dict) -> List[float]:
    temporal = features["temporal"]
    linguistic = features["linguistic"]
    grammar = features["grammar"]
    prosodic = features["prosodic"]

    vector = [
        temporal["num_segments"],
        temporal["avg_segment_duration"],
        temporal["avg_pause_duration"],
        temporal["speech_rate"],
        temporal["pause_ratio"],
        temporal["max_pause"],
        temporal["segment_variability"],
        temporal["speaking_time"],
        temporal["total_time"],
        linguistic["word_count"],
        linguistic["unique_word_ratio"],
        linguistic["avg_word_length"],
        linguistic["filler_word_ratio"],
        linguistic["repetition_ratio"],
        linguistic.get("immediate_repetition_ratio", 0.0),
        linguistic["sentence_count"],
        linguistic["has_meaning"],
        linguistic["content_word_ratio"],
        linguistic.get("semantic_coherence", 0.0),
        grammar["grammar_errors"],
        grammar["errors_per_word"],
        1.0 if grammar["has_complete_sentences"] else 0.0,
        grammar["missing_function_words"],
        grammar["function_word_ratio"],
        prosodic[0], prosodic[1], prosodic[2], prosodic[3],
        prosodic[4], prosodic[5], prosodic[6],
        0.5, 0.5, 0.5, 0.5
    ]

    return vector


def _load_samples(metadata_path: Path) -> List[Dict]:
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_sample_from_results(
    results: Dict,
    audio_source_path: str,
    severity_label: str,
    aphasia_type: str,
    training_data_dir: str = "training_data"
) -> Dict:
    training_dir = Path(training_data_dir)
    training_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sample_{timestamp}.wav"
    target_path = training_dir / filename

    if Path(audio_source_path).resolve() != target_path.resolve():
        shutil.copy2(audio_source_path, target_path)

    temporal = results.get("temporal_features", {})
    duration = float(temporal.get("total_time", 0.0))

    prosodic = results.get("prosodic_features", [])
    if isinstance(prosodic, np.ndarray):
        prosodic = prosodic.tolist()

    sample = {
        "timestamp": timestamp,
        "filename": filename,
        "filepath": str(target_path),
        "scenario": results.get("scenario", "Unknown"),
        "duration": duration,
        "severity_label": severity_label,
        "severity_level": SEVERITY_MAP[severity_label],
        "aphasia_type": aphasia_type,
        "transcript": results.get("transcript", ""),
        "word_count": results.get("linguistic_features", {}).get("word_count", 0),
        "features": {
            "temporal": temporal,
            "linguistic": results.get("linguistic_features", {}),
            "grammar": results.get("grammar_features", {}),
            "prosodic": prosodic
        }
    }

    metadata_path = training_dir / "samples_metadata.json"
    samples = _load_samples(metadata_path)
    samples.append(sample)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    return sample


def train_and_generate_plots(
    training_data_dir: str = "training_data",
    plots_dir: str = "training_plots",
    min_samples: int = 10
) -> Tuple[bool, str]:
    training_dir = Path(training_data_dir)
    metadata_path = training_dir / "samples_metadata.json"
    samples = _load_samples(metadata_path)

    if len(samples) < min_samples:
        return False, f"Not enough samples to train (have {len(samples)}, need {min_samples})."

    X = []
    y = []
    for sample in samples:
        features = sample.get("features", {})
        if not features:
            continue
        X.append(_extract_feature_vector(features))
        y.append(sample.get("severity_level", 0))

    if len(X) < min_samples:
        return False, "Not enough feature-complete samples to train."

    X = np.array(X)
    y = np.array(y)

    class_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    if len(class_counts) < 2:
        return False, "Need at least 2 classes to train and generate evaluation plots."

    stratify = y if min(class_counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    classifier = MultimodalClassifier()
    classifier.build_models()
    classifier.train_tree_models(X_train, y_train)

    y_pred = classifier.lgb_model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(SEVERITY_LABELS))))

    cv_folds = 3 if min(class_counts.values()) >= 3 else 2
    train_sizes, train_scores, val_scores = learning_curve(
        classifier.lgb_model,
        X,
        y,
        cv=cv_folds,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    visualizer = TrainingVisualizer(save_dir=plots_dir)
    visualizer.plot_confusion_matrix(cm, SEVERITY_LABELS)
    visualizer.plot_learning_curves(
        train_sizes=train_sizes.tolist(),
        train_scores=train_scores_mean.tolist(),
        val_scores=val_scores_mean.tolist()
    )

    if hasattr(classifier.lgb_model, "feature_importances_"):
        visualizer.plot_feature_importance(
            features=FEATURE_NAMES,
            importance=classifier.lgb_model.feature_importances_.tolist()
        )

    metrics = {
        "accuracy": acc,
        "samples": int(len(y)),
        "class_counts": class_counts,
        "model": "LightGBM",
        "plots_dir": plots_dir
    }
    visualizer.create_training_report(metrics)

    model_dir = Path("trained_models") / datetime.now().strftime("%Y%m%d_%H%M%S")
    classifier.save_models(str(model_dir))

    return True, f"Training complete. Model saved to {model_dir}."
