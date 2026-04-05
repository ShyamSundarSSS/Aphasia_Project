# Multimodal Aphasia Detection System - Architecture

```mermaid
flowchart TB
    %% Input Layer
    A[Raw Audio Signal<br/>16kHz, 16-bit]
    
    %% Stage 1: Preprocessing
    B[Preprocessing Pipeline<br/>• Bandpass Filter<br/>• VAD<br/>• Normalization]
    C[Whisper ASR Model<br/>Automatic Speech Recognition]
    
    %% Stage 2: Feature Extraction - Audio
    D[Audio Feature Extraction]
    E[MFCC<br/>13 + Δ + ΔΔ<br/>= 39 features]
    F[Prosodic Features<br/>Pitch, Energy<br/>Rate, Pauses]
    G[Voice Quality<br/>Jitter, Shimmer]
    
    %% Stage 2: Feature Extraction - Text
    H[Text Feature Extraction]
    I[BERT Tokenization<br/>WordPiece]
    J[Lexical Features<br/>TTR, Diversity]
    K[Syntactic Features<br/>FK Grade, Complexity]
    
    %% Stage 3: Deep Learning Models
    L[BiLSTM<br/>Temporal Encoder<br/>128 units<br/>→ 256-dim output]
    M[BERT<br/>Linguistic Encoder<br/>768-dim output]
    
    %% Stage 4: Fusion & Ensemble
    N[Multimodal Fusion Layer<br/>Concatenation + Batch Norm<br/>512-dim vector]
    O[XGBoost Classifier<br/>100 trees<br/>L1/L2 Regularization]
    P[LightGBM Classifier<br/>100 trees<br/>Histogram Boosting]
    Q[Ensemble Voting<br/>Weighted avg α=0.6]
    
    %% Stage 5: Output
    R[Severity Classification<br/>5 Classes:<br/>Normal to Very Severe]
    S[Confidence Scores]
    
    %% Connections - Stage 1
    A --> B
    B --> |Clean Signal| D
    B --> C
    C --> |Text Transcript| H
    
    %% Connections - Audio Features
    D --> E
    D --> F
    D --> G
    
    %% Connections - Text Features
    H --> I
    H --> J
    H --> K
    
    %% Connections - To Deep Learning Models
    E --> L
    F --> L
    G --> L
    
    I --> M
    J --> M
    K --> M
    
    %% Connections - To Fusion
    L --> N
    M --> N
    
    %% Connections - To Ensemble
    N --> O
    N --> P
    O --> Q
    P --> Q
    
    %% Connections - Final Output
    Q --> R
    R --> S
    
    %% Professional Styling - Minimal Colors
    style A fill:#f0f0f0,stroke:#333,stroke-width:2px,color:#000
    style B fill:#fff,stroke:#333,stroke-width:1.5px,color:#000
    style C fill:#fff,stroke:#333,stroke-width:1.5px,color:#000
    style D fill:#fff,stroke:#333,stroke-width:1.5px,color:#000
    style E fill:#f9f9f9,stroke:#666,stroke-width:1px,color:#000
    style F fill:#f9f9f9,stroke:#666,stroke-width:1px,color:#000
    style G fill:#f9f9f9,stroke:#666,stroke-width:1px,color:#000
    style H fill:#fff,stroke:#333,stroke-width:1.5px,color:#000
    style I fill:#f9f9f9,stroke:#666,stroke-width:1px,color:#000
    style J fill:#f9f9f9,stroke:#666,stroke-width:1px,color:#000
    style K fill:#f9f9f9,stroke:#666,stroke-width:1px,color:#000
    style L fill:#e8e8e8,stroke:#333,stroke-width:1.5px,color:#000
    style M fill:#e8e8e8,stroke:#333,stroke-width:1.5px,color:#000
    style N fill:#fff,stroke:#333,stroke-width:1.5px,color:#000
    style O fill:#e8e8e8,stroke:#333,stroke-width:1.5px,color:#000
    style P fill:#e8e8e8,stroke:#333,stroke-width:1.5px,color:#000
    style Q fill:#fff,stroke:#333,stroke-width:1.5px,color:#000
    style R fill:#e0e0e0,stroke:#333,stroke-width:2px,color:#000
    style S fill:#e0e0e0,stroke:#333,stroke-width:2px,color:#000
```

## Component Legend

- **Input**: Raw audio signal (16kHz, 16-bit)
- **Processing**: Preprocessing and feature extraction pipelines
- **Features**: Extracted acoustic and linguistic features
- **Models**: Deep learning encoders and ensemble classifiers
- **Output**: Classification results and confidence scores

## Architecture Overview

This multimodal system combines:

1. **Audio Stream**: MFCC, prosodic, and voice quality features → BiLSTM encoder
2. **Text Stream**: BERT tokenization, lexical, and syntactic features → BERT encoder
3. **Fusion**: Concatenation of both streams into unified representation
4. **Ensemble**: XGBoost + LightGBM with weighted voting
5. **Output**: 5-level severity classification with confidence scores

## Key Features

- **Multimodal**: Leverages both audio and text information
- **Deep Learning**: BiLSTM for temporal patterns, BERT for linguistic understanding
- **Ensemble**: Combines XGBoost and LightGBM for robust predictions
- **Real-time**: Optimized for production deployment

## Stage Details

### Stage 1: Preprocessing
- **Bandpass Filter**: 80-8000 Hz to remove noise
- **VAD**: Voice Activity Detection for segment extraction
- **Normalization**: Peak normalization to standardize amplitude

### Stage 2: Feature Extraction
**Audio Features (256-dim):**
- MFCC: 13 coefficients + delta + delta-delta = 39 features
- Prosodic: pitch (F0), energy, speaking rate, pause statistics
- Voice quality: jitter, shimmer, HNR

**Text Features (from Whisper ASR):**
- BERT tokenization with WordPiece
- Lexical: TTR, word diversity, vocabulary richness
- Syntactic: Flesch-Kincaid grade, sentence complexity

### Stage 3: Deep Learning Encoders
- **BiLSTM**: 128 hidden units, bidirectional, outputs 256-dim temporal representation
- **BERT**: Pre-trained base model, outputs 768-dim linguistic representation

### Stage 4: Ensemble Classification
- **Fusion**: Concatenate audio + text features → 1024-dim vector
- **XGBoost**: 100 trees, L1/L2 regularization for sparsity
- **LightGBM**: 100 trees, histogram-based boosting for speed
- **Voting**: Weighted average with α=0.6 favoring XGBoost

### Stage 5: Output
- **5 severity classes**: Normal, Mild, Moderate, Severe, Very Severe
- **Confidence scores**: Per-class probabilities for interpretability
