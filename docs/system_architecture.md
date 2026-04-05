# Multimodal Aphasia Detection System Architecture

## Overview
A complete end-to-end system for aphasia detection and severity classification using audio and text modalities with clinical validation.

## System Phases

### Phase 1: Data Collection and Clinical Validation
**Input Sources:**
1. **Patient Audio Samples**
   - Raw audio recordings (16kHz, 16-bit)
   - Duration: 10-30 seconds per sample
   - Source: Clinical interviews, reading tasks, spontaneous speech

2. **Clinical Diagnosis**
   - Expert clinician assessment
   - Severity classification: Normal → Mild → Moderate → Severe → Very Severe
   - Aphasia type annotation: Broca's, Wernicke's, Global, etc.
   - Ground truth labels for supervised learning

### Phase 2: Multimodal Feature Extraction

#### Audio Processing Pipeline
1. **Preprocessing**
   - Bandpass filtering (300Hz - 8kHz)
   - Voice Activity Detection (VAD)
   - Normalization

2. **Acoustic Features (39-dimensional MFCC)**
   - 13 MFCCs + Δ + ΔΔ
   - Captures prosody, pitch, speech rate

3. **Temporal Features**
   - Pause detection and duration
   - Speech rate (words per minute)
   - Segment variability

4. **Deep Audio Embeddings**
   - BiLSTM Temporal Encoder (128 units → 256-dim output)
   - Captures temporal speech patterns

#### Text Processing Pipeline
1. **Speech-to-Text Conversion**
   - Whisper ASR Model (OpenAI)
   - Automatic transcription with high accuracy

2. **Linguistic Features**
   - BERT tokenization (WordPiece)
   - POS tagging (TTR, lexical diversity)
   - Syntactic complexity (FK grade, PK complexity)

3. **Semantic Embeddings**
   - BERT Linguistic Encoder (768-dim output)
   - Contextual word representations

### Phase 3: Multimodal Fusion and Training

#### Fusion Architecture
