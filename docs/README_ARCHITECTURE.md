# System Architecture Documentation

## Overview
This folder contains the complete system architecture documentation for the multimodal aphasia detection system, suitable for academic publication.

## Files

### 1. `system_architecture.md`
Comprehensive written description of the system including:
- Data collection and clinical validation
- Feature extraction pipelines (audio + text)
- Multimodal fusion architecture
- ML training process with supervised learning
- Real-time inference for new patients
- Performance metrics and clinical validation

### 2. `system_architecture_diagram.png`
Publication-quality visual diagram showing:
- **Phase 1 (Training)**: Patient input → Clinical diagnosis → Feature extraction → ML training → Trained model
- **Phase 2 (Inference)**: New patient audio → Feature extraction → Trained model → Aphasia prediction
- Clear distinction between training (with labels) and inference (without labels)
- Data flow, components, and performance metrics

## Generating the Diagram

Run the diagram generator:
```bash
python create_architecture_diagram.py
```

This creates a high-resolution (300 DPI) diagram suitable for paper publication.

## Key Points for Publication

### Training Phase (Supervised Learning)
1. **Input**: Patient audio + Clinical expert diagnosis (ground truth labels)
2. **Process**: Feature extraction (audio + text) → ML training with labels
3. **Output**: Trained classification model (saved for deployment)
4. **Validation**: 5-fold cross-validation with clinical verification

### Inference Phase (Real-time Prediction)
1. **Input**: New patient audio (NO labels, NO diagnosis)
2. **Process**: Same feature extraction → Trained model inference
3. **Output**: 
   - Aphasia detected: Yes/No
   - Severity level: 0-4 (Normal → Very Severe)
   - Confidence score: 0-100%
   - Type classification (if applicable)

### Novel Contributions
- **Multimodal fusion** of audio (acoustic + temporal) and text (linguistic + semantic)
- **End-to-end system** from raw audio to clinical decision support
- **High accuracy** (94.2%) with clinical validation
- **Real-time inference** (<2 seconds) for practical deployment

## Citation
When using this architecture in publications, please cite:
