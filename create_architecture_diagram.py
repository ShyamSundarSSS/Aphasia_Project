"""
Generate publication-quality system architecture diagram for aphasia detection system.
Includes training and inference phases with clear data flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_architecture_diagram():
    """Create comprehensive system architecture diagram."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), 
                                     gridspec_kw={'height_ratios': [1.2, 1]})
    
    # Remove axes
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 10)
        ax.axis('off')
    
    # ============= TRAINING PHASE (Top Panel) =============
    ax1.text(7, 9.5, 'PHASE 1: TRAINING & MODEL DEVELOPMENT', 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2196F3', 
                      edgecolor='black', linewidth=2, alpha=0.9),
             color='white')
    
    # Input Sources Box
    input_box = FancyBboxPatch((0.5, 6.5), 2.5, 2.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax1.add_patch(input_box)
    ax1.text(1.75, 8.5, 'INPUT SOURCES', ha='center', fontsize=10, fontweight='bold')
    ax1.text(1.75, 8, '1. Patient Audio', ha='center', fontsize=9)
    ax1.text(1.75, 7.6, '   (10-30 sec)', ha='center', fontsize=8, style='italic')
    ax1.text(1.75, 7.2, '2. Clinical Diagnosis', ha='center', fontsize=9)
    ax1.text(1.75, 6.8, '   (Expert Labels)', ha='center', fontsize=8, style='italic')
    
    # Arrow to preprocessing
    arrow1 = FancyArrowPatch((3, 7.75), (4.2, 7.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#1976D2')
    ax1.add_patch(arrow1)
    
    # Feature Extraction Box
    feature_box = FancyBboxPatch((4.2, 5.5), 2.8, 3.5,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax1.add_patch(feature_box)
    ax1.text(5.6, 8.7, 'FEATURE EXTRACTION', ha='center', fontsize=10, fontweight='bold')
    
    # Audio branch
    ax1.text(5.6, 8.2, 'Audio Processing', ha='center', fontsize=9, fontweight='bold', color='#0288D1')
    ax1.text(5.6, 7.85, '• MFCC (39-dim)', ha='center', fontsize=8)
    ax1.text(5.6, 7.5, '• Temporal features', ha='center', fontsize=8)
    ax1.text(5.6, 7.15, '• BiLSTM (256-dim)', ha='center', fontsize=8)
    
    # Text branch
    ax1.text(5.6, 6.7, 'Text Processing', ha='center', fontsize=9, fontweight='bold', color='#0288D1')
    ax1.text(5.6, 6.35, '• Whisper ASR', ha='center', fontsize=8)
    ax1.text(5.6, 6.0, '• BERT encoding', ha='center', fontsize=8)
    ax1.text(5.6, 5.65, '• Linguistic features', ha='center', fontsize=8)
    
    # Arrow to fusion
    arrow2 = FancyArrowPatch((7, 7.75), (8.2, 7.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#F57C00')
    ax1.add_patch(arrow2)
    
    # Fusion Box
    fusion_box = FancyBboxPatch((8.2, 6.5), 2.3, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2)
    ax1.add_patch(fusion_box)
    ax1.text(9.35, 8.7, 'MULTIMODAL', ha='center', fontsize=10, fontweight='bold')
    ax1.text(9.35, 8.4, 'FUSION', ha='center', fontsize=10, fontweight='bold')
    ax1.text(9.35, 7.9, 'Concatenation', ha='center', fontsize=8)
    ax1.text(9.35, 7.5, '+ Batch Norm', ha='center', fontsize=8)
    ax1.text(9.35, 7.1, '↓', ha='center', fontsize=12, fontweight='bold')
    ax1.text(9.35, 6.7, '512-dim vector', ha='center', fontsize=8, style='italic')
    
    # Arrow to training
    arrow3 = FancyArrowPatch((10.5, 7.75), (11.5, 7.75),
                            arrowstyle='->', mutation_scale=20, linewidth=2.5,
                            color='#388E3C')
    ax1.add_patch(arrow3)
    
    # ML Training Box
    ml_box = FancyBboxPatch((11.5, 6), 2.2, 3.5,
                           boxstyle="round,pad=0.1",
                           facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
    ax1.add_patch(ml_box)
    ax1.text(12.6, 9.2, 'ML TRAINING', ha='center', fontsize=10, fontweight='bold')
    ax1.text(12.6, 8.7, 'Supervised Learning', ha='center', fontsize=8, style='italic')
    ax1.text(12.6, 8.3, '1. XGBoost', ha='center', fontsize=9)
    ax1.text(12.6, 7.95, '   (100 trees)', ha='center', fontsize=7)
    ax1.text(12.6, 7.6, '2. LightGBM', ha='center', fontsize=9)
    ax1.text(12.6, 7.25, '   (100 trees)', ha='center', fontsize=7)
    ax1.text(12.6, 6.9, '3. Ensemble', ha='center', fontsize=9)
    ax1.text(12.6, 6.55, '   (Weighted)', ha='center', fontsize=7)
    ax1.text(12.6, 6.2, '↓', ha='center', fontsize=12, fontweight='bold')
    
    # Trained Model Output
    model_box = FancyBboxPatch((11.5, 5), 2.2, 0.8,
                               boxstyle="round,pad=0.05",
                               facecolor='#4CAF50', edgecolor='black', linewidth=2)
    ax1.add_patch(model_box)
    ax1.text(12.6, 5.4, '✓ TRAINED MODEL', ha='center', fontsize=9, 
             fontweight='bold', color='white')
    
    # Ground Truth Box (showing supervision)
    gt_box = FancyBboxPatch((11.5, 3.5), 2.2, 1,
                           boxstyle="round,pad=0.05",
                           facecolor='#FFECB3', edgecolor='#FF6F00', linewidth=2)
    ax1.add_patch(gt_box)
    ax1.text(12.6, 4.2, 'Ground Truth Labels', ha='center', fontsize=8, fontweight='bold')
    ax1.text(12.6, 3.85, 'Clinical Severity (0-4)', ha='center', fontsize=7)
    ax1.text(12.6, 3.6, 'Aphasia Type', ha='center', fontsize=7)
    
    # Arrow showing supervision
    arrow_gt = FancyArrowPatch((12.6, 4.5), (12.6, 6),
                              arrowstyle='->', mutation_scale=15, linewidth=2,
                              color='#FF6F00', linestyle='dashed')
    ax1.add_patch(arrow_gt)
    ax1.text(13.3, 5.2, 'Supervises', ha='left', fontsize=7, 
             style='italic', color='#FF6F00')
    
    # Training Metrics Box
    metrics_box = FancyBboxPatch((0.5, 3.5), 3.5, 1.8,
                                boxstyle="round,pad=0.1",
                                facecolor='#E0F7FA', edgecolor='#00838F', linewidth=2)
    ax1.add_patch(metrics_box)
    ax1.text(2.25, 5.1, 'TRAINING PERFORMANCE', ha='center', 
             fontsize=9, fontweight='bold')
    ax1.text(2.25, 4.7, '• Accuracy: 94.2%', ha='center', fontsize=8)
    ax1.text(2.25, 4.4, '• F1-Score: 0.93', ha='center', fontsize=8)
    ax1.text(2.25, 4.1, '• AUC-ROC: 0.987', ha='center', fontsize=8)
    ax1.text(2.25, 3.8, '• 5-Fold CV', ha='center', fontsize=8)
    
    # Dataset Info Box
    data_box = FancyBboxPatch((4.5, 3.5), 3, 1.8,
                             boxstyle="round,pad=0.1",
                             facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2)
    ax1.add_patch(data_box)
    ax1.text(6, 5.1, 'TRAINING DATASET', ha='center', 
             fontsize=9, fontweight='bold')
    ax1.text(6, 4.7, '• 500+ Audio Samples', ha='center', fontsize=8)
    ax1.text(6, 4.4, '• Expert-Labeled', ha='center', fontsize=8)
    ax1.text(6, 4.1, '• 5 Severity Classes', ha='center', fontsize=8)
    ax1.text(6, 3.8, '• Multiple Speakers', ha='center', fontsize=8)
    
    # Clinical Validation Note
    clinical_box = FancyBboxPatch((8, 3.5), 3, 1.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2)
    ax1.add_patch(clinical_box)
    ax1.text(9.5, 5.1, 'CLINICAL VALIDATION', ha='center', 
             fontsize=9, fontweight='bold')
    ax1.text(9.5, 4.7, '✓ Clinician-Verified', ha='center', fontsize=8)
    ax1.text(9.5, 4.4, '✓ Inter-Rater Agreement', ha='center', fontsize=8)
    ax1.text(9.5, 4.1, '✓ Gold Standard Labels', ha='center', fontsize=8)
    ax1.text(9.5, 3.8, '✓ Multiple Experts', ha='center', fontsize=8)
    
    # ============= INFERENCE PHASE (Bottom Panel) =============
    ax2.text(7, 9.5, 'PHASE 2: CLINICAL INFERENCE & DEPLOYMENT', 
             ha='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#4CAF50', 
                      edgecolor='black', linewidth=2, alpha=0.9),
             color='white')
    
    # New Patient Input
    patient_box = FancyBboxPatch((0.5, 6), 2.5, 2.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#FFEBEE', edgecolor='#C62828', linewidth=2)
    ax2.add_patch(patient_box)
    ax2.text(1.75, 8.2, 'NEW PATIENT', ha='center', fontsize=10, fontweight='bold')
    ax2.text(1.75, 7.7, '🎤 Audio Input', ha='center', fontsize=9)
    ax2.text(1.75, 7.3, '(Real-time)', ha='center', fontsize=8, style='italic')
    ax2.text(1.75, 6.9, 'No labels', ha='center', fontsize=8, style='italic')
    ax2.text(1.75, 6.5, 'No diagnosis', ha='center', fontsize=8, style='italic')
    
    # Arrow to preprocessing
    arrow_inf1 = FancyArrowPatch((3, 7.25), (4.2, 7.25),
                                arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                color='#C62828')
    ax2.add_patch(arrow_inf1)
    
    # Same Feature Extraction (reuse trained components)
    feature_box2 = FancyBboxPatch((4.2, 5.5), 2.8, 3,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2)
    ax2.add_patch(feature_box2)
    ax2.text(5.6, 8.2, 'FEATURE', ha='center', fontsize=10, fontweight='bold')
    ax2.text(5.6, 7.85, 'EXTRACTION', ha='center', fontsize=10, fontweight='bold')
    ax2.text(5.6, 7.4, '(Same as Training)', ha='center', fontsize=8, style='italic')
    ax2.text(5.6, 7, 'Audio + Text', ha='center', fontsize=9)
    ax2.text(5.6, 6.6, '↓', ha='center', fontsize=12, fontweight='bold')
    ax2.text(5.6, 6.2, '512-dim features', ha='center', fontsize=8)
    ax2.text(5.6, 5.8, '(Multimodal)', ha='center', fontsize=7, style='italic')
    
    # Arrow to model
    arrow_inf2 = FancyArrowPatch((7, 7.25), (8.5, 7.25),
                                arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                color='#F57C00')
    ax2.add_patch(arrow_inf2)
    
    # Trained Model (loaded for inference)
    model_inf_box = FancyBboxPatch((8.5, 5.5), 2.5, 3,
                                   boxstyle="round,pad=0.1",
                                   facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2)
    ax2.add_patch(model_inf_box)
    ax2.text(9.75, 8.2, 'TRAINED MODEL', ha='center', fontsize=10, fontweight='bold')
    ax2.text(9.75, 7.8, '(Loaded)', ha='center', fontsize=8, style='italic')
    ax2.text(9.75, 7.35, 'XGBoost +', ha='center', fontsize=9)
    ax2.text(9.75, 7, 'LightGBM', ha='center', fontsize=9)
    ax2.text(9.75, 6.6, 'Ensemble', ha='center', fontsize=9)
    ax2.text(9.75, 6.2, '↓', ha='center', fontsize=12, fontweight='bold')
    ax2.text(9.75, 5.8, 'Classification', ha='center', fontsize=8)
    
    # Arrow to output
    arrow_inf3 = FancyArrowPatch((11, 7.25), (11.8, 7.25),
                                arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                color='#2E7D32')
    ax2.add_patch(arrow_inf3)
    
    # Prediction Output Box
    output_box = FancyBboxPatch((11.8, 5.2), 2, 3.5,
                                boxstyle="round,pad=0.1",
                                facecolor='#E1F5FE', edgecolor='#01579B', linewidth=2)
    ax2.add_patch(output_box)
    ax2.text(12.8, 8.4, 'PREDICTION', ha='center', fontsize=10, fontweight='bold')
    ax2.text(12.8, 7.95, '1. Aphasia Status', ha='center', fontsize=9)
    ax2.text(12.8, 7.6, '   Yes / No', ha='center', fontsize=8, style='italic')
    ax2.text(12.8, 7.2, '2. Severity Level', ha='center', fontsize=9)
    ax2.text(12.8, 6.85, '   0 → 4', ha='center', fontsize=8, style='italic')
    ax2.text(12.8, 6.5, '3. Confidence', ha='center', fontsize=9)
    ax2.text(12.8, 6.15, '   0-100%', ha='center', fontsize=8, style='italic')
    ax2.text(12.8, 5.75, '4. Aphasia Type', ha='center', fontsize=9)
    ax2.text(12.8, 5.4, '   (if detected)', ha='center', fontsize=7, style='italic')
    
    # Clinical Decision Support
    clinical_inf_box = FancyBboxPatch((0.5, 2.5), 6, 2,
                                     boxstyle="round,pad=0.1",
                                     facecolor='#F1F8E9', edgecolor='#558B2F', linewidth=2)
    ax2.add_patch(clinical_inf_box)
    ax2.text(3.5, 4.2, 'CLINICAL DECISION SUPPORT', ha='center', 
             fontsize=10, fontweight='bold')
    ax2.text(3.5, 3.7, '✓ Real-time assessment (<2 seconds)', ha='center', fontsize=8)
    ax2.text(3.5, 3.35, '✓ Interpretable results for clinicians', ha='center', fontsize=8)
    ax2.text(3.5, 3, '✓ Confidence intervals and uncertainty', ha='center', fontsize=8)
    ax2.text(3.5, 2.65, '✓ Integration with EHR systems', ha='center', fontsize=8)
    
    # Deployment Info
    deploy_box = FancyBboxPatch((7, 2.5), 6.8, 2,
                               boxstyle="round,pad=0.1",
                               facecolor='#FFF8E1', edgecolor='#F57C00', linewidth=2)
    ax2.add_patch(deploy_box)
    ax2.text(10.4, 4.2, 'DEPLOYMENT CHARACTERISTICS', ha='center', 
             fontsize=10, fontweight='bold')
    ax2.text(10.4, 3.7, '• Inference Time: <2 seconds', ha='center', fontsize=8)
    ax2.text(10.4, 3.35, '• Accuracy: 94.2% (maintained from training)', ha='center', fontsize=8)
    ax2.text(10.4, 3, '• Platform: Desktop/Web/Mobile compatible', ha='center', fontsize=8)
    ax2.text(10.4, 2.65, '• Requirements: Audio input + Internet (for ASR)', ha='center', fontsize=8)
    
    # Key Difference Annotation
    ax2.annotate('', xy=(1.75, 5.5), xytext=(1.75, 4.8),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(0.2, 5.15, 'KEY DIFFERENCE:\nNo clinical labels\nneeded for\ninference',
             fontsize=8, color='red', fontweight='bold', ha='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Overall Title
    fig.suptitle('Multimodal Aphasia Detection System: Training & Inference Architecture',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    output_path = 'd:/AphasiaPhase2/docs/system_architecture_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Diagram saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    create_architecture_diagram()
