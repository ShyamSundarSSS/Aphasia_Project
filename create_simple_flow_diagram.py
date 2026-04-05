"""
Generate a simple, clear flow chart showing audio input to severity classification.
Matches the caption: "Depicts a flow chart demonstrating how audio input flows 
through the entire system until it arrives at the classification of severity."
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_simple_flow_diagram():
    """Create simple vertical flow diagram from audio to classification."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Title
    ax.text(5, 13.4, 'Aphasia Detection System Flow', 
            ha='center', fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#1976D2', 
                     edgecolor='#0D47A1', linewidth=2, alpha=1),
            color='white')
    
    # Step 1: Audio Input
    step1_box = FancyBboxPatch((1.5, 12.3), 7, 0.9,
                               boxstyle="round,pad=0.1",
                               facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=2.5)
    ax.add_patch(step1_box)
    ax.text(5, 13.05, '🎤 AUDIO INPUT', ha='center', fontsize=13, fontweight='bold')
    ax.text(5, 12.65, 'Patient Speech Sample (16 kHz, 10-30 sec)', 
            ha='center', fontsize=9, style='italic')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((5.0, 12.3), (5.0, 11.65),
                            arrowstyle='->', mutation_scale=28, linewidth=3,
                            color='#D32F2F', connectionstyle="arc3,rad=0")
    ax.add_patch(arrow1)
    
    # Step 2: Audio Preprocessing
    step2_box = FancyBboxPatch((1.5, 10.4), 7, 1.15,
                               boxstyle="round,pad=0.1",
                               facecolor='#FFF8E1', edgecolor='#F57C00', linewidth=2.5)
    ax.add_patch(step2_box)
    ax.text(5, 11.3, '⚙️ AUDIO PREPROCESSING', ha='center', fontsize=13, fontweight='bold')
    ax.text(5, 10.95, '• Noise Reduction (80 Hz - 8 kHz)  • Voice Activity Detection  • Signal Normalization', 
            ha='center', fontsize=8.5)
    
    # Arrow 2
    arrow2 = FancyArrowPatch((5.0, 10.4), (5.0, 9.8),
                            arrowstyle='->', mutation_scale=28, linewidth=3,
                            color='#F57C00', connectionstyle="arc3,rad=0")
    ax.add_patch(arrow2)
    
    # Step 3: Speech-to-Text Conversion
    step3_box = FancyBboxPatch((1.5, 9.05), 7, 0.65,
                               boxstyle="round,pad=0.1",
                               facecolor='#F3E5F5', edgecolor='#8E24AA', linewidth=2.5)
    ax.add_patch(step3_box)
    ax.text(5, 9.55, '📝 SPEECH-TO-TEXT CONVERSION', ha='center', fontsize=13, fontweight='bold')
    ax.text(5, 9.2, 'Whisper ASR → Text Transcript', ha='center', fontsize=9)
    
    # Split arrow
    arrow3a = FancyArrowPatch((5.0, 9.05), (2.5, 8.4),
                             arrowstyle='->', mutation_scale=23, linewidth=2.5,
                             color='#8E24AA', connectionstyle="arc3,rad=0")
    ax.add_patch(arrow3a)
    
    arrow3b = FancyArrowPatch((5.0, 9.05), (7.5, 8.4),
                             arrowstyle='->', mutation_scale=23, linewidth=2.5,
                             color='#8E24AA', connectionstyle="arc3,rad=0")
    ax.add_patch(arrow3b)
    
    # Step 4a: Audio Feature Extraction (LEFT)
    step4a_box = FancyBboxPatch((0.5, 6.15), 4, 2.15,
                                boxstyle="round,pad=0.1",
                                facecolor='#E1F5FE', edgecolor='#0277BD', linewidth=2.5)
    ax.add_patch(step4a_box)
    ax.text(2.5, 8.1, '🔊 AUDIO FEATURES', ha='center', fontsize=12, fontweight='bold')
    ax.text(2.5, 7.8, '• MFCC (39 coefficients)', ha='center', fontsize=8.5)
    ax.text(2.5, 7.55, '• Prosodic features (pitch, energy)', ha='center', fontsize=8.5)
    ax.text(2.5, 7.3, '• Temporal dynamics', ha='center', fontsize=8.5)
    ax.text(2.5, 7.05, '↓ BiLSTM → 256-dim', ha='center', fontsize=8.5, fontweight='bold')
    
    # Step 4b: Text Feature Extraction (RIGHT)
    step4b_box = FancyBboxPatch((5.5, 6.15), 4, 2.15,
                                boxstyle="round,pad=0.1",
                                facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2.5)
    ax.add_patch(step4b_box)
    ax.text(7.5, 8.1, '📄 TEXT FEATURES', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 7.8, '• BERT embeddings (contextual)', ha='center', fontsize=8.5)
    ax.text(7.5, 7.55, '• Semantic & syntactic analysis', ha='center', fontsize=8.5)
    ax.text(7.5, 7.3, '• Lexical diversity (TTR)', ha='center', fontsize=8.5)
    ax.text(7.5, 7.05, '↓ BERT → 768-dim', ha='center', fontsize=8.5, fontweight='bold')
    
    # Merge arrows
    arrow4a = FancyArrowPatch((2.5, 6.15), (5.0, 5.5),
                             arrowstyle='->', mutation_scale=23, linewidth=2.5,
                             color='#0277BD', connectionstyle="arc3,rad=0")
    ax.add_patch(arrow4a)
    
    arrow4b = FancyArrowPatch((7.5, 6.15), (5.0, 5.5),
                             arrowstyle='->', mutation_scale=23, linewidth=2.5,
                             color='#388E3C', connectionstyle="arc3,rad=0")
    ax.add_patch(arrow4b)
    
    # Step 5: Multimodal Fusion
    step5_box = FancyBboxPatch((1.5, 4.85), 7, 0.55,
                               boxstyle="round,pad=0.1",
                               facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2.5)
    ax.add_patch(step5_box)
    ax.text(5, 5.3, '⚡ MULTIMODAL FUSION', ha='center', fontsize=13, fontweight='bold')
    ax.text(5, 5.0, 'Concatenate: 256 + 768 = 1024-dimensional vector', ha='center', fontsize=9)
    
    # Arrow 5
    arrow5 = FancyArrowPatch((5.0, 4.85), (5.0, 4.25),
                            arrowstyle='->', mutation_scale=28, linewidth=3,
                            color='#C2185B', connectionstyle="arc3,rad=0")
    ax.add_patch(arrow5)
    
    # Step 6: Ensemble Classification
    step6_box = FancyBboxPatch((1.5, 3.35), 7, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor='#E8F5E9', edgecolor='#2E7D32', linewidth=2.5)
    ax.add_patch(step6_box)
    ax.text(5, 4.0, '🤖 ENSEMBLE CLASSIFIERS', ha='center', fontsize=13, fontweight='bold')
    ax.text(5, 3.65, 'XGBoost + LightGBM with Weighted Voting (α = 0.6)', 
            ha='center', fontsize=9)
    
    # Arrow 6
    arrow6 = FancyArrowPatch((5.0, 3.35), (5.0, 2.8),
                            arrowstyle='->', mutation_scale=28, linewidth=3,
                            color='#2E7D32', connectionstyle="arc3,rad=0")
    ax.add_patch(arrow6)
    
    # Step 7: Classification Output
    step7_box = FancyBboxPatch((1, 2.05), 8, 0.65,
                               boxstyle="round,pad=0.1",
                               facecolor='#4CAF50', edgecolor='#1B5E20', linewidth=3)
    ax.add_patch(step7_box)
    ax.text(5, 2.5, '✓ SEVERITY CLASSIFICATION', ha='center', fontsize=12, fontweight='bold', color='white')
    ax.text(5, 2.2, 'Level 0-4 (Normal → Very Severe) + Confidence Score', 
            ha='center', fontsize=9, color='white')
    
    # Performance box at bottom
    perf_box = FancyBboxPatch((1.5, 0.3), 7, 0.65,
                              boxstyle="round,pad=0.08",
                              facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2)
    ax.add_patch(perf_box)
    ax.text(5, 0.85, '📊 Performance: Accuracy 94.2% | F1-Score 0.93 | Inference Time <2 sec', 
            ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save with high DPI
    output_path = 'D:/AphasiaPhase2/system_flow.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ High-quality flow diagram saved to: {output_path}")
    
    # Also save to docs folder
    docs_path = 'D:/AphasiaPhase2/docs/system_flow_high_res.png'
    plt.savefig(docs_path, dpi=400, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Extra high-quality version saved to: {docs_path}")
    
    plt.show()


if __name__ == "__main__":
    create_simple_flow_diagram()
