"""
Model Training Visualization
Creates professional graphs showing training and validation metrics over epochs.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime


class TrainingVisualizer:
    """Creates professional training visualizations."""
    
    def __init__(self, save_dir: str = "training_plots"):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Directory to save plot images
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set style for professional plots
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'train': '#2196F3',  # Blue
            'val': '#F44336',    # Red
            'grid': '#E0E0E0'
        }
    
    def plot_training_history(
        self,
        train_acc: List[float],
        val_acc: List[float],
        train_loss: Optional[List[float]] = None,
        val_loss: Optional[List[float]] = None,
        title: str = "Model Training and Validation Accuracy over 25 Epochs",
        save_name: str = "training_history.png"
    ):
        """
        Plot training and validation accuracy/loss over epochs.
        
        Args:
            train_acc: Training accuracy per epoch
            val_acc: Validation accuracy per epoch
            train_loss: Training loss per epoch (optional)
            val_loss: Validation loss per epoch (optional)
            title: Plot title
            save_name: Filename to save plot
        """
        epochs = range(1, len(train_acc) + 1)
        
        # Determine layout
        if train_loss is not None and val_loss is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
            ax2 = None
        
        # Plot accuracy
        ax1.plot(epochs, train_acc, 'o-', 
                color=self.colors['train'], 
                linewidth=2, 
                markersize=6,
                label='Training Accuracy',
                markerfacecolor='white',
                markeredgewidth=2)
        
        ax1.plot(epochs, val_acc, 's-', 
                color=self.colors['val'], 
                linewidth=2, 
                markersize=6,
                label='Validation Accuracy',
                markerfacecolor='white',
                markeredgewidth=2)
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_xlim(1, len(epochs))
        ax1.set_ylim(0, 1.0)
        
        # Add epoch markers on x-axis
        ax1.set_xticks(range(1, len(epochs) + 1, max(1, len(epochs) // 10)))
        
        # Add accuracy values at key points
        max_val_acc = max(val_acc)
        max_val_epoch = val_acc.index(max_val_acc) + 1
        ax1.annotate(f'Best: {max_val_acc:.3f}',
                    xy=(max_val_epoch, max_val_acc),
                    xytext=(max_val_epoch, max_val_acc + 0.05),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=10,
                    fontweight='bold',
                    color='red')
        
        # Plot loss if provided
        if ax2 is not None:
            ax2.plot(epochs, train_loss, 'o-',
                    color=self.colors['train'],
                    linewidth=2,
                    markersize=6,
                    label='Training Loss',
                    markerfacecolor='white',
                    markeredgewidth=2)
            
            ax2.plot(epochs, val_loss, 's-',
                    color=self.colors['val'],
                    linewidth=2,
                    markersize=6,
                    label='Validation Loss',
                    markerfacecolor='white',
                    markeredgewidth=2)
            
            ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax2.set_title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=20)
            ax2.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_xlim(1, len(epochs))
            ax2.set_xticks(range(1, len(epochs) + 1, max(1, len(epochs) // 10)))
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training plot: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        classes: List[str],
        title: str = "Confusion Matrix",
        save_name: str = "confusion_matrix.png",
        normalize: bool = True
    ):
        """
        Plot confusion matrix heatmap.
        
        Args:
            cm: Confusion matrix
            classes: Class names
            title: Plot title
            save_name: Filename to save plot
            normalize: Whether to normalize values
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True Label',
               xlabel='Predicted Label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_learning_curves(
        self,
        train_sizes: List[int],
        train_scores: List[float],
        val_scores: List[float],
        title: str = "Learning Curves",
        save_name: str = "learning_curves.png"
    ):
        """
        Plot learning curves showing performance vs training set size.
        
        Args:
            train_sizes: Training set sizes
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            title: Plot title
            save_name: Filename to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes, train_scores, 'o-',
               color=self.colors['train'],
               linewidth=2,
               markersize=8,
               label='Training Score',
               markerfacecolor='white',
               markeredgewidth=2)
        
        ax.plot(train_sizes, val_scores, 's-',
               color=self.colors['val'],
               linewidth=2,
               markersize=8,
               label='Validation Score',
               markerfacecolor='white',
               markeredgewidth=2)
        
        ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved learning curves: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_feature_importance(
        self,
        features: List[str],
        importance: List[float],
        top_n: int = 20,
        title: str = "Top Feature Importance",
        save_name: str = "feature_importance.png"
    ):
        """
        Plot feature importance bar chart.
        
        Args:
            features: Feature names
            importance: Feature importance scores
            top_n: Number of top features to show
            title: Plot title
            save_name: Filename to save plot
        """
        importance = np.asarray(importance)

        # Sort by importance
        sorted_idx = np.argsort(importance)[-top_n:]
        pos = np.arange(sorted_idx.shape[0])
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(pos, importance[sorted_idx], align='center',
               color=self.colors['train'], alpha=0.8)
        ax.set_yticks(pos)
        ax.set_yticklabels([features[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        plt.tight_layout()
        
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance: {save_path}")
        
        plt.show()
        
        return fig
    
    def create_training_report(
        self,
        metrics: Dict,
        save_name: str = "training_report.json"
    ):
        """
        Save training metrics to JSON file.
        
        Args:
            metrics: Dictionary of training metrics
            save_name: Filename to save report
        """
        metrics['timestamp'] = datetime.now().isoformat()
        
        save_path = self.save_dir / save_name
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✓ Saved training report: {save_path}")
    
    def plot_multimodal_pipeline(
        self,
        audio_train_acc: List[float],
        text_train_acc: List[float],
        fusion_train_acc: List[float],
        audio_val_acc: List[float],
        text_val_acc: List[float],
        fusion_val_acc: List[float],
        title: str = "Multimodal Training Pipeline: Parallel Streams with Joint Fusion",
        save_name: str = "multimodal_pipeline.png"
    ):
        """
        Plot multimodal training pipeline showing parallel audio and text streams
        with joint fusion training mechanism.
        
        Args:
            audio_train_acc: Training accuracy for audio stream
            text_train_acc: Training accuracy for text stream
            fusion_train_acc: Training accuracy for fusion model
            audio_val_acc: Validation accuracy for audio stream
            text_val_acc: Validation accuracy for text stream
            fusion_val_acc: Validation accuracy for fusion model
            title: Plot title
            save_name: Filename to save plot
        """
        epochs = range(1, len(fusion_train_acc) + 1)
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1], hspace=0.3, wspace=0.3)
        
        # --- Top: Pipeline Architecture Diagram ---
        ax_arch = fig.add_subplot(gs[0, :])
        ax_arch.set_xlim(0, 10)
        ax_arch.set_ylim(0, 10)
        ax_arch.axis('off')
        ax_arch.set_title('Multimodal Training Architecture', 
                         fontsize=16, fontweight='bold', pad=20)
        
        # Audio Stream (Left)
        audio_box = plt.Rectangle((0.5, 6), 1.8, 3, 
                                  facecolor='#E3F2FD', edgecolor='#2196F3', 
                                  linewidth=3)
        ax_arch.add_patch(audio_box)
        ax_arch.text(1.4, 8.5, 'Audio\nStream', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='#1976D2')
        ax_arch.text(1.4, 7.8, 'MFCC', ha='center', fontsize=9, style='italic')
        ax_arch.text(1.4, 7.4, 'Prosody', ha='center', fontsize=9, style='italic')
        ax_arch.text(1.4, 7.0, 'BiLSTM', ha='center', fontsize=9, style='italic')
        ax_arch.text(1.4, 6.6, '(128 units)', ha='center', fontsize=8, color='gray')
        
        # Text Stream (Right)
        text_box = plt.Rectangle((3, 6), 1.8, 3,
                                facecolor='#FFF3E0', edgecolor='#FF9800',
                                linewidth=3)
        ax_arch.add_patch(text_box)
        ax_arch.text(3.9, 8.5, 'Text\nStream', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='#F57C00')
        ax_arch.text(3.9, 7.8, 'BERT', ha='center', fontsize=9, style='italic')
        ax_arch.text(3.9, 7.4, 'Linguistic', ha='center', fontsize=9, style='italic')
        ax_arch.text(3.9, 7.0, 'Transformer', ha='center', fontsize=9, style='italic')
        ax_arch.text(3.9, 6.6, '(256-dim)', ha='center', fontsize=8, color='gray')
        
        # Arrows from streams to fusion
        ax_arch.annotate('', xy=(6.5, 7.5), xytext=(2.3, 7.5),
                        arrowprops=dict(arrowstyle='->', lw=2.5, color='#2196F3'))
        ax_arch.annotate('', xy=(6.5, 7.5), xytext=(4.8, 7.5),
                        arrowprops=dict(arrowstyle='->', lw=2.5, color='#FF9800'))
        
        # Joint Fusion Module
        fusion_box = plt.Rectangle((6.5, 6), 2.5, 3,
                                   facecolor='#E8F5E9', edgecolor='#4CAF50',
                                   linewidth=3)
        ax_arch.add_patch(fusion_box)
        ax_arch.text(7.75, 8.5, 'Joint Fusion\nModule', ha='center', va='center',
                    fontsize=11, fontweight='bold', color='#2E7D32')
        ax_arch.text(7.75, 7.8, 'Concatenate', ha='center', fontsize=9, style='italic')
        ax_arch.text(7.75, 7.4, 'Dense (128)', ha='center', fontsize=9, style='italic')
        ax_arch.text(7.75, 7.0, 'Dropout (0.3)', ha='center', fontsize=9, style='italic')
        ax_arch.text(7.75, 6.6, 'Softmax (5)', ha='center', fontsize=8, color='gray')
        
        # Training labels
        ax_arch.text(1.4, 5.5, 'Audio Loss\n& Gradients', ha='center',
                    fontsize=9, color='#1976D2', style='italic')
        ax_arch.text(3.9, 5.5, 'Text Loss\n& Gradients', ha='center',
                    fontsize=9, color='#F57C00', style='italic')
        ax_arch.text(7.75, 5.5, 'Joint Loss\n& Backprop', ha='center',
                    fontsize=9, color='#2E7D32', style='italic')
        
        # Loss backprop arrows
        ax_arch.annotate('', xy=(1.4, 6), xytext=(1.4, 5.8),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                      linestyle='--'))
        ax_arch.annotate('', xy=(3.9, 6), xytext=(3.9, 5.8),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                      linestyle='--'))
        ax_arch.annotate('', xy=(7.75, 6), xytext=(7.75, 5.8),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red',
                                      linestyle='--'))
        
        # --- Middle Left: Audio Stream Performance ---
        ax_audio = fig.add_subplot(gs[1, 0])
        ax_audio.plot(epochs, audio_train_acc, 'o-',
                     color='#2196F3', linewidth=2, markersize=5,
                     label='Audio Train', markerfacecolor='white', markeredgewidth=2)
        ax_audio.plot(epochs, audio_val_acc, 's-',
                     color='#1565C0', linewidth=2, markersize=5,
                     label='Audio Val', markerfacecolor='white', markeredgewidth=2)
        ax_audio.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax_audio.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax_audio.set_title('Audio Stream Training', fontsize=12, fontweight='bold')
        ax_audio.legend(loc='lower right', fontsize=9)
        ax_audio.grid(True, alpha=0.3, linestyle='--')
        ax_audio.set_ylim(0, 1.0)
        
        # --- Middle Right: Text Stream Performance ---
        ax_text = fig.add_subplot(gs[1, 1])
        ax_text.plot(epochs, text_train_acc, 'o-',
                    color='#FF9800', linewidth=2, markersize=5,
                    label='Text Train', markerfacecolor='white', markeredgewidth=2)
        ax_text.plot(epochs, text_val_acc, 's-',
                    color='#EF6C00', linewidth=2, markersize=5,
                    label='Text Val', markerfacecolor='white', markeredgewidth=2)
        ax_text.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax_text.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax_text.set_title('Text Stream Training', fontsize=12, fontweight='bold')
        ax_text.legend(loc='lower right', fontsize=9)
        ax_text.grid(True, alpha=0.3, linestyle='--')
        ax_text.set_ylim(0, 1.0)
        
        # --- Bottom: Joint Fusion Performance ---
        ax_fusion = fig.add_subplot(gs[2, :])
        
        # Plot all three for comparison
        ax_fusion.plot(epochs, audio_val_acc, '--',
                      color='#2196F3', linewidth=1.5, alpha=0.5,
                      label='Audio Only (Val)')
        ax_fusion.plot(epochs, text_val_acc, '--',
                      color='#FF9800', linewidth=1.5, alpha=0.5,
                      label='Text Only (Val)')
        ax_fusion.plot(epochs, fusion_train_acc, 'o-',
                      color='#4CAF50', linewidth=2.5, markersize=6,
                      label='Fusion Train', markerfacecolor='white', markeredgewidth=2)
        ax_fusion.plot(epochs, fusion_val_acc, 's-',
                      color='#2E7D32', linewidth=2.5, markersize=6,
                      label='Fusion Val', markerfacecolor='white', markeredgewidth=2)
        
        ax_fusion.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax_fusion.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax_fusion.set_title('Joint Fusion Training (Combined Modalities)',
                           fontsize=12, fontweight='bold')
        ax_fusion.legend(loc='lower right', fontsize=9, ncol=2)
        ax_fusion.grid(True, alpha=0.3, linestyle='--')
        ax_fusion.set_ylim(0, 1.0)
        
        # Highlight best fusion accuracy
        max_fusion_acc = max(fusion_val_acc)
        max_fusion_epoch = fusion_val_acc.index(max_fusion_acc) + 1
        ax_fusion.annotate(f'Best: {max_fusion_acc:.3f}',
                          xy=(max_fusion_epoch, max_fusion_acc),
                          xytext=(max_fusion_epoch, max_fusion_acc + 0.08),
                          arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.5),
                          fontsize=10, fontweight='bold', color='#2E7D32')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved multimodal pipeline plot: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_roc_comparison(
        self,
        audio_fpr: List[float],
        audio_tpr: List[float],
        audio_auc: float,
        text_fpr: List[float],
        text_tpr: List[float],
        text_auc: float,
        fusion_fpr: List[float],
        fusion_tpr: List[float],
        fusion_auc: float,
        title: str = "ROC Curve Comparison: Unimodal vs. Multimodal System",
        save_name: str = "roc_comparison.png"
    ):
        """
        Plot ROC curve comparison showing unimodal (audio/text) vs multimodal (fusion).
        FIXED: Better positioning to avoid label overlaps.
        """
        # Create figure with better spacing
        fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(17, 7))
        
        # Define cleaner color palette
        color_audio = '#1976D2'     # Blue
        color_text = '#F57C00'      # Orange  
        color_fusion = '#388E3C'    # Green
        color_random = '#757575'    # Gray
        
        # --- MAIN ROC PLOT ---
        # Plot diagonal reference line (random classifier)
        ax_main.plot([0, 1], [0, 1], color=color_random, linestyle='--', 
                    linewidth=1.5, alpha=0.4, label='Random (AUC=0.50)', zorder=1)
        
        # Plot audio unimodal
        ax_main.plot(
            audio_fpr, audio_tpr,
            color=color_audio, linewidth=2.5, linestyle='--',
            marker='o', markersize=3, markevery=15,
            label=f'Audio Only (AUC={audio_auc:.3f})',
            alpha=0.8, zorder=2
        )
        
        # Plot text unimodal
        ax_main.plot(
            text_fpr, text_tpr,
            color=color_text, linewidth=2.5, linestyle='--',
            marker='s', markersize=3, markevery=15,
            label=f'Text Only (AUC={text_auc:.3f})',
            alpha=0.8, zorder=3
        )
        
        # Plot multimodal fusion (emphasized)
        ax_main.plot(
            fusion_fpr, fusion_tpr,
            color=color_fusion, linewidth=3.5,
            marker='D', markersize=4, markevery=15,
            label=f'Multimodal Fusion (AUC={fusion_auc:.3f})',
            alpha=0.9, zorder=4
        )
        
        # Find optimal operating points (closest to top-left corner)
        def find_optimal_point(fpr, tpr):
            distances = np.sqrt(np.array(fpr)**2 + (1 - np.array(tpr))**2)
            optimal_idx = np.argmin(distances)
            return fpr[optimal_idx], tpr[optimal_idx], optimal_idx
        
        # FIXED: Collect optimal points and plot with smart positioning
        audio_opt_fpr, audio_opt_tpr, _ = find_optimal_point(audio_fpr, audio_tpr)
        text_opt_fpr, text_opt_tpr, _ = find_optimal_point(text_fpr, text_tpr)
        fusion_opt_fpr, fusion_opt_tpr, _ = find_optimal_point(fusion_fpr, fusion_tpr)
        
        # Plot star markers
        ax_main.scatter(audio_opt_fpr, audio_opt_tpr, s=150, c=color_audio, 
                       edgecolors='black', linewidths=2, zorder=10, marker='*')
        ax_main.scatter(text_opt_fpr, text_opt_tpr, s=150, c=color_text,
                       edgecolors='black', linewidths=2, zorder=10, marker='*')
        ax_main.scatter(fusion_opt_fpr, fusion_opt_tpr, s=150, c=color_fusion,
                       edgecolors='black', linewidths=2, zorder=10, marker='*')
        
        # FIXED: Smart label positioning to avoid overlaps
        # Calculate positions based on where points are
        points = [
            (audio_opt_fpr, audio_opt_tpr, color_audio, 'Audio', audio_auc),
            (text_opt_fpr, text_opt_tpr, color_text, 'Text', text_auc),
            (fusion_opt_fpr, fusion_opt_tpr, color_fusion, 'Fusion', fusion_auc)
        ]
        
        # Sort by FPR to determine left-to-right order
        points.sort(key=lambda x: x[0])
        
        # Position labels strategically
        for i, (opt_fpr, opt_tpr, color, name, auc) in enumerate(points):
            # Determine best label position based on location and neighbors
            if i == 0:  # Leftmost point
                # Place label to the left and slightly up
                text_x = opt_fpr - 0.08
                text_y = opt_tpr + 0.08
                ha = 'right'
            elif i == 1:  # Middle point
                # Place label above
                text_x = opt_fpr
                text_y = opt_tpr + 0.10
                ha = 'center'
            else:  # Rightmost point (i == 2)
                # Place label to the right
                text_x = opt_fpr + 0.08
                text_y = opt_tpr + 0.05
                ha = 'left'
            
            # Ensure labels stay within plot bounds
            text_x = max(0.05, min(0.95, text_x))
            text_y = max(0.1, min(0.98, text_y))
            
            # Add annotation with arrow
            ax_main.annotate(
                f'{name}\n({opt_fpr:.3f}, {opt_tpr:.3f})',
                xy=(opt_fpr, opt_tpr),
                xytext=(text_x, text_y),
                fontsize=9,
                fontweight='bold',
                color=color,
                ha=ha,
                va='bottom',
                arrowprops=dict(
                    arrowstyle='->', 
                    color=color, 
                    lw=1.5, 
                    alpha=0.7,
                    connectionstyle='arc3,rad=0.2'
                ),
                bbox=dict(
                    boxstyle='round,pad=0.4', 
                    facecolor='white', 
                    edgecolor=color, 
                    alpha=0.95,
                    linewidth=1.5
                ),
                zorder=11
            )
        
        ax_main.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax_main.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax_main.set_title('ROC Curves: Unimodal vs. Multimodal', fontsize=13, fontweight='bold')
        ax_main.legend(loc='lower right', fontsize=10, framealpha=0.95, 
                      edgecolor='gray', shadow=True)
        ax_main.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax_main.set_xlim([-0.02, 1.02])
        ax_main.set_ylim([-0.02, 1.05])
        
        # --- ZOOMED VIEW ---
        zoom_range_x = (0.0, 0.35)
        zoom_range_y = (0.65, 1.02)
        
        def zoom_data(fpr, tpr):
            mask = (np.array(fpr) <= zoom_range_x[1]) & (np.array(tpr) >= zoom_range_y[0])
            return np.array(fpr)[mask], np.array(tpr)[mask]
        
        audio_fpr_zoom, audio_tpr_zoom = zoom_data(audio_fpr, audio_tpr)
        text_fpr_zoom, text_tpr_zoom = zoom_data(text_fpr, text_tpr)
        fusion_fpr_zoom, fusion_tpr_zoom = zoom_data(fusion_fpr, fusion_tpr)
        
        if len(audio_fpr_zoom) > 0:
            ax_zoom.plot(audio_fpr_zoom, audio_tpr_zoom, 
                        color=color_audio, linewidth=2.5, linestyle='--',
                        marker='o', markersize=4, alpha=0.8, zorder=2)
        
        if len(text_fpr_zoom) > 0:
            ax_zoom.plot(text_fpr_zoom, text_tpr_zoom,
                        color=color_text, linewidth=2.5, linestyle='--',
                        marker='s', markersize=4, alpha=0.8, zorder=3)
        
        if len(fusion_fpr_zoom) > 0:
            ax_zoom.plot(fusion_fpr_zoom, fusion_tpr_zoom,
                        color=color_fusion, linewidth=3.5,
                        marker='D', markersize=5, alpha=0.9, zorder=4)
        
        # Mark optimal points in zoom view
        for opt_fpr, opt_tpr, color, name, _ in points:
            if zoom_range_x[0] <= opt_fpr <= zoom_range_x[1] and \
               zoom_range_y[0] <= opt_tpr <= zoom_range_y[1]:
                ax_zoom.scatter(opt_fpr, opt_tpr, s=120, c=color,
                              edgecolors='black', linewidths=2, zorder=10,
                              marker='*')
                
                # Add small label in zoom
                ax_zoom.text(opt_fpr, opt_tpr - 0.025, name,
                           fontsize=8, fontweight='bold', color=color,
                           ha='center', va='top',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', 
                                   edgecolor=color,
                                   alpha=0.9, linewidth=1))
        
        ax_zoom.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        ax_zoom.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        ax_zoom.set_title('Zoomed View (High Sensitivity Region)', fontsize=12, fontweight='bold')
        ax_zoom.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax_zoom.set_xlim(zoom_range_x)
        ax_zoom.set_ylim(zoom_range_y)
        
        # FIXED: Better positioned metrics table
        metrics_text = (
            f"AUC SCORES\n"
            f"{'─'*25}\n"
            f"Audio:   {audio_auc:.4f}\n"
            f"Text:    {text_auc:.4f}\n"
            f"Fusion:  {fusion_auc:.4f}\n"
            f"{'─'*25}\n"
            f"IMPROVEMENT\n"
            f"vs Audio: +{(fusion_auc - audio_auc)*100:.2f}%\n"
            f"vs Text:  +{(fusion_auc - text_auc)*100:.2f}%"
        )
        
        # Place metrics box in bottom-right of zoom plot
        ax_zoom.text(
            0.98, 0.02, metrics_text,
            transform=ax_zoom.transAxes,
            fontsize=8.5,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(
                boxstyle='round,pad=0.6', 
                facecolor='white', 
                edgecolor='#424242', 
                alpha=0.95, 
                linewidth=1.5
            ),
            family='monospace',
            zorder=12
        )
        
        plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save plot
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved ROC comparison: {save_path}")
        
        plt.show()
        
        return fig














































































































    # ...existing code...        print("\n7. Creating training report...")  # Changed from 7 to 8    # Create training report        )        fusion_auc=fusion_auc        fusion_tpr=fusion_tpr.tolist(),        fusion_fpr=fusion_fpr.tolist(),        text_auc=text_auc,        text_tpr=text_tpr.tolist(),        text_fpr=text_fpr.tolist(),        audio_auc=audio_auc,        audio_tpr=audio_tpr.tolist(),        audio_fpr=audio_fpr.tolist(),    visualizer.plot_roc_comparison(        fusion_tpr = np.maximum.accumulate(fusion_tpr)    text_tpr = np.maximum.accumulate(text_tpr)    audio_tpr = np.maximum.accumulate(audio_tpr)    # Ensure monotonic increase        fusion_auc = 0.95    fusion_tpr = np.clip(fusion_tpr, 0, 1)    fusion_tpr = 1 - (1 - fusion_fpr) ** 2.8 + np.random.normal(0, 0.01, n_points)    fusion_fpr = np.linspace(0, 1, n_points)    # Fusion model (best performance)        text_auc = 0.91    text_tpr = np.clip(text_tpr, 0, 1)    text_tpr = 1 - (1 - text_fpr) ** 2.2 + np.random.normal(0, 0.015, n_points)    text_fpr = np.linspace(0, 1, n_points)    # Text model (better than audio)        audio_auc = 0.87    audio_tpr = np.clip(audio_tpr, 0, 1)    audio_tpr = 1 - (1 - audio_fpr) ** 1.8 + np.random.normal(0, 0.02, n_points)    audio_fpr = np.linspace(0, 1, n_points)    n_points = 100    # Audio model (good but not perfect)    # Generate realistic ROC curves        print("\n6. Generating ROC curve comparison...")    # Example 6: ROC Curve Comparison        )        save_name="example_multimodal_pipeline.png"        title="Sample Multimodal Training Pipeline",        fusion_val_acc=[0.25, 0.45, 0.65, 0.85, 0.9],        text_val_acc=[0.3, 0.5, 0.7, 0.85, 0.92],        audio_val_acc=[0.2, 0.4, 0.6, 0.8, 0.9],        fusion_train_acc=[0.15, 0.35, 0.55, 0.75, 0.85],        text_train_acc=[0.2, 0.4, 0.6, 0.8, 0.95],        audio_train_acc=[0.1, 0.3, 0.5, 0.7, 0.9],    visualizer.plot_multimodal_pipeline(    print("\n5. Generating multimodal pipeline plot...")    # Example 5: Multimodal Pipeline        )        save_name="example_feature_importance.png"        title="Sample Feature Importance",        top_n=10,        importance=np.random.rand(20),        features=[f'Feature {i}' for i in range(1, 21)],    visualizer.plot_feature_importance(    print("\n4. Generating feature importance plot...")    # Example 4: Feature Importance        )        save_name="example_learning_curves.png"        title="Sample Learning Curves",        val_scores=[0.55, 0.6, 0.65, 0.7, 0.75],        train_scores=[0.6, 0.65, 0.7, 0.75, 0.8],        train_sizes=[100, 200, 300, 400, 500],    visualizer.plot_learning_curves(    print("\n3. Generating learning curves plot...")    # Example 3: Learning Curves        )        normalize=True        save_name="example_confusion_matrix.png",        title="Sample Confusion Matrix",        classes=['Class A', 'Class B', 'Class C'],        cm=cm,    visualizer.plot_confusion_matrix(                   [ 0,  3, 47]])                   [ 5, 45, 5],    cm = np.array([[50, 2, 1],    print("\n2. Generating confusion matrix plot...")    # Example 2: Confusion Matrix        )        save_name="example_training_history.png"        title="Sample Model Training History",        val_loss=[2.5, 2.0, 1.5, 1.2, 0.8],        train_loss=[2.3, 1.8, 1.2, 0.9, 0.5],        val_acc=[0.2, 0.3, 0.5, 0.7, 0.85],        train_acc=[0.1, 0.4, 0.6, 0.8, 0.9],    visualizer.plot_training_history(    print("1. Generating simple training history plot...")    # Example 1: Simple Training History        visualizer = TrainingVisualizer()        print("="*60 + "\n")    print("CREATING EXAMPLE TRAINING VISUALIZATIONS")    print("\n" + "="*60)    """Create example training plots with synthetic data."""def create_example_plots():        return fig


def create_example_plots():
    """Create example training plots with synthetic data."""
    print("\n" + "="*60)
    print("CREATING EXAMPLE TRAINING VISUALIZATIONS")
    print("="*60 + "\n")
    
    visualizer = TrainingVisualizer()
    
    # Example 1: Basic Training History
    print("1. Generating basic training history plot...")
    epochs = list(range(1, 26))
    train_acc = [0.1*i + np.random.normal(0, 0.02) for i in epochs]
    val_acc = [0.1*i + np.random.normal(0, 0.025) for i in epochs]
    train_loss = [1.0/(i+1) + np.random.normal(0, 0.01) for i in epochs]
    val_loss = [1.0/(i+1) + np.random.normal(0, 0.015) for i in epochs]
    
    visualizer.plot_training_history(train_acc, val_acc, train_loss, val_loss)
    
    # Example 2: Confusion Matrix
    print("\n2. Generating confusion matrix...")
    cm = np.array([[50, 2, 1],
                   [ 5, 45, 5],
                   [ 0,  3, 47]])
    classes = ['Class A', 'Class B', 'Class C']
    visualizer.plot_confusion_matrix(cm, classes)
    
    # Example 3: Learning Curves
    print("\n3. Generating learning curves...")
    train_sizes = [100, 200, 300, 400, 500]
    train_scores = [0.2, 0.4, 0.6, 0.8, 0.9]
    val_scores = [0.18, 0.38, 0.58, 0.78, 0.88]
    visualizer.plot_learning_curves(train_sizes, train_scores, val_scores)
    
    # Example 4: Feature Importance
    print("\n4. Generating feature importance plot...")
    features = [f'Feature {i}' for i in range(1, 21)]
    importance = np.random.rand(20)
    visualizer.plot_feature_importance(features, importance)
    
    # Example 5: Training Report
    print("\n5. Creating training report...")
    metrics = {
        'accuracy': 0.85,
        'loss': 0.35,
        'val_accuracy': 0.82,
        'val_loss': 0.38
    }
    visualizer.create_training_report(metrics)
    
    # Example 6: Multimodal Pipeline (NEW)
    print("\n6. Generating multimodal training pipeline...")
    
    # Simulate realistic multimodal training
    epochs = 25
    
    # Audio stream (slower convergence)
    audio_train = [0.45 + 0.014*i + np.random.normal(0, 0.02) for i in range(epochs)]
    audio_val = [0.45 + 0.011*i + np.random.normal(0, 0.025) for i in range(epochs)]
    
    # Text stream (faster convergence, higher ceiling)
    text_train = [0.52 + 0.016*i + np.random.normal(0, 0.02) for i in range(epochs)]
    text_val = [0.52 + 0.013*i + np.random.normal(0, 0.025) for i in range(epochs)]
    
    # Fusion (best performance, benefits from both)
    fusion_train = [0.58 + 0.018*i + np.random.normal(0, 0.015) for i in range(epochs)]
    fusion_val = [0.58 + 0.015*i + np.random.normal(0, 0.02) for i in range(epochs)]
    
    # Cap accuracies
    audio_train = [min(0.85, x) for x in audio_train]
    audio_val = [min(0.80, x) for x in audio_val]
    text_train = [min(0.90, x) for x in text_train]
    text_val = [min(0.85, x) for x in text_val]
    fusion_train = [min(0.95, x) for x in fusion_train]
    fusion_val = [min(0.92, x) for x in fusion_val]
    
    visualizer.plot_multimodal_pipeline(
        audio_train_acc=audio_train,
        text_train_acc=text_train,
        fusion_train_acc=fusion_train,
        audio_val_acc=audio_val,
        text_val_acc=text_val,
        fusion_val_acc=fusion_val
    )
    
    # Create training report
    print("\n7. Creating training report...")
    metrics = {
        'accuracy': 0.92,
        'loss': 0.28,
        'val_accuracy': 0.90,
        'val_loss': 0.30,
        'train_time': 150.5,
        'epochs': 25,
        'model_size_mb': 15.2
    }
    visualizer.create_training_report(metrics, save_name="training_report_v2.json")
    
    print("\n" + "="*60)
    print("VISUALIZATION CREATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    create_example_plots()
