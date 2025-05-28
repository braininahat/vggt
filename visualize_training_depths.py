#!/usr/bin/env python3
"""
Quick visualization of depth predictions during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

def visualize_batch_depths(batch, predictions, epoch, save_dir="./depth_visualizations"):
    """
    Visualize depth predictions from a training batch.
    
    Args:
        batch: Training batch dict
        predictions: Model predictions dict
        epoch: Current epoch number
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Get first sequence from batch
    images = batch['images'][0].cpu()  # Shape: (S, 3, H, W)
    gt_depths = batch['depths'][0].cpu()  # Shape: (S, H, W)
    pred_depths = predictions['depth'][0].cpu()  # Shape: (S, H, W, 1)
    pred_depths = pred_depths.squeeze(-1)  # Remove last dimension
    
    # Convert images from normalized to RGB
    images = (images * 0.225 + 0.45) * 255  # Denormalize
    images = images.clamp(0, 255).byte()
    images = images.permute(0, 2, 3, 1)  # (S, H, W, 3)
    
    # Number of frames to visualize
    num_frames = min(4, images.shape[0])
    
    fig, axes = plt.subplots(3, num_frames, figsize=(4*num_frames, 12))
    if num_frames == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_frames):
        # Original image
        ax = axes[0, i]
        ax.imshow(images[i])
        ax.set_title(f'Frame {i}')
        ax.axis('off')
        
        # Ground truth depth
        ax = axes[1, i]
        im = ax.imshow(gt_depths[i], cmap='viridis')
        ax.set_title(f'GT Depth (range: {gt_depths[i].min():.1f}-{gt_depths[i].max():.1f})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Predicted depth
        ax = axes[2, i]
        im = ax.imshow(pred_depths[i], cmap='viridis')
        ax.set_title(f'Pred Depth (range: {pred_depths[i].min():.1f}-{pred_depths[i].max():.1f})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'Depth Predictions - Epoch {epoch}')
    plt.tight_layout()
    
    # Save figure
    save_path = save_dir / f'depth_epoch_{epoch}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved depth visualization to {save_path}")
    
    # Also save depth error visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Compute absolute error for first frame
    depth_error = torch.abs(pred_depths[0] - gt_depths[0])
    im = ax.imshow(depth_error, cmap='hot')
    ax.set_title(f'Absolute Depth Error - Epoch {epoch}\nMean: {depth_error.mean():.2f}, Max: {depth_error.max():.2f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    
    error_path = save_dir / f'depth_error_epoch_{epoch}.png'
    plt.savefig(error_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error visualization to {error_path}")


if __name__ == "__main__":
    # Example usage with dummy data
    print("This script is meant to be called from the training loop.")
    print("Add this to your training loop:")
    print("  from visualize_training_depths import visualize_batch_depths")
    print("  if epoch % 5 == 0:  # Every 5 epochs")
    print("      visualize_batch_depths(batch, predictions, epoch)")