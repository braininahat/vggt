#!/usr/bin/env python3
"""
Test script for UltrasoundHDF5Dataset to verify it loads data correctly.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add the training directory to Python path
import sys
sys.path.append('training')

from training.data.datasets.ultrasound_hdf5 import UltrasoundHDF5Dataset


def visualize_batch(batch_data, save_path=None):
    """Visualize a batch of data from the dataset."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Get first sequence
    images = batch_data['images'][0]  # Shape: (S, C, H, W)
    depths = batch_data['depths'][0]  # Shape: (S, H, W)
    world_points = batch_data['world_points'][0]  # Shape: (S, H, W, 3)
    masks = batch_data['point_masks'][0]  # Shape: (S, H, W)
    
    num_frames = min(4, len(images))
    
    for i in range(num_frames):
        # Show image
        ax_img = axes[0, i]
        img = images[i].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * 0.225 + 0.45
        img = np.clip(img, 0, 1)
        ax_img.imshow(img)
        ax_img.set_title(f'Frame {i}')
        ax_img.axis('off')
        
        # Show depth
        ax_depth = axes[1, i]
        depth = depths[i].numpy()
        im = ax_depth.imshow(depth, cmap='viridis')
        ax_depth.set_title(f'Depth {i}')
        ax_depth.axis('off')
        plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)
    
    plt.suptitle(f"Sequence: {batch_data['seq_name'][0]}")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def test_dataset():
    """Test the UltrasoundHDF5Dataset."""
    
    # Create mock configuration
    common_conf = OmegaConf.create({
        'img_size': 518,
        'patch_size': 14,
        'augs': {'scales': [0.8, 1.2]},
        'rescale': True,
        'rescale_aug': False,  # Disable augmentation for now
        'landscape_check': False,  # Ultrasound images are typically portrait
        'debug': False,
        'training': True,
        'get_nearby': False,
        'load_depth': True,
        'inside_random': False,
    })
    
    # Initialize dataset
    dataset = UltrasoundHDF5Dataset(
        common_conf=common_conf,
        hdf5_root_dir='/home/varun/XiaLabSync/datasets/freehand_May212025/processed',
        split='train',
        sequence_length=8,
        stride=5,  # Skip some frames to see more variation
        max_sequences_per_file=5,
        enable_augmentation=False,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a single item
    print("\nLoading first item...")
    item = dataset[0]
    
    # Print shapes and info
    print(f"\nData shapes:")
    print(f"  Images: {item['images'].shape}")
    print(f"  Extrinsics: {item['extrinsics'].shape}")
    print(f"  Intrinsics: {item['intrinsics'].shape}")
    print(f"  Depths: {item['depths'].shape}")
    print(f"  World points: {item['world_points'].shape}")
    print(f"  Point masks: {item['point_masks'].shape}")
    print(f"  Sequence name: {item['seq_name']}")
    
    # Check data statistics
    print(f"\nData statistics:")
    print(f"  Image range: [{item['images'].min():.3f}, {item['images'].max():.3f}]")
    print(f"  Depth range: [{item['depths'].min():.3f}, {item['depths'].max():.3f}] mm")
    print(f"  World points X range: [{item['world_points'][..., 0].min():.3f}, {item['world_points'][..., 0].max():.3f}] mm")
    print(f"  World points Y range: [{item['world_points'][..., 1].min():.3f}, {item['world_points'][..., 1].max():.3f}] mm")
    print(f"  World points Z range: [{item['world_points'][..., 2].min():.3f}, {item['world_points'][..., 2].max():.3f}] mm")
    
    # Test with different aspect ratios
    print("\nTesting different aspect ratios...")
    for aspect_ratio in [0.75, 1.0, 1.25]:
        item = dataset.get_data(0, img_per_seq=4, aspect_ratio=aspect_ratio)
        print(f"  Aspect ratio {aspect_ratio}: Image shape = {item['images'].shape}")
    
    # Visualize
    print("\nVisualizing data...")
    batch_data = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in item.items()}
    visualize_batch(batch_data, save_path='ultrasound_dataset_test.png')
    
    print("\nDataset test completed successfully!")


if __name__ == '__main__':
    test_dataset()