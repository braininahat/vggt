# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import h5py
import numpy as np
from pathlib import Path
from glob import glob
from natsort import natsorted
import torch
from torch.utils.data import Dataset
import cv2

from training.data.base_dataset import BaseDataset
from training.data.dataset_util import *


class UltrasoundHDF5Dataset(BaseDataset):
    """
    Dataset for loading ultrasound HDF5 files with pointmaps.
    
    This dataset:
    - Loads sequences of ultrasound frames with corresponding 3D pointmaps
    - Generates synthetic camera parameters from pointmap geometry
    - Treats sequences as continuous volume sweeps (temporal continuity)
    - No augmentation by default (can be enabled later)
    """
    
    def __init__(
        self,
        common_conf,
        hdf5_root_dir: str,
        split: str = "train",
        sequence_length: int = 8,
        stride: int = 1,
        max_sequences_per_file: int = 10,
        enable_augmentation: bool = False,
        train_split_ratio: float = 0.8,
        random_seed: int = 42,
    ):
        """
        Initialize the UltrasoundHDF5Dataset.
        
        Args:
            common_conf: Common configuration from BaseDataset
            hdf5_root_dir: Root directory containing HDF5 files
            split: 'train' or 'test' split
            sequence_length: Number of frames per sequence
            stride: Stride between frames in a sequence
            max_sequences_per_file: Maximum sequences to sample from each HDF5 file
            enable_augmentation: Whether to enable data augmentation (default: False)
            train_split_ratio: Ratio of data to use for training
            random_seed: Random seed for reproducible splits
        """
        super().__init__(common_conf=common_conf)
        
        self.hdf5_root_dir = Path(hdf5_root_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_sequences_per_file = max_sequences_per_file
        self.enable_augmentation = enable_augmentation
        self.training = (split == "train")
        
        # Find all HDF5 files
        self.hdf5_files = natsorted(glob(str(self.hdf5_root_dir / "**" / "*_volume.h5"), recursive=True))
        if not self.hdf5_files:
            raise ValueError(f"No HDF5 files found in {self.hdf5_root_dir}")
        
        print(f"Found {len(self.hdf5_files)} HDF5 files")
        
        # Create train/test split
        np.random.seed(random_seed)
        indices = np.random.permutation(len(self.hdf5_files))
        split_idx = int(len(self.hdf5_files) * train_split_ratio)
        
        if split == "train":
            self.hdf5_files = [self.hdf5_files[i] for i in indices[:split_idx]]
        else:
            self.hdf5_files = [self.hdf5_files[i] for i in indices[split_idx:]]
        
        print(f"{split} split: {len(self.hdf5_files)} files")
        
        # Build sequence index
        self._build_sequence_index()
        
        # Set dataset length for BaseDataset
        self.len_train = len(self.sequences)
        
    def _build_sequence_index(self):
        """Build an index of all possible sequences across all HDF5 files."""
        self.sequences = []
        
        for hdf5_path in self.hdf5_files:
            with h5py.File(hdf5_path, 'r') as hf:
                num_frames = len(hf['frames'])
                
                # Sample sequences with stride
                max_start_idx = num_frames - (self.sequence_length - 1) * self.stride
                if max_start_idx <= 0:
                    print(f"Skipping {hdf5_path}: not enough frames")
                    continue
                
                # Sample evenly spaced starting indices
                num_sequences = min(max_start_idx, self.max_sequences_per_file)
                if num_sequences > 1:
                    start_indices = np.linspace(0, max_start_idx - 1, num_sequences, dtype=int)
                else:
                    start_indices = [0]
                
                for start_idx in start_indices:
                    self.sequences.append({
                        'hdf5_path': hdf5_path,
                        'start_idx': start_idx,
                        'indices': list(range(start_idx, start_idx + self.sequence_length * self.stride, self.stride))
                    })
        
        print(f"Built index with {len(self.sequences)} sequences")
    
    def generate_synthetic_cameras_from_pointmap(self, pointmap, frame_idx=0, total_frames=1):
        """
        Generate synthetic camera parameters from a pointmap.
        Same logic as in visualization script.
        """
        H, W = pointmap.shape[:2]
        
        # Get corners of the imaging plane
        corners = np.array([
            pointmap[0, 0],          # top-left
            pointmap[0, W-1],        # top-right
            pointmap[H-1, W-1],      # bottom-right
            pointmap[H-1, 0]         # bottom-left
        ])
        
        # Calculate center of imaging plane
        plane_center = pointmap[H//2, W//2]
        
        # Calculate normal to the imaging plane
        u_vec = corners[1] - corners[0]
        v_vec = corners[3] - corners[0]
        normal = np.cross(u_vec, v_vec)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        
        # Position camera along the normal
        plane_size = max(np.linalg.norm(u_vec), np.linalg.norm(v_vec))
        camera_distance = plane_size * 0.5
        camera_center = plane_center + normal * camera_distance
        
        # Create camera coordinate system
        z_axis = -normal
        x_axis = u_vec / (np.linalg.norm(u_vec) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        
        # Build rotation matrix (world-to-camera)
        R_w2c = np.stack([x_axis, y_axis, z_axis], axis=0)
        
        # Camera-to-world transformation
        R_c2w = R_w2c.T
        t_c2w = camera_center
        
        # Build 4x4 extrinsic matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_c2w
        extrinsic[:3, 3] = t_c2w
        
        # Create intrinsic matrix
        fov = 2 * np.arctan(plane_size / (2 * camera_distance))
        focal_length = (W / 2) / np.tan(fov / 2)
        
        intrinsic = np.array([
            [focal_length, 0, W / 2],
            [0, focal_length, H / 2],
            [0, 0, 1]
        ])
        
        return extrinsic, intrinsic
    
    def get_data(self, seq_index, img_per_seq=None, aspect_ratio=1.0):
        """
        Get a sequence of data from the dataset.
        
        Args:
            seq_index: Index of the sequence
            img_per_seq: Number of images per sequence (overrides default)
            aspect_ratio: Target aspect ratio
            
        Returns:
            Dictionary containing:
                - images: (S, 3, H, W) tensor of images
                - extrinsics: (S, 4, 4) camera extrinsics
                - intrinsics: (S, 3, 3) camera intrinsics
                - depths: (S, H, W) depth maps
                - world_points: (S, H, W, 3) 3D world coordinates
                - point_masks: (S, H, W) valid point masks
                - seq_name: Name of the sequence
        """
        sequence_info = self.sequences[seq_index]
        hdf5_path = sequence_info['hdf5_path']
        frame_indices = sequence_info['indices']
        
        if img_per_seq is not None:
            # Adjust sequence length if requested
            frame_indices = frame_indices[:img_per_seq]
        
        # Load data from HDF5
        with h5py.File(hdf5_path, 'r') as hf:
            # Load frames and pointmaps
            frames = []
            pointmaps = []
            
            for idx in frame_indices:
                frame = hf['frames'][idx]
                pointmap = hf['pointmaps'][idx]
                
                frames.append(frame)
                pointmaps.append(pointmap)
            
            frames = np.stack(frames)
            pointmaps = np.stack(pointmaps)
            
            # Get metadata
            is_grayscale = hf.attrs.get('is_grayscale', False)
        
        # Convert grayscale to RGB if needed
        if is_grayscale:
            frames = np.stack([frames] * 3, axis=-1)
        
        # Generate synthetic cameras and depth maps
        extrinsics = []
        intrinsics = []
        depths = []
        
        for i, pointmap in enumerate(pointmaps):
            # Generate camera parameters
            extrinsic, intrinsic = self.generate_synthetic_cameras_from_pointmap(
                pointmap, i, len(pointmaps)
            )
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            
            # Calculate depth from camera center
            camera_center = extrinsic[:3, 3]
            depth = np.linalg.norm(pointmap - camera_center, axis=2)
            depths.append(depth)
        
        extrinsics = np.stack(extrinsics)
        intrinsics = np.stack(intrinsics)
        depths = np.stack(depths)
        
        # Create validity masks (non-zero regions)
        # For ultrasound, we typically have valid data throughout the imaging region
        point_masks = np.ones_like(depths, dtype=bool)
        
        # Get target shape based on aspect ratio
        target_shape = self.get_target_shape(aspect_ratio)
        
        # Process each frame
        processed_data = []
        for i in range(len(frames)):
            # For ultrasound, we need to handle pointmaps specially
            # We'll pass a dummy depth map and then use our pointmap
            dummy_depth = np.zeros(frames[i].shape[:2])
            
            # Process image (this handles resizing, cropping, etc.)
            image, _, extri, intri, _, _, mask, _ = self.process_one_image(
                frames[i],
                dummy_depth,
                extrinsics[i],
                intrinsics[i],
                np.array(frames[i].shape[:2]),
                target_shape,
                track=None,
                safe_bound=4,
            )
            
            # Apply the same transformations to pointmap
            # First, we need to resize/crop the pointmap to match the processed image
            pointmap = pointmaps[i]
            original_shape = frames[i].shape[:2]
            processed_shape = image.shape[:2]
            
            # If shapes don't match, we need to resize the pointmap
            if original_shape != processed_shape:
                # Resize each coordinate channel
                pointmap_resized = np.zeros((*processed_shape, 3))
                for c in range(3):
                    pointmap_resized[:, :, c] = cv2.resize(
                        pointmap[:, :, c], 
                        (processed_shape[1], processed_shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    )
                pointmap = pointmap_resized
            
            # Recalculate depth with the processed pointmap
            camera_center = extri[:3, 3]
            depth = np.linalg.norm(pointmap - camera_center, axis=2)
            
            processed_data.append({
                'image': image,
                'extrinsic': extri,
                'intrinsic': intri,
                'depth': depth,
                'world_points': pointmap,
                'mask': mask,
            })
        
        # Stack all processed data
        images = np.stack([d['image'] for d in processed_data])
        extrinsics = np.stack([d['extrinsic'] for d in processed_data])
        intrinsics = np.stack([d['intrinsic'] for d in processed_data])
        depths_proc = np.stack([d['depth'] for d in processed_data])
        world_points = np.stack([d['world_points'] for d in processed_data])
        masks = np.stack([d['mask'] for d in processed_data])
        
        # Convert to torch tensors and proper format
        images = torch.from_numpy(images).float()
        # Convert from (S, H, W, C) to (S, C, H, W)
        images = images.permute(0, 3, 1, 2)
        # Normalize images (ImageNet normalization)
        images = (images / 255.0 - 0.45) / 0.225
        
        extrinsics = torch.from_numpy(extrinsics).float()
        intrinsics = torch.from_numpy(intrinsics).float()
        depths_proc = torch.from_numpy(depths_proc).float()
        world_points = torch.from_numpy(world_points).float()
        masks = torch.from_numpy(masks).bool()
        
        # Create sequence name
        seq_name = f"{Path(hdf5_path).stem}_seq{seq_index}"
        
        return {
            'images': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'depths': depths_proc,
            'world_points': world_points,
            'point_masks': masks,
            'seq_name': seq_name,
            'dataset': 'ultrasound',
        }
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """Override to handle simple indexing."""
        if isinstance(idx, int):
            # Default values for img_per_seq and aspect_ratio
            return self.get_data(idx, img_per_seq=self.sequence_length, aspect_ratio=1.0)
        else:
            # Handle tuple input from BaseDataset
            return super().__getitem__(idx)