"""
Ultrasound HDF5 dataset with padding to preserve aspect ratio and geometric accuracy.
"""
import numpy as np
import torch
import h5py
from pathlib import Path
from training.data.base_dataset import BaseDataset
import cv2


class UltrasoundHDF5PaddedDataset(BaseDataset):
    """
    Dataset for ultrasound HDF5 files that preserves aspect ratio using padding.
    """
    
    def __init__(
        self,
        common_conf,
        hdf5_root_dir,
        split="train",
        sequence_length=8,
        stride=5,
        max_sequences_per_file=100,
        enable_augmentation=False,
        train_split_ratio=0.8,
    ):
        super().__init__(common_conf)
        
        self.hdf5_root_dir = Path(hdf5_root_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_sequences_per_file = max_sequences_per_file
        self.enable_augmentation = enable_augmentation
        self.train_split_ratio = train_split_ratio
        
        # Build dataset index
        self.sequences = self._build_sequences()
        
    def regenerate_pointmap(self, corners, H, W):
        """
        Regenerate pointmap at new resolution using bilinear interpolation.
        
        Args:
            corners: Dict with c1, c2, c3, c4 corner coordinates
            H, W: Target dimensions
            
        Returns:
            pointmap: (H, W, 3) array
        """
        # Create normalized coordinates
        u = np.linspace(0, 1, W, dtype=np.float32)
        v = np.linspace(0, 1, H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)
        
        # Bilinear interpolation
        c1, c2, c3, c4 = corners['c1'], corners['c2'], corners['c3'], corners['c4']
        
        # Top edge: interpolate between c1 and c2
        top = c1[None, None, :] * (1 - uu[:, :, None]) + c2[None, None, :] * uu[:, :, None]
        # Bottom edge: interpolate between c4 and c3
        bottom = c4[None, None, :] * (1 - uu[:, :, None]) + c3[None, None, :] * uu[:, :, None]
        # Final interpolation between top and bottom
        pointmap = top * (1 - vv[:, :, None]) + bottom * vv[:, :, None]
        
        return pointmap.astype(np.float32)
    
    def process_frame_with_padding(self, frame, pointmap, corners, target_size=518):
        """
        Process frame and pointmap with padding to preserve aspect ratio.
        
        Args:
            frame: Original frame (H, W) or (H, W, 3)
            pointmap: Original pointmap (H, W, 3)
            corners: Corner coordinates
            target_size: Target square size
            
        Returns:
            padded_frame, padded_pointmap, padding_mask
        """
        H, W = frame.shape[:2]
        
        # Calculate scaling to fit in target_size while preserving aspect ratio
        scale = min(target_size / H, target_size / W)
        new_H = int(H * scale)
        new_W = int(W * scale)
        
        # Ensure dimensions are divisible by patch_size (14)
        patch_size = 14
        new_H = (new_H // patch_size) * patch_size
        new_W = (new_W // patch_size) * patch_size
        
        # Resize frame
        if len(frame.shape) == 3:
            resized_frame = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
        else:
            resized_frame = cv2.resize(frame, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            resized_frame = resized_frame[:, :, None]
        
        # Regenerate pointmap at new resolution (don't interpolate 3D coords!)
        resized_pointmap = self.regenerate_pointmap(corners, new_H, new_W)
        
        # Calculate padding
        pad_top = (target_size - new_H) // 2
        pad_bottom = target_size - new_H - pad_top
        pad_left = (target_size - new_W) // 2
        pad_right = target_size - new_W - pad_left
        
        # Pad frame
        if len(resized_frame.shape) == 3:
            padded_frame = np.pad(
                resized_frame,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0
            )
        else:
            padded_frame = np.pad(
                resized_frame,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )
        
        # Pad pointmap with zeros (invalid regions)
        padded_pointmap = np.pad(
            resized_pointmap,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0
        )
        
        # Create mask for valid regions
        padding_mask = np.ones((new_H, new_W), dtype=bool)
        padding_mask = np.pad(
            padding_mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=False
        )
        
        return padded_frame, padded_pointmap, padding_mask, scale
    
    def get_data(self, seq_index, img_per_seq=None, aspect_ratio=1.0):
        """
        Get a sequence with padding to preserve aspect ratio.
        """
        sequence_info = self.sequences[seq_index]
        hdf5_path = sequence_info['hdf5_path']
        frame_indices = sequence_info['indices']
        
        if img_per_seq is not None:
            frame_indices = frame_indices[:img_per_seq]
        
        # Load data from HDF5
        with h5py.File(hdf5_path, 'r') as hf:
            # Load frames and metadata
            frames = []
            pointmaps = []
            corners_list = []
            
            # Get corner data
            corners_group = hf['corners']
            
            for idx in frame_indices:
                frame = hf['frames'][idx]
                frames.append(frame)
                
                # Get corners for this frame
                corners = {
                    'c1': np.array([corners_group[f'c1_{ax}'][idx] for ax in ['x', 'y', 'z']]),
                    'c2': np.array([corners_group[f'c2_{ax}'][idx] for ax in ['x', 'y', 'z']]),
                    'c3': np.array([corners_group[f'c3_{ax}'][idx] for ax in ['x', 'y', 'z']]),
                    'c4': np.array([corners_group[f'c4_{ax}'][idx] for ax in ['x', 'y', 'z']]),
                }
                corners_list.append(corners)
            
            frames = np.stack(frames)
            is_grayscale = hf.attrs.get('is_grayscale', False)
        
        # Convert grayscale to RGB if needed
        if is_grayscale:
            frames = np.stack([frames] * 3, axis=-1)
        
        # Process each frame with padding
        processed_frames = []
        processed_pointmaps = []
        processed_masks = []
        scales = []
        
        target_size = self.img_size  # 518
        
        for i in range(len(frames)):
            frame = frames[i]
            corners = corners_list[i]
            
            # Process with padding
            padded_frame, padded_pointmap, mask, scale = self.process_frame_with_padding(
                frame, None, corners, target_size
            )
            
            processed_frames.append(padded_frame)
            processed_pointmaps.append(padded_pointmap)
            processed_masks.append(mask)
            scales.append(scale)
        
        # Stack processed data
        images = np.stack(processed_frames)
        world_points = np.stack(processed_pointmaps)
        masks = np.stack(processed_masks)
        
        # Generate synthetic cameras (same as before)
        extrinsics = []
        intrinsics = []
        depths = []
        
        for i in range(len(images)):
            # Use the scale to adjust intrinsics
            scale = scales[i]
            
            # Generate synthetic camera
            extrinsic, intrinsic = self.generate_synthetic_camera(
                world_points[i], i, len(images), scale
            )
            extrinsics.append(extrinsic)
            intrinsics.append(intrinsic)
            
            # Calculate depth
            camera_center = extrinsic[:3, 3]
            depth = np.linalg.norm(world_points[i] - camera_center, axis=2)
            depths.append(depth)
        
        # Convert to torch tensors
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2)
        images = (images / 255.0 - 0.45) / 0.225
        
        extrinsics = torch.from_numpy(np.stack(extrinsics)).float()
        intrinsics = torch.from_numpy(np.stack(intrinsics)).float()
        depths = torch.from_numpy(np.stack(depths)).float()
        world_points = torch.from_numpy(world_points).float()
        masks = torch.from_numpy(masks).bool()
        
        seq_name = f"{Path(hdf5_path).stem}_seq{seq_index}"
        
        return {
            'images': images,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
            'depths': depths,
            'world_points': world_points,
            'point_masks': masks,
            'seq_name': seq_name,
            'dataset': 'ultrasound',
        }
    
    def generate_synthetic_camera(self, pointmap, frame_idx, total_frames, scale=1.0):
        """Generate synthetic camera with scale adjustment for intrinsics."""
        H, W = pointmap.shape[:2]
        
        # Find valid region
        valid_mask = np.any(pointmap != 0, axis=2)
        valid_points = pointmap[valid_mask]
        
        if len(valid_points) < 4:
            # Fallback to center region
            center_y, center_x = H // 2, W // 2
            valid_points = pointmap[center_y-10:center_y+10, center_x-10:center_x+10].reshape(-1, 3)
        
        # Get bounding box of valid region
        ys, xs = np.where(valid_mask)
        if len(ys) > 0:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            
            # Get corners of valid region
            corners = np.array([
                pointmap[y_min, x_min],
                pointmap[y_min, x_max],
                pointmap[y_max, x_max],
                pointmap[y_max, x_min]
            ])
        else:
            # Use image corners
            corners = np.array([
                pointmap[0, 0],
                pointmap[0, W-1],
                pointmap[H-1, W-1],
                pointmap[H-1, 0]
            ])
        
        # Calculate plane center and normal
        plane_center = np.mean(valid_points, axis=0)
        
        # Calculate normal using SVD
        centered_points = valid_points - plane_center
        _, _, vt = np.linalg.svd(centered_points)
        normal = vt[2]  # Smallest singular value direction
        
        # Ensure normal points towards camera
        if normal[2] < 0:
            normal = -normal
        
        # Position camera
        plane_size = np.max(np.std(valid_points, axis=0)) * 2
        camera_distance = plane_size * 0.8
        camera_center = plane_center + normal * camera_distance
        
        # Create camera coordinate system
        z_axis = -normal
        
        # Find the most horizontal edge for x-axis
        if len(corners) >= 2:
            x_axis = corners[1] - corners[0]
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        else:
            # Fallback
            x_axis = np.array([1, 0, 0])
            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
        
        # Build matrices
        R_w2c = np.stack([x_axis, y_axis, z_axis], axis=0)
        R_c2w = R_w2c.T
        t_c2w = camera_center
        
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R_c2w
        extrinsic[:3, 3] = t_c2w
        
        # Intrinsics adjusted for scale
        fov = 2 * np.arctan(plane_size / (2 * camera_distance))
        focal_length = (W / 2) / np.tan(fov / 2) * scale
        
        # Principal point at image center (accounting for padding)
        intrinsic = np.array([
            [focal_length, 0, W / 2],
            [0, focal_length, H / 2],
            [0, 0, 1]
        ])
        
        return extrinsic, intrinsic