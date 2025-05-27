# Simplified dataset for ultrasound frame corner prediction
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
from natsort import natsorted
from typing import Dict, List, Tuple, Optional
import logging


class UltrasoundFrameDataset(Dataset):
    """Dataset for ultrasound frame corner prediction.
    
    Loads sequences of ultrasound images with their 3D frame corner positions.
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_length: int = 8,
        image_size: Tuple[int, int] = (518, 518),
        sampling_stride: int = 1,  # Frame sampling stride
        augment: bool = False,
        min_sequence_length: int = 10,
    ):
        """Initialize dataset.
        
        Args:
            data_root: Root directory containing organized data
            sequence_length: Number of frames per sequence
            image_size: Target image size (H, W)
            sampling_stride: Stride for sampling frames
            augment: Whether to apply data augmentation
            min_sequence_length: Minimum frames required
        """
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.sampling_stride = sampling_stride
        self.augment = augment
        self.min_sequence_length = min_sequence_length
        
        # Load all sequences
        self.sequences = self._load_sequences()
        
        # Create sampling indices for each sequence
        self.sample_indices = self._create_sample_indices()
        
        logging.info(f"Loaded {len(self.sequences)} sequences, "
                    f"{len(self.sample_indices)} samples")
    
    def _load_sequences(self) -> List[Dict]:
        """Load all valid sequences."""
        sequences = []
        
        # Find all interpolated CSV files
        csv_files = list(self.data_root.glob("**/*_interpolated.csv"))
        
        for csv_path in csv_files:
            try:
                # Load CSV
                df = pd.read_csv(csv_path)
                
                if len(df) < self.min_sequence_length:
                    continue
                
                # Find frames directory with various naming patterns
                scan_dir = csv_path.parent
                frames_dir = None
                frame_files = []
                
                # Try different directory names
                possible_dirs = [
                    scan_dir / "frames",
                    scan_dir / "roi_frames",
                    scan_dir / f"{scan_dir.name}_frame_rois",
                    scan_dir / "frame_rois"
                ]
                
                for dir_candidate in possible_dirs:
                    if dir_candidate.exists():
                        # Get frame files
                        frame_files = natsorted(list(dir_candidate.glob("frame_*.png")))
                        if not frame_files:
                            frame_files = natsorted(list(dir_candidate.glob("*.png")))
                        
                        if frame_files:
                            frames_dir = dir_candidate
                            break
                
                if not frames_dir:
                    logging.warning(f"No frames directory found for {csv_path}")
                    continue
                
                if len(frame_files) != len(df):
                    logging.warning(
                        f"Frame mismatch: {len(frame_files)} files, "
                        f"{len(df)} CSV rows for {csv_path.name}"
                    )
                    # Use minimum
                    max_frames = min(len(frame_files), len(df))
                    frame_files = frame_files[:max_frames]
                    df = df.iloc[:max_frames]
                
                # Extract frame corner data
                corner_cols = [
                    'frame_c1_x_mm', 'frame_c1_y_mm', 'frame_c1_z_mm',
                    'frame_c2_x_mm', 'frame_c2_y_mm', 'frame_c2_z_mm',
                    'frame_c3_x_mm', 'frame_c3_y_mm', 'frame_c3_z_mm',
                    'frame_c4_x_mm', 'frame_c4_y_mm', 'frame_c4_z_mm',
                ]
                
                # Check if corner columns exist
                if not all(col in df.columns for col in corner_cols):
                    logging.warning(f"Missing corner columns in {csv_path}")
                    continue
                
                sequence = {
                    'name': csv_path.stem.replace('_interpolated', ''),
                    'csv_path': csv_path,
                    'frame_files': frame_files,
                    'dataframe': df,
                    'num_frames': len(frame_files),
                }
                
                sequences.append(sequence)
                
            except Exception as e:
                logging.error(f"Error loading {csv_path}: {e}")
                continue
        
        return sequences
    
    def _create_sample_indices(self) -> List[Tuple[int, int]]:
        """Create indices for sampling sequences."""
        indices = []
        
        for seq_idx, seq in enumerate(self.sequences):
            num_frames = seq['num_frames']
            
            # Calculate valid starting positions
            max_start = num_frames - (self.sequence_length - 1) * self.sampling_stride
            
            if max_start <= 0:
                continue
                
            # Add all valid starting positions
            for start_idx in range(max_start):
                indices.append((seq_idx, start_idx))
        
        return indices
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sequence sample.
        
        Returns:
            Dictionary containing:
            - images: (S, 3, H, W) image sequence
            - frame_corners: (S, 4, 3) corner positions in mm
            - frame_centers: (S, 3) geometric centers
            - sequence_name: Name of the sequence
            - frame_indices: (S,) indices of frames used
        """
        seq_idx, start_idx = self.sample_indices[idx]
        sequence = self.sequences[seq_idx]
        
        # Sample frame indices
        frame_indices = []
        for i in range(self.sequence_length):
            frame_idx = start_idx + i * self.sampling_stride
            frame_indices.append(frame_idx)
        
        # Load images
        images = []
        for frame_idx in frame_indices:
            img_path = sequence['frame_files'][frame_idx]
            img = Image.open(img_path).convert('RGB')
            
            # Resize if needed
            if img.size != (self.image_size[1], self.image_size[0]):
                img = img.resize(
                    (self.image_size[1], self.image_size[0]), 
                    Image.BILINEAR
                )
            
            # Convert to tensor
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            images.append(img_tensor)
        
        images = torch.stack(images)  # (S, 3, H, W)
        
        # Extract frame data
        df = sequence['dataframe']
        frame_corners = []
        frame_centers = []
        frame_quaternions = []
        
        for frame_idx in frame_indices:
            row = df.iloc[frame_idx]
            
            # Extract 4 corners
            corners = []
            for corner_idx in range(1, 5):
                x = row[f'frame_c{corner_idx}_x_mm']
                y = row[f'frame_c{corner_idx}_y_mm']
                z = row[f'frame_c{corner_idx}_z_mm']
                corners.append([x, y, z])
            
            frame_corners.append(corners)
            
            # Extract frame center (should already be standardized in CSV)
            if 'frame_center_x_mm' in row:
                center = [
                    row['frame_center_x_mm'],
                    row['frame_center_y_mm'],
                    row['frame_center_z_mm']
                ]
            else:
                # Fallback: compute from corners
                center = np.array(corners).mean(axis=0).tolist()
            frame_centers.append(center)
            
            # Extract quaternion (wxyz format)
            if 'quat_w' in row:
                quat = [
                    row['quat_w'],
                    row['quat_x'],
                    row['quat_y'],
                    row['quat_z']
                ]
                # Normalize
                quat = np.array(quat)
                quat = quat / np.linalg.norm(quat)
                frame_quaternions.append(quat.tolist())
            else:
                # Default to identity
                frame_quaternions.append([1., 0., 0., 0.])
        
        frame_corners = torch.tensor(frame_corners, dtype=torch.float32)  # (S, 4, 3)
        frame_centers = torch.tensor(frame_centers, dtype=torch.float32)  # (S, 3)
        frame_quaternions = torch.tensor(frame_quaternions, dtype=torch.float32)  # (S, 4)
        
        # Data augmentation
        if self.augment:
            images, frame_corners, frame_centers = self._augment(
                images, frame_corners, frame_centers
            )
        
        return {
            'images': images,
            'frame_corners': frame_corners,
            'frame_centers': frame_centers,
            'frame_quaternions': frame_quaternions,
            'sequence_name': sequence['name'],
            'frame_indices': torch.tensor(frame_indices),
        }
    
    def _augment(self, images, frame_corners, frame_centers):
        """Apply data augmentation.
        
        Currently just a placeholder - you could add:
        - Random crops
        - Color jitter
        - Random flips (with corresponding corner adjustments)
        """
        return images, frame_corners, frame_centers
    
    def get_frame_statistics(self) -> Dict[str, float]:
        """Compute statistics of frame dimensions across dataset."""
        all_corners = []
        
        for seq in self.sequences:
            df = seq['dataframe']
            for i in range(len(df)):
                row = df.iloc[i]
                corners = []
                for corner_idx in range(1, 5):
                    x = row[f'frame_c{corner_idx}_x_mm']
                    y = row[f'frame_c{corner_idx}_y_mm'] 
                    z = row[f'frame_c{corner_idx}_z_mm']
                    corners.append([x, y, z])
                all_corners.append(corners)
        
        all_corners = np.array(all_corners)  # (N, 4, 3)
        
        # Compute frame dimensions
        # Width: distance between corners 1-2 or 3-4
        widths = np.linalg.norm(all_corners[:, 1] - all_corners[:, 0], axis=1)
        # Height: distance between corners 1-4 or 2-3  
        heights = np.linalg.norm(all_corners[:, 3] - all_corners[:, 0], axis=1)
        
        stats = {
            'mean_width_mm': float(np.mean(widths)),
            'std_width_mm': float(np.std(widths)),
            'mean_height_mm': float(np.mean(heights)),
            'std_height_mm': float(np.std(heights)),
            'num_sequences': len(self.sequences),
            'total_frames': sum(seq['num_frames'] for seq in self.sequences),
        }
        
        return stats