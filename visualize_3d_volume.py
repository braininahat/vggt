import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from transformers import AutoModel
import yaml
from torch.utils.data import DataLoader
import h5py

from vggt.models.vggt import VGGT


def load_model(checkpoint_path, config_path, device="cuda"):
    """Load the trained VGGT model from checkpoint."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = VGGT(
        img_size=config['data']['img_size'],
        patch_size=config['data']['patch_size'],
        embed_dim=1024  # Default for VGGT-1B
    )
    
    # Load checkpoint (weights_only=False for compatibility with older checkpoints)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def resize_and_crop_to_square(image, target_size):
    """Resize image preserving aspect ratio and center crop to square, matching training."""
    h, w = image.shape[:2]
    
    # Calculate scale to fit the target size
    scale = max(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize preserving aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Center crop to target size
    y_start = (new_h - target_size) // 2
    x_start = (new_w - target_size) // 2
    cropped = resized[y_start:y_start + target_size, x_start:x_start + target_size]
    
    return cropped


def process_sequence(model, images, target_size, device="cuda"):
    """Process a sequence of images through the model to get 3D predictions."""
    # Process numpy array
    if isinstance(images, np.ndarray):
        # Resize and crop each frame to match training preprocessing
        processed_frames = []
        for i in range(len(images)):
            frame = images[i]
            # Resize and crop to square
            processed = resize_and_crop_to_square(frame, target_size)
            processed_frames.append(processed)
        
        images = np.stack(processed_frames)
        images = torch.from_numpy(images).float()
    
    # Normalize to [0, 1] if needed
    if images.max() > 1.0:
        images = images / 255.0
    
    # Apply ImageNet normalization to match training
    # From training: images = (images / 255.0 - 0.45) / 0.225
    images = (images - 0.45) / 0.225
    
    # Add batch dimension if needed
    if len(images.shape) == 4:  # [S, H, W, C]
        images = images.permute(0, 3, 1, 2)  # [S, C, H, W]
        images = images.unsqueeze(0)  # [1, S, C, H, W]
    elif len(images.shape) == 5:  # [B, S, H, W, C]
        images = images.permute(0, 1, 4, 2, 3)  # [B, S, C, H, W]
    
    images = images.to(device)
    
    with torch.no_grad():
        predictions = model(images)
    
    return predictions


def process_long_sequence_sliding_window(model, frames, window_size, target_size, stride=1, device="cuda"):
    """Process a long video sequence using sliding window inference."""
    n_frames = len(frames)
    all_predictions = {
        'world_points': [],
        'world_points_conf': [],
        'depth': [],
        'depth_conf': [],
        'pose_enc': []
    }
    
    # Process with sliding window
    for start_idx in tqdm(range(0, n_frames - window_size + 1, stride), desc="Processing windows"):
        end_idx = start_idx + window_size
        window_frames = frames[start_idx:end_idx]
        
        # Process this window
        predictions = process_sequence(model, window_frames, target_size, device)
        
        # Store predictions for all frames in the window
        if stride == 1:
            # Only take the last frame to avoid duplicates
            for key in all_predictions:
                if key in predictions:
                    if start_idx == 0:
                        # First window: take all frames
                        all_predictions[key].append(predictions[key][0])
                    else:
                        # Subsequent windows: only take the last frame
                        all_predictions[key].append(predictions[key][0, -1:])
        else:
            # Take all frames when stride > 1
            for key in all_predictions:
                if key in predictions:
                    all_predictions[key].append(predictions[key][0])
    
    # Handle remaining frames if any
    if stride > 1 and n_frames % stride != 0:
        start_idx = n_frames - window_size
        window_frames = frames[start_idx:]
        predictions = process_sequence(model, window_frames, target_size, device)
        
        # Take only the new frames
        new_frames = n_frames - (start_idx + window_size)
        for key in all_predictions:
            if key in predictions and new_frames > 0:
                all_predictions[key].append(predictions[key][0, -new_frames:])
    
    # Concatenate all predictions
    final_predictions = {}
    for key in all_predictions:
        if all_predictions[key]:
            final_predictions[key] = torch.cat([torch.from_numpy(p) if isinstance(p, np.ndarray) else p 
                                               for p in all_predictions[key]], dim=0)
            final_predictions[key] = final_predictions[key].unsqueeze(0)  # Add batch dimension back
    
    # Add the original frames
    final_predictions['images'] = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    
    return final_predictions


def visualize_3d_volume(predictions, save_path=None, show_confidence=True):
    """Visualize the 3D point cloud from model predictions."""
    # Extract world points
    world_points = predictions['world_points'][0].cpu().numpy()  # [S, H, W, 3]
    if show_confidence and 'world_points_conf' in predictions:
        confidences = predictions['world_points_conf'][0].cpu().numpy()  # [S, H, W]
    else:
        confidences = np.ones_like(world_points[..., 0])
    
    # Create figure with subplots for each frame
    S = world_points.shape[0]
    fig = plt.figure(figsize=(20, 5))
    
    for i in range(S):
        ax = fig.add_subplot(1, S, i+1, projection='3d')
        
        # Flatten points and filter by confidence
        points = world_points[i].reshape(-1, 3)
        conf = confidences[i].flatten()
        
        # Filter points with low confidence
        mask = conf > 0.5
        points = points[mask]
        conf = conf[mask]
        
        # Downsample for visualization if too many points
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points = points[indices]
            conf = conf[indices]
        
        # Plot points
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=conf, cmap='viridis', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {i}')
        
        # Set equal aspect ratio
        max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                            points[:, 1].max()-points[:, 1].min(),
                            points[:, 2].max()-points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    return fig


def save_volume_projections(predictions, output_dir, prefix="volume", min_confidence=0.5):
    """Save 2D projections of the 3D volume as PNG images."""
    output_dir = Path(output_dir)
    
    # Extract data
    world_points = predictions['world_points'][0].cpu().numpy()  # [S, H, W, 3]
    world_points_conf = predictions['world_points_conf'][0].cpu().numpy() if 'world_points_conf' in predictions else np.ones_like(world_points[..., 0])
    
    # Create projection directory
    proj_dir = output_dir / "projections"
    proj_dir.mkdir(exist_ok=True)
    
    # Process each frame
    for frame_idx in range(world_points.shape[0]):
        points = world_points[frame_idx]  # [H, W, 3]
        conf = world_points_conf[frame_idx]  # [H, W]
        
        # Apply confidence threshold
        mask = conf > min_confidence
        
        # Create figure with 3 projections
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # XY projection (top-down view)
        ax = axes[0, 0]
        valid_points = points[mask]
        if len(valid_points) > 0:
            scatter = ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                               c=valid_points[:, 2], cmap='viridis', s=1, alpha=0.6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'XY Projection (Top View) - Frame {frame_idx}')
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax, label='Z depth')
        
        # XZ projection (side view)
        ax = axes[0, 1]
        if len(valid_points) > 0:
            scatter = ax.scatter(valid_points[:, 0], valid_points[:, 2], 
                               c=valid_points[:, 1], cmap='viridis', s=1, alpha=0.6)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title(f'XZ Projection (Side View) - Frame {frame_idx}')
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax, label='Y')
        
        # YZ projection (front view)
        ax = axes[1, 0]
        if len(valid_points) > 0:
            scatter = ax.scatter(valid_points[:, 1], valid_points[:, 2], 
                               c=valid_points[:, 0], cmap='viridis', s=1, alpha=0.6)
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')
            ax.set_title(f'YZ Projection (Front View) - Frame {frame_idx}')
            ax.set_aspect('equal')
            plt.colorbar(scatter, ax=ax, label='X')
        
        # 3D view
        ax = fig.add_subplot(2, 2, 4, projection='3d')
        if len(valid_points) > 0:
            # Downsample for 3D view if too many points
            if len(valid_points) > 10000:
                indices = np.random.choice(len(valid_points), 10000, replace=False)
                plot_points = valid_points[indices]
            else:
                plot_points = valid_points
            
            ax.scatter(plot_points[:, 0], plot_points[:, 1], plot_points[:, 2], 
                      c=conf[mask].flatten()[:len(plot_points)], cmap='viridis', s=1, alpha=0.6)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D View - Frame {frame_idx}')
        
        plt.tight_layout()
        
        # Save figure
        proj_path = proj_dir / f"{prefix}_frame_{frame_idx:04d}_projections.png"
        plt.savefig(proj_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    # Create a summary projection using all frames
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Combine all frames
    all_points = world_points.reshape(-1, 3)
    all_conf = world_points_conf.flatten()
    mask = all_conf > min_confidence
    valid_points = all_points[mask]
    
    if len(valid_points) > 50000:
        indices = np.random.choice(len(valid_points), 50000, replace=False)
        valid_points = valid_points[indices]
    
    # XY projection
    ax = axes[0]
    if len(valid_points) > 0:
        scatter = ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                           c=valid_points[:, 2], cmap='viridis', s=0.5, alpha=0.4)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('XY Projection - All Frames')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Z depth')
    
    # XZ projection
    ax = axes[1]
    if len(valid_points) > 0:
        scatter = ax.scatter(valid_points[:, 0], valid_points[:, 2], 
                           c=valid_points[:, 1], cmap='viridis', s=0.5, alpha=0.4)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_title('XZ Projection - All Frames')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Y')
    
    # YZ projection
    ax = axes[2]
    if len(valid_points) > 0:
        scatter = ax.scatter(valid_points[:, 1], valid_points[:, 2], 
                           c=valid_points[:, 0], cmap='viridis', s=0.5, alpha=0.4)
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title('YZ Projection - All Frames')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='X')
    
    plt.tight_layout()
    summary_path = proj_dir / f"{prefix}_all_frames_projections.png"
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved projection images to {proj_dir}")
    print(f"  - Individual frame projections: {len(world_points)} files")
    print(f"  - Summary projection: {summary_path}")


def save_volume_data(predictions, output_dir, prefix="volume"):
    """Save volume data as NPZ archive."""
    output_dir = Path(output_dir)
    
    # Extract data
    world_points = predictions['world_points'][0].cpu().numpy()  # [S, H, W, 3]
    world_points_conf = predictions['world_points_conf'][0].cpu().numpy() if 'world_points_conf' in predictions else np.ones_like(world_points[..., 0])
    depth = predictions['depth'][0].cpu().numpy() if 'depth' in predictions else None
    depth_conf = predictions['depth_conf'][0].cpu().numpy() if 'depth_conf' in predictions else None
    pose_enc = predictions['pose_enc'][0].cpu().numpy() if 'pose_enc' in predictions else None
    
    # Save as NumPy compressed archive
    npz_path = output_dir / f"{prefix}_data.npz"
    save_dict = {
        'world_points': world_points,
        'world_points_conf': world_points_conf,
    }
    if depth is not None:
        save_dict['depth'] = depth
    if depth_conf is not None:
        save_dict['depth_conf'] = depth_conf
    if pose_enc is not None:
        save_dict['pose_enc'] = pose_enc
    
    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved volume data to {npz_path}")
    print(f"  - world_points shape: {world_points.shape}")
    print(f"  - world_points_conf shape: {world_points_conf.shape}")
    if depth is not None:
        print(f"  - depth shape: {depth.shape}")
    if depth_conf is not None:
        print(f"  - depth_conf shape: {depth_conf.shape}")
    if pose_enc is not None:
        print(f"  - pose_enc shape: {pose_enc.shape}")
    
    return npz_path


def create_rotating_visualization(predictions, output_path, fps=30, duration=10):
    """Create a rotating 3D visualization animation."""
    world_points = predictions['world_points'][0].cpu().numpy()  # [S, H, W, 3]
    confidences = predictions['world_points_conf'][0].cpu().numpy() if 'world_points_conf' in predictions else np.ones_like(world_points[..., 0])
    
    # Use the first frame for the rotating visualization
    points = world_points[0].reshape(-1, 3)
    conf = confidences[0].flatten()
    
    # Filter and downsample
    mask = conf > 0.5
    points = points[mask]
    conf = conf[mask]
    
    if len(points) > 20000:
        indices = np.random.choice(len(points), 20000, replace=False)
        points = points[indices]
        conf = conf[indices]
    
    # Create frames
    frames = []
    n_frames = fps * duration
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                       c=conf, cmap='viridis', s=1, alpha=0.6)
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                        points[:, 1].max()-points[:, 1].min(),
                        points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Generate frames
    for i in tqdm(range(n_frames), desc="Creating animation"):
        ax.view_init(elev=20, azim=i * 360 / n_frames)
        
        # Save frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
    
    plt.close(fig)
    
    # Save as video
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved rotating visualization to {output_path}")


def load_video_frames(video_path, max_frames=None):
    """Load frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    return np.array(frames)


def load_image_sequence(image_dir, pattern="*.png"):
    """Load a sequence of images from a directory."""
    image_dir = Path(image_dir)
    image_paths = sorted(image_dir.glob(pattern))
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir} with pattern {pattern}")
    
    frames = []
    for path in image_paths:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    
    return np.array(frames)


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D volume from trained VGGT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/ultrasound_finetune_complete.yaml", 
                        help="Path to config file used for training")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to video file, image directory, or HDF5 file")
    parser.add_argument("--output_dir", type=str, default="./visualizations", 
                        help="Directory to save visualizations")
    parser.add_argument("--max_frames", type=int, default=None, 
                        help="Maximum number of frames to process (None for all frames)")
    parser.add_argument("--create_animation", action="store_true", 
                        help="Create rotating 3D animation")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--hdf5_volume_idx", type=int, default=0,
                        help="Volume index to use from HDF5 file")
    parser.add_argument("--window_size", type=int, default=None,
                        help="Window size for sliding window inference (default: from config)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride for sliding window inference")
    parser.add_argument("--save_volume", action="store_true",
                        help="Save 3D volume data (NPZ format)")
    parser.add_argument("--save_projections", action="store_true",
                        help="Save 2D projections of 3D volume as PNG images")
    parser.add_argument("--min_confidence", type=float, default=0.5,
                        help="Minimum confidence threshold for point cloud export")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint, args.config, args.device)
    
    # Load input data
    input_path = Path(args.input)
    
    if input_path.suffix in ['.mp4', '.avi', '.mov']:
        print(f"Loading video from {input_path}")
        frames = load_video_frames(input_path, args.max_frames)
    elif input_path.suffix in ['.h5', '.hdf5']:
        print(f"Loading from HDF5 file {input_path}")
        with h5py.File(input_path, 'r') as f:
            volume_keys = list(f.keys())
            if args.hdf5_volume_idx >= len(volume_keys):
                raise ValueError(f"Volume index {args.hdf5_volume_idx} out of range. File has {len(volume_keys)} volumes.")
            
            volume_key = volume_keys[args.hdf5_volume_idx]
            if args.max_frames:
                frames = f[volume_key]['frames'][:args.max_frames]
            else:
                frames = f[volume_key]['frames'][:]
            print(f"Loaded volume '{volume_key}' with shape {frames.shape}")
    elif input_path.is_dir():
        print(f"Loading image sequence from {input_path}")
        if args.max_frames:
            frames = load_image_sequence(input_path)[:args.max_frames]
        else:
            frames = load_image_sequence(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_path}")
    
    print(f"Loaded {len(frames)} frames")
    
    # Determine window size
    window_size = args.window_size or config['data']['sequence_length']
    
    # Process through model
    print("Processing frames through model...")
    target_size = config['data']['img_size']
    if len(frames) <= window_size:
        # Process all frames at once
        predictions = process_sequence(model, frames, target_size, args.device)
    else:
        # Use sliding window for long sequences
        print(f"Using sliding window inference with window_size={window_size}, stride={args.stride}")
        predictions = process_long_sequence_sliding_window(model, frames, window_size, target_size, args.stride, args.device)
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Save volume data if requested
    if args.save_volume:
        print("Saving 3D volume data...")
        save_volume_data(predictions, output_dir)
    
    # Save projections if requested
    if args.save_projections:
        print("Saving 2D projections...")
        save_volume_projections(predictions, output_dir, min_confidence=args.min_confidence)
    
    # Static multi-frame visualization
    static_path = output_dir / "3d_volume_multiframe.png"
    visualize_3d_volume(predictions, save_path=static_path)
    
    # Rotating animation if requested
    if args.create_animation:
        animation_path = output_dir / "3d_volume_rotating.mp4"
        create_rotating_visualization(predictions, animation_path)
    
    # Also save depth maps
    if 'depth' in predictions:
        depth_maps = predictions['depth'][0].cpu().numpy()  # [S, H, W, 1]
        depth_conf = predictions['depth_conf'][0].cpu().numpy() if 'depth_conf' in predictions else None
        
        n_frames_to_show = min(5, len(frames))
        fig, axes = plt.subplots(2, n_frames_to_show, figsize=(15, 6))
        if n_frames_to_show == 1:
            axes = axes.reshape(2, 1)
        
        for i in range(n_frames_to_show):
            # Show depth
            axes[0, i].imshow(depth_maps[i, :, :, 0], cmap='viridis')
            axes[0, i].set_title(f'Depth {i}')
            axes[0, i].axis('off')
            
            # Show confidence if available
            if depth_conf is not None:
                axes[1, i].imshow(depth_conf[i], cmap='hot')
                axes[1, i].set_title(f'Confidence {i}')
            else:
                axes[1, i].imshow(frames[i])
                axes[1, i].set_title(f'Frame {i}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "depth_maps.png", dpi=150)
        plt.close()
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()