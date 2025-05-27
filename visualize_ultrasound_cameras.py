#!/usr/bin/env python3
"""
Visualize synthetic cameras generated from ultrasound pointmap data.
This script helps verify the camera generation before training.
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
from tqdm import tqdm
import torch

def generate_synthetic_cameras_from_pointmap(pointmap, frame_idx, total_frames):
    """
    Generate synthetic camera parameters from a pointmap.
    
    For ultrasound imaging:
    - The imaging plane is defined by the pointmap
    - Camera is positioned perpendicular to the imaging plane
    - Camera looks at the center of the imaging plane
    
    Args:
        pointmap: (H, W, 3) array of 3D coordinates
        frame_idx: Current frame index
        total_frames: Total number of frames in sequence
        
    Returns:
        extrinsic: 4x4 camera-to-world transformation matrix
        intrinsic: 3x3 camera intrinsic matrix
        camera_center: 3D position of camera
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
    
    # Calculate normal to the imaging plane using cross product
    # Vector from top-left to top-right
    u_vec = corners[1] - corners[0]
    # Vector from top-left to bottom-left
    v_vec = corners[3] - corners[0]
    
    # Normal vector (pointing towards the probe/camera)
    normal = np.cross(u_vec, v_vec)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    
    # Position camera along the normal, at a distance proportional to plane size
    plane_size = max(np.linalg.norm(u_vec), np.linalg.norm(v_vec))
    camera_distance = plane_size * 0.5  # Adjust this factor as needed
    camera_center = plane_center + normal * camera_distance
    
    # Log key values
    print(f"\nFrame {frame_idx} camera generation:")
    print(f"  Plane size: {plane_size:.2f} mm")
    print(f"  Camera distance: {camera_distance:.2f} mm")
    print(f"  Plane center: {plane_center}")
    print(f"  Camera center: {camera_center}")
    print(f"  Normal vector: {normal}")
    
    # Create camera coordinate system
    # Z-axis points from camera to plane center (opposite of normal)
    z_axis = -normal
    
    # X-axis aligned with u_vec (image horizontal)
    x_axis = u_vec / (np.linalg.norm(u_vec) + 1e-8)
    
    # Y-axis from cross product (image vertical, pointing down)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    # Build rotation matrix (world-to-camera)
    R_w2c = np.stack([x_axis, y_axis, z_axis], axis=0)
    
    # Translation (world-to-camera)
    t_w2c = -R_w2c @ camera_center
    
    # Camera-to-world transformation (what VGGT expects)
    R_c2w = R_w2c.T
    t_c2w = camera_center
    
    # Build 4x4 extrinsic matrix (camera-to-world)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R_c2w
    extrinsic[:3, 3] = t_c2w
    
    # Create intrinsic matrix
    # Field of view based on imaging plane size and distance
    fov = 2 * np.arctan(plane_size / (2 * camera_distance))
    focal_length = (W / 2) / np.tan(fov / 2)
    
    print(f"  FOV: {np.degrees(fov):.2f} degrees")
    print(f"  Focal length: {focal_length:.2f} pixels")
    
    intrinsic = np.array([
        [focal_length, 0, W / 2],
        [0, focal_length, H / 2],
        [0, 0, 1]
    ])
    
    return extrinsic, intrinsic, camera_center


def visualize_camera_trajectory(hdf_path, num_frames=20, skip_frames=None):
    """
    Visualize the synthetic camera trajectory for an ultrasound sequence.
    
    Args:
        hdf_path: Path to HDF5 file
        num_frames: Number of frames to visualize
        skip_frames: Number of frames to skip between visualizations
    """
    with h5py.File(hdf_path, 'r') as hf:
        pointmaps = hf['pointmaps']
        frames = hf['frames']
        total_frames = len(pointmaps)
        
        if skip_frames is None:
            skip_frames = max(1, total_frames // num_frames)
        
        # Select frames to visualize
        frame_indices = range(0, min(total_frames, num_frames * skip_frames), skip_frames)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 10))
        
        # 3D plot for camera positions and orientations
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title('Camera Trajectory and Orientations')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        
        # 2D plot for frame samples
        ax2 = fig.add_subplot(122)
        ax2.set_title('Sample Frames')
        
        camera_centers = []
        camera_orientations = []
        plane_centers = []
        
        # Process each frame
        for i, frame_idx in enumerate(tqdm(frame_indices, desc="Processing frames")):
            pointmap = pointmaps[frame_idx]
            frame = frames[frame_idx]
            
            # Generate synthetic camera
            extrinsic, intrinsic, camera_center = generate_synthetic_cameras_from_pointmap(
                pointmap, frame_idx, total_frames
            )
            
            camera_centers.append(camera_center)
            
            # Extract camera orientation (z-axis of camera)
            camera_z = extrinsic[:3, 2]  # Third column is z-axis in camera-to-world
            camera_orientations.append(camera_z)
            
            # Get plane center
            H, W = pointmap.shape[:2]
            plane_center = pointmap[H//2, W//2]
            plane_centers.append(plane_center)
            
            # Plot imaging plane corners
            corners = np.array([
                pointmap[0, 0],
                pointmap[0, W-1],
                pointmap[H-1, W-1],
                pointmap[H-1, 0],
                pointmap[0, 0]  # Close the rectangle
            ])
            
            # Use color gradient for temporal visualization
            color = plt.cm.viridis(i / len(frame_indices))
            ax1.plot(corners[:, 0], corners[:, 1], corners[:, 2], 
                    color=color, alpha=0.6, linewidth=2)
            
            # Draw camera frustum
            # Lines from camera to plane corners
            for corner in corners[:-1]:
                ax1.plot([camera_center[0], corner[0]], 
                        [camera_center[1], corner[1]], 
                        [camera_center[2], corner[2]], 
                        color=color, alpha=0.3, linestyle='--')
        
        # Plot camera trajectory
        camera_centers = np.array(camera_centers)
        ax1.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], 
                'r-', linewidth=3, label='Camera trajectory')
        ax1.scatter(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], 
                   c='red', s=50, marker='o')
        
        # Plot camera orientations as arrows
        for i, (center, orientation) in enumerate(zip(camera_centers, camera_orientations)):
            if i % max(1, len(camera_centers) // 10) == 0:  # Show subset of arrows
                ax1.quiver(center[0], center[1], center[2],
                          orientation[0], orientation[1], orientation[2],
                          length=10, color='red', alpha=0.7)
        
        # Plot plane centers
        plane_centers = np.array(plane_centers)
        ax1.plot(plane_centers[:, 0], plane_centers[:, 1], plane_centers[:, 2], 
                'b--', linewidth=2, alpha=0.5, label='Plane centers')
        
        ax1.legend()
        ax1.set_box_aspect([1,1,1])
        
        # Show sample frames in 2D subplot
        n_samples = min(6, len(frame_indices))
        sample_indices = np.linspace(0, len(frame_indices)-1, n_samples, dtype=int)
        
        for i, idx in enumerate(sample_indices):
            ax2_sub = fig.add_subplot(2, 3, i+1)
            frame_idx = frame_indices[idx]
            frame = frames[frame_idx]
            
            if len(frame.shape) == 2:  # Grayscale
                ax2_sub.imshow(frame, cmap='gray')
            else:  # RGB
                ax2_sub.imshow(frame)
            
            ax2_sub.set_title(f'Frame {frame_idx}')
            ax2_sub.axis('off')
        
        ax2.axis('off')
        
        plt.tight_layout()
        return fig


def visualize_single_frame_geometry(hdf_path, frame_idx=0):
    """
    Detailed visualization of a single frame's geometry.
    """
    with h5py.File(hdf_path, 'r') as hf:
        pointmap = hf['pointmaps'][frame_idx]
        frame = hf['frames'][frame_idx]
        total_frames = len(hf['pointmaps'])
        
        # Generate synthetic camera
        extrinsic, intrinsic, camera_center = generate_synthetic_cameras_from_pointmap(
            pointmap, frame_idx, total_frames
        )
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Original frame
        ax1 = fig.add_subplot(2, 3, 1)
        if len(frame.shape) == 2:
            ax1.imshow(frame, cmap='gray')
        else:
            ax1.imshow(frame)
        ax1.set_title(f'Ultrasound Frame {frame_idx}')
        ax1.axis('off')
        
        # 2. Pointmap depth visualization
        ax2 = fig.add_subplot(2, 3, 2)
        depth_from_camera = np.linalg.norm(pointmap - camera_center, axis=2)
        im2 = ax2.imshow(depth_from_camera, cmap='viridis')
        ax2.set_title('Depth from Synthetic Camera')
        plt.colorbar(im2, ax=ax2)
        ax2.axis('off')
        
        # 3. 3D visualization
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        
        # Subsample points for visualization
        step = 20
        points_vis = pointmap[::step, ::step].reshape(-1, 3)
        colors_vis = frame[::step, ::step].reshape(-1, 3) / 255.0 if len(frame.shape) == 3 else None
        
        if colors_vis is not None:
            ax3.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                       c=colors_vis, s=1, alpha=0.5)
        else:
            ax3.scatter(points_vis[:, 0], points_vis[:, 1], points_vis[:, 2], 
                       c=frame[::step, ::step].flatten(), cmap='gray', s=1, alpha=0.5)
        
        # Add camera
        ax3.scatter(*camera_center, c='red', s=200, marker='^', label='Camera')
        
        # Add coordinate frame at camera
        axis_length = 20
        axes = extrinsic[:3, :3] * axis_length
        origin = camera_center
        ax3.quiver(origin[0], origin[1], origin[2], axes[0, 0], axes[1, 0], axes[2, 0], color='r', arrow_length_ratio=0.1)
        ax3.quiver(origin[0], origin[1], origin[2], axes[0, 1], axes[1, 1], axes[2, 1], color='g', arrow_length_ratio=0.1)
        ax3.quiver(origin[0], origin[1], origin[2], axes[0, 2], axes[1, 2], axes[2, 2], color='b', arrow_length_ratio=0.1)
        
        ax3.set_xlabel('X (mm)')
        ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)')
        ax3.set_title('3D Geometry')
        ax3.legend()
        
        # 4. Camera parameters
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.text(0.1, 0.9, 'Camera Parameters:', transform=ax4.transAxes, fontsize=12, fontweight='bold')
        ax4.text(0.1, 0.8, f'Position: [{camera_center[0]:.1f}, {camera_center[1]:.1f}, {camera_center[2]:.1f}]', transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Focal length: {intrinsic[0, 0]:.1f}', transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'Principal point: [{intrinsic[0, 2]:.1f}, {intrinsic[1, 2]:.1f}]', transform=ax4.transAxes)
        ax4.text(0.1, 0.5, 'Extrinsic matrix:', transform=ax4.transAxes)
        for i in range(4):
            ax4.text(0.1, 0.4 - i*0.05, f'  [{extrinsic[i, 0]:.3f}, {extrinsic[i, 1]:.3f}, {extrinsic[i, 2]:.3f}, {extrinsic[i, 3]:.3f}]', 
                    transform=ax4.transAxes, fontsize=10, family='monospace')
        ax4.axis('off')
        
        # 5. Pointmap statistics
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.text(0.1, 0.9, 'Pointmap Statistics:', transform=ax5.transAxes, fontsize=12, fontweight='bold')
        ax5.text(0.1, 0.8, f'Shape: {pointmap.shape}', transform=ax5.transAxes)
        ax5.text(0.1, 0.7, f'X range: [{pointmap[:,:,0].min():.1f}, {pointmap[:,:,0].max():.1f}] mm', transform=ax5.transAxes)
        ax5.text(0.1, 0.6, f'Y range: [{pointmap[:,:,1].min():.1f}, {pointmap[:,:,1].max():.1f}] mm', transform=ax5.transAxes)
        ax5.text(0.1, 0.5, f'Z range: [{pointmap[:,:,2].min():.1f}, {pointmap[:,:,2].max():.1f}] mm', transform=ax5.transAxes)
        ax5.text(0.1, 0.4, f'Mean depth from camera: {depth_from_camera.mean():.1f} mm', transform=ax5.transAxes)
        ax5.axis('off')
        
        # 6. Projected points using camera parameters
        ax6 = fig.add_subplot(2, 3, 6)
        # Project 3D points back to 2D using synthetic camera
        world_points = pointmap.reshape(-1, 3)
        world_points_h = np.hstack([world_points, np.ones((len(world_points), 1))])
        
        # World to camera
        cam_points = (np.linalg.inv(extrinsic) @ world_points_h.T).T[:, :3]
        
        # Project to image
        valid_points = cam_points[:, 2] > 0
        proj_points = (intrinsic @ cam_points[valid_points].T).T
        proj_points = proj_points[:, :2] / proj_points[:, 2:3]
        
        H, W = frame.shape[:2]
        ax6.scatter(proj_points[:, 0], proj_points[:, 1], c='red', s=0.1, alpha=0.5)
        ax6.set_xlim(0, W)
        ax6.set_ylim(H, 0)  # Flip y-axis for image coordinates
        ax6.set_title('Reprojected Points')
        ax6.set_aspect('equal')
        
        plt.tight_layout()
        return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize synthetic cameras for ultrasound data')
    parser.add_argument('hdf_path', type=str, help='Path to HDF5 file')
    parser.add_argument('--num_frames', type=int, default=20, help='Number of frames to visualize in trajectory')
    parser.add_argument('--skip_frames', type=int, default=None, help='Frames to skip between visualizations')
    parser.add_argument('--single_frame', type=int, default=None, help='Visualize single frame geometry')
    parser.add_argument('--output_dir', type=str, default='./camera_visualizations', help='Output directory for figures')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    hdf_path = Path(args.hdf_path)
    base_name = hdf_path.stem
    
    if args.single_frame is not None:
        # Single frame visualization
        fig = visualize_single_frame_geometry(args.hdf_path, args.single_frame)
        output_path = output_dir / f'{base_name}_frame_{args.single_frame}_geometry.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved single frame visualization to {output_path}")
    else:
        # Trajectory visualization
        fig = visualize_camera_trajectory(args.hdf_path, args.num_frames, args.skip_frames)
        output_path = output_dir / f'{base_name}_camera_trajectory.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization to {output_path}")
    
    plt.show()


if __name__ == '__main__':
    main()