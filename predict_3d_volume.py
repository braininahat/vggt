#!/usr/bin/env python3
"""
Run VGGT prediction and visualize 3D volume
"""

import argparse
import glob
import os
import torch
import numpy as np
from pathlib import Path

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def run_prediction(image_folder, model_path=None, device="cuda"):
    """Run VGGT prediction on a folder of images"""
    
    # Load images
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_paths:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"Found {len(image_paths)} images")
    
    # Load model
    if model_path:
        print(f"Loading model from {model_path}")
        model = VGGT.from_pretrained(model_path)
    else:
        print("Loading pretrained VGGT model from HuggingFace")
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    
    model = model.to(device).eval()
    
    # Load and preprocess images
    images = load_and_preprocess_images(image_paths, device=device)
    print(f"Images shape: {images.shape}")
    
    # Run inference
    with torch.no_grad():
        predictions = model(images)
    
    # Extract predictions
    pose_enc = predictions["pose_encoding"]  # (S, 10)
    depth_map = predictions["depth_map"]  # (S, H, W)
    depth_conf = predictions["depth_confidence"]  # (S, H, W)
    
    # Convert to camera parameters
    extri, intri = pose_encoding_to_extri_intri(pose_enc, height=depth_map.shape[-2], width=depth_map.shape[-1])
    
    # Option 1: Use predicted point maps if available
    if "point_map" in predictions:
        world_points = predictions["point_map"]  # (S, H, W, 3)
        point_conf = predictions["point_confidence"]  # (S, H, W)
    else:
        # Option 2: Unproject depth maps to 3D points
        world_points = []
        for i in range(len(images)):
            points = unproject_depth_map_to_point_map(
                depth_map[i:i+1], 
                extri[i:i+1], 
                intri[i:i+1]
            )
            world_points.append(points)
        world_points = torch.cat(world_points, dim=0)
        point_conf = depth_conf
    
    # Prepare output dictionary for visualization
    pred_dict = {
        "images": images.cpu(),
        "world_points": world_points.cpu(),
        "confidence": point_conf.cpu(),
        "extrinsics": extri.cpu(),
        "intrinsics": intri.cpu(),
        "depth": depth_map.cpu(),
    }
    
    # Add tracks if available
    if "tracks" in predictions:
        pred_dict["tracks"] = predictions["tracks"].cpu()
        pred_dict["track_confidence"] = predictions["track_confidence"].cpu()
    
    return pred_dict


def main():
    parser = argparse.ArgumentParser(description="Run VGGT prediction and visualize 3D volume")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model checkpoint (optional)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--visualize", action="store_true", help="Launch viser visualization")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save predictions")
    parser.add_argument("--port", type=int, default=8080, help="Port for viser server")
    parser.add_argument("--conf_threshold", type=float, default=50.0, help="Confidence threshold percentage")
    
    args = parser.parse_args()
    
    # Run prediction
    print("Running VGGT prediction...")
    pred_dict = run_prediction(args.image_folder, args.model_path, args.device)
    
    # Save predictions if requested
    if args.save_path:
        print(f"Saving predictions to {args.save_path}")
        torch.save(pred_dict, args.save_path)
    
    # Visualize if requested
    if args.visualize:
        print(f"Starting viser visualization on port {args.port}")
        print("Open http://localhost:{args.port} in your browser")
        
        # Import and run viser visualization
        from demo_viser import viser_wrapper
        viser_wrapper(
            pred_dict,
            port=args.port,
            init_conf_threshold=args.conf_threshold,
            use_point_map=True,
            image_folder=args.image_folder
        )
    
    print("Done!")


if __name__ == "__main__":
    main()