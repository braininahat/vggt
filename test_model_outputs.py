#!/usr/bin/env python3
"""Quick test to check VGGT model outputs"""

import torch
from vggt.models.vggt import VGGT

# Create model
model = VGGT.from_pretrained("facebook/VGGT-1B")
model.eval()

# Create dummy input
batch_size = 1
seq_len = 4
height = 518
width = 518

dummy_images = torch.rand(batch_size, seq_len, 3, height, width)

# Run forward pass
with torch.no_grad():
    predictions = model(dummy_images)

print("Model outputs:")
for k, v in predictions.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        if v.numel() > 0:
            print(f"    min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}")
    else:
        print(f"  {k}: {type(v)}")

# Check if heads exist
print("\nModel components:")
print(f"  aggregator: {model.aggregator is not None}")
print(f"  camera_head: {model.camera_head is not None}")
print(f"  point_head: {model.point_head is not None}")
print(f"  depth_head: {model.depth_head is not None}")
print(f"  track_head: {model.track_head is not None}")