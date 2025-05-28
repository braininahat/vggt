#!/usr/bin/env python3
"""Check the structure of HDF5 files to understand what data is available."""

import h5py
import numpy as np
from pathlib import Path
from glob import glob

# Find an HDF5 file
hdf5_files = glob("/home/varun/XiaLabSync/datasets/freehand_May212025/processed/**/*.h5", recursive=True)
if not hdf5_files:
    print("No HDF5 files found!")
    exit(1)

print(f"Found {len(hdf5_files)} HDF5 files")
print(f"\nChecking first file: {hdf5_files[0]}")

with h5py.File(hdf5_files[0], 'r') as hf:
    print("\nFile structure:")
    
    def print_structure(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{indent}{name}/")
    
    hf.visititems(print_structure)
    
    print("\nAttributes:")
    for key, value in hf.attrs.items():
        print(f"  {key}: {value}")
    
    # Check specific datasets
    if 'frames' in hf:
        print(f"\nFrames shape: {hf['frames'].shape}")
        print(f"First frame shape: {hf['frames'][0].shape}")
    
    if 'pointmaps' in hf:
        print(f"\nPointmaps shape: {hf['pointmaps'].shape}")
        print(f"First pointmap range: [{hf['pointmaps'][0].min():.2f}, {hf['pointmaps'][0].max():.2f}]")
    
    # Check if corners exist
    if 'corners' in hf:
        print("\nCorners structure:")
        corners = hf['corners']
        for key in corners.keys():
            print(f"  {key}: shape={corners[key].shape}")
    else:
        print("\nNo 'corners' group found in HDF5 file")