# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT (Visual Geometry Grounded Transformer) is a feed-forward neural network that directly infers 3D scene attributes including camera parameters, depth maps, point maps, and 3D point tracks from one or multiple input views. The model is built with PyTorch and published by Meta AI and University of Oxford.

## Setup and Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# Install demo dependencies (for visualization and web interface)
pip install -r requirements_demo.txt

# Alternative: install as package
pip install -e .
```

The model automatically downloads pretrained weights from Hugging Face on first use.

## Common Commands

### Running Demos
```bash
# Interactive web interface
python demo_gradio.py

# 3D viewer with viser
python demo_viser.py --image_folder path/to/images

# Export to COLMAP format (for NeRF/Gaussian training)
python demo_colmap.py --scene_dir path/to/images [--use_ba]
```

### Development
- No specific linting or test commands are configured in the project
- Use standard Python testing frameworks if adding tests
- Follow PyTorch conventions for neural network code

## Architecture Overview

### Core Model Structure
The VGGT model (`vggt/models/vggt.py`) consists of four main components:

1. **Aggregator** (`vggt/models/aggregator.py`): Backbone transformer with alternating attention
   - Processes input images through patch embedding (DINOv2-based or conv)
   - Applies alternating frame-level and global attention across multiple blocks
   - Uses rotary position embeddings (RoPE) for spatial understanding
   - Outputs aggregated token representations for downstream heads

2. **Camera Head** (`vggt/heads/camera_head.py`): Predicts camera parameters
   - Estimates extrinsic and intrinsic camera matrices
   - Uses iterative refinement with transformer blocks
   - Outputs pose encoding (translation, quaternion rotation, field of view)

3. **Depth/Point Heads** (`vggt/heads/dpt_head.py`): Dense prediction heads
   - DPT-style architecture for pixel-wise predictions
   - Depth head: predicts depth maps with confidence scores
   - Point head: directly predicts 3D world coordinates with confidence

4. **Track Head** (`vggt/heads/track_head.py`): Point tracking across frames
   - Tracks query points through the sequence
   - Outputs tracks with visibility and confidence scores

### Key Design Patterns

**Alternating Attention**: The aggregator alternates between frame-level attention (within each image) and global attention (across all images) to capture both local and inter-frame relationships.

**Multi-Head Architecture**: Each head operates on the same aggregated tokens but produces different outputs (cameras, depth, points, tracks).

**Iterative Refinement**: Camera and track heads use multiple transformer blocks for iterative improvement of predictions.

**Token Organization**: Special tokens (camera, register) are prepended to patch tokens, with `patch_start_idx` indicating where image patches begin.

### Coordinate Systems
- Camera matrices follow OpenCV convention (camera-to-world transformation)
- Input images are normalized using ImageNet statistics
- Depth maps can be unprojected to 3D points using camera parameters

### Dependencies Integration
- **VGGSfM Integration** (`vggt/dependency/`): Provides tracking and SfM utilities from the predecessor VGGSfM model
- **COLMAP Export** (`vggt/dependency/np_to_pycolmap.py`): Converts predictions to COLMAP format for downstream 3D applications

## Usage Patterns

### Basic Inference
```python
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

model = VGGT.from_pretrained("facebook/VGGT-1B")
images = load_and_preprocess_images(image_paths)
predictions = model(images)
```

### Selective Prediction
Call individual heads to predict only specific attributes:
```python
aggregated_tokens, ps_idx = model.aggregator(images)
pose_enc = model.camera_head(aggregated_tokens)
depth, conf = model.depth_head(aggregated_tokens, images, ps_idx)
```

### Coordinate Transformations
- Use `vggt/utils/pose_enc.py` for camera parameter conversions
- Use `vggt/utils/geometry.py` for 3D geometry operations
- Use `vggt/utils/load_fn.py` for image preprocessing

## File Organization

- `vggt/models/`: Core model definitions
- `vggt/heads/`: Task-specific prediction heads  
- `vggt/layers/`: Transformer building blocks (attention, MLP, etc.)
- `vggt/utils/`: Utility functions for geometry, loading, visualization
- `vggt/dependency/`: Integration with VGGSfM and external tools
- `demo_*.py`: Demonstration scripts for different use cases
- `examples/`: Sample image datasets for testing

## Development Notes

- Model uses mixed precision training (bfloat16/float16) for efficiency
- Supports batch processing of multiple scenes
- Memory usage scales with number of input frames (see README benchmarks)
- Flash Attention is used for efficient transformer computation
- No specific code style enforcement tools are configured