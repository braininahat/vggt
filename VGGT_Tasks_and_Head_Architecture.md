# VGGT: Tasks, Demos, and Head Architecture Analysis

## Task Coverage Overview

VGGT provides three main demo applications, each targeting different use cases and showcasing different combinations of the four prediction heads:

| **Demo** | **Primary Task** | **Heads Used** | **Output Format** | **Use Case** |
|----------|------------------|----------------|-------------------|--------------|
| `demo_gradio.py` | Interactive 3D Reconstruction | Camera + Depth + Point | GLB (Web 3D) | Exploration/Demonstration |
| `demo_viser.py` | Professional Visualization | Camera + Depth + Point | Viser Server | Research/Analysis |
| `demo_colmap.py` | Production Export | Camera + Depth + Track | COLMAP Format | NeRF/3DGS Training |

## Demo-Specific Task Analysis

### 1. demo_gradio.py - Interactive Web Interface

**Core Task**: Real-time 3D reconstruction with interactive visualization

**Workflow**:
```python
# 1. Load and preprocess images
images = load_and_preprocess_images(image_names)

# 2. Run full model inference
predictions = model(images)  # Uses all heads except tracking

# 3. Convert pose encoding to camera matrices
extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], ...)

# 4. Generate 3D points (two modes available)
if "Depthmap and Camera Branch":
    world_points = unproject_depth_map_to_point_map(depth, extrinsic, intrinsic)
else:  # "Pointmap Branch" 
    world_points = predictions["world_points"]

# 5. Convert to interactive GLB scene
glbscene = predictions_to_glb(predictions, ...)
```

**Heads Utilized**:
- **Camera Head**: Pose estimation for camera matrices
- **Depth Head**: Dense depth prediction with confidence
- **Point Head**: Direct 3D world coordinate regression (alternative mode)
- **Track Head**: Not used

**Unique Features**:
- **Dual prediction modes**: Switch between depth-based and direct point regression
- **Real-time parameter tuning**: Confidence threshold, frame filtering, masking
- **Browser-based 3D viewer**: Interactive GLB visualization
- **Sky segmentation**: Optional filtering for outdoor scenes

### 2. demo_viser.py - Professional 3D Visualization

**Core Task**: High-quality 3D scene visualization with professional controls

**Workflow**:
```python
# 1. Run VGGT inference (similar to gradio)
predictions = model(images)

# 2. Choose visualization mode
if use_point_map:
    world_points = predictions["world_points"]  # Point head output
    conf = predictions["world_points_conf"]
else:
    world_points = unproject_depth_map_to_point_map(...)  # Depth head output
    conf = predictions["depth_conf"]

# 3. Apply optional sky segmentation
if mask_sky:
    conf = apply_sky_segmentation(conf, image_folder)

# 4. Create viser server with interactive controls
server = viser.ViserServer(...)
# Add point clouds, cameras, interactive widgets
```

**Heads Utilized**:
- **Camera Head**: Camera pose estimation
- **Depth Head**: Primary dense prediction (default mode)
- **Point Head**: Alternative direct 3D prediction (`--use_point_map`)
- **Track Head**: Not used

**Unique Features**:
- **Professional visualization**: Viser server with advanced 3D controls
- **Clickable camera frustums**: Jump to specific viewpoints
- **Real-time confidence filtering**: Percentage-based point filtering
- **Background server mode**: Can run as daemon process

### 3. demo_colmap.py - Production Pipeline Export

**Core Task**: Export reconstruction to COLMAP format for downstream NeRF/3DGS training

**Workflow**:
```python
# 1. VGGT inference for initial reconstruction
extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype)

# 2. Optional bundle adjustment workflow
if args.use_ba:
    # 2a. Extract and track keypoints
    query_points = extract_keypoints(images, method="aliked+sp")
    tracks, vis_score, conf_score = predict_tracks(
        model, images, query_points, fine_tracking=True
    )
    
    # 2b. Run bundle adjustment
    cameras_colmap, images_colmap, points3d_colmap = run_bundle_adjustment(
        extrinsic, intrinsic, tracks, vis_score, conf_score, ...
    )
else:
    # Feedforward only: convert VGGT output to COLMAP format
    cameras_colmap, images_colmap = batch_np_matrix_to_pycolmap_wo_track(...)

# 3. Export to COLMAP files
write_cameras_binary(cameras_colmap, cameras_bin_path)
write_images_binary(images_colmap, images_bin_path)
write_points3D_binary(points3d_colmap, points3d_bin_path)
```

**Heads Utilized**:
- **Camera Head**: Initial pose estimation
- **Depth Head**: Depth maps for 3D point generation
- **Point Head**: Not used directly (depth-based approach preferred)
- **Track Head**: Used indirectly via VGGSfM tracker for bundle adjustment

**Unique Features**:
- **COLMAP format export**: Direct compatibility with NeRF/3DGS pipelines
- **Bundle adjustment support**: Refinement using tracked correspondences
- **Resolution preservation**: Scales results from 518×518 back to original resolution
- **Professional tracking**: VGGSfM tracker with keypoint extraction

## Head Architecture Details

### 1. Camera Head Architecture

```python
class CameraHead(nn.Module):
    def __init__(self, dim_in=2048, trunk_depth=4, ...):
        # Transformer trunk for iterative refinement
        self.trunk = nn.Sequential(*[Block(...) for _ in range(trunk_depth)])
        
        # Pose embedding and modulation
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, 9))
        self.embed_pose = nn.Linear(9, dim_in)
        self.poseLN_modulation = nn.Sequential(...)
        
        # Output prediction branch
        self.pose_branch = Mlp(in_features=dim_in, out_features=9)
```

**Key Specializations**:
- **Iterative refinement**: 4 iterations of pose updates with modulated attention
- **Pose parameterization**: 9D encoding (3D translation + 4D quaternion + 2D FoV)
- **Adaptive normalization**: Modulated layer normalization for refinement
- **Dedicated camera tokens**: Uses only camera tokens (index 0) from aggregator

**Architecture Flow**:
```
Camera Tokens [B, S, C] → Trunk [4×Block] → Pose Branch → [B, S, 9]
     ↑                           ↑
Empty Pose Init           Modulated Attention
```

### 2. DPT Head Architecture (Depth & Point Heads)

```python
class DPTHead(nn.Module):
    def __init__(self, dim_in, output_dim=4, activation="inv_log", ...):
        # Multi-scale feature projection
        self.projects = nn.ModuleList([Conv2d(dim_in, oc, 1) for oc in out_channels])
        
        # Multi-scale fusion (DPT architecture)
        self.resize_layers = nn.ModuleList([...])  # Upsample/downsample
        self.scratch.refinenet1-4 = ...  # Feature fusion blocks
        
        # Final prediction layers
        self.scratch.output_conv1 = Conv2d(features, features//2, 3)
        self.scratch.output_conv2 = Sequential(Conv2d, ReLU, Conv2d)
```

**Key Specializations**:

#### Depth Head Configuration:
- **Output dim**: 2 (depth + confidence)
- **Activation**: "exp" (ensures positive depth)
- **Conf activation**: "expp1" (confidence ≥ 1)
- **Multi-scale fusion**: Combines features from layers [4, 11, 17, 23]

#### Point Head Configuration:
- **Output dim**: 4 (3D coordinates + confidence) 
- **Activation**: "inv_log" (handles large coordinate ranges)
- **Conf activation**: "expp1" (confidence ≥ 1)
- **Direct regression**: Predicts world coordinates without camera unprojection

**Architecture Flow**:
```
Aggregated Tokens → Multi-layer Features → DPT Fusion → Dense Prediction
[B,S,P,2C]          [4 different scales]    [256 features]   [B,S,H,W,output_dim]
```

### 3. Track Head Architecture

```python
class TrackHead(nn.Module):
    def __init__(self, dim_in, features=128, iters=4, ...):
        # Feature extractor (DPT-based)
        self.feature_extractor = DPTHead(
            dim_in=dim_in, features=features, 
            feature_only=True, down_ratio=2
        )
        
        # Tracker module  
        self.tracker = BaseTrackerPredictor(
            latent_dim=features, predict_conf=True,
            corr_levels=7, corr_radius=4, hidden_size=384
        )
```

**Key Specializations**:
- **Two-stage design**: DPT feature extraction + specialized tracker
- **Correlation-based**: Multi-level correlation pyramids for feature matching
- **Iterative tracking**: 4 iterations of coordinate refinement
- **Spatial resolution**: Operates at H//2 × W//2 due to down_ratio=2

**BaseTrackerPredictor Architecture**:
```python
class BaseTrackerPredictor(nn.Module):
    def __init__(self, corr_levels=7, corr_radius=4, ...):
        # Correlation processing
        self.corr_mlp = Mlp(corr_features, hidden_size, latent_dim)
        
        # Transformer for temporal reasoning
        self.updateformer = EfficientUpdateFormer(
            space_depth=6, time_depth=6, ...
        )
        
        # Prediction heads
        self.vis_predictor = Linear(latent_dim, 1)
        self.conf_predictor = Linear(latent_dim, 1)
```

**Architecture Flow**:
```
Feature Maps [B,S,C,H//2,W//2] → Correlation → UpdateFormer → Tracks [B,S,N,2]
Query Points [B,N,2]              Pyramid      [Space+Time]    + Vis/Conf
```

## Head Specialization Summary

### Architectural Differences

| **Head** | **Input** | **Architecture** | **Output** | **Specialization** |
|----------|-----------|------------------|------------|-------------------|
| **Camera** | Camera tokens only | Transformer trunk + iterative refinement | [B,S,9] pose encoding | Dedicated pose reasoning |
| **Depth** | All patch tokens | DPT multi-scale fusion | [B,S,H,W,2] depth+conf | Dense depth estimation |
| **Point** | All patch tokens | DPT multi-scale fusion | [B,S,H,W,4] xyz+conf | Direct 3D regression |
| **Track** | Feature extraction + correlation | DPT→Tracker pipeline | [B,S,N,2] tracks+vis+conf | Temporal correspondence |

### Task-Specific Optimizations

#### Camera Head:
- **Global reasoning**: Uses camera tokens that aggregate global scene information
- **Iterative refinement**: Multiple passes for accurate pose estimation
- **Geometric consistency**: Quaternion parameterization prevents singularities

#### Depth Head:
- **Local-to-global**: Multi-scale fusion captures both fine details and global structure
- **Metric scale**: Exp activation ensures positive depth values
- **Confidence modeling**: Separate confidence stream for uncertainty estimation

#### Point Head:
- **Direct regression**: Bypasses camera parameters for end-to-end 3D prediction
- **Large dynamic range**: Inv_log activation handles world coordinate scales
- **Alternative to depth**: Provides backup when camera estimation is unreliable

#### Track Head:
- **Feature correlation**: Multi-level pyramids for robust feature matching
- **Temporal modeling**: UpdateFormer captures motion patterns
- **Iterative refinement**: Progressive coordinate updates with visibility scoring

This specialization allows VGGT to excel at different aspects of 3D scene understanding while maintaining a unified backbone architecture.