# Minimal VGGT adaptation using frame centers and quaternions directly
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

from vggt.models.vggt import VGGT
from vggt.heads.camera_head import CameraHead
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.rotation import quat_to_mat, mat_to_quat


class FrameCenterPredictor(VGGT):
    """Simplified VGGT for predicting frame centers and orientations.
    
    Uses the standardized coordinate system from notebook 008:
    - First frame center at origin
    - First frame has identity rotation
    - Predicts relative motion for subsequent frames
    """
    
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        frame_width_mm: float = 38.0,
        frame_height_mm: float = 50.0,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
    ):
        super().__init__(img_size, patch_size, embed_dim)
        
        # Frame dimensions for corner reconstruction
        self.frame_width_mm = frame_width_mm
        self.frame_height_mm = frame_height_mm
        
        # Replace camera head with a simpler version if using LoRA
        if use_lora:
            # Keep existing camera head but add LoRA layers
            self._add_lora_to_camera_head(lora_rank, lora_alpha)
        
    def _add_lora_to_camera_head(self, rank: int, alpha: float):
        """Add LoRA layers to existing camera head."""
        # Add LoRA to trunk blocks in camera head
        for block in self.camera_head.trunk:
            # Original dimensions
            dim = block.attn.qkv.in_features
            
            # Add LoRA to QKV projection
            self._add_lora_layer(block.attn, 'qkv', dim, dim * 3, rank, alpha)
            
            # Add LoRA to MLP
            if hasattr(block, 'mlp'):
                hidden_dim = block.mlp.fc1.out_features
                self._add_lora_layer(block.mlp, 'fc1', dim, hidden_dim, rank, alpha)
                self._add_lora_layer(block.mlp, 'fc2', hidden_dim, dim, rank, alpha)
    
    def _add_lora_layer(self, module, attr_name, in_features, out_features, rank, alpha):
        """Add a LoRA layer to a specific module attribute."""
        original_layer = getattr(module, attr_name)
        
        # Create LoRA parameters
        lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Store as new attributes
        setattr(module, f'{attr_name}_lora_A', lora_A)
        setattr(module, f'{attr_name}_lora_B', lora_B)
        setattr(module, f'{attr_name}_lora_scaling', alpha / rank)
        
        # Replace forward method to include LoRA
        original_forward = original_layer.forward
        
        def lora_forward(x):
            # Original output
            out = original_forward(x)
            # Add LoRA contribution
            lora_out = x @ lora_A.T @ lora_B.T * getattr(module, f'{attr_name}_lora_scaling')
            return out + lora_out
        
        original_layer.forward = lora_forward
    
    def forward(
        self,
        images: torch.Tensor,
        return_corners: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass predicting frame centers and orientations.
        
        Args:
            images: Input images (B, S, 3, H, W) or (S, 3, H, W)
            return_corners: Whether to compute frame corners from center+orientation
            
        Returns:
            Dictionary containing:
            - frame_centers: (B, S, 3) frame center positions (first at origin)
            - frame_quaternions: (B, S, 4) quaternions wxyz (first is identity)
            - frame_corners: (B, S, 4, 3) if return_corners=True
        """
        # Add batch dimension if needed
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
            
        B, S, _, H, W = images.shape
        
        # Get aggregated tokens
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        
        predictions = {}
        
        with torch.cuda.amp.autocast(enabled=False):
            # Predict camera pose encoding
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                pose_enc = pose_enc_list[-1]  # Last iteration
                
                # Extract translation and rotation
                # Pose encoding format: [T(3), quat(4), fov(2)]
                frame_centers = pose_enc[:, :, :3]  # (B, S, 3)
                frame_quats = pose_enc[:, :, 3:7]   # (B, S, 4)
                
                # Normalize quaternions
                frame_quats = F.normalize(frame_quats, p=2, dim=-1)
                
                # Enforce first frame constraints
                # First frame should be at origin with identity rotation
                if S > 1:
                    # Adjust centers relative to first frame
                    frame_centers = frame_centers - frame_centers[:, 0:1, :]
                    
                    # Adjust rotations relative to first frame
                    # This is handled by the loss function during training
                    # For inference, we can optionally enforce it here
                
                predictions["frame_centers"] = frame_centers
                predictions["frame_quaternions"] = frame_quats
                
                # Compute frame corners if requested
                if return_corners:
                    frame_corners = self.compute_frame_corners_from_center_quat(
                        frame_centers, frame_quats
                    )
                    predictions["frame_corners"] = frame_corners
        
        predictions["images"] = images
        
        return predictions
    
    def compute_frame_corners_from_center_quat(
        self,
        centers: torch.Tensor,
        quaternions: torch.Tensor
    ) -> torch.Tensor:
        """Compute frame corners from center positions and orientations.
        
        Args:
            centers: (B, S, 3) frame center positions
            quaternions: (B, S, 4) quaternions in wxyz format
            
        Returns:
            corners: (B, S, 4, 3) corner positions
        """
        B, S = centers.shape[:2]
        device = centers.device
        
        # Convert quaternions to rotation matrices
        # quaternions are wxyz, need to convert to rotation matrices
        R_list = []
        for b in range(B):
            for s in range(S):
                q = quaternions[b, s]
                # Convert wxyz to rotation matrix
                R = quat_to_mat(q.unsqueeze(0)).squeeze(0)
                R_list.append(R)
        
        R_all = torch.stack(R_list).reshape(B, S, 3, 3)
        
        # Define corners in frame-local coordinates
        # Assuming frame lies in XY plane, centered at origin
        half_width = self.frame_width_mm / 2
        half_height = self.frame_height_mm / 2
        
        local_corners = torch.tensor([
            [-half_width, -half_height, 0],  # Corner 1: bottom-left
            [-half_width,  half_height, 0],  # Corner 2: top-left
            [ half_width,  half_height, 0],  # Corner 3: top-right
            [ half_width, -half_height, 0],  # Corner 4: bottom-right
        ], device=device, dtype=centers.dtype)
        
        # Transform corners to world coordinates
        corners = []
        for b in range(B):
            for s in range(S):
                # Rotate local corners
                rotated_corners = (R_all[b, s] @ local_corners.T).T  # (4, 3)
                # Translate to frame center
                world_corners = rotated_corners + centers[b, s].unsqueeze(0)
                corners.append(world_corners)
        
        corners = torch.stack(corners).reshape(B, S, 4, 3)
        
        return corners
    
    def compute_loss(self, predictions: dict, targets: dict) -> dict:
        """Compute losses for training.
        
        Expects targets to have standardized coordinates:
        - First frame center at origin
        - First frame with identity rotation
        """
        losses = {}
        
        # Center position loss
        if 'frame_centers' in predictions and 'frame_centers' in targets:
            center_loss = F.l1_loss(
                predictions['frame_centers'],
                targets['frame_centers']
            )
            losses['center'] = center_loss
        
        # Quaternion loss (with special handling for first frame)
        if 'frame_quaternions' in predictions and 'frame_quaternions' in targets:
            pred_quats = predictions['frame_quaternions']
            target_quats = targets['frame_quaternions']
            
            # First frame should have identity quaternion [1, 0, 0, 0]
            identity_quat = torch.tensor([1., 0., 0., 0.], device=pred_quats.device)
            first_frame_loss = F.mse_loss(
                pred_quats[:, 0, :],
                identity_quat.expand(pred_quats.shape[0], 4)
            )
            
            # Other frames: quaternion distance
            if pred_quats.shape[1] > 1:
                # Quaternion loss (considering q and -q represent same rotation)
                quat_dot = (pred_quats[:, 1:] * target_quats[:, 1:]).sum(dim=-1)
                quat_loss = 1 - quat_dot.abs().mean()
                losses['quaternion'] = quat_loss + 0.1 * first_frame_loss
            else:
                losses['quaternion'] = first_frame_loss
        
        # Corner loss if available
        if 'frame_corners' in predictions and 'frame_corners' in targets:
            corner_loss = F.l1_loss(
                predictions['frame_corners'],
                targets['frame_corners']
            )
            losses['corners'] = corner_loss * 0.5  # Lower weight since derived
        
        return losses
    
    def load_pretrained_with_lora(self, pretrained_path: str, freeze_base: bool = True):
        """Load pretrained VGGT and prepare for LoRA fine-tuning."""
        # Load pretrained weights
        if pretrained_path.endswith('.pt'):
            state_dict = torch.load(pretrained_path, map_location='cpu')
        else:
            # Load from HuggingFace
            from vggt.models.vggt import VGGT as VGGTBase
            base_model = VGGTBase.from_pretrained(pretrained_path)
            state_dict = base_model.state_dict()
        
        # Load only matching keys (ignore LoRA weights)
        model_state = self.state_dict()
        pretrained_state = {}
        
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                if 'lora' not in k:  # Skip LoRA parameters
                    pretrained_state[k] = v
        
        self.load_state_dict(pretrained_state, strict=False)
        
        print(f"Loaded {len(pretrained_state)} pretrained parameters")
        
        # Freeze base model if requested
        if freeze_base:
            for name, param in self.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
            
            # Count trainable parameters
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        return self