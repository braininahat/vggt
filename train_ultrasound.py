#!/usr/bin/env python3
"""
Training script for finetuning VGGT on ultrasound data.
Adapted from VGGT's training framework with specific configurations for ultrasound pointmap data.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf
from datetime import datetime

# Add training directory to path
sys.path.append("training")

from vggt.models.vggt import VGGT
from training.data.datasets.ultrasound_hdf5 import UltrasoundHDF5Dataset
from training.loss import camera_loss, depth_loss, point_loss


class UltrasoundTrainer:
    """Trainer for finetuning VGGT on ultrasound data."""

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = self._init_model()

        # Initialize datasets
        self.train_dataset, self.val_dataset = self._init_datasets()

        # Initialize optimizer
        self.optimizer = self._init_optimizer()

        # Initialize scheduler
        self.scheduler = self._init_scheduler()

        # Initialize wandb if enabled
        if config.logging.use_wandb:
            self._init_wandb()

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler() if config.training.use_amp else None

    def _init_model(self):
        """Initialize VGGT model."""
        print("Loading pretrained VGGT model...")

        # Load pretrained model
        model = VGGT.from_pretrained(self.config.model.pretrained_name)
        model = model.to(self.device)

        # Optionally freeze parts of the model
        if self.config.model.freeze_aggregator:
            print("Freezing aggregator...")
            for param in model.aggregator.parameters():
                param.requires_grad = False

        if self.config.model.freeze_camera_head:
            print("Freezing camera head...")
            for param in model.camera_head.parameters():
                param.requires_grad = False

        return model

    def _init_datasets(self):
        """Initialize training and validation datasets."""
        # Common configuration for datasets
        common_conf = OmegaConf.create(
            {
                "img_size": self.config.data.img_size,
                "patch_size": self.config.data.patch_size,
                "augs": {"scales": self.config.data.aug_scales},
                "rescale": True,
                "rescale_aug": self.config.data.enable_augmentation,
                "landscape_check": False,  # Ultrasound images are typically portrait
                "debug": False,
                "training": True,
                "get_nearby": False,
                "load_depth": True,
                "inside_random": False,
            }
        )

        # Training dataset
        train_dataset = UltrasoundHDF5Dataset(
            common_conf=common_conf,
            hdf5_root_dir=self.config.data.hdf5_root_dir,
            split="train",
            sequence_length=self.config.data.sequence_length,
            stride=self.config.data.stride,
            max_sequences_per_file=self.config.data.max_sequences_per_file,
            enable_augmentation=self.config.data.enable_augmentation,
            train_split_ratio=self.config.data.train_split_ratio,
        )

        # Validation dataset (no augmentation)
        val_common_conf = common_conf.copy()
        val_common_conf.training = False
        val_common_conf.rescale_aug = False

        val_dataset = UltrasoundHDF5Dataset(
            common_conf=val_common_conf,
            hdf5_root_dir=self.config.data.hdf5_root_dir,
            split="test",
            sequence_length=self.config.data.sequence_length,
            stride=self.config.data.stride,
            max_sequences_per_file=self.config.data.max_sequences_per_file,
            enable_augmentation=False,
            train_split_ratio=self.config.data.train_split_ratio,
        )

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")

        return train_dataset, val_dataset

    def _init_optimizer(self):
        """Initialize optimizer."""
        # Filter parameters that require gradients
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.config.optimizer.type == "adamw":
            optimizer = optim.AdamW(
                params,
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
                betas=(self.config.optimizer.beta1, self.config.optimizer.beta2),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer.type}")

        return optimizer

    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        if self.config.scheduler.type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.scheduler.min_lr,
            )
        elif self.config.scheduler.type == "constant":
            scheduler = optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler.type}")

        return scheduler

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.logging.wandb_project,
            name=self.config.logging.exp_name,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

    def compute_losses(self, batch, predictions):
        """Compute all losses for ultrasound data."""
        losses = {}

        # Camera loss (if not frozen)
        if not self.config.model.freeze_camera_head and "pose_enc" in predictions:
            camera_loss_dict, _ = camera_loss(
                [predictions["pose_enc"]],  # camera_loss expects a list
                batch,
                loss_type=self.config.loss.camera_loss_type,
                weight_T=self.config.loss.weight_camera_T,
                weight_R=self.config.loss.weight_camera_R,
                weight_fl=self.config.loss.weight_camera_fl,
            )
            losses.update(camera_loss_dict)
        else:
            losses["loss_camera"] = torch.tensor(0.0).to(self.device)

        # Depth loss
        if self.config.loss.weight_depth > 0 and "depth" in predictions:
            depth_loss_dict = depth_loss(
                predictions["depth"],
                predictions["depth_conf"],
                batch,
                alpha=self.config.loss.depth_conf_alpha,
                gradient_loss=self.config.loss.depth_gradient_loss,
                disable_conf=self.config.loss.disable_depth_conf,
            )
            losses.update(depth_loss_dict)
        else:
            losses["loss_conf_depth"] = torch.tensor(0.0).to(self.device)

        # Point loss (main loss for ultrasound)
        if self.config.loss.weight_point > 0 and "world_points" in predictions:
            point_loss_dict = point_loss(
                predictions["world_points"],
                predictions["world_points_conf"],
                batch,
                normalize_pred=self.config.loss.normalize_pred_points,
                alpha=self.config.loss.point_conf_alpha,
                gradient_loss=self.config.loss.point_gradient_loss,
                disable_conf=self.config.loss.disable_point_conf,
            )
            losses.update(point_loss_dict)
        else:
            losses["loss_conf"] = torch.tensor(0.0).to(self.device)

        # Total loss
        total_loss = (
            self.config.loss.weight_camera * losses["loss_camera"]
            + self.config.loss.weight_depth * losses.get("loss_conf_depth", 0)
            + self.config.loss.weight_point * losses.get("loss_conf", 0)
        )

        losses["total_loss"] = total_loss

        return losses

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        # Create data loader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

        epoch_losses = []
        accumulation_steps = self.config.training.gradient_accumulation_steps

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {self.epoch}")
        ):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    predictions = self.model(
                        batch["images"],
                        query_points=None,  # Not using tracks for ultrasound
                    )
                    # Compute losses
                    losses = self.compute_losses(batch, predictions)
                    # Scale loss for gradient accumulation
                    losses["total_loss"] = losses["total_loss"] / accumulation_steps
                
                # Backward pass with scaled gradients
                self.scaler.scale(losses["total_loss"]).backward()
            else:
                # Standard forward/backward pass
                predictions = self.model(
                    batch["images"],
                    query_points=None,  # Not using tracks for ultrasound
                )
                losses = self.compute_losses(batch, predictions)
                losses["total_loss"] = losses["total_loss"] / accumulation_steps
                losses["total_loss"].backward()

            # Update weights after accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler is not None:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                if self.config.training.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip
                    )
                
                if self.scaler is not None:
                    # Step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()

            # Record losses (unscaled)
            loss_dict = {}
            for k, v in losses.items():
                if hasattr(v, 'item'):
                    loss_dict[k] = v.item() * accumulation_steps
                else:
                    loss_dict[k] = float(v) * accumulation_steps
            epoch_losses.append(loss_dict)

            # Log to wandb
            if (
                self.config.logging.use_wandb
                and batch_idx % self.config.logging.log_freq == 0
            ):
                wandb.log(
                    {f"train/{k}": v for k, v in epoch_losses[-1].items()},
                    step=self.global_step,
                )

            self.global_step += 1

        # Average epoch losses
        avg_losses = {}
        for key in epoch_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in epoch_losses])

        return avg_losses

    def validate(self):
        """Validate the model."""
        self.model.eval()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )

        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Forward pass
                predictions = self.model(
                    batch["images"],
                    query_points=None,  # Not using tracks for ultrasound
                )

                # Compute losses
                losses = self.compute_losses(batch, predictions)

                loss_dict = {}
                for k, v in losses.items():
                    if hasattr(v, 'item'):
                        loss_dict[k] = v.item()
                    else:
                        loss_dict[k] = float(v)
                val_losses.append(loss_dict)

        # Average validation losses
        avg_losses = {}
        for key in val_losses[0].keys():
            avg_losses[key] = np.mean([l[key] for l in val_losses])

        return avg_losses

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save latest checkpoint
        checkpoint_path = Path(self.config.training.checkpoint_dir) / "latest.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.training.checkpoint_dir) / "best.pth"
            torch.save(checkpoint, best_path)

        # Save periodic checkpoint
        if self.epoch % self.config.training.save_freq == 0:
            epoch_path = (
                Path(self.config.training.checkpoint_dir) / f"epoch_{self.epoch}.pth"
            )
            torch.save(checkpoint, epoch_path)

    def train(self):
        """Main training loop."""
        print("Starting training...")

        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch

            # Train
            train_losses = self.train_epoch()
            print(f"\nEpoch {epoch} - Train losses:")
            for k, v in train_losses.items():
                print(f"  {k}: {v:.4f}")

            # Validate
            val_losses = self.validate()
            print(f"Epoch {epoch} - Val losses:")
            for k, v in val_losses.items():
                print(f"  {k}: {v:.4f}")

            # Update learning rate
            self.scheduler.step()

            # Log to wandb
            if self.config.logging.use_wandb:
                wandb.log(
                    {f"val/{k}": v for k, v in val_losses.items()},
                    step=self.global_step,
                )
                wandb.log(
                    {
                        "epoch": epoch,
                        "lr": self.scheduler.get_last_lr()[0],
                    },
                    step=self.global_step,
                )

            # Save checkpoint
            is_best = val_losses["total_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses["total_loss"]
            self.save_checkpoint(is_best=is_best)

        print("Training completed!")


def create_default_config():
    """Create default configuration for ultrasound training."""
    config = OmegaConf.create(
        {
            "model": {
                "pretrained_name": "facebook/VGGT-1B",  # Only available model
                "freeze_aggregator": True,  # Freeze to save memory
                "freeze_camera_head": True,  # Freeze to save memory
                "predict_depth": True,
                "predict_points": True,
            },
            "data": {
                "hdf5_root_dir": "/home/varun/XiaLabSync/datasets/freehand_May212025/processed",
                "img_size": 518,
                "patch_size": 14,
                "sequence_length": 4,  # Reduced from 8 to save memory
                "stride": 10,  # Increased stride to get more variation
                "batch_size": 1,  # Minimum batch size for memory
                "num_workers": 2,  # Reduced workers to save memory
                "max_sequences_per_file": 10,
                "train_split_ratio": 0.8,
                "enable_augmentation": False,  # Start without augmentation
                "aug_scales": [0.9, 1.1],
            },
            "optimizer": {
                "type": "adamw",
                "lr": 1e-5,  # Lower learning rate for finetuning
                "weight_decay": 0.05,
                "beta1": 0.9,
                "beta2": 0.999,
            },
            "scheduler": {
                "type": "cosine",
                "min_lr": 1e-6,
            },
            "loss": {
                # Loss weights - emphasize point loss for ultrasound
                "weight_camera": 0.1,  # Lower weight for synthetic cameras
                "weight_depth": 0.2,  # Medium weight for depth
                "weight_point": 1.0,  # High weight for pointmaps (main supervision)
                # Camera loss settings
                "camera_loss_type": "l1",
                "weight_camera_T": 1.0,
                "weight_camera_R": 1.0,
                "weight_camera_fl": 0.5,
                # Depth loss settings
                "depth_conf_alpha": 0.2,
                "depth_gradient_loss": "grad",
                "disable_depth_conf": False,
                # Point loss settings
                "normalize_pred_points": True,
                "point_conf_alpha": 0.2,
                "point_gradient_loss": "grad",
                "disable_point_conf": False,
            },
            "training": {
                "num_epochs": 50,
                "grad_clip": 1.0,
                "save_freq": 5,
                "checkpoint_dir": "./checkpoints/ultrasound",
                "use_amp": True,  # Use automatic mixed precision
                "gradient_accumulation_steps": 4,  # Accumulate gradients
            },
            "logging": {
                "use_wandb": False,  # Set to True if you want to use wandb
                "wandb_project": "vggt-ultrasound",
                "exp_name": f'ultrasound_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                "log_freq": 10,
            },
        }
    )

    return config


def main():
    parser = argparse.ArgumentParser(description="Train VGGT on ultrasound data")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    # Load config
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config = create_default_config()

    # Override config with command line arguments
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.lr is not None:
        config.optimizer.lr = args.lr
    if args.use_wandb:
        config.logging.use_wandb = True

    # Create trainer and start training
    trainer = UltrasoundTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
