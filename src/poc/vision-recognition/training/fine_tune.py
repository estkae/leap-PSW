#!/usr/bin/env python3
"""
Fine-Tuning Pipeline for LFM2-VL Vision Recognition Model
Adapts pre-trained models for specific tasks or domains
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from train_model import LFM2VLModel, TrainingConfig, VisionDataset, ModelTrainer
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ Training dependencies not available. Using mock implementation.")
    TORCH_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning process"""
    # Base model
    pretrained_model_path: str = "checkpoints/best_model.pth"
    freeze_backbone: bool = True
    freeze_layers: List[str] = None

    # Fine-tuning parameters
    learning_rate: float = 0.0001  # Lower than training
    num_epochs: int = 20
    batch_size: int = 16  # Smaller batch for fine-tuning
    warmup_epochs: int = 3

    # Learning rate scheduling
    use_cosine_schedule: bool = True
    min_lr: float = 1e-6

    # Data augmentation (usually more aggressive)
    augmentation_strength: float = 0.8
    mixup_alpha: float = 0.2

    # Regularization
    dropout: float = 0.3
    weight_decay: float = 0.01
    label_smoothing: float = 0.1

    # Task-specific
    new_num_classes: Optional[int] = None  # None = keep original
    task_type: str = "classification"  # "classification", "detection", "segmentation"

    # Progressive unfreezing
    progressive_unfreezing: bool = False
    unfreeze_schedule: List[int] = None  # Epochs at which to unfreeze layers


class FineTuningDataset(VisionDataset):
    """Enhanced dataset for fine-tuning with domain adaptation"""

    def __init__(self, data_dir: str, config: FineTuningConfig, mode: str = "train"):
        """
        Initialize fine-tuning dataset

        Args:
            data_dir: Path to domain-specific dataset
            config: Fine-tuning configuration
            mode: "train", "val", or "test"
        """
        self.fine_tune_config = config

        # Create base config for parent class
        base_config = TrainingConfig()
        super().__init__(data_dir, base_config, mode)

        # Override with fine-tuning specific transforms
        self.transform = self._get_fine_tuning_transforms()

    def _get_fine_tuning_transforms(self):
        """Get fine-tuning specific transformations"""
        if not TORCH_AVAILABLE:
            return None

        import torchvision.transforms as transforms

        if self.mode == "train":
            # More aggressive augmentation for fine-tuning
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.3 * self.fine_tune_config.augmentation_strength,
                    contrast=0.3 * self.fine_tune_config.augmentation_strength,
                    saturation=0.3 * self.fine_tune_config.augmentation_strength,
                    hue=0.1 * self.fine_tune_config.augmentation_strength
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

        return transform


class FineTuningModel(nn.Module if TORCH_AVAILABLE else object):
    """Enhanced model for fine-tuning with task-specific heads"""

    def __init__(self, base_model: LFM2VLModel, config: FineTuningConfig):
        if TORCH_AVAILABLE:
            super().__init__()

        self.config = config
        self.base_model = base_model

        if TORCH_AVAILABLE:
            # Freeze layers if specified
            if config.freeze_backbone:
                self._freeze_backbone()

            # Add task-specific heads
            self._add_task_heads()

    def _freeze_backbone(self):
        """Freeze backbone layers for fine-tuning"""
        if not TORCH_AVAILABLE:
            return

        # Freeze vision backbone
        for param in self.base_model.vision_backbone.parameters():
            param.requires_grad = False

        logger.info("Backbone frozen for fine-tuning")

    def _add_task_heads(self):
        """Add task-specific classification/detection heads"""
        if not TORCH_AVAILABLE:
            return

        # Replace classifier if needed
        if self.config.new_num_classes:
            # Get feature size
            feature_size = self.base_model.classifier[1].in_features

            # New classifier with dropout
            self.base_model.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(self.config.dropout),
                nn.Linear(feature_size, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
                nn.Linear(512, self.config.new_num_classes)
            )

            logger.info(f"Added new classifier for {self.config.new_num_classes} classes")

    def forward(self, x):
        """Forward pass"""
        return self.base_model(x)

    def unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specific layers during progressive unfreezing"""
        if not TORCH_AVAILABLE:
            return

        for name, param in self.base_model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = True

        logger.info(f"Unfrozen layers: {layer_names}")


class FineTuner:
    """Main fine-tuning coordinator"""

    def __init__(self, config: FineTuningConfig):
        """
        Initialize fine-tuner

        Args:
            config: Fine-tuning configuration
        """
        self.config = config

        # Load pre-trained model
        self.base_model = self._load_pretrained_model()

        # Create fine-tuning model
        self.model = FineTuningModel(self.base_model, config)

        # Setup training components
        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            # Optimizer with different learning rates for different parts
            self.optimizer = self._get_fine_tuning_optimizer()

            # Loss function
            self.criterion = self._get_loss_function()

            # Learning rate scheduler
            self.scheduler = self._get_scheduler()

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def _load_pretrained_model(self) -> LFM2VLModel:
        """Load pre-trained base model"""
        logger.info(f"Loading pre-trained model: {self.config.pretrained_model_path}")

        if not TORCH_AVAILABLE:
            return None

        # Create base config (this should match the pre-trained model)
        base_config = TrainingConfig()
        model = LFM2VLModel(base_config)

        if os.path.exists(self.config.pretrained_model_path):
            checkpoint = torch.load(self.config.pretrained_model_path, map_location='cpu')
            model.load_state_dict(checkpoint.get('model_state', checkpoint))
            logger.info("Pre-trained model loaded successfully")
        else:
            logger.warning(f"Pre-trained model not found: {self.config.pretrained_model_path}")

        return model

    def _get_fine_tuning_optimizer(self):
        """Get optimizer with different learning rates for different parts"""
        if not TORCH_AVAILABLE:
            return None

        # Different learning rates for frozen and unfrozen parts
        param_groups = []

        # Backbone parameters (lower learning rate)
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name or 'detector' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': self.config.learning_rate * 0.1  # 10x lower for backbone
            })

        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': self.config.learning_rate  # Full learning rate for new heads
            })

        return optim.AdamW(param_groups, weight_decay=self.config.weight_decay)

    def _get_loss_function(self):
        """Get loss function with label smoothing"""
        if not TORCH_AVAILABLE:
            return None

        if self.config.label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            return nn.CrossEntropyLoss()

    def _get_scheduler(self):
        """Get learning rate scheduler"""
        if not TORCH_AVAILABLE or self.optimizer is None:
            return None

        if self.config.use_cosine_schedule:
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.min_lr
            )
        else:
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )

    def prepare_data(self, train_dir: str, val_dir: str):
        """Prepare fine-tuning datasets"""
        train_dataset = FineTuningDataset(train_dir, self.config, mode="train")
        val_dataset = FineTuningDataset(val_dir, self.config, mode="val")

        if TORCH_AVAILABLE:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        else:
            # Mock loaders
            train_loader = [("mock", i, {}) for i in range(20)]
            val_loader = [("mock", i, {}) for i in range(10)]

        return train_loader, val_loader

    def fine_tune(self, train_dir: str, val_dir: str):
        """Main fine-tuning loop"""
        logger.info("Starting fine-tuning process...")

        # Prepare data
        train_loader, val_loader = self.prepare_data(train_dir, val_dir)

        # Warmup phase
        if self.config.warmup_epochs > 0:
            logger.info(f"Warmup phase: {self.config.warmup_epochs} epochs")
            self._warmup_phase(train_loader, val_loader)

        # Main fine-tuning
        for epoch in range(self.config.num_epochs):
            start_time = time.time()

            # Progressive unfreezing
            if self.config.progressive_unfreezing and self.config.unfreeze_schedule:
                if epoch in self.config.unfreeze_schedule:
                    self._progressive_unfreeze(epoch)

            # Training
            train_loss, train_acc = self._train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self._validate_epoch(val_loader)

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log results
            current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001
            epoch_time = time.time() - start_time

            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            logger.info(f"  Learning Rate: {current_lr:.6f}")
            logger.info(f"  Time: {epoch_time:.2f}s")

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

        logger.info("Fine-tuning completed!")
        return self.history

    def _warmup_phase(self, train_loader, val_loader):
        """Warmup phase with gradual learning rate increase"""
        original_lr = self.config.learning_rate
        warmup_lr_schedule = [original_lr * (i + 1) / self.config.warmup_epochs
                             for i in range(self.config.warmup_epochs)]

        for epoch, lr in enumerate(warmup_lr_schedule):
            # Set learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Train epoch
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate_epoch(val_loader)

            logger.info(f"Warmup {epoch+1}/{self.config.warmup_epochs}")
            logger.info(f"  LR: {lr:.6f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    def _progressive_unfreeze(self, epoch: int):
        """Progressively unfreeze layers"""
        if epoch == self.config.unfreeze_schedule[0]:
            # Unfreeze classifier layers
            layers_to_unfreeze = ['classifier']
        elif epoch == self.config.unfreeze_schedule[1]:
            # Unfreeze some backbone layers
            layers_to_unfreeze = ['layer4', 'layer3']
        else:
            # Unfreeze more backbone layers
            layers_to_unfreeze = ['layer2', 'layer1']

        self.model.unfreeze_layers(layers_to_unfreeze)

    def _train_epoch(self, dataloader):
        """Train for one epoch"""
        if not TORCH_AVAILABLE:
            return 0.3, 0.89  # Mock values

        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels, metadata) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs, _ = self.model(images)

            # Loss calculation
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(dataloader), correct / total

    def _validate_epoch(self, dataloader):
        """Validate for one epoch"""
        if not TORCH_AVAILABLE:
            return 0.25, 0.92  # Mock values

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels, metadata in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs, _ = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return total_loss / len(dataloader), correct / total

    def save_fine_tuned_model(self, output_path: str):
        """Save fine-tuned model"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)

        model_path = output_dir / "fine_tuned_model.pth"
        config_path = output_dir / "fine_tuning_config.json"

        if TORCH_AVAILABLE:
            # Save model
            torch.save({
                'model_state': self.model.state_dict(),
                'config': asdict(self.config),
                'history': self.history
            }, model_path)

        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        logger.info(f"Fine-tuned model saved to: {output_path}")


def main():
    """Main fine-tuning function"""
    print("=" * 60)
    print("LFM2-VL Model Fine-Tuning")
    print("=" * 60)

    # Configuration
    config = FineTuningConfig(
        pretrained_model_path="checkpoints/best_model.pth",
        freeze_backbone=True,
        learning_rate=0.0001,
        num_epochs=15,
        batch_size=16,
        new_num_classes=5,  # New task with 5 classes
        progressive_unfreezing=True,
        unfreeze_schedule=[5, 10]
    )

    print(f"Configuration:")
    print(f"  Pre-trained model: {config.pretrained_model_path}")
    print(f"  New classes: {config.new_num_classes}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Freeze backbone: {config.freeze_backbone}")
    print()

    # Create fine-tuner
    fine_tuner = FineTuner(config)

    # Fine-tune
    print("Starting fine-tuning...")
    history = fine_tuner.fine_tune("data/custom_train", "data/custom_val")

    # Save fine-tuned model
    fine_tuner.save_fine_tuned_model("fine_tuned_models")

    print("\n✅ Fine-tuning completed successfully!")
    return fine_tuner, history


if __name__ == "__main__":
    fine_tuner, history = main()