import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import timm
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
import kornia.augmentation as K

from ..transforms.normalize import NormalizeMeanStd


from ..backbones.registry import BACKBONE_REGISTRY


class DINOModule(LightningModule):
    """DINO module for hyperspectral self-supervised learning.
    
    This module implements the DINO (self-DIstillation with NO labels) approach
    adapted for hyperspectral imagery. It trains using a student-teacher architecture
    where the teacher is updated via momentum averaging of student parameters.
    The model supports both standard and multi-crop training strategies.
    
    """

    def __init__(
        self,
        backbone_name: str = "spec_resnet50",
        in_channels: int = 202,
        hidden_dim: float = 2048,
        bottleneck_dim: float = 256,
        output_dim: int = 32768,
        lr: float = 9.6,
        warmup_epochs: int = 20,
        weight_decay: float = 1e-6,
        momentum: float = 0.9,
        warmup_teacher_temp_epochs: int = 10,
        size: int = 128,
        multicrop: bool = False,
        n_views: int = 0,  # Number of extra local views when multicrop is enabled.
        token_patch_size: int = 4,
    ) -> None:
        """Initialize the DINO module.
        
        Args:
            backbone_name: Backbone architecture from registry or timm
            in_channels: Number of input spectral channels (202 for EnMAP)
            hidden_dim: Hidden dimension in projection head
            bottleneck_dim: Bottleneck dimension in projection head
            output_dim: Output dimension for final representations
            lr: Learning rate for optimizer
            warmup_epochs: Number of warmup epochs for learning rate
            weight_decay: Weight decay factor for regularization
            momentum: Initial momentum value for teacher update
            warmup_teacher_temp_epochs: Epochs to warm up teacher temperature
            size: Input patch size
            multicrop: Whether to use multi-crop strategy
            n_views: Number of additional local views for multi-crop
            token_patch_size: Size of patches for ViT models
        """
        super().__init__()
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.size = size
        self.multicrop = multicrop
        self.n_views = n_views
        self.in_channels = in_channels

        # Load normalization statistics for enMAP (202 channels)
        if in_channels == 202:
            mean = torch.tensor(np.load('data/statistics/mu.npy'))
            std = torch.tensor(np.load('data/statistics/sigma.npy'))
        else:
            raise ValueError("This module only supports in_channels==202 for enMAP data.")

        # Compute augmentation parameters.
        ks = size // 10 // 2 * 2 + 1
        if multicrop:
            max_scale_global = 1.0
            max_scale_local = 0.4
            global_size = size
            local_size = 48
        else:
            max_scale_global = 1.0
            max_scale_local = 0.05
            global_size = local_size = size
        local_ks = local_size // 10 // 2 * 2 + 1

        # Build global augmentation pipeline.
        global_pipeline = [
            K.RandomResizedCrop(size=(global_size, global_size), scale=(0.05, max_scale_global)),
            K.RandomGaussianBlur(kernel_size=(ks, ks), sigma=(0.1, 2), p=0.5),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            NormalizeMeanStd(mean=mean, std=std),
        ]
        # Build local augmentation pipeline.
        local_pipeline = [
            K.RandomResizedCrop(size=(local_size, local_size), scale=(0.05, max_scale_local)),
            K.RandomGaussianBlur(kernel_size=(local_ks, local_ks), sigma=(0.1, 2), p=0.5),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            NormalizeMeanStd(mean=mean, std=std),
        ]
        self.augmentation1 = K.AugmentationSequential(*global_pipeline, data_keys=["input"])
        # When multicrop is enabled, augmentation2 will generate local views.
        self.augmentation2 = K.AugmentationSequential(*local_pipeline, data_keys=["input"])

        # Create backbone.
        if backbone_name in BACKBONE_REGISTRY:
            if "vit" in backbone_name:
                backbone = BACKBONE_REGISTRY[backbone_name](
                    num_classes=0,
                    token_patch_size=token_patch_size,
                )
            else:
                backbone = BACKBONE_REGISTRY[backbone_name](
                    num_classes=0,
                )
        else:
            backbone = timm.create_model(backbone_name, in_chans=in_channels, num_classes=0, pretrained=False)
        
        self.student_backbone = backbone
        self.teacher_backbone = copy.deepcopy(backbone)
        # Create DINO projection heads.
        self.student_head = DINOProjectionHead(backbone.num_features, hidden_dim, bottleneck_dim, output_dim, freeze_last_layer=1)
        self.teacher_head = DINOProjectionHead(backbone.num_features, hidden_dim, bottleneck_dim, output_dim)
        # Freeze teacher parameters.
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

        self.criterion = DINOLoss(output_dim=output_dim, warmup_teacher_temp_epochs=warmup_teacher_temp_epochs)
        self.avg_output_std = 0.0

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the student network."""
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x: Tensor) -> Tensor:
        """Forward pass through the teacher network."""
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx) -> Tensor:
        # Update teacher momentum using cosine schedule.
        momentum = cosine_schedule(self.current_epoch, 10, 0.996, 1)
        update_momentum(self.student_backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.student_head, self.teacher_head, m=momentum)

        # Get input views: if temporal, expect "image1" and "image2"; otherwise, use "image".
        if "image1" in batch and "image2" in batch:
            x1 = batch["image1"].float()
            x2 = batch["image2"].float()
            assert x1.size(1) == self.in_channels
        else:
            x = batch["image"].float()
            assert x.size(1) == self.in_channels
            x1 = x2 = x

        with torch.no_grad():
            x1 = self.augmentation1(x1)
            x2 = self.augmentation1(x2)

        views = [x1, x2]
        global_views = views[:]
        if self.multicrop:
            local_views = []
            for i in range(self.n_views):
                if "image1" in batch and "image2" in batch:
                    # If temporal, randomly select one of the temporal images.
                    x = batch["image1"].float() if np.random.rand() > 0.5 else batch["image2"].float()
                else:
                    x = batch["image"].float()
                local_views.append(self.augmentation2(x))
            views = global_views + local_views

        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]

        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss", loss)

        with torch.no_grad():
            features = self.student_backbone(global_views[0]).flatten(start_dim=1)
            norm_features = F.normalize(features, dim=1)
            output_std = torch.std(norm_features, dim=0).mean().item()
            self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std
            self.log("train_ssl_std", self.avg_output_std)

        return loss

    def on_after_backward(self):
        self.student_head.cancel_last_layer_gradients(current_epoch=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1 / self.warmup_epochs, total_iters=self.warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
            ],
            milestones=[self.warmup_epochs],
        )
        return [optimizer], [lr_scheduler]
