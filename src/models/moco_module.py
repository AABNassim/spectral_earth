from collections.abc import Sequence
from typing import Tuple

import kornia.augmentation as K
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch import Tensor
import numpy as np
from lightning import LightningModule

from ..transforms.normalize import NormalizeMeanStd

from ..backbones.registry import BACKBONE_REGISTRY

# ------------------------------------------------------------------------------
# Baseline augmentation (without color augmentations)
# ------------------------------------------------------------------------------
def moco_augmentations(size: int, mean: Tensor, std: Tensor) -> Tuple[nn.Module, nn.Module]:
    """Create the baseline MoCo augmentation pipeline for hyperspectral imagery.
    
    This function creates two identical augmentation pipelines used for the query and key
    branches in MoCo. The augmentations include spatial transformations (cropping, flipping)
    and Gaussian blur, but explicitly exclude color augmentations to avoid distorting the spectrum.
    
    Args:
        size (int): The size of the output images.
        mean (torch.Tensor): Mean values for normalization, with shape matching input channels.
        std (torch.Tensor): Standard deviation values for normalization, with shape matching input channels.
        
    Returns:
        Tuple[nn.Module, nn.Module]: Two identical augmentation pipelines for the query and key encoders.
    """
    max_scale = 0.2 if size < 32 else 1.0
    ks = (size // 10 // 2 * 2) + 1

    base_pipeline = [
        K.RandomResizedCrop(size=(size, size), scale=(0.08, max_scale)),
        K.RandomGaussianBlur(kernel_size=(ks, ks), sigma=(0.1, 2), p=0.5),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
        NormalizeMeanStd(mean=mean, std=std)
    ]
    aug = K.AugmentationSequential(*base_pipeline, data_keys=["input"])
    return aug, aug

# ------------------------------------------------------------------------------
# MoCo Module (v2 only)
# ------------------------------------------------------------------------------
class MoCoModule(LightningModule):
    """MoCo v2 implementation for self-supervised learning with hyperspectral imagery.
    
    This module implements Momentum Contrast (MoCo) v2 for hyperspectral imagery. 
    
    The implementation is specifically adapted for hyperspectral data by:
    1. Using specialized hyperspectral backbones (SpectralAdapter + ResNet/ViT)
    2. Avoiding color augmentations that would distort meaningful spectral information
    
    """
    def __init__(
        self,
        backbone_name: str = "spec_resnet50",
        in_channels: int = 202,
        layers: int = 2,            # Fixed to 2 for v2.
        hidden_dim: int = 2048,      # Recommended for v2.
        output_dim: int = 128,       # Recommended for v2.
        lr: float = 0.03,
        weight_decay: float = 1e-4,
        momentum: float = 0.9,
        schedule: Sequence[int] = (120, 160),
        temperature: float = 0.07,
        memory_bank_size: int = 65536,
        moco_momentum: float = 0.999,
        gather_distributed: bool = False,
        size: int = 128,
        token_patch_size: int = 8,
        freeze_patch_embed: bool = False,
    ) -> None:
        """Initialize the MoCo module with configuration parameters.
        
        Args:
            backbone_name (str): Name of the backbone architecture. Defaults to "spec_resnet50".
            in_channels (int): Number of input spectral channels. Defaults to 202 (EnMAP).
            layers (int): Number of layers in the projection head. Fixed to 2 for MoCo v2.
            hidden_dim (int): Hidden dimension in the projection head. Defaults to 2048.
            output_dim (int): Output dimension of the projection head. Defaults to 128.
            lr (float): Base learning rate. Defaults to 0.03.
            weight_decay (float): Weight decay factor. Defaults to 1e-4.
            momentum (float): Momentum factor for SGD. Defaults to 0.9.
            schedule (Sequence[int]): Learning rate schedule milestones. Defaults to (120, 160).
            temperature (float): Temperature parameter for NTXent loss. Defaults to 0.07.
            memory_bank_size (int): Size of the memory bank. Defaults to 65536.
            moco_momentum (float): Momentum for updating the key encoder. Defaults to 0.999.
            gather_distributed (bool): Whether to gather features from distributed training. Defaults to False.
            size (int): Input image size. Defaults to 128.
            token_patch_size (int): Size of patches for ViT models. Defaults to 8.
            freeze_patch_embed (bool): Whether to freeze the patch embedding layer. Defaults to False.
        """
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.temperature = temperature
        self.memory_bank_size = memory_bank_size
        self.moco_momentum = moco_momentum
        self.gather_distributed = gather_distributed
        self.size = size
        self.schedule = schedule
        self.freeze_patch_embed = freeze_patch_embed

        # Load normalization statistics for enmap (202 channels)
        if in_channels == 202:
            mean = torch.tensor(np.load('data/statistics/mu.npy'))
            std = torch.tensor(np.load('data/statistics/sigma.npy'))
        else:
            raise ValueError("This module only supports in_channels==202 for enmap data.")

        # Create baseline augmentations (same for both branches)
        self.augmentation1, self.augmentation2 = moco_augmentations(size, mean, std)

        # Create backbone and momentum backbone.
        if backbone_name in BACKBONE_REGISTRY:
            if "vit" in backbone_name:
                self.backbone = BACKBONE_REGISTRY[backbone_name](
                    num_classes=0,
                    token_patch_size=token_patch_size,
                )
                self.backbone_momentum = BACKBONE_REGISTRY[backbone_name](
                    num_classes=0,
                    token_patch_size=token_patch_size,
                )
            else:
                self.backbone = BACKBONE_REGISTRY[backbone_name](num_classes=0)
                self.backbone_momentum = BACKBONE_REGISTRY[backbone_name](num_classes=0)
        else:
            self.backbone = timm.create_model(backbone_name, in_chans=in_channels, num_classes=0, pretrained=False)
            self.backbone_momentum = timm.create_model(backbone_name, in_chans=in_channels, num_classes=0, pretrained=False)

        # Optionally freeze patch embedding for ViT-based models.
        if self.freeze_patch_embed and hasattr(self.backbone, "vit_core"):
            if hasattr(self.backbone.vit_core, "patch_embed") and hasattr(self.backbone.vit_core.patch_embed, "proj"):
                self.backbone.vit_core.patch_embed.proj.weight.requires_grad = False
                self.backbone.vit_core.patch_embed.proj.bias.requires_grad = False

        # Freeze momentum backbone parameters.
        deactivate_requires_grad(self.backbone_momentum)

        # Create projection head and its momentum version.
        input_dim = self.backbone.num_features
        self.projection_head = MoCoProjectionHead(input_dim, hidden_dim, output_dim, layers, batch_norm=True)
        self.projection_head_momentum = MoCoProjectionHead(input_dim, hidden_dim, output_dim, layers, batch_norm=True)
        deactivate_requires_grad(self.projection_head_momentum)

        # Define NTXent loss.
        self.criterion = NTXentLoss(temperature, memory_bank_size, gather_distributed)

        self.avg_output_std = 0.0

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through the main encoder.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, height, width]
                where channels is the number of spectral bands.
                
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - The projection output (q) of shape [batch_size, output_dim]
                - The backbone features of shape [batch_size, num_features]
        """
        features = self.backbone(x)
        q = self.projection_head(features)
        return q, features

    def forward_momentum(self, x: Tensor) -> Tensor:
        """Forward pass through the momentum encoder.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, channels, height, width]
                where channels is the number of spectral bands.
                
        Returns:
            Tensor: The projection output (k) of shape [batch_size, output_dim]
        """
        features = self.backbone_momentum(x)
        q = self.projection_head_momentum(features)
        return q

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Execute a single training step with contrastive learning.
        
        This method:
        1. Applies different augmentations to the same input (or uses provided views)
        2. Computes query representations using the main encoder
        3. Updates the momentum encoder
        4. Computes key representations using the momentum encoder
        5. Calculates the contrastive loss between query and key
        
        Args:
            batch (dict): Input batch containing either:
                - "image": Single view to be augmented twice
                - "image1" and "image2": Two pre-augmented views
            batch_idx (int): Index of the current batch
            
        Returns:
            Tensor: The calculated contrastive loss
        """
        if "image1" in batch and "image2" in batch:
            x1 = batch["image1"].float()
            x2 = batch["image2"].float()
            assert x1.size(1) == 202
        else:
            x = batch["image"].float()
            assert x.size(1) == 202
            x1 = x
            x2 = x

        with torch.no_grad():
            x1 = self.augmentation1(x1)
            x2 = self.augmentation2(x2)

        q, features = self.forward(x1)
        update_momentum(self.backbone, self.backbone_momentum, self.moco_momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, self.moco_momentum)
        with torch.no_grad():
            k = self.forward_momentum(x2)
        loss = self.criterion(q, k)

        # Compute mean normalized standard deviation of the backbone features.
        output = features.detach()
        output = F.normalize(output, dim=1)
        output_std = torch.std(output, dim=0).mean()
        self.avg_output_std = 0.9 * self.avg_output_std + 0.1 * output_std.item()

        self.log("train_ssl_std", self.avg_output_std)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        """Configure optimizers and learning rate schedulers.
        
        Returns:
            Tuple[list[Optimizer], list]: A tuple containing:
                - A list with the SGD optimizer
                - A list with the MultiStepLR scheduler
        """
        optimizer = SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = MultiStepLR(optimizer, milestones=list(self.schedule), gamma=0.1)
        return [optimizer], [scheduler]
