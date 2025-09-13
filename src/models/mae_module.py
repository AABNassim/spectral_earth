from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import timm
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

import kornia.augmentation as K

from ..transforms.normalize import NormalizeMeanStd

from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from ..backbones.registry import BACKBONE_REGISTRY


###############################################################################
# PatchEmbedWrapper
###############################################################################
class PatchEmbedWrapper(nn.Module):
    """Wrapper for patch embedding that incorporates spectral adapter.
    
    This wrapper ensures the spectral adapter is applied to hyperspectral inputs
    before patch embedding, maintaining compatibility with standard ViT architectures.
    """
    
    def __init__(self, spectral_adapter: nn.Module, original_patch_embed: nn.Module):
        """Initialize the wrapper with spectral adapter and patch embedding.
        
        Args:
            spectral_adapter: Module to process spectral bands
            original_patch_embed: Original patch embedding module to wrap
        """
        super().__init__()
        self.spectral_adapter = spectral_adapter
        self.original_patch_embed = original_patch_embed
        self.proj = original_patch_embed.proj

    def forward(self, x: Tensor) -> Tensor:
        # Apply the spectral adapter exactly once.
        x = self.spectral_adapter(x)
        # Then compute the patch embeddings.
        return self.original_patch_embed(x)

    @property
    def num_patches(self):
        return self.original_patch_embed.num_patches

    def __getattr__(self, name):
        # Forward attribute lookups to the original patch embedding module.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_patch_embed, name)


###############################################################################
# WrappedVisionTransformer
###############################################################################
class WrappedVisionTransformer(nn.Module):
    """Adapts a spectral ViT model to work with MAE framework.
    
    This wrapper makes custom spectral vision transformers compatible with the 
    lightly MAE implementation by exposing the necessary attributes and integrating
    the spectral adapter into the patch embedding process.
    """
    
    def __init__(self, spec_vit: nn.Module):
        """Initialize the wrapper for a spectral vision transformer.
        
        Args:
            spec_vit: A spectral vision transformer model to wrap
        """
        super().__init__()
        self.spec_vit = spec_vit  
        self.vit_core = spec_vit.vit_core
        
        self.vit_core.patch_embed = PatchEmbedWrapper(self.spec_vit.spectral_adapter,
                                                        self.vit_core.patch_embed)
        
        self.patch_embed = self.vit_core.patch_embed

        # Copy expected attributes from the underlying timm model.
        self.embed_dim = self.vit_core.embed_dim
        self.pos_embed = self.vit_core.pos_embed
        self.num_prefix_tokens = getattr(self.vit_core, "num_prefix_tokens", 1)
        self.norm_pre = getattr(self.vit_core, "norm_pre", None)
        self.blocks = getattr(self.vit_core, "blocks", None)
        self.norm = getattr(self.vit_core, "norm", None)
        self.attn_pool = getattr(self.vit_core, "attn_pool", None)
        self.global_pool = getattr(self.vit_core, "global_pool", None)
        self.cls_token = getattr(self.vit_core, "cls_token", None)
        self.reg_token = getattr(self.vit_core, "reg_token", None)
        self.no_embed_class = getattr(self.vit_core, "no_embed_class", False)
        self.has_class_token = getattr(self.vit_core, "has_class_token", False)
        self.dynamic_img_size = getattr(self.vit_core, "dynamic_img_size", False)
        self.pos_drop = getattr(self.vit_core, "pos_drop", None)

    def forward(self, x: Tensor) -> Tensor:
        return self.vit_core.forward(x)


###############################################################################
# MAEModule
###############################################################################
class MAEModule(LightningModule):
    """Masked Autoencoder (MAE) module for self-supervised learning on hyperspectral data.
    
    This module implements MAE adapted for hyperspectral imagery,
    enabling self-supervised pretraining by reconstructing randomly masked patches.
    The implementation is compatible with custom spectral backbone models.
    """

    def __init__(
        self,
        backbone_name: str = "spec_vit_small",  
        decoder_dim: int = 512,
        patch_size: int = 128,     
        token_patch_size: int = 4, 
        in_channels: int = 202,
        mask_ratio: float = 0.75,
        num_layers: int = 1,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        lr: float = 0.001,
        T_max: int = 100,
        eta_min: float = 0.0,
        norm_pix: bool = True,
    ) -> None:
        """Initialize the MAE module for hyperspectral data.
        
        Args:
            backbone_name: Model name from registry or timm
            decoder_dim: Embedding dimension in decoder
            patch_size: Size of input image patches
            token_patch_size: Size of token patches for ViT
            in_channels: Number of input spectral bands
            mask_ratio: Ratio of tokens to mask
            num_layers: Number of transformer layers in decoder
            num_heads: Number of attention heads in decoder
            mlp_ratio: MLP hidden dim to embedding dim ratio
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            lr: Learning rate
            T_max: Max iterations for cosine scheduler
            eta_min: Min learning rate for scheduler
            norm_pix: Whether to normalize pixels when computing loss
        """
        super().__init__()
        self.mask_ratio = mask_ratio
        self.lr = lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.norm_pix = norm_pix

        # Load normalization statistics for EnMAP (202 channels).
        if in_channels != 202:
            raise ValueError("This module only supports in_channels==202 for EnMAP data.")
        self.mean = torch.tensor(np.load('data/statistics/mu.npy'))
        self.std = torch.tensor(np.load('data/statistics/sigma.npy'))

        # Setup an augmentation pipeline.
        self.augmentations = K.AugmentationSequential(
            K.RandomResizedCrop(size=(patch_size, patch_size), scale=(0.4, 1.0)),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            NormalizeMeanStd(mean=self.mean, std=self.std),
            data_keys=["input"],
        )

        # Create the backbone.
        if backbone_name in BACKBONE_REGISTRY:
            vit = BACKBONE_REGISTRY[backbone_name](token_patch_size=token_patch_size)
        else:
            vit = timm.create_model(backbone_name, in_chans=in_channels, img_size=patch_size, patch_size=token_patch_size, num_classes=0, pretrained=False)
        
        # If the backbone is a custom ViT (i.e. has vit_core), wrap it.
        if hasattr(vit, "vit_core"):
            vit = WrappedVisionTransformer(vit)

        self.image_size = patch_size
        self.patch_size = token_patch_size

        self.encoder = MaskedVisionTransformerTIMM(vit=vit)
        
        self.sequence_length = self.encoder.sequence_length
        
        self.decoder = MAEDecoderTIMM(
            in_chans=in_channels,
            num_patches=vit.patch_embed.num_patches,
            patch_size=token_patch_size,
            decoder_depth=num_layers,
            decoder_num_heads=num_heads,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            mlp_ratio=mlp_ratio,
            proj_drop_rate=dropout,
            attn_drop_rate=attention_dropout,
        )

        self.criterion = nn.MSELoss()
        

    def forward_encoder(self, images: Tensor, idx_keep: Tensor = None) -> Tensor:
        return self.encoder.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred


    def training_step(self, batch, batch_idx) -> Tensor:
        images = batch["image"].float()
        with torch.no_grad():
            images_aug = self.augmentations(images)
        batch_size = images_aug.shape[0]
        # Create a random token mask.
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images_aug, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )
        # Get target patches.
        patches = utils.patchify(images_aug, self.patch_size)
        target = utils.get_at_index(patches, idx_mask - 1)
        if self.norm_pix:
            mean_patch = target.mean(dim=-1, keepdim=True)
            var_patch = target.var(dim=-1, keepdim=True)
            target = (target - mean_patch) / (var_patch + 1e-6).sqrt()
        loss = self.criterion(x_pred, target)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Tuple[list[Optimizer], list]:
        optimizer = AdamW(self.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=self.eta_min)
        return [optimizer], [scheduler]
