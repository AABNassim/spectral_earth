from timm.models.vision_transformer import VisionTransformer    
from functools import partial
from torch import nn
import torch
from typing import Union, Sequence
from typing import Tuple
from .spectral_adapter import SpectralAdapter



class SpecVisionTransformer(nn.Module):
    """Spectral Vision Transformer for hyperspectral image processing.
    
    This model combines a Spectral Adapter for processing the spectral dimension
    with a standard Vision Transformer for spatial feature extraction. The adapter
    processes the spectral bands and outputs features that are then tokenized and
    processed by the ViT architecture.
    
    """
    def __init__(self, token_patch_size=4, patch_size=128, embed_dim=768, reduced_channels=128, 
                 dynamic_img_size=False, depth=12, num_heads=6, mlp_ratio=4, **kwargs):
        """Initialize the Spectral Vision Transformer model.
        
        Args:
            token_patch_size (int): Size of patches for the ViT tokenization. Default: 4.
            patch_size (int): Expected size of input images. Default: 128.
            embed_dim (int): Embedding dimension for transformer. Default: 768.
            reduced_channels (int): Number of channels after spectral adaptation. Default: 128.
            dynamic_img_size (bool): Whether to handle variable image sizes. Default: False.
            depth (int): Number of transformer blocks. Default: 12.
            num_heads (int): Number of attention heads per block. Default: 6.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
            **kwargs: Additional arguments passed to the VisionTransformer.
        """
        super(SpecVisionTransformer, self).__init__()

        self.spectral_adapter = SpectralAdapter()

        # Initialize Vision Transformer
        self.vit_core = VisionTransformer(
            img_size=patch_size, patch_size=token_patch_size, in_chans=reduced_channels, 
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            dynamic_img_size=dynamic_img_size, **kwargs)
        
        self.num_features = self.vit_core.num_features
    
    def forward(self, x):
        """Forward pass through the Spectral ViT.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, depth, height, width]
                where depth is the spectral dimension.
                
        Returns:
            torch.Tensor: Output tensor of shape determined by the ViT configuration.
                For classification tasks, this will be [batch_size, num_classes].
        """
        x = self.spectral_adapter(x)
        
        # Pass through Vision Transformer
        x = self.vit_core(x)
        
        return x
    
    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ):
        """Extract features from intermediate transformer blocks.
        
        This internal method extracts outputs from the last n transformer blocks
        or from specific blocks indexed by n.
        
        Args:
            x (torch.Tensor): Input tensor.
            n (Union[int, Sequence]): If int, the last n blocks are used.
                If sequence, the blocks at those indices are used. Default: 1.
                
        Returns:
            list: List of tensor outputs from selected transformer blocks.
        """
        outputs, num_blocks = [], len(self.vit_core.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.spectral_adapter(x)
        x = self.vit_core.patch_embed(x)
        x = self.vit_core._pos_embed(x)
        x = self.vit_core.patch_drop(x)
        x = self.vit_core.norm_pre(x)
        for i, blk in enumerate(self.vit_core.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs
    
    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface.
        
        Args:
            x (torch.Tensor): Input tensor.
            n (Union[int, Sequence]): If int, the last n blocks are used.
                If sequence, the blocks at those indices are used. Default: 1.
            reshape (bool): If True, reshape output tensors to spatial feature maps. Default: False.
            return_prefix_tokens (bool): If True, return prefix tokens (e.g., cls_token) separately. Default: False.
            norm (bool): If True, apply normalization to output features. Default: False.
                
        Returns:
            Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]: Tuple of tensors or tuple of
                (tensor, prefix_token) pairs if return_prefix_tokens is True.
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.vit_core.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.vit_core.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.vit_core.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.vit_core.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)
    
    def get_classifier(self):
        """Get the classification head of the model.
        
        Returns:
            nn.Module: The classification head of the Vision Transformer.
        """
        return self.vit_core.head

class SpecViTTiny(SpecVisionTransformer):
    """Tiny variant of the Spectral Vision Transformer.
    
    This is a tiny version of the Spectral Vision Transformer with:
    - embed_dim=192
    - depth=12
    - num_heads=3
    """
    def __init__(self, **kwargs):
        """Initialize the Tiny Spectral Vision Transformer.
        
        Args:
            **kwargs: Additional arguments passed to the SpecVisionTransformer.
        """
        super().__init__(
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4,
            **kwargs
        )


class SpecViTSmall(SpecVisionTransformer):
    """Small variant of the Spectral Vision Transformer.
    
    This is a smaller version of the Spectral Vision Transformer with:
    - embed_dim=384
    - depth=12
    - num_heads=6
    
    This variant has approximately 28M parameters and is suitable for tasks
    with limited computational resources or datasets.
    """
    def __init__(self, **kwargs):
        """Initialize the Small Spectral Vision Transformer.
        
        Args:
            **kwargs: Additional arguments passed to the SpecVisionTransformer.
        """
        super().__init__(
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            **kwargs
        )

class SpecViTBase(SpecVisionTransformer):
    """Base variant of the Spectral Vision Transformer.
    
    This is the standard version of the Spectral Vision Transformer with:
    - embed_dim=768
    - depth=12
    - num_heads=12
    
    This variant has approximately 86M parameters and offers a good balance between
    performance and computational requirements.
    """
    def __init__(self, **kwargs):
        """Initialize the Base Spectral Vision Transformer.
        
        Args:
            **kwargs: Additional arguments passed to the SpecVisionTransformer.
        """
        super().__init__(
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            **kwargs
        )

class SpecViTLarge(SpecVisionTransformer):
    """Large variant of the Spectral Vision Transformer.
    
    This is a larger version of the Spectral Vision Transformer with:
    - embed_dim=1024
    - depth=24
    - num_heads=16
    
    """
    def __init__(self, **kwargs):
        """Initialize the Large Spectral Vision Transformer.
        
        Args:
            **kwargs: Additional arguments passed to the SpecVisionTransformer.
        """
        super().__init__(
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            **kwargs
        )

class SpecViTHuge(SpecVisionTransformer):
    """Huge variant of the Spectral Vision Transformer.
    
    This is a very large version of the Spectral Vision Transformer with:
    - embed_dim=1280
    - depth=32
    - num_heads=16

    """
    def __init__(self, **kwargs):
        """Initialize the Huge Spectral Vision Transformer.
        
        Args:
            **kwargs: Additional arguments passed to the SpecVisionTransformer.
        """
        super().__init__(
            embed_dim=1280,
            depth=32,
            num_heads=16,
            mlp_ratio=4,
            **kwargs
        )
        
class SpecViTGiant(SpecVisionTransformer):
    """Giant variant of the Spectral Vision Transformer.
    
    This is the largest version of the Spectral Vision Transformer with:
    - embed_dim=1536
    - depth=40
    - num_heads=24
    

    """
    def __init__(self, **kwargs):
        """Initialize the Giant Spectral Vision Transformer.
        
        Args:
            **kwargs: Additional arguments passed to the SpecVisionTransformer.
        """
        super().__init__(
            embed_dim=1536,
            depth=40,
            num_heads=24,
            mlp_ratio=4,
            **kwargs
        )

class SpectralAdapterProjection(nn.Module):
    """Combines spectral adapter with projection layer for end-to-end spectral-to-token processing.
    
    This module connects a SpectralAdapter with a 2D convolutional projection layer
    that directly produces embedded tokens compatible with transformer architectures.
    It offers an alternative to the two-step process of spectral adaptation followed by
    patch embedding.
    """
    def __init__(self, spectral_adapter, reduced_channels, embed_dim, token_patch_size):
        """Initialize the SpectralAdapterProjection module.
        
        Args:
            spectral_adapter (nn.Module): The spectral adapter for processing input bands.
            reduced_channels (int): Number of channels output by the spectral adapter.
            embed_dim (int): Embedding dimension for the output tokens.
            token_patch_size (int): Size of patches for the projection layer.
        """
        super(SpectralAdapterProjection, self).__init__()
        self.spectral_adapter = spectral_adapter
        self.conv2d = nn.Conv2d(
            in_channels=reduced_channels, 
            out_channels=embed_dim, 
            kernel_size=token_patch_size, 
            stride=token_patch_size
        )

    def forward(self, x):
        """Forward pass through the adapter and projection layers.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, depth, height, width]
                where depth is the spectral dimension.
                
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, embed_dim, height/token_patch_size, 
                width/token_patch_size] containing projected tokens.
        """
        # Process through spectral adapter
        x = self.spectral_adapter(x)

        # Pass through the final 2D convolution layer
        x = self.conv2d(x)
        return x
