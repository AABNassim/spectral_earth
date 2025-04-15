import os
from typing import Any, Optional, Dict
import timm
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex

from ..backbones.registry import BACKBONE_REGISTRY

# Temporarily import wandb and matplotlib to log images
import wandb
import matplotlib.pyplot as plt
import warnings
from torchgeo.datasets.utils import unbind_samples

from typing import List


class ConvHead(nn.Module):
    def __init__(self, embedding_size: int = 384, num_classes: int = 5, patch_size: int = 4):
        super(ConvHead, self).__init__()

        # Ensure patch_size is a positive power of 2
        if not (patch_size > 0 and ((patch_size & (patch_size - 1)) == 0)):
            raise ValueError("patch_size must be a positive power of 2.")

        num_upsampling_steps = int(math.log2(patch_size))

        # Determine the initial number of filters (maximum 128 or embedding_size)
        initial_filters = 128

        # Generate the sequence of filters: 128, 64, 32, ..., down to num_classes
        filters = [initial_filters // (2 ** i) for i in range(num_upsampling_steps - 1)]
        filters.append(num_classes)  # Ensure the last layer outputs num_classes channels

        layers = []
        in_channels = embedding_size

        for i in range(num_upsampling_steps):
            out_channels = filters[i]

            # Upsampling layer
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

            # Convolutional layer
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

            # Apply BatchNorm and ReLU only if not the last layer
            if i < num_upsampling_steps - 1:
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))

            in_channels = out_channels  # Update in_channels for the next iteration

        self.segmentation_conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.segmentation_conv(x)


class ViTSegmentor(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module, patch_size: int = 4):
        super(ViTSegmentor, self).__init__()
        self.encoder = backbone
        self.num_classes = num_classes  # Add a class for background if needed
        self.embedding_size = backbone.num_features
        self.patch_size = patch_size
        self.head = ConvHead(self.embedding_size, self.num_classes, self.patch_size)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size)
        x = self.encoder.get_intermediate_layers(x, norm=True)[0]
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size, self.embedding_size, int(mask_dim[0]), int(mask_dim[1]))
        x = self.head(x)
        return x


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        inter_channels = in_channels // 8
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, out_channels, 1),
        ]
        super().__init__(*layers)


class FCNSegmenter(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.encoder = backbone
        self.classifier = classifier

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape[-2:]
        features = self.encoder(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


def create_fcn_model(backbone: nn.Module, num_classes: int, embedding_size: int) -> FCNSegmenter:
    """Creates an FCN segmentation model with the given backbone and classifier."""
    # Create the FCN classifier head
    classifier = FCNHead(embedding_size, num_classes)

    # Return the full segmentation model
    return FCNSegmenter(backbone, classifier)

class SemanticSegmentationModule(LightningModule):
    """Semantic segmentation module for hyperspectral imagery.
    
    This module handles training, validation, and testing for pixel-level 
    classification using hyperspectral data. It supports different segmentation
    architectures (FCN or ConvHead) with various backbone networks, and includes
    flexible fine-tuning options for transfer learning from pretrained models.
    """

    def __init__(
        self,
        seg_model: str,
        backbone: str,
        num_classes: int,
        in_channels: int = 202,
        img_size: int = 128,
        pretrained_weights: Optional[str] = None,
        token_patch_size: int = 4,
        freeze_backbone: bool = False,
        finetune_adapter: bool = False,
        finetune_first_n_layers: int = 0,
        class_weights: Optional[Tensor] = None,
        ignore_index: Optional[int] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        t_max: int = 10,
        eta_min: float = 1e-5,
    ) -> None:
        """Initialize the Lightning semantic segmentation module.

        Args:
            seg_model: Segmentation architecture ("fcn" or "conv_head")
            backbone: Backbone model name from registry or timm
            num_classes: Number of segmentation classes
            in_channels: Number of input spectral bands (default: 202 for EnMAP)
            img_size: Input image size (default: 128)
            pretrained_weights: Optional path to pretrained weights
            token_patch_size: Size of patches for ViT models
            freeze_backbone: If True, freeze backbone for linear probing
            finetune_adapter: If True, only finetune spectral adapter
            finetune_first_n_layers: Number of backbone layers to unfreeze
            class_weights: Optional tensor of weights for balancing classes
            ignore_index: Optional class index to ignore in loss/metrics
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            t_max: Maximum iterations for cosine scheduler
            eta_min: Minimum learning rate for scheduler
        """
        super().__init__()

        # Validate ignore_index
        if not isinstance(ignore_index, (int, type(None))):
            raise ValueError("ignore_index must be an int or None")

        # Assign hyperparameters as explicit attributes
        self.seg_model = seg_model
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size
        self.pretrained_weights = pretrained_weights
        self.token_patch_size = token_patch_size
        self.freeze_backbone = freeze_backbone
        self.finetune_adapter = finetune_adapter
        self.finetune_first_n_layers = finetune_first_n_layers
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.eta_min = eta_min


        

        # Configure the task (initialize model, loss, etc.)
        self._config_task()

        # Initialize metrics
        self.train_metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                MulticlassJaccardIndex(
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index,
                    average="micro",
                    
                ),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        
        self.max_val_metrics = {}
        for key in self.val_metrics.keys():
            self.max_val_metrics['max_' + key] = 0.0

    def _config_task(self) -> None:
        """Configures the task by initializing the model, loss function, and freezing layers if necessary."""
        # Create the backbone based on model_type
        backbone = self._initialize_backbone()
        

        self._load_pretrained_weights(backbone)

        # Initialize the segmentation model based on model_type
        self._initialize_model(backbone)

        # Set the loss function
        if self.class_weights is not None:
            self.loss = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=-1000 if self.ignore_index is None else self.ignore_index
            )
        else:
            self.loss = nn.CrossEntropyLoss(
                ignore_index=-1000 if self.ignore_index is None else self.ignore_index
            )

        # Freeze backbone if specified
        if self.freeze_backbone:
            self._freeze_encoder()

        # Unfreeze adapter layers if finetuning is enabled
        if self.finetune_adapter:
            self._unfreeze_adapter_layers()
            
        self._unfreeze_first_n_layers()
            
    def _unfreeze_first_n_layers(self) -> None:
        """Unfreezes the first n backbone layers (ignoring the spectral adapter)."""
        n = self.finetune_first_n_layers
        if n <= 0:
            return

        # For a ResNet-based encoder:
        if hasattr(self.model.encoder, "resnet"):
            resnet = self.model.encoder.resnet
            layers = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
            # Unfreeze the first n layers (ensure n doesn't exceed available layers)
            for i, layer in enumerate(layers):
                if i < n:
                    for param in layer.parameters():
                        param.requires_grad = True

        # For a ViT-based encoder:
        elif hasattr(self.model.encoder, "vit_core"):
            vit = self.model.encoder.vit_core
            # Unfreeze the first n transformer blocks (ignoring the spectral adapter)
            for i, block in enumerate(vit.blocks):
                if i < n:
                    for param in block.parameters():
                        param.requires_grad = True

    def _initialize_backbone(self) -> nn.Module:
        """Initializes the backbone based on the model type.

        Returns:
            nn.Module: Initialized backbone module.
        """
        if self.backbone_name in BACKBONE_REGISTRY:
            if self.seg_model == "conv_head":
                backbone = BACKBONE_REGISTRY[self.backbone_name](
                    num_classes=0,
                    token_patch_size=self.token_patch_size,
                )
            elif self.seg_model == "fcn":
                if "in_channels" in BACKBONE_REGISTRY[self.backbone_name].__init__.__code__.co_varnames:
                    backbone = BACKBONE_REGISTRY[self.backbone_name](
                        num_classes=0,
                        replace_stride_with_dilation=[True, True, True, True],
                        in_channels=self.in_channels,
                        return_features=True,
                    )
                else:
                    backbone = BACKBONE_REGISTRY[self.backbone_name](
                        num_classes=0,
                        replace_stride_with_dilation=[True, True, True, True],
                        return_features=True,
                    )
        else:
            # Use timm
            if "vit" in self.backbone_name:
                backbone = timm.create_model(
                    self.backbone_name,
                    pretrained=False,
                    in_chans=self.in_channels,
                    img_size=self.img_size,
                    patch_size=self.token_patch_size,
                    num_classes=0,
                )
            elif "resnet" in self.backbone_name:
                backbone = timm.create_model(
                    self.backbone_name,
                    pretrained=False,
                    in_chans=self.in_channels,
                    num_classes=0,
                    replace_stride_with_dilation=[True, True, True, True],
                )
            else:
                raise ValueError("Backbone not supported.")


        return backbone

    def _load_pretrained_weights(self, backbone: nn.Module) -> None:
        """Loads pretrained weights into the backbone if a path is provided.

        Args:
            backbone (nn.Module): The backbone module to load weights into.

        Raises:
            FileNotFoundError: If the pretrained_weights path does not exist.
        """
        if self.pretrained_weights:
            if not os.path.exists(self.pretrained_weights):
                raise FileNotFoundError(f"Pretrained weights not found at {self.pretrained_weights}")
            state_dict = torch.load(self.pretrained_weights)
            msg = backbone.load_state_dict(state_dict, strict=False)
            print("Encoder weights loaded:", msg)

    def _initialize_model(self, backbone: nn.Module) -> None:
        """Initializes the segmentation model based on the model type.

        Args:
            backbone (nn.Module): The initialized backbone module.
        """
        if self.seg_model == "conv_head":
            self.model = ViTSegmentor(
                num_classes=self.num_classes,
                backbone=backbone,
                patch_size=self.token_patch_size
            )
        elif self.seg_model == "fcn":
            self.model = create_fcn_model(
                backbone=backbone,
                num_classes=self.num_classes,
                embedding_size=backbone.num_features
            )

    def _unfreeze_adapter_layers(self) -> None:
        """Unfreezes the adapter layers."""
        try:
            for param in self.model.encoder.spectral_adapter.parameters():
                param.requires_grad = True
        except AttributeError:
            raise AttributeError("The backbone does not have 'spectral_adapter' attributes.")

    def _freeze_encoder(self) -> None:
        """Freezes the encoder parameters."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        """Unfreezes the encoder parameters."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Model output.
        """
        return self.model(x)

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch (Dict[str, Tensor]): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            Tensor: Training loss.
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # Log training loss
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        self.train_metrics(y_hat_hard, y)

        return loss

    def on_train_epoch_end(self) -> None:
        """Logs epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Compute validation loss and log metrics.

        Args:
            batch (Dict[str, Tensor]): Batch of data.
            batch_idx (int): Batch index.
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # Log validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics(y_hat_hard, y)
        
        # Only log every 5 epochs
        current_epoch = self.trainer.current_epoch

        if (
            batch_idx < 1
            and hasattr(self.trainer, "datamodule")
            and self.logger  
            and current_epoch % 20 == 0    
        ):
            try:
                #print("Trying to logg validation images")
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                # Loop over the batch and log the first 10 images
                figures = []
                for i in range(min(8, batch["image"].shape[0])):
                    sample = unbind_samples(batch)[i]
                    fig = datamodule.plot(sample)
                    figures.append(fig)
                    #print(self.logger)
                    summary_writer = self.logger.experiment
                    #print(summary_writer)
                summary_writer.log({
                        "samples": [wandb.Image(fig) for fig in figures]
                }, step=self.global_step)

                plt.close()
            except ValueError:
                warnings.warn(
                    "Could not log validation images. "
                )
                pass

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """Compute test loss and log metrics.

        Args:
            batch (Dict[str, Tensor]): Batch of data.
            batch_idx (int): Batch index.
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # Log test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics(y_hat_hard, y)

    def on_validation_epoch_end(self) -> None:
        """Logs epoch-level validation metrics."""
        val_metrics = self.val_metrics.compute()
        for key, value in val_metrics.items():
            self.max_val_metrics['max_' + key] = value
        self.log_dict(val_metrics)
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Logs epoch-level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
        for key, value in self.max_val_metrics.items():
            self.log(key, value)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Dict[str, Any]: Dictionary containing optimizer and scheduler configurations.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Optional: Specify a metric to monitor
                "interval": "epoch",
                "frequency": 1,
            },
        }
