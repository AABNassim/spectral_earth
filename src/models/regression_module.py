import os
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
from lightning.pytorch import LightningModule
from torchmetrics import MeanSquaredError, MetricCollection
from ..backbones.registry import BACKBONE_REGISTRY

from typing import Any, Dict, Optional
from typing import List

# =============================================================================
# CUSTOM MSE LOSS
# =============================================================================
def custom_mse_loss(
    y_pred: Tensor,
    y_true: Tensor,
    baseline_outputs: Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0])
) -> Tensor:
    """
    Computes the custom MSE loss as the ratio between the model's MSE and the MSE 
    of a naive baseline (which returns the empirical mean, here assumed to be 0).

    Args:
        y_pred: Tensor of shape (batch_size, 4), predictions from the model.
        y_true: Tensor of shape (batch_size, 4), ground truth values.
        baseline_outputs: Tensor of shape (4,), baseline predictions (default zeros).

    Returns:
        A scalar tensor representing the normalized MSE loss.
    """
    # Ensure baseline_outputs is on the same device as y_true.
    baseline_outputs = baseline_outputs.to(y_true.device)
    
    # Compute MSE for the model's predictions (per target).
    mse_model = F.mse_loss(y_pred, y_true, reduction="none").mean(dim=0)
    
    # Compute MSE for the naive baseline (mean predictor).
    baseline_tensor = baseline_outputs.unsqueeze(0).expand_as(y_true)
    mse_baseline = F.mse_loss(baseline_tensor, y_true, reduction="none").mean(dim=0)
    
    # Compute normalized MSE (per target) and average them.
    normalized_mse = mse_model / mse_baseline
    loss = normalized_mse.mean()
    return loss

class RegressionModule(LightningModule):
    """
    LightningModule for regression on hyperspectral data. 
    The module uses a custom MSE loss that computes the ratio between the
    model's MSE and the MSE of a naive baseline (which returns the normalized target mean, 0).
    """
    def __init__(
        self,
        backbone: str,  
        pretrained_weights: str,
        in_channels: int,
        num_outputs: int,
        learning_rate: float,
        weight_decay: float,
        t_max: int,
        mlp_head: bool = False,
        mlp_dims: list = None,  
        token_patch_size: int = None,
        freeze_backbone: bool = False,
        finetune_adapter: bool = False,
        baseline: Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0]), 
        wave_list: Optional[Dict[str, List[float]]] = None,
        sensor: Optional[str] = None,
        img_size: int = 128,
    ) -> None:
        """
        Args:
            model: Name of the backbone model (key in MODEL_REGISTRY).
            weights: Pretrained flag (True/False) or path to weights.
            in_channels: Number of input channels.
            num_outputs: Number of regression targets.
            learning_rate: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
            t_max: Maximum iterations for the cosine annealing scheduler.
            mlp_head: If True, replace the default head with an MLP.
            token_patch_size: (For ViT models) Token patch size.
            freeze_backbone: If True, freeze the backbone parameters.
            baseline: Baseline outputs for computing the custom loss (default zeros).
        """
        super().__init__()
        self.backbone_name = backbone
        self.pretrained_weights = pretrained_weights
        self.in_channels = in_channels
        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.t_max = t_max
        self.mlp_head = mlp_head
        self.token_patch_size = token_patch_size
        self.freeze_backbone = freeze_backbone
        self.finetune_adapter = finetune_adapter
        self.baseline = torch.tensor(baseline, dtype=torch.float32)
        self.mlp_dims = mlp_dims or []
        self.best_val_loss = float("inf") 
        self.wave_list = wave_list
        self.sensor = sensor
        self.img_size = img_size
        
        

        # Configure the backbone model.
        self._configure_backbone()

        # Use the custom MSE loss.
        self.loss_fn = custom_mse_loss

        # Set up metrics (here we use RMSE for monitoring).
        self.train_metrics = MetricCollection({
            "RMSE": MeanSquaredError(squared=False)
        }, prefix="train_")
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def _load_pretrained_weights(self) -> None:
        """Load pretrained weights into the model if a path is provided."""
        if self.pretrained_weights:
            if not os.path.exists(self.pretrained_weights):
                raise FileNotFoundError(f"Pretrained weights not found at {self.pretrained_weights}")
            state_dict = torch.load(self.pretrained_weights, map_location="cpu")
            msg = self.model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded:", msg)
            
    def _configure_backbone(self) -> None:
        """
        Build the model backbone using MODEL_REGISTRY and apply head modifications if needed.
        """

        if self.backbone_name in BACKBONE_REGISTRY:
            model_cls = BACKBONE_REGISTRY[self.backbone_name]
            if "vit" in self.backbone_name:
                self.model = model_cls(
                    num_classes=0,
                    token_patch_size=self.token_patch_size,
                )
            else:
                self.model = model_cls(num_classes=0)
            
        else:
            # Use a timm model if the backbone is not in our registry
            if "vit" in self.backbone_name:
                self.model = timm.create_model(
                    self.backbone_name,
                    num_classes=0,
                    pretrained=False,
                    in_chans=self.in_channels,
                    img_size=self.img_size,
                    patch_size=self.token_patch_size,
                )
            else:
                self.model = timm.create_model(
                    self.backbone_name,
                    num_classes=0,
                    pretrained=False,
                    in_chans=self.in_channels,
                )
                
        # Load pretrained weights if specified
        self._load_pretrained_weights()
        
        in_features = self.model.num_features

        # Build the list of dimensions: input -> hidden layers -> output.
        dims = [in_features] + self.mlp_dims + [self.num_outputs]
        layers = []
        layers.append(nn.BatchNorm1d(in_features))
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))

        mlp = nn.Sequential(*layers)

        if "vit" in self.backbone_name:
            self.model.vit_core.head = mlp
        else:
            self.model.resnet.fc = mlp
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            self._freeze_backbone()
            
        # Unfreeze adapter layers if finetuning is enabled
        if self.finetune_adapter:
            self._unfreeze_adapter_layers()
    
    def _freeze_backbone(self) -> None:
        """Freeze the backbone parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the classifier head if present
        if hasattr(self.model, 'get_classifier'):
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            print("Warning: Model does not have a recognizable classifier head to unfreeze.")

    def _unfreeze_adapter_layers(self) -> None:
        """Unfreeze the adapter layers."""
        if hasattr(self.model, 'spectral_adapter'):
            for param in self.model.spectral_adapter.parameters():
                param.requires_grad = True
        else:
            raise AttributeError("The backbone does not have 'spectral_adapter' attributes.")

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        # Ensure dimensions match.
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(1)
        loss = self.loss_fn(y_hat, y.float(), self.baseline)
        self.log("train_loss", loss, sync_dist=True)
        self.train_metrics(y_hat, y.float())
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(1)
        loss = self.loss_fn(y_hat, y.float(), self.baseline)
        self.log("val_loss", loss, sync_dist=True)
        self.val_metrics(y_hat, y.float())

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        # Retrieve the current validation loss from callback metrics.
        current_val_loss = self.trainer.callback_metrics.get("val_loss")
        if current_val_loss is not None:
            current_val_loss = (
                current_val_loss.item()
                if isinstance(current_val_loss, torch.Tensor)
                else current_val_loss
            )
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
        

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(1)
        loss = self.loss_fn(y_hat, y.float(), self.baseline)
        self.log("test_loss", loss, sync_dist=True)
        self.test_metrics(y_hat, y.float())

    def on_test_epoch_end(self) -> None:
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
        self.log("best_val_loss", self.best_val_loss)

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        return self(batch["image"])

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.t_max, eta_min=0.0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}