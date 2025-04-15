from typing import Any, Union

import kornia.augmentation as K
from kornia.constants import DataKey, Resample

from torchgeo.datamodules.geo import NonGeoDataModule
from torch import Tensor
import torch
import numpy as np
from torchgeo.transforms import AugmentationSequential

from torchgeo.datamodules.utils import MisconfigurationException
from torchgeo.samplers.utils import _to_tuple

from ..datasets.desis_cdl import DESISCDLDataset
from ..transforms.normalize import NormalizeMeanStd  


class DESISCDLDataModule(NonGeoDataModule):
    """LightningDataModule for DESIS-CDL hyperspectral dataset. """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        stats_path: str = "data/desis_statistics",
        **kwargs: Any,
    ) -> None:
        """Initialize DESIS-CDL datamodule.
        
        Args:
            batch_size: Mini-batch size for training/validation
            patch_size: Size of image patches to extract
            num_workers: Number of parallel data loading workers
            stats_path: Path to normalization statistics for DESIS
            **kwargs: Additional arguments passed to DESISCDLDataset
        """
        super().__init__(DESISCDLDataset, batch_size, num_workers, **kwargs)
        self.patch_size = _to_tuple(patch_size)
        try:
            mean = torch.tensor(np.load(f"{stats_path}/mu.npy"))
            std = torch.tensor(np.load(f"{stats_path}/sigma.npy"))
        except FileNotFoundError:
            raise MisconfigurationException("Missing statistics! Ensure mu.npy and sigma.npy are available.")

        self.train_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.4, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image", "mask"],
            extra_args={DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}},
        )
        self.val_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image", "mask"],
        )
        self.test_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image", "mask"],
        )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply augmentations after transferring the batch to device."""
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug
            elif self.trainer.testing or self.trainer.predicting:
                aug = self.test_aug
            else:
                raise NotImplementedError("Trainer mode not found")
            batch["image"] = batch["image"].float()
            batch = aug(batch)
            batch["image"] = batch["image"].to(batch["mask"].device)
        return batch
