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

from ..datasets.eo1_cdl import EO1CDLDataset
from ..transforms.normalize import NormalizeMeanStd


class EO1CDLDataModule(NonGeoDataModule):
    """LightningDataModule for EO-1 CDL (file-based) dataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        stats_path: str = "data/eo1_statistics",
        **kwargs: Any,
    ) -> None:
        """
        Args:
            batch_size: Mini-batch size.
            patch_size: Size of the input patch.
            num_workers: Number of workers for data loading.
            stats_path: Path to the directory containing normalization statistics (mu.npy and sigma.npy).
            **kwargs: Additional keyword arguments passed to the dataset.
        """
        super().__init__(EO1CDLDataset, batch_size, num_workers, **kwargs)
        self.patch_size = _to_tuple(patch_size)
        try:
            # Load EO1 statistics
            mean = torch.tensor(np.load(f"{stats_path}/mu.npy"))
            std = torch.tensor(np.load(f"{stats_path}/sigma.npy"))
        except FileNotFoundError:
            raise MisconfigurationException("Missing DESIS statistics!")

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
