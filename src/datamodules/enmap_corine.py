from typing import Any, Union

import numpy as np
import torch
from torch import Tensor
import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datamodules.utils import MisconfigurationException

from ..datasets.enmap_corine import EnMAPCorineDataset
from ..transforms.normalize import NormalizeMeanStd  

class EnMAPCorineDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EnMAP CORINE dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        stats_path: str = "data/statistics",
        **kwargs: Any,
    ) -> None:
        """Initialize the EnMAP Corine Benchmark DataModule.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either an integer or a tuple (height, width).
            num_workers: Number of workers for parallel data loading.
            stats_path: Path to the directory containing normalization statistics (mu.npy and sigma.npy).
            **kwargs: Additional keyword arguments passed to the dataset.
        """
        super().__init__(EnMAPCorineDataset, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)

        try:
            mean = torch.tensor(np.load(f"{stats_path}/mu.npy"))
            std = torch.tensor(np.load(f"{stats_path}/sigma.npy"))
        except FileNotFoundError:
            raise MisconfigurationException("Missing statistics! Ensure mu.npy and sigma.npy are available.")


        # Define data augmentations
        self.train_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.RandomResizedCrop(self.patch_size, scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"],
        )

        self.val_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"],
        )

        self.test_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"],
        )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations after transferring batch to device.

        Args:
            batch: A batch of data that needs to be augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            Augmented batch of data.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug
            elif self.trainer.testing or self.trainer.predicting:
                aug = self.test_aug
            else:
                raise NotImplementedError("Unknown trainer state")

            batch["image"] = batch["image"].float()
            batch = aug(batch)
            batch["image"] = batch["image"].to(batch["label"].device)


        return batch
