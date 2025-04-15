from typing import Any, Union

import kornia.augmentation as K
from kornia.constants import DataKey, Resample

from ..datasets.enmap_cdl_nlcd import EnMAPCDLNLCDDataset

from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential
from torchgeo.datamodules.geo import NonGeoDataModule

from torch import Tensor
import torch
import numpy as np
from ..transforms.normalize import NormalizeMeanStd

from torchgeo.datamodules.utils import MisconfigurationException


class EnMAPCDLNLCDDataModule(NonGeoDataModule):
    """LightningDataModule for the EnMAP-CDL/NLCD dataset."""

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        stats_path: str = "data/statistics",
        **kwargs: Any,
    ) -> None:
        """Initialize an EnMAP-CDL/NLCD datamodule.

        Args:
            batch_size: Size of each mini-batch for training, validation, and testing.
            patch_size: Size of each image patch, either a single int or (height, width) tuple.
                All images will be resized to this size. Default: 128.
            num_workers: Number of workers for parallel data loading. Default: 0.
            stats_path: Path to the directory containing normalization statistics (mu.npy and sigma.npy).
                Default: "data/statistics".
            **kwargs: Additional keyword arguments passed to the EnMAPCDLNLCDDataset constructor.
                
        Raises:
            MisconfigurationException: If the normalization statistics files are not found
                at the specified stats_path.
        """
        super().__init__(EnMAPCDLNLCDDataset, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)

        self.aug = None

        try:
            mean = torch.tensor(np.load(f"{stats_path}/mu.npy"))
            std = torch.tensor(np.load(f"{stats_path}/sigma.npy"))
        except FileNotFoundError:
            raise MisconfigurationException("Missing statistics! Ensure mu.npy and sigma.npy are available.")


        # Training augmentations: resize, random crop, flips, and normalization
        self.train_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.2, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )
        
        # Validation augmentations: resize, center crop, and normalization
        self.val_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image", "mask"],
        )
        
        # Test augmentations: same as validation
        self.test_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image", "mask"],
        )
    
    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply augmentations to the batch after it is transferred to the device.

        This method is called automatically by PyTorch Lightning after the batch
        is transferred to the device (GPU/CPU). It applies the appropriate augmentations
        based on the current trainer state (training/validation/testing).
        
        
        Args:
            batch: A dictionary batch of data containing 'image' and 'mask' tensors.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            The augmented batch with normalized images and masks.
            
        Raises:
            NotImplementedError: If the trainer mode cannot be determined.
        """
        if self.trainer:
            if self.trainer.training:
                aug = self.train_aug 
            elif self.trainer.validating or self.trainer.sanity_checking:
                aug = self.val_aug
            elif self.trainer.testing:
                aug = self.test_aug 
            elif self.trainer.predicting:
                aug = self.test_aug
            else:
                print("No trainer mode found")
                raise NotImplementedError 
            batch["image"] = batch["image"].float()
            batch = aug(batch)
            batch["image"] = batch["image"].to(batch["mask"].device)

        return batch

