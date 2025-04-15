import kornia.augmentation as K


from typing import Any, Union

from torchgeo.samplers.utils import _to_tuple
from torchgeo.transforms import AugmentationSequential
from torchgeo.datamodules.geo import NonGeoDataModule

from ..datasets.hyperview import HyperviewDataset
from ..transforms.normalize import NormalizeMeanStd
import torch
import numpy as np
from torchgeo.datamodules.utils import MisconfigurationException



class HyperviewDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Hyperview dataset.

    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 128,
        num_workers: int = 0,
        stats_path: str = "data/hyperview_statistics",
        **kwargs: Any,
    ) -> None:
        """Initialize a new Hyperview instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            num_workers: Number of workers for parallel data loading.
            stats_path: Path to the directory containing normalization statistics (mu.npy and sigma.npy).
            **kwargs: Additional keyword arguments passed to
        """
        super().__init__(HyperviewDataset, batch_size, num_workers, **kwargs)

        

        self.patch_size = _to_tuple(patch_size)
        
        self.aug = None
        
        
        try:
            mean = torch.tensor(np.load(f"{stats_path}/mu.npy"))
            std = torch.tensor(np.load(f"{stats_path}/sigma.npy"))
        except FileNotFoundError:
            raise MisconfigurationException("Missing statistics! Ensure mu.npy and sigma.npy are available.")



        self.train_aug = AugmentationSequential(
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.4, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"],
        )
        self.val_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            #K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"],
        )
        self.test_aug = AugmentationSequential(
            K.Resize(_to_tuple(self.patch_size)),
            #K.CenterCrop(self.patch_size),
            NormalizeMeanStd(mean=mean, std=std),
            data_keys=["image"],
        )

    
    
        
