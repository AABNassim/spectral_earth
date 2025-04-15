from typing import Any, Optional

from torch import Tensor
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

# Import your SpectralEarthDataset class
from ..datasets.spectral_earth import SpectralEarthDataset


class SpectralEarthDataModule(LightningDataModule):
    """LightningDataModule for the Spectral Earth Dataset."""

    def __init__(
        self,
        root: str = "data",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sensor: str = "enmap",
        num_bands: int = 202,
        patch_size: int = 128,
        return_two_views: bool = False,
        transforms: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the SpectralEarthDataModule. Data augmentation is performed in the SSL model.

        Args:
            root: Root directory where the dataset can be found.
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for data loading.
            pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            sensor: Sensor type ('enmap', 'enmap_vnir', 'enmap_swir').
            num_bands: Number of spectral bands.
            patch_size: Size of the patches.
            return_two_views: If True, returns two random views of the same patch.
            transforms: Optional transforms to apply to the samples.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.sensor = sensor
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.return_two_views = return_two_views
        self.transforms = transforms
        self.kwargs = kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the dataset.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SpectralEarthDataset(
            root=self.root,
            sensor=self.sensor,
            num_bands=self.num_bands,
            patch_size=self.patch_size,
            transforms=self.transforms,
            return_two_views=self.return_two_views,
            **self.kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader.

        Returns:
            A DataLoader for the training dataset.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Optionally implement a validation DataLoader if needed.

        Returns:
            A DataLoader for the validation dataset.
        """
        pass  

    def test_dataloader(self) -> DataLoader:
        """Optionally implement a test DataLoader if needed.

        Returns:
            A DataLoader for the test dataset.
        """
        pass  

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Optionally apply any necessary transformations after batch transfer.

        Args:
            batch: A batch of data.
            dataloader_idx: Index of the dataloader.

        Returns:
            The transformed batch.
        """
        # Convert images to float if they are not already
        for key in batch:
            if "image" in key:
                batch[key] = batch[key].float()
        return batch