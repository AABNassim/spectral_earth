import os
import random
from typing import Callable, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor

from torchgeo.datasets.geo import NonGeoDataset


class SpectralEarthDataset(NonGeoDataset):
    """Spectral Earth Dataset for hyperspectral foundation model pretraining.
    
    This dataset handles the large-scale hyperspectral data collection from EnMAP
    for self-supervised pretraining. It supports both single and multi-view loading
    modes for contrastive learning approaches.
    
    Each patch has the following properties:
    - 128 x 128 pixels
    - Single multispectral GeoTIFF file
    - 202 spectral bands (EnMAP)
    - 30 m spatial resolution
    """

    class _Metadata(TypedDict):
        num_bands: int
        rgb_bands: list[int]

    rgb_indices = {
        "enmap": [43, 28, 10],
    }

    valid_sensors = ["enmap"]

    def __init__(
        self,
        root: str = "data",
        sensor: str = "enmap",
        num_bands: int = 202,
        patch_size: int = 128,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        return_two_views: bool = False,
    ) -> None:
        """
        Initialize a new SpectralEarthDataset instance.

        Args:
            root: Root directory where the dataset can be found.
            sensor: Sensor type ('enmap', 'enmap_vnir', 'enmap_swir').
            num_bands: Number of spectral bands.
            patch_size: Size of the patches.
            transforms: Optional transforms to apply to the samples.
            return_two_views: If True, returns two random views of the same patch.
        """
        assert (
            sensor in self.valid_sensors
        ), f"Only supports one of {self.valid_sensors}, but found '{sensor}'."
        self.sensor = sensor
        self.root = root
        self.patch_size = patch_size
        self.transforms = transforms
        self.num_bands = num_bands
        self.return_two_views = return_two_views
        self.patch_paths = {}

        # Build the patch_paths dictionary
        patch_dirs = os.listdir(os.path.join(self.root, self.sensor))
        for patch_dir in patch_dirs:
            patch_path = os.path.join(self.root, self.sensor, patch_dir)
            if os.path.isdir(patch_path):
                image_files = [
                    os.path.join(patch_path, f)
                    for f in os.listdir(patch_path)
                    if f.endswith(".tif")
                ]
                if image_files:
                    self.patch_paths[patch_dir] = image_files

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Return a data sample from the dataset.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            A dictionary containing the image(s).
        """
        patch_id = list(self.patch_paths.keys())[index]
        image_paths = self.patch_paths[patch_id]

        if self.return_two_views:
            if len(image_paths) >= 2:
                chosen_paths = random.sample(image_paths, 2)
            else:
                # If less than 2 images are available, duplicate the same image
                chosen_paths = [random.choice(image_paths)] * 2

            images = []
            for path in chosen_paths:
                with rasterio.open(path) as f:
                    image = torch.from_numpy(f.read().astype(np.int16))
                images.append(image)

            sample = {"image1": images[0], "image2": images[1]}
        else:
            chosen_path = random.choice(image_paths)
            with rasterio.open(chosen_path) as f:
                image = torch.from_numpy(f.read().astype(np.int16))
            sample = {"image": image}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of patches in the dataset."""
        return len(self.patch_paths)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a sample from the dataset.

        Args:
            sample: A sample returned by `__getitem__`.
            show_titles: Whether to show titles above each panel.
            suptitle: Optional string for the figure's suptitle.

        Returns:
            A matplotlib Figure with the rendered sample.
        """
        if self.return_two_views:
            image = sample["image1"][self.rgb_indices[self.sensor]].numpy()
        else:
            image = sample["image"][self.rgb_indices[self.sensor]].numpy()

        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image)
        ax.axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
