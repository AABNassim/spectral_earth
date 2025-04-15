import os
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor
import rasterio
import matplotlib.pyplot as plt

from torchgeo.datasets.cdl import CDL
from torchgeo.datasets.geo import NonGeoDataset

# -----------------------------------------------------------------------------
# DESISCDLDataset
# -----------------------------------------------------------------------------
class DESISCDLDataset(NonGeoDataset):
    """DESIS CDL dataset.

    This dataset is refactored in the style of your EnMAPCDLNLCDDataset. It
    assumes that a split file exists (e.g. "data/splits/desis_cdl/train.txt")
    that contains one image identifier per line. Full paths for images and masks
    are constructed from common root directories.
    """

    # Format strings (adapt these to your file structure)
    image_root: str = "{}"  # e.g. sensor name ("desis") is inserted here
    mask_root: str = "{}"     # product; in our case always "cdl"
    split_path: str = "data/splits/{}_{}/{}.txt"  # (sensor, product, split)

    rgb_indices = {"desis": [43, 28, 10]}
    split_percentages = [0.75, 0.1, 0.15]  # for reference only

    cmaps = {"cdl": CDL.cmap}

    def __init__(
        self,
        root: str = "data",
        sensor: str = "desis",
        split: str = "train",
        classes: Optional[List[int]] = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        num_bands: int = 202,
        raw_mask: bool = False,
    ) -> None:
        """
        Args:
            root: Root directory where the dataset is stored.
            sensor: Sensor type (should be "desis").
            split: One of "train", "val", or "test".
            classes: List of classes to include (must include 0 for background).
            transforms: Optional callable applied to each sample.
            num_bands: Number of image bands.
            raw_mask: If True, do not remap the mask (e.g. for debugging).
        """
        self.sensor = sensor
        self.product = "cdl"  # only CDL is supported
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split must be one of 'train', 'val', 'test'. Got {split}.")
        self.split = split

        self.cmap = self.cmaps[self.product]
        if classes is None:
            classes = list(self.cmap.keys())
        if 0 not in classes:
            raise ValueError("Classes must include the background class: 0")
        self.root = root
        self.classes = classes
        self.transforms = transforms
        self.num_bands = num_bands
        self.raw_mask = raw_mask

        # Construct full paths for the image and mask directories
        self.img_dir = os.path.join(self.root, self.image_root.format(self.sensor))
        self.mask_dir = os.path.join(self.root, self.mask_root.format(self.product))
        self.split_file = self.split_path.format(self.sensor, self.product, self.split)
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file {self.split_file} not found.")

        # Ensure that background (class 0) is moved to the end of the class list.
        self.classes.remove(0)
        self.classes.append(0)

        # Create mapping from original class to ordinal index.
        # Any class not in self.classes gets mapped to background (last index).
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=torch.long) + (len(self.classes) - 1)
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])

    def read_split_file(self) -> List[Tuple[str, str]]:
        """Reads a split file containing image identifiers (one per line) and
        returns a list of (image_path, mask_path) tuples.
        """
        with open(self.split_file, "r") as f:
            sample_ids = [line.strip() for line in f.readlines()]
        sample_collection = [
            (
                os.path.join(self.img_dir, sample_id),
                os.path.join(self.mask_dir, sample_id)
            )
            for sample_id in sample_ids
        ]
        return sample_collection

    def _load_image(self, path: str) -> Tensor:
        with rasterio.open(path) as src:
            image = torch.from_numpy(src.read()).float()
        return image

    def _load_mask(self, path: str) -> Tensor:
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read()).long()
        if self.raw_mask:
            return mask
        return self.ordinal_map[mask]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        img_path, mask_path = self.sample_collection[index]
        sample = {
            "image": self._load_image(img_path),
            "mask": self._load_mask(mask_path),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
            
        return sample

    def __len__(self) -> int:
        return len(self.sample_collection)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plots a sample.
        
        Note: The image is normalized to [0, 1] for display.
        """
        ncols = 2
        image = sample["image"][self.rgb_indices[self.sensor]].numpy()
        image = image.transpose(1, 2, 0)
        # Normalize the image for display.
        image = (image - image.min()) / (image.max() - image.min())
        mask = sample["mask"].squeeze(0)
        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(self.ordinal_cmap[mask], interpolation="none")
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig