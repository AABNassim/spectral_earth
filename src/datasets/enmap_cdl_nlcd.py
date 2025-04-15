# Based on: https://github.com/microsoft/torchgeo/blob/main/torchgeo/datasets/ssl4eo_benchmark.py

import os
from typing import Callable, Optional

from torchgeo.datasets.cdl import CDL
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datasets.nlcd import NLCD

import torch
from torch import Tensor

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import random




class EnMAPCDLNLCDDataset(NonGeoDataset):
    """EnMAP-CDL/NLCD dataset for crop type/land cover classification.
    
    This dataset handles hyperspectral EnMAP imagery paired with either 
    Cropland Data Layer (CDL) or National Land Cover Database (NLCD) labels.
    It supports train/val/test splits and class mapping for land cover analysis.
    """

    valid_products = ["cdl", "nlcd"]
    valid_splits = ["train", "val", "test"]

    image_root = "{}"
    mask_root = "{}"
    split_path = "data/splits/{}_{}/{}.txt"


    rgb_indices = {
        "enmap": [43, 28, 10],
    }

    split_percentages = [0.75, 0.1, 0.15]

    cmaps = {"nlcd": NLCD.cmap, "cdl": CDL.cmap}

    def __init__(
        self,
        root: str = "data",
        sensor: str = "enmap",
        product: str = "cdl",
        split: str = "train",
        classes: Optional[list[int]] = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        num_bands: int = 202,
        raw_mask: bool = False,
        subset_percent: Optional[float] = None,

    ) -> None:
        """Initialize the EnMAP-CDL/NLCD dataset.
        
        Args:
            root: Root directory containing the dataset
            sensor: Sensor type (currently only 'enmap' supported)
            product: Label type, either 'cdl' or 'nlcd'
            split: Dataset split ('train', 'val', or 'test')
            classes: List of class IDs to include (default: all classes)
            transforms: Optional transformation function
            num_bands: Number of spectral bands (default: 202 for EnMAP)
            raw_mask: If True, return raw mask without class mapping
            subset_percent: Optional fraction of data to use (for debugging)
        """
        self.sensor = sensor
        assert (
            product in self.valid_products
        ), f"Only supports one of {self.valid_products}, but found {product}."
        self.product = product
        assert (
            split in self.valid_splits
        ), f"Only supports one of {self.valid_splits}, but found {split}."
        self.split = split

        self.cmap = self.cmaps[product]
        if classes is None:
            classes = list(self.cmap.keys())

        assert 0 in classes, "Classes must include the background class: 0"

        self.root = root
        self.classes = classes
        self.transforms = transforms
        self.ordinal_map = torch.zeros(max(self.cmap.keys()) + 1, dtype=torch.long) + len(self.classes) - 1
        self.ordinal_cmap = torch.zeros((len(self.classes), 4), dtype=torch.uint8)
        self.img_dir_name = self.image_root.format(self.sensor)
        self.mask_dir_name = self.mask_root.format(self.product)
        self.num_bands = num_bands
        self.raw_mask = raw_mask
        self.subset_percent = subset_percent

        # Check if split file exists, if not create it
        self.split_file = self.split_path.format(self.sensor, self.product, self.split)
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file {self.split_file} not found.")

        # First remove class 0 from the list of classes
        self.classes.remove(0)
        # Then add it back to the end of the list. This ensures that the background class is always the last class, which can be ignored during training.
        self.classes.append(0)

        # Map chosen classes to ordinal numbers, all others mapped to background class
        for v, k in enumerate(self.classes):
            self.ordinal_map[k] = v
            self.ordinal_cmap[v] = torch.tensor(self.cmap[k])
            
        if self.subset_percent is not None and self.split in ["train", "val"]:
            # Load the original train and val split files to get the total counts
            with open(self.split_path.format(self.sensor, self.product, "train"), "r") as f:
                train_ids = [line.strip() for line in f.readlines()]
            with open(self.split_path.format(self.sensor, self.product, "val"), "r") as f:
                val_ids = [line.strip() for line in f.readlines()]

            total_tv = len(train_ids) + len(val_ids)
            # Determine the target total number of samples to use from train+val.
            subset_total = int(total_tv * self.subset_percent)

            # Choose partitioning ratios based on the subset size.
            # For smaller subsets (< 50%), use 80% train, 20% val.
            # Otherwise, use 70% train, 30% val.
            if self.subset_percent < 0.5:
                target_train = int(subset_total * 0.8)
            else:
                target_train = int(subset_total * 0.7)
            target_val = subset_total - target_train

            # Based on the current split, determine the new sample count.
            if self.split == "train":
                new_count = min(len(self.sample_collection), target_train)
            elif self.split == "val":
                new_count = min(len(self.sample_collection), target_val)

            # Use a fixed seed for reproducibility.
            rng = random.Random(42)
            self.sample_collection = rng.sample(self.sample_collection, new_count)
            

    def split_train_val_test(self) -> list:
        """Random Split Train/Val/Test. Not used in the current implementation. The function was used to generate the split files."""
        np.random.seed(0)
        sizes = (np.array(self.split_percentages) * len(self.sample_collection)).astype(int)
        cutoffs = np.cumsum(sizes)[:-1]
        sample_indices = np.arange(len(self.sample_collection))
        np.random.shuffle(sample_indices)
        groups = np.split(sample_indices, cutoffs)
        split_indices = {"train": groups[0], "val": groups[1], "test": groups[2]}

        train_val_test_images = {"train": [self.sample_collection[idx] for idx in split_indices["train"]],
                                 "val": [self.sample_collection[idx] for idx in split_indices["val"]],
                                 "test": [self.sample_collection[idx] for idx in split_indices["test"]]}
        
        
        return train_val_test_images
    
    def read_split_file(self) -> list:
        """Read .txt file containing train/val/test split with only image identifiers.
        """
        with open(self.split_file, "r") as f:
            sample_ids = [x.strip() for x in f.readlines()]

        # Construct the full paths for image and mask
        sample_collection = [
            (
                os.path.join(self.root, self.image_root.format(self.sensor), sample_id),
                os.path.join(self.root, self.mask_root.format(self.product), sample_id)
            )
            for sample_id in sample_ids
        ]
        
        return sample_collection

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            image and sample
        """
        img_path, mask_path = self.sample_collection[index]

        sample = {
            "image": self._load_image(img_path),
            "mask": self._load_mask(mask_path),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.sample_collection)
    


    def _load_image(self, path: str) -> Tensor:
        """Load the input image.

        Args:
            path: path to input image

        Returns:
            image
        """
        with rasterio.open(path) as src:
            image = torch.from_numpy(src.read()).float()
        return image

    def _load_mask(self, path: str) -> Tensor:
        """Load the mask.

        Args:
            path: path to mask

        Retuns:
            mask
        """
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read()).long()
        if self.raw_mask:
            return mask
        return self.ordinal_map[mask]

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        ncols = 2

        image = sample["image"][self.rgb_indices[self.sensor]].numpy()

        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())
        mask = sample["mask"].squeeze(0)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0)
            ncols = 3

        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(self.ordinal_cmap[mask], interpolation="none")
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")

        if showing_predictions:
            ax[2].imshow(self.ordinal_cmap[pred], interpolation="none")
            if show_titles:
                ax[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
