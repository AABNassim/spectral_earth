from collections import defaultdict
import os
import random
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets.geo import NonGeoDataset


class EnMAPCorineDataset(NonGeoDataset):
    """EnMAP Corine Dataset.

    Dataset intended for evaluating SSL techniques with a few thousand images
    and corresponding land cover classification masks.

    Supports 43 or 19 classes based on the Corine Land Cover classification.
    """

    # Class definitions for 19 and 43 classes
    class_sets = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
        ],
        43: [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean",
        ],
    }

    # Mapping from Corine codes to label indices
    corine_to_label_dict = {
        111: 0,
        112: 1,
        121: 2,
        122: 3,
        123: 4,
        124: 5,
        131: 6,
        132: 7,
        133: 8,
        141: 9,
        142: 10,
        211: 11,
        212: 12,
        213: 13,
        221: 14,
        222: 15,
        223: 16,
        231: 17,
        241: 18,
        242: 19,
        243: 20,
        244: 21,
        311: 22,
        312: 23,
        313: 24,
        321: 25,
        322: 26,
        323: 27,
        324: 28,
        331: 29,
        332: 30,
        333: 31,
        334: 32,
        411: 33,
        412: 34,
        421: 35,
        422: 36,
        423: 37,
        511: 38,
        512: 39,
        521: 40,
        522: 41,
        523: 42,
    }

    # Mapping from 43-class labels to 19-class labels
    label_converter_dict = {
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }

    # Default dicts with 43 as default value for unmapped keys
    corine_to_label = defaultdict(lambda: 43, corine_to_label_dict)
    label_converter = defaultdict(lambda: 43, label_converter_dict)

    valid_sensors = ["enmap"]
    valid_products = ["corine"]
    valid_splits = ["train", "val", "test"]

    image_root = "{}"
    mask_root = "{}"
    split_path = "data/splits/{}_{}/{}.txt"

    rgb_indices = {
        "enmap": [43, 28, 10],
        "enmap_vnir": [2, 1, 0],
        "enmap_swir": [2, 1, 0],
        "s2": [3, 2, 1],
    }

    split_percentages = [0.75, 0.1, 0.15]  # Train, Val, Test percentages

    def __init__(
        self,
        root: str = "data",
        sensor: str = "enmap",
        product: str = "corine",
        split: str = "train",
        num_classes: int = 19,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        num_bands: int = 202,
        return_mask: bool = False,
        subset_percent: Optional[float] = None,
    ) -> None:
        """Initialize a new EnMAP Corine Benchmark instance.

        Args:
            root: Root directory where the dataset can be found.
            sensor: One of ['enmap', 'enmap_vnir', 'enmap_swir', 's2'].
            product: Mask target, one of ['corine', 'corine_v2', 'corine_v2_small'].
            split: Dataset split, one of ['train', 'val', 'test'].
            num_classes: Number of classes (19 or 43).
            transforms: A function/transform that takes input sample and its target as
                entry and returns a transformed version.
            return_mask: If True, returns the raw Corine mask.
        """
        # Validate inputs
        assert sensor in self.valid_sensors, f"Invalid sensor: {sensor}."
        assert product in self.valid_products, f"Invalid product: {product}."
        assert split in self.valid_splits, f"Invalid split: {split}."
        assert num_classes in [19, 43], f"num_classes must be 19 or 43, got {num_classes}."

        self.root = root
        self.sensor = sensor
        self.product = product
        self.split = split
        self.num_classes = num_classes
        self.transforms = transforms
        self.num_bands = num_bands
        self.return_mask = return_mask
        self.subset_percent = subset_percent

        self.img_dir_name = self.image_root.format(self.sensor)
        self.mask_dir_name = self.mask_root.format(self.product)

        # Check if split file exists; if not, create it
        self.split_file = self.split_path.format(self.sensor, self.product, self.split)
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise FileNotFoundError(f"Split file not found: {self.split_file}")
        
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
    
    def __len__(self) -> int:
        """Return the number of data points in the dataset."""
        return len(self.sample_collection)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return a data sample corresponding to the given index.

        Args:
            index: Index of the data sample.

        Returns:
            Dictionary containing image and label tensors.
        """
        img_path, mask_path = self.sample_collection[index]

        if self.return_mask:
            sample = {
                "image": self._load_image(img_path),
                "mask": self._load_mask(mask_path),
            }
        else:
            sample = {
                "image": self._load_image(img_path),
                "label": self._load_label(mask_path),
            }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_image(self, path: str) -> Tensor:
        """Load the input image.

        Args:
            path: Path to the input image.

        Returns:
            Image tensor.
        """
        with rasterio.open(path) as src:
            image = torch.from_numpy(src.read())
        return image

    def _load_mask(self, path: str) -> Tensor:
        """Load the raw Corine mask.

        Args:
            path: Path to the raw Corine mask.

        Returns:
            Raw Corine mask tensor.
        """
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read()).long()
        return mask

    def _load_label(self, path: str) -> Tensor:
        """Load and process the mask.

        Args:
            path: Path to the mask.

        Returns:
            Processed mask tensor.
        """
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read()).long()

        # Map Corine codes to label indices
        mask = mask.apply_(lambda x: self.corine_to_label[x])

        # Convert to 19-class labels if needed
        if self.num_classes == 19:
            mask = mask.apply_(lambda x: self.label_converter[x])

        # Convert mask to one-hot encoding
        indices = torch.unique(mask)
        if indices[-1] == 43:
            indices = indices[:-1]  # Remove the default class

        target = torch.zeros(self.num_classes, dtype=torch.int)
        target[indices] = 1

        return target

    def _onehot_labels_to_names(self, label_mask: np.ndarray) -> list[str]:
        """Get a list of class names given a label mask.

        Args:
            label_mask: Boolean mask corresponding to labels.

        Returns:
            List of class names.
        """
        labels = [
            self.class_sets[self.num_classes][i]
            for i, mask in enumerate(label_mask)
            if mask
        ]
        return labels

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: A sample returned by __getitem__.
            show_titles: Whether to show titles above each panel.
            suptitle: Optional string for the figure's suptitle.

        Returns:
            Matplotlib Figure with the rendered sample.
        """
        image = sample["image"][self.rgb_indices[self.sensor]].numpy()
        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        label_vec = sample["label"].numpy().astype(bool)
        label_names = self._onehot_labels_to_names(label_vec)

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_vec = sample["prediction"].numpy().astype(bool)
            prediction_names = self._onehot_labels_to_names(prediction_vec)

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image.astype(float))
        ax.axis("off")

        if show_titles:
            title = f"Labels: {', '.join(label_names)}"
            if showing_predictions:
                title += f"\nPredictions: {', '.join(prediction_names)}"
            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig

    def plot_with_mask(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot a sample along with its mask.

        Args:
            sample: A sample returned by __getitem__.
            show_titles: Whether to show titles above each panel.
            suptitle: Optional string for the figure's suptitle.

        Returns:
            Matplotlib Figure with the rendered sample and mask.
        """
        ncols = 2
        image = sample["image"][self.rgb_indices[self.sensor]].numpy()
        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample["mask"].squeeze(0).numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0).numpy()
            ncols = 3

        fig, axes = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))

        axes[0].imshow(image.astype(float))
        axes[0].axis("off")
        if show_titles:
            axes[0].set_title("Image")

        axes[1].imshow(mask, interpolation="none")
        axes[1].axis("off")
        if show_titles:
            axes[1].set_title("Mask")

        if showing_predictions:
            axes[2].imshow(pred, interpolation="none")
            axes[2].axis("off")
            if show_titles:
                axes[2].set_title("Prediction")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
