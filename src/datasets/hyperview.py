from torchgeo.datasets.geo import NonGeoDataset
from torch import Tensor
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt

def resize_hyperspectral_tensor(img_tensor: Tensor, output_shape: Tuple[int, int, int] = (150, 128, 128)) -> Tensor:
    """
    Resize a hyperspectral image tensor using bicubic interpolation.
    
    Args:
        img_tensor: A tensor of shape (channels, height, width).
        output_shape: The target shape (channels, height, width). Default is (150, 128, 128).
        
    Returns:
        A resized tensor of shape output_shape.
    """
    # Add batch dimension (required by F.interpolate) and resize.
    img_tensor = img_tensor.unsqueeze(0)
    resized_tensor = F.interpolate(img_tensor, size=output_shape[1:], mode='bicubic', align_corners=True)
    return resized_tensor.squeeze(0)


class HyperviewDataset(NonGeoDataset):
    """HyperView dataset for soil property prediction from hyperspectral imagery.
    
    This dataset handles hyperspectral imagery paired with soil property measurements
    (P, K, Mg, pH) for regression tasks. It loads image data from NPZ files and 
    corresponding ground truth values from CSV files.
    """
    valid_splits = ["train", "val", "test"]
    gt_file_path = "data/splits/hyperview/{}.csv"  # expects files like "train.csv", "val.csv", "test.csv"

    def __init__( 
        self,
        root: str = "data",
        data_dir: str = "images",
        split: str = "train",
        num_bands: int = 150,
        patch_size: int = 128,
    ) -> None:
        """Initialize the HyperView dataset.

        Args:
            root: Root directory containing the dataset
            data_dir: Directory within root containing image patches
            split: Dataset split ("train", "val", or "test")
            num_bands: Number of spectral bands to use
            patch_size: Size to resize images to (height and width)
            
        Raises:
            ValueError: If split is invalid or ground truth file is missing
        """
        if split not in self.valid_splits:
            raise ValueError(f"Split must be one of {self.valid_splits}, but got '{split}'.")
        self.split = split
        self.root = root
        self.img_path = os.path.join(self.root, data_dir)
        self.num_bands = num_bands
        self.patch_size = patch_size

        # Load the ground truth CSV.
        gt_csv_path = self.gt_file_path.format(self.split)
        if not os.path.exists(gt_csv_path):
            raise ValueError(f"GT CSV file not found at {gt_csv_path}.")
        self.gt = self._load_gt(gt_csv_path)

        # Build sample_collection based on the sample_index column in the GT CSV.
        self.sample_collection = [
            os.path.join(self.img_path, f"{int(idx)}.npz") for idx in self.gt["sample_index"].values
        ]

        # Read all images and store them in memory.
        self.images = [self._read_image(img_path) for img_path in self.sample_collection]
        self.labels = self.gt

    def _read_image(self, img_path: str) -> np.ndarray:
        """
        Read an image patch from a .npz file.
        """
        with np.load(img_path) as npz_file:
            # Assuming that the file stores an array under the key 'data'
            arr = npz_file['data']
        return arr

    def _load_gt(self, file_path: str) -> pd.DataFrame:
        """
        Load ground truth CSV from a given file path.
        The CSV is expected to contain at least the columns 'sample_index', 'P', 'K', 'Mg', 'pH'.
        """
        return pd.read_csv(file_path)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """
        Return a dictionary containing the image tensor and its corresponding label.

        Args:
            index: Index of the desired sample.

        Returns:
            A dictionary with keys 'image' and 'label'.
        """
        image = torch.tensor(self.images[index]).float()
        image = resize_hyperspectral_tensor(image, output_shape=(self.num_bands, self.patch_size, self.patch_size))
        # Select only the columns 'P', 'K', 'Mg', 'pH' for labels.
        label_values = self.labels.iloc[index][["P", "K", "Mg", "pH"]].values
        # Convert to an array of floats
        label_values = label_values.astype(np.float32)
        return {"image": image, "label": torch.tensor(label_values, dtype=torch.float)}

    def __len__(self) -> int:
        """
        Return the total number of samples.
        """
        return len(self.sample_collection)

    def plot(self, sample: dict[str, Tensor], show_titles: bool = True, suptitle: Optional[str] = None) -> plt.Figure:
        """
        Plot a sample from the dataset.
        
        Args:
            sample: A dictionary containing an image tensor and label.
            show_titles: Whether to display titles on the plot.
            suptitle: An optional suptitle for the plot.
            
        Returns:
            A matplotlib Figure with the rendered sample.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        band_idx = self.num_bands // 2  # Display the middle band
        ax.imshow(sample["image"][band_idx].cpu(), cmap="gray")
        if show_titles:
            ax.set_title(f"Band {band_idx}")
        if suptitle:
            fig.suptitle(suptitle)
        return fig