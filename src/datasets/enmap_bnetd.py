import os
from typing import Callable, Optional, List
import torch
from torch import Tensor
import rasterio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from torchgeo.datasets.geo import NonGeoDataset

class EnMAPBNETDDataset(NonGeoDataset):
    """EnMAP-BNETD dataset for land cover monitoring with hyperspectral imagery.
    
    This dataset pairs EnMAP hyperspectral imagery with a land cover map from
    the BNETD (Bureau National d'Études Techniques et de Développement) database
    over Ivory Coast. 
    """

    valid_splits = ["train", "val", "test"]

    # Relative directories (with respect to the root)
    image_root = "enmap"         # e.g., directory containing TreeMap images
    mask_root = "bnetd"    # e.g., directory containing TreeMap masks
    split_path = os.path.join("data", "splits", "enmap_bnetd", "{}.txt")
    
    rgb_indices = {"enmap": [43, 28, 10]}

    def __init__(
        self,
        root: str = "data",
        sensor: str = "enmap",
        split: str = "train",
        classes: Optional[List[int]] = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        num_bands: int = 202,
        raw_mask: bool = False,
    ) -> None:
        """Initialize the EnMAP-BNETD dataset.
        
        Args:
            root: Root directory where dataset is stored
            sensor: Sensor name (default: "enmap")
            split: Dataset split ("train", "val", or "test")
            classes: List of forest class codes (must include 0 for background)
            transforms: Optional transforms to apply to samples
            num_bands: Number of spectral bands (default: 202 for EnMAP)
            raw_mask: If True, don't remap mask classes
            
        Raises:
            ValueError: If split is invalid, classes is None, 0 not in classes,
                or split file is not found
        """
        if split not in self.valid_splits:
            raise ValueError(f"Split '{split}' not one of {self.valid_splits}.")
        if classes is None:
            raise ValueError("Please provide a list of classes. All other classes will be mapped to background.")
        if 0 not in classes:
            raise ValueError("The provided classes must include the background code 0.")
        
        self.split = split
        self.root = root
        self.transforms = transforms
        self.raw_mask = raw_mask
        self.sensor = sensor
        self.num_bands = num_bands

        # Store the provided classes.
        self.classes = classes
        # Foreground classes are all classes except 0, in the order provided.
        self.foreground_classes = [c for c in classes if c != 0]
        # The background/ignore index is set as the last index.
        self.ignore_index = len(self.foreground_classes)
        # Build mapping for foreground classes: each foreground class code is mapped to a new ordinal (0-based)
        self.mapping = {code: idx for idx, code in enumerate(self.foreground_classes)}
        
        # Build colormap for visualization: total classes = len(foreground_classes) + 1 (background)
        num_vis_classes = len(self.foreground_classes) + 1
        cmap = cm.get_cmap("tab20", num_vis_classes)
        self.ordinal_cmap = torch.zeros((num_vis_classes, 4), dtype=torch.uint8)
        for i in range(num_vis_classes):
            color = cmap(i)  # (r, g, b, a) in [0, 1]
            self.ordinal_cmap[i] = torch.tensor([int(255 * c) for c in color])
        
        # Read split file containing sample identifiers
        self.split_file = self.split_path.format(self.split)
        if os.path.exists(self.split_file):
            self.sample_collection = self.read_split_file()
        else:
            raise ValueError(f"Split file {self.split_file} not found.")

    def read_split_file(self) -> List:
        """Read the split file (a text file with one sample identifier per line) and return a list of (image_path, mask_path) tuples."""
        with open(self.split_file, "r") as f:
            sample_ids = [x.strip() for x in f.readlines()]
        sample_collection = [
            (
                os.path.join(self.root, self.image_root, sample_id),
                os.path.join(self.root, self.mask_root, sample_id)
            )
            for sample_id in sample_ids
        ]
        return sample_collection

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return a sample given its index."""
        img_path, mask_path = self.sample_collection[index]
        sample = {
            "image": self._load_image(img_path),
            "mask": self._load_mask(mask_path)
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sample_collection)

    def _load_image(self, path: str) -> Tensor:
        """Load the TreeMap image using rasterio."""
        with rasterio.open(path) as src:
            image = torch.from_numpy(src.read()).float()  # shape: (bands, H, W)
        return image

    def _load_mask(self, path: str) -> Tensor:
        """Load the TreeMap mask using rasterio, and remap classes if needed."""
        with rasterio.open(path) as src:
            mask = torch.from_numpy(src.read(1)).long()  # shape: (1, H, W)
        if self.raw_mask:
            return mask
        return self._remap_mask(mask)

    def _remap_mask(self, mask: Tensor) -> Tensor:
        """
        Remap the raw mask to ordinal labels.
        All pixels in the foreground classes (as provided) are remapped to [0, 1, 2, ...],
        while any other pixel (including original background, i.e. 0) is set to self.ignore_index.
        """
        mask = mask.squeeze(0)  # shape: (H, W)
        new_mask = torch.full_like(mask, fill_value=self.ignore_index)
        for code, ordinal in self.mapping.items():
            new_mask[mask == code] = ordinal
        return new_mask.unsqueeze(0)

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """Plot the sample image and mask."""
        ncols = 2
        image = sample["image"][self.rgb_indices[self.sensor]].numpy()
        image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min())

        mask = sample["mask"].squeeze(0).numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze(0).numpy()
            ncols = 3

        fig, ax = plt.subplots(ncols=ncols, figsize=(4 * ncols, 4))
        ax[0].imshow(image)
        ax[0].axis("off")
        ax[1].imshow(self.ordinal_cmap.numpy()[mask], interpolation="none")
        ax[1].axis("off")
        if show_titles:
            ax[0].set_title("Image")
            ax[1].set_title("Mask")
        if showing_predictions:
            ax[2].imshow(self.ordinal_cmap.numpy()[pred], interpolation="none")
            if show_titles:
                ax[2].set_title("Prediction")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig