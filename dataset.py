"""
This module creates a custom PyTorch Dataset for loading prepared and pre-processed image data and corresponding mask from a CSV DataFrame.
Each row in the DataFrame contains image_id which is the file name for the image and mask files.
"""

import os
from typing import Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class CSVDataset(Dataset):
    """
    Attributes:
        df (pd.DataFrame): DataFrame containing image file names.
        args: Object containing configuration parameters such as dataset directories and image settings.
        img_transform (transforms.Compose): Composed transformation pipeline applied to each input image.
        mask_transform (transforms.Compose): Composed transformation pipeline applied to each mask.
    """

    def __init__(self, args, df: pd.DataFrame) -> None:
        """
        Initialize the dataset with arguments and data.
        Initialize the image and mask transformation pipeline for pre-processing.

        Args:
            args: Configuration ArgumentParser.
            df (pd.DataFrame): DataFrame containing 'image_id' column.
        """
        self.df = df
        self.args = args

        self.images_location = os.path.join(args.dataset_dir, "Images")
        self.masks_location = os.path.join(args.dataset_dir, "Masks")

        random_seed = args.seed

        self.img_transform = transforms.Compose(
            [
                # add augmentation, resize, flip, etc.
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                # add augmentation, resize, flip, etc.
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Retrieve a single image_id from the dataset and use it to open input image and its mask.

        Args:
            i (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Image tensor of shape (C, H, W) after transformations.
                - Mask tensor of shape (H, W) after transformation, typically 0 or 1 for binary classification.
                - Image ID of type string used to save the prediction output as the same name.
        """
        row = self.df.iloc[i]
        image_path = os.path.join(self.images_location, row["image_id"] + ".jpg")
        mask_path = os.path.join(self.masks_location, row["image_id"] + "_segmentation.png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Augmentation
        torch.manual_seed(self.args.seed)

        if torch.rand(1) > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        rotation_angle = (torch.rand(1).item() - 0.5) * 30
        image = transforms.functional.rotate(image, rotation_angle)
        mask = transforms.functional.rotate(mask, rotation_angle)

        image = self.img_transform(image)
        mask = self.mask_transform(mask).squeeze(0).long()

        return image, mask, row["image_id"]
