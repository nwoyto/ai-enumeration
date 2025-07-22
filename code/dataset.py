import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import rasterio
import numpy as np
from PIL import Image
from pathlib import Path
import os
import re

class SpaceNetBuildingDataset(Dataset):
    def __init__(self, image_root_dir, mask_root_dir, augment=None, num_input_channels=3):
        self.image_root_dir = Path(image_root_dir)
        self.mask_root_dir = Path(mask_root_dir)
        self.augment = augment if augment is not None else self._default_augment()
        self.num_input_channels = num_input_channels

        # Collect image and mask files based on a common ID
        self.image_id_to_path = {self._extract_id(p): p for p in self.image_root_dir.glob('*.tif')}
        self.mask_id_to_path = {self._extract_id(p): p for p in self.mask_root_dir.glob('*.tif')}

        common_ids = sorted(list(self.image_id_to_path.keys() & self.mask_id_to_path.keys()))
        self.data_pairs = [{
            'image_path': self.image_id_to_path[img_id],
            'mask_path': self.mask_id_to_path[img_id],
            'id': img_id
        } for img_id in common_ids]
        
        print(f"Initialized SpaceNetBuildingDataset with {len(self.data_pairs)} valid image-mask pairs.")

    def _extract_id(self, path):
        # Extracts ID like 'AOI_2_Vegas_imgXXXX'
        match = re.search(r'(AOI_2_Vegas_img\d+)', path.name)
        return match.group(1) if match else path.stem

    def __len__(self):
        return len(self.data_pairs)

    def _default_augment(self):
        import albumentations as A
        # Default: geometric + photometric augmentations
        # Resize to 224x224, then augment
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), shear=(-10, 10), p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=10, p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            # Note: HueSaturationValue would only be appropriate for RGB bands
        ], additional_targets={'mask': 'mask'})

    def __getitem__(self, idx):
        import albumentations as A
        data_info = self.data_pairs[idx]
        image_path = data_info['image_path']
        mask_path = data_info['mask_path']

        # Load image (multi-band TIFF)
        with rasterio.open(image_path) as src:
            bands_to_read = list(range(1, min(src.count + 1, self.num_input_channels + 1)))
            image = src.read(bands_to_read).astype(np.float32) / 65535.0 # (C, H, W) float32, 0-1 range

        # Load mask (binary TIFF)
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32) # (H, W) float32, 0 or 1

        # Albumentations expects HWC
        image = np.transpose(image, (1, 2, 0)) # (H, W, C)
        mask = mask.astype(np.uint8) # (H, W), 0 or 1

        augmented = self.augment(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']

        # Convert back to CHW for PyTorch
        image = np.transpose(image, (2, 0, 1)) # (C, H, W)
        mask = np.expand_dims(mask, axis=0) # (1, H, W)

        image_tensor = torch.from_numpy(image).float()
        mask_tensor = torch.from_numpy(mask).float()

        return image_tensor, mask_tensor