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
    def __init__(self, image_root_dir, mask_root_dir, transform=None, num_input_channels=3):
        self.image_root_dir = Path(image_root_dir)
        self.mask_root_dir = Path(mask_root_dir)
        self.transform = transform
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

    def __getitem__(self, idx):
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
        mask = np.expand_dims(mask, axis=0) # Add channel dimension (1, H, W)

        image_tensor = torch.from_numpy(image)
        mask_tensor = torch.from_numpy(mask)

        # Apply transformations
        if self.transform:
            # You can chain more transforms here if needed, or refine the internal ones.
            pass # Currently, transformations are handled internally in dataset logic if `transform` is None.

        # Ensure consistent resizing and normalization
        resize_transform = transforms.Resize((224, 224), antialias=True)
        resize_mask_transform = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST_EXACT)
        
        image_tensor = resize_transform(image_tensor)
        mask_tensor = resize_mask_transform(mask_tensor)

        mean_vals = [0.5] * self.num_input_channels
        std_vals = [0.5] * self.num_input_channels
        normalize_transform = transforms.Normalize(mean=mean_vals, std=std_vals)
        image_tensor = normalize_transform(image_tensor)

        return image_tensor, mask_tensor