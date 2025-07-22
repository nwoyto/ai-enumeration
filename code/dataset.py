import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import rasterio
import numpy as np
from PIL import Image
from pathlib import Path
import os
import re
import s3fs
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

# --- S3 Configuration ---
S3_BUCKET = "sagemaker-us-east-1-040571275415"
# Make sure this points to your 8-channel multispectral images
MS_IMAGE_PATH = "spacenet-building-detection/raw_data/AOI_2_Vegas/PS-MS"
MASK_PATH = "spacenet-building-detection/processed_masks/train/masks"
NUM_MS_CHANNELS = 8 # Update this based on your actual multispectral band count

# Initialize s3fs globally for shared access across functions and classes
fs = s3fs.S3FileSystem(anon=False)

# --- Placeholder for calculated dataset statistics ---
# These values MUST be computed from your actual training data using the
# `calculate_dataset_stats_from_s3` function below.
# Once calculated, update these global variables with your specific values
# during the execution of your script.
# Example values (replace with your actual calculated means/stds when you run the calculation step):
DATASET_MEAN = [0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08] # These are just placeholders!
DATASET_STD = [0.05, 0.04, 0.05, 0.04, 0.03, 0.04, 0.03, 0.02]   # These are just placeholders!

# --- Dataset Class for Loading and Augmenting SpaceNet Data from S3 ---
class SpaceNetBuildingDataset(Dataset):
    def __init__(self, s3_bucket, image_s3_path_prefix, mask_s3_path_prefix, augment_mode='train', num_input_channels=3):
        self.s3_bucket = s3_bucket
        self.image_s3_path_prefix = image_s3_path_prefix
        self.mask_s3_path_prefix = mask_s3_path_prefix
        self.num_input_channels = num_input_channels

        # s3fs instance for this dataset (can be shared with global `fs`)
        self.fs = fs 

        # Define transformations based on mode (train/validation/test)
        self.transform = self._get_transforms(augment_mode)

        # Collect image and mask files based on a common ID, using s3fs.glob
        all_image_s3_paths = self.fs.glob(f"s3://{self.s3_bucket}/{self.image_s3_path_prefix}/*.tif")
        all_mask_s3_paths = self.fs.glob(f"s3://{self.s3_bucket}/{self.mask_s3_path_prefix}/*.tif")

        self.image_id_to_path = {self._extract_id(os.path.basename(p)): p for p in all_image_s3_paths}
        self.mask_id_to_path = {self._extract_id(os.path.basename(p)): p for p in all_mask_s3_paths}

        # Filter for common IDs to ensure every image has a corresponding mask and vice versa
        common_ids = sorted(list(self.image_id_to_path.keys() & self.mask_id_to_path.keys()))
        self.data_pairs = [{
            'image_path': self.image_id_to_path[img_id],
            'mask_path': self.mask_id_to_path[img_id],
            'id': img_id
        } for img_id in common_ids]
        
        print(f"Initialized SpaceNetBuildingDataset with {len(self.data_pairs)} valid image-mask pairs in '{augment_mode}' mode.")

    def _extract_id(self, filename):
        """
        Extracts a standardized image ID from various SpaceNet filenames.
        Examples:
        - SN2_buildings_train_AOI_2_Vegas_PS-RGB_img10.tif -> AOI_2_Vegas_img10
        - AOI_2_Vegas_img10.tif -> AOI_2_Vegas_img10
        """
        match = re.search(r'(AOI_2_Vegas_img\d+)', filename)
        if match:
            return match.group(1)
        # Fallback for RGB file naming if not already matched
        match = re.search(r'_img(\d+)\.tif', filename)
        if match:
            return f"AOI_2_Vegas_img{match.group(1)}" # Standardize to 'AOI_2_Vegas_imgXXX'
        
        return Path(filename).stem # Fallback to filename stem

    def __len__(self):
        return len(self.data_pairs)

    def _get_transforms(self, augment_mode):
        """
        Defines the Albumentations compose pipeline for different modes (train/validation/test).
        Includes resize, augmentations, normalization, and conversion to PyTorch tensor.
        """
        # Define common transforms for both training and validation/test
        base_transforms = [
            A.Resize(224, 224), # All images are resized to a fixed size for model input
            # Normalization should be one of the last transforms, applied AFTER other pixel-level ops.
            # max_pixel_value=1.0 is crucial because we already scaled images to 0-1 range.
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD, max_pixel_value=1.0),
            ToTensorV2(), # Converts NumPy array (H, W, C) to PyTorch tensor (C, H, W) and handles data types
        ]

        if augment_mode == 'train':
            return A.Compose([
                # Geometric augmentations (applied to image and mask)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.9, 1.1),             # Random scaling
                    translate_percent=(0.05, 0.05), # Random translation
                    shear=(-10, 10),              # Random shear
                    p=0.5,
                    interpolation=1,              # cv2.INTER_LINEAR for images
                    mask_interpolation=0          # cv2.INTER_NEAREST for masks to keep binary
                ),
                A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=10, p=0.3, # Subtle elastic distortions
                    interpolation=1, mask_interpolation=0
                ),
                # Photometric augmentations (applied to image only, often better for RGB-like bands)
                A.RandomBrightnessContrast(p=0.5), # Adjusts brightness and contrast
                A.GaussNoise(var_limit=(0.001, 0.01), p=0.3), # Add Gaussian noise (tuned for 0-1 range)
                A.GaussianBlur(blur_limit=(3, 7), p=0.2), # Apply Gaussian blur
            ] + base_transforms, additional_targets={'mask': 'mask'})
        
        elif augment_mode == 'val' or augment_mode == 'test':
            return A.Compose(base_transforms, additional_targets={'mask': 'mask'})
        
        else:
            raise ValueError(f"Unknown augment_mode: {augment_mode}")

    def __getitem__(self, idx):
        data_info = self.data_pairs[idx]
        image_path = data_info['image_path']
        mask_path = data_info['mask_path']

        # Load image (multi-band TIFF) from S3
        with self.fs.open(image_path, 'rb') as f_img:
            with rasterio.open(f_img) as src_img:
                # Read only up to num_input_channels
                bands_to_read = list(range(1, min(src_img.count + 1, self.num_input_channels + 1)))
                image = src_img.read(bands_to_read).astype(np.float32) / 65535.0 # (C, H, W) float32, scaled to 0-1

        # Load mask (binary TIFF) from S3
        with self.fs.open(mask_path, 'rb') as f_mask:
            with rasterio.open(f_mask) as src_mask:
                mask = src_mask.read(1).astype(np.float32) # (H, W) float32, 0 or 1

        # Albumentations expects images in HWC format and masks as uint8
        image = np.transpose(image, (1, 2, 0)) # (H, W, C)
        mask = mask.astype(np.uint8) # (H, W), values 0 or 1

        # Apply transformations (includes resizing, augmentations, normalization, ToTensorV2)
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image'] # Already a tensor (C, H, W), normalized
        mask_tensor = augmented['mask']   # Already a tensor (H, W)

        # Add channel dimension to mask (e.g., from (H, W) to (1, H, W)) for PyTorch segmentation models
        mask_tensor = mask_tensor.unsqueeze(0)

        return image_tensor, mask_tensor

# --- Function to Calculate Dataset Mean and Standard Deviation from S3 ---
def calculate_dataset_stats_from_s3(s3_bucket, s3_image_path, num_channels, sample_frac=1.0):
    """
    Calculates per-channel mean and standard deviation for normalization
    by sampling images directly from S3.

    Args:
        s3_bucket (str): The S3 bucket name.
        s3_image_path (str): The S3 path prefix for the images (e.g., "raw_data/AOI_2_Vegas/PS-MS").
        num_channels (int): The number of channels to calculate statistics for.
        sample_frac (float): Fraction of images to sample for calculation (0.0 to 1.0).
                              Set to 1.0 to use all images (recommended for accuracy, but slower).

    Returns:
        tuple: (mean_values, std_values) lists, where each list contains `num_channels` floats.
    """
    s3_image_prefix_full = f"{s3_bucket}/{s3_image_path}"
    all_s3_image_files = fs.glob(f"s3://{s3_image_prefix_full}/*.tif")
    
    if not all_s3_image_files:
        raise ValueError(f"No TIFF images found in s3://{s3_image_prefix_full}")

    # Randomly sample images if sample_frac is less than 1.0
    if sample_frac < 1.0:
        num_samples = int(len(all_s3_image_files) * sample_frac)
        if num_samples == 0 and len(all_s3_image_files) > 0: # Ensure at least one sample if files exist
            num_samples = 1
        all_s3_image_files = random.sample(all_s3_image_files, num_samples)
    
    print(f"Calculating stats for {len(all_s3_image_files)} images across {num_channels} channels from S3. (Sample Fraction: {sample_frac*100:.1f}%)")

    channel_sums = np.zeros(num_channels, dtype=np.float64)
    channel_sq_sums = np.zeros(num_channels, dtype=np.float64)
    total_pixels_per_channel = 0 # Accumulator for total pixels used in calculation

    for i, s3_img_path in enumerate(all_s3_image_files):
        try:
            with fs.open(s3_img_path, 'rb') as f:
                with rasterio.open(f) as src:
                    # Read relevant bands (min(src.count + 1, num_channels + 1) ensures we don't exceed actual bands)
                    bands_to_read = list(range(1, min(src.count + 1, num_channels + 1)))
                    img_data = src.read(bands_to_read).astype(np.float32) / 65535.0 # Scale to 0-1 range

                    # Handle cases where an image might have fewer channels than `num_channels` expected for stats
                    if img_data.shape[0] < num_channels:
                        padded_img_data = np.zeros((num_channels, img_data.shape[1], img_data.shape[2]), dtype=np.float32)
                        padded_img_data[:img_data.shape[0], :, :] = img_data
                        img_data = padded_img_data
                    elif img_data.shape[0] > num_channels: # Truncate if more channels than needed
                        img_data = img_data[:num_channels, :, :]

                    # Accumulate sums and sum of squares for each channel
                    channel_sums += img_data.sum(axis=(1, 2))
                    channel_sq_sums += (img_data**2).sum(axis=(1, 2))
                    total_pixels_per_channel += img_data.shape[1] * img_data.shape[2] # Add H * W for this image

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(all_s3_image_files)} images for stats.")
        except Exception as e:
            print(f"  Skipping image {os.path.basename(s3_img_path)} due to error: {e}")
            continue

    # Calculate final mean and standard deviation for each channel
    mean_values = channel_sums / total_pixels_per_channel
    std_values = np.sqrt(channel_sq_sums / total_pixels_per_channel - mean_values**2)

    return mean_values.tolist(), std_values.tolist()

# --- Main Execution Block ---
# This block demonstrates how to calculate the stats and then use them to initialize datasets.
if __name__ == '__main__':
    print("--- Starting Data Processing Workflow ---")
    
    # Step 1: Calculate dataset statistics (MEAN and STD DEV)
    print("\n--- STEP 1: Calculating Dataset Statistics ---")
    # This step can be time-consuming for large datasets from S3.
    # Set `sample_frac` to 1.0 for the most accurate statistics, ideally on your full training set.
    # For initial testing/development, a smaller sample_frac (e.g., 0.1) can be used.
    
    try:
        calculated_mean, calculated_std = calculate_dataset_stats_from_s3(
            S3_BUCKET, MS_IMAGE_PATH, num_channels=NUM_MS_CHANNELS, sample_frac=0.1
        ) # Using 10% sample for faster demonstration. Change to 1.0 for full accuracy.

        print(f"\nCalculated Dataset Mean (per channel): {calculated_mean}")
        print(f"Calculated Dataset Std Dev (per channel): {calculated_std}")

        # --- IMPORTANT: Update the global DATASET_MEAN and DATASET_STD ---
        DATASET_MEAN = calculated_mean
        DATASET_STD = calculated_std
        print("\nGlobal DATASET_MEAN and DATASET_STD variables successfully updated.")

    except Exception as e:
        print(f"FAILED to calculate dataset statistics: {e}")
        print("Please ensure your S3 paths are correct and your SageMaker role has `s3:GetObject` permissions for the image data.")
        # Depending on your workflow, you might want to `raise e` here or `sys.exit()`
        # if the statistics are critical for proceeding. For a notebook, print is often fine.
        exit() # Exit the script if stats calculation fails

    # Step 2: Initialize your training and validation datasets using the calculated stats
    print("\n--- STEP 2: Initializing Datasets with Calculated Statistics ---")
    try:
        print("\nInitializing SpaceNetBuildingDataset for Training...")
        train_dataset = SpaceNetBuildingDataset(
            s3_bucket=S3_BUCKET,
            image_s3_path_prefix=MS_IMAGE_PATH,
            mask_s3_path_prefix=MASK_PATH,
            augment_mode='train',
            num_input_channels=NUM_MS_CHANNELS
        )

        print("\nInitializing SpaceNetBuildingDataset for Validation...")
        val_dataset = SpaceNetBuildingDataset(
            s3_bucket=S3_BUCKET,
            image_s3_path_prefix=MS_IMAGE_PATH,
            mask_s3_path_prefix=MASK_PATH,
            augment_mode='val', # Use 'val' for validation to disable random augmentations
            num_input_channels=NUM_MS_CHANNELS
        )

        # Step 3: Test retrieving a sample from the training dataset to verify the pipeline
        print("\n--- STEP 3: Testing Dataset Sample Retrieval ---")
        if len(train_dataset) > 0:
            sample_image, sample_mask = train_dataset[0]
            print(f"Successfully retrieved sample from training dataset.")
            print(f"Sample Image Tensor Shape: {sample_image.shape}") # Expected: (C, H, W) e.g., (8, 224, 224)
            print(f"Sample Mask Tensor Shape: {sample_mask.shape}")   # Expected: (1, H, W) e.g., (1, 224, 224)
            print(f"Sample Image Tensor Dtype: {sample_image.dtype}") # Expected: torch.float32
            print(f"Sample Mask Tensor Dtype: {sample_mask.dtype}")   # Expected: torch.float32
            
            # Optional: Verify normalization by checking mean/std of a sample channel
            # For a truly normalized sample, the mean should be close to 0 and std close to 1
            # (though a single sample won't be perfectly 0/1, but shouldn't be 0.x/0.y range)
            print(f"Sample (normalized) Channel 0 Mean: {sample_image[0].mean():.4f}")
            print(f"Sample (normalized) Channel 0 Std Dev: {sample_image[0].std():.4f}")

            # You can also visualize a channel if you like, but it will be normalized (often grayscale)
            # and may not be visually appealing like the original RGB.
            # import matplotlib.pyplot as plt
            # plt.figure(figsize=(5,5))
            # plt.imshow(sample_image[0].numpy(), cmap='gray') # Display first channel as grayscale
            # plt.title("Normalized Sample Image (Channel 0)")
            # plt.axis('off')
            # plt.show()

        else:
            print("Training dataset is empty. Cannot retrieve a sample.")

    except Exception as e:
        print(f"FAILED to initialize or retrieve samples from dataset: {e}")
        print("Please check your S3 paths, IAM permissions, and the content of your buckets.")

    print("\n--- Data Processing Workflow Complete ---")