import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List your dependencies here
for pkg in ["rasterio", "geopandas", "shapely", "tqdm", "scikit-learn", "pandas"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

import os
import tarfile
from pathlib import Path
import time
import re
import json
import shutil
import subprocess
from tqdm import tqdm # Use regular tqdm for standalone script

# For mask generation
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
from sklearn.model_selection import train_test_split

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def extract_image_id_full(filename):
    """Extract full image ID from filename (e.g., AOI_2_Vegas_img1234)."""
    match = re.search(r'(AOI_2_Vegas_img\d+)', filename)
    return match.group(1) if match else Path(filename).stem

def create_mask_from_geojson(image_path, geojson_path, output_mask_path):
    """
    Creates a binary mask from a GeoJSON file, reprojecting to image pixel coordinates,
    and saves the mask as a TIFF.
    """
    try:
        with rasterio.open(image_path) as src:
            transform = src.transform
            width = src.width
            height = src.height
            image_crs = src.crs
    except rasterio.errors.RasterioIOError as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        return False

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as e:
        logger.warning(f"Skipping {geojson_path}: Error reading GeoJSON file - {e}")
        return False

    if gdf.crs and gdf.crs != image_crs:
        gdf = gdf.to_crs(image_crs)
        logger.info(f"Reprojected GeoJSON {geojson_path.name} from {gdf.crs} to {image_crs}")

    geometries = []
    for geom in gdf.geometry:
        if geom and geom.is_valid:
            if isinstance(geom, (Polygon, MultiPolygon)):
                geometries.append(geom)
            else:
                logger.debug(f"Skipping unsupported geometry type '{geom.geom_type}' in {geojson_path.name}")
        # else:
            # logger.debug(f"Skipping invalid/empty geometry in {geojson_path.name}")

    mask = np.zeros((height, width), dtype=np.uint8)
    if geometries:
        shapes_to_rasterize = [(geom, 1) for geom in geometries]
        if shapes_to_rasterize:
            try:
                temp_mask = rasterize(
                    shapes=shapes_to_rasterize,
                    out_shape=(height, width),
                    transform=transform,
                    fill=0, # Background value
                    all_touched=True, # Include all pixels touched by the geometry
                    dtype=np.uint8
                )
                mask = np.logical_or(mask, temp_mask).astype(np.uint8)
            except Exception as e:
                logger.error(f"Error rasterizing geometries from {geojson_path}: {e}")
                return False

    mask_meta = src.meta.copy()
    mask_meta.update(dtype=rasterio.uint8, count=1, driver='GTiff') # Explicitly set driver

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(output_mask_path, 'w', **mask_meta) as dst:
            dst.write(mask, 1)
        return True
    except Exception as e:
        logger.error(f"Failed to write mask to {output_mask_path}: {e}")
        return False

# This function was incorrectly interjected into process_spacenet_dataset_cloud.
# It should be a standalone helper function.
def extract_tar(tar_path, dest_dir):
    if not tar_path.exists():
        logger.error(f"Tarball not found for extraction: {tar_path}")
        return False
    
    # Determine the expected folder name after extraction for idempotency check
    if 'train' in tar_path.name:
        expected_folder_name = 'AOI_2_Vegas_Train'
    elif 'test' in tar_path.name:
        expected_folder_name = 'AOI_2_Vegas_Test_Public'
    else:
        expected_folder_name = tar_path.stem.replace('.tar', '') # Fallback

    extracted_dir_check = dest_dir / expected_folder_name

    if extracted_dir_check.exists() and list(extracted_dir_check.iterdir()):
        logger.info(f"Directory '{extracted_dir_check.name}' already exists and is not empty. Skipping extraction of {tar_path.name}.")
        return True # Assume already extracted
    else:
        if extracted_dir_check.exists():
            logger.warning(f"Removing empty/incomplete '{extracted_dir_check.name}' for fresh extraction.")
            shutil.rmtree(extracted_dir_check)

    logger.info(f"Extracting {tar_path.name} to {dest_dir}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=dest_dir) # Use extractall for efficiency
        logger.info(f"Successfully extracted {tar_path.name}.")
        return True
    except Exception as e:
        logger.error(f"Error extracting {tar_path.name}: {e}")
        return False


def process_spacenet_dataset_cloud(raw_input_dir, processed_output_dir, split_ratio=0.8, rgb_only=True):
    """
    Manages mask generation for SpaceNet data within a cloud processing environment.
    Assumes tarballs have already been extracted in the appropriate subdirectories.
    """
    raw_input_path = Path(raw_input_dir)
    processed_output_path = Path(processed_output_dir)

    # Set up paths to extracted data (assume already extracted by main block)
    # The tarballs extract their content into the directory where they are.
    # So, AOI_2_Vegas_Train is under /opt/ml/processing/input/data/train/AOI_2_Vegas_Train
    extracted_train_data_base = raw_input_path / 'train' / 'AOI_2_Vegas_Train'
    # If you later process the test data, it would be:
    # extracted_test_data_base = raw_input_path / 'test' / 'AOI_2_Vegas_Test_Public'

    geojson_train_dir = extracted_train_data_base / 'geojson' / 'buildings'
    if rgb_only:
        input_images_path_train = extracted_train_data_base / 'RGB-PanSharpen'
        image_suffix = '.tif'
        logger.info("Using RGB-PanSharpen images (3 channels) for mask generation.")
    else:
        input_images_path_train = extracted_train_data_base / 'MUL-PanSharpen'
        image_suffix = '.tif'
        logger.info("Using MUL-PanSharpen images (8 channels) for mask generation.")

    logger.info(f"Collecting images from: {input_images_path_train}")
    all_image_files = sorted(list(input_images_path_train.glob(f'*{image_suffix}')))
    if not all_image_files:
        logger.error(f"No image files found in {input_images_path_train}. Check extraction and paths.")
        sys.exit(1)

    image_id_to_paths = {extract_image_id_full(p.name): p for p in all_image_files}
    all_geojson_files = sorted(list(geojson_train_dir.glob('*.geojson')))
    geojson_id_to_paths = {extract_image_id_full(p.name): p for p in all_geojson_files}

    common_image_ids = sorted(list(image_id_to_paths.keys() & geojson_id_to_paths.keys()))
    logger.info(f"Found {len(common_image_ids)} image/GeoJSON pairs for mask generation.")

    # Define output directories within the processing output path
    output_train_masks_dir = processed_output_path / 'train' / 'masks'
    output_val_masks_dir = processed_output_path / 'val' / 'masks'
    output_train_masks_dir.mkdir(parents=True, exist_ok=True)
    output_val_masks_dir.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = train_test_split(common_image_ids, test_size=1-split_ratio, random_state=42)
    logger.info(f"Splitting data: {len(train_ids)} for training, {len(val_ids)} for validation.")

    for split_type, ids in [('train', train_ids), ('val', val_ids)]:
        logger.info(f"Generating {split_type} masks...")
        current_output_mask_dir = output_train_masks_dir if split_type == 'train' else output_val_masks_dir
        for img_id in tqdm(ids, desc=f"Generating {split_type} masks"):
            image_path = image_id_to_paths[img_id]
            geojson_path = geojson_id_to_paths[img_id]
            output_mask_path = current_output_mask_dir / f"{img_id}.tif"
            create_mask_from_geojson(image_path, geojson_path, output_mask_path)
            
    logger.info("Mask generation complete.")
    train_mask_count = len(list(output_train_masks_dir.iterdir()))
    val_mask_count = len(list(output_val_masks_dir.iterdir()))
    logger.info(f"Generated {train_mask_count} train masks and {val_mask_count} val masks.")


if __name__ == '__main__':
    # SageMaker Processing Job arguments are passed as environment variables
    # SM_CHANNEL_INPUT_DATA: Where S3 input data (tarballs) are mounted
    # SM_OUTPUT_DATA_DIR: Where outputs (masks) should be written to be uploaded to S3
    
    # Default paths for SageMaker Processing container
    input_data_path = os.environ.get('SM_CHANNEL_INPUT_DATA', '/opt/ml/processing/input/data/')
    output_masks_path = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output/') 

    logger.info(f"Processing Job Input Data Path: {input_data_path}")
    logger.info(f"Processing Job Output Masks Path: {output_masks_path}")

    # Debug: Print directory contents to locate tarballs
    logger.info(f"DEBUG: input_data_path ({input_data_path}) contents: {os.listdir(input_data_path)}")
    tarballs_found = False
    for subdir in os.listdir(input_data_path):
        subdir_path = os.path.join(input_data_path, subdir)
        if os.path.isdir(subdir_path):
            logger.info(f"DEBUG: Subdir {subdir_path} contents: {os.listdir(subdir_path)}")
            # Extract any .tar.gz files in this subdirectory
            tarballs = [f for f in os.listdir(subdir_path) if f.endswith('.tar.gz')]
            if tarballs:
                tarballs_found = True
            for tarball in tarballs:
                tarball_path = os.path.join(subdir_path, tarball)
                logger.info(f"Starting tarball extraction: {tarball_path} -> {subdir_path}")
                try:
                    with tarfile.open(tarball_path, 'r:gz') as tar:
                        tar.extractall(path=subdir_path)
                except Exception as e:
                    logger.error(f"Error extracting {tarball_path}: {e}")
                    raise
    if not tarballs_found:
        logger.warning(f"No .tar.gz files found in any subdirectories of {input_data_path}. No extraction performed.")

    # You can pass rgb_only as an argument to the processing job if needed.
    # For now, it's hardcoded to True.
    rgb_only_choice = False # Set to False if you want to preprocess MUL-PanSharpen (8-band) data

    process_spacenet_dataset_cloud(
        raw_input_dir=input_data_path,
        processed_output_dir=output_masks_path,
        rgb_only=rgb_only_choice
    )