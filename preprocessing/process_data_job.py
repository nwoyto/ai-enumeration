import subprocess
import sys

# --- Dependency Installation ---
def install(package):
    """Installs a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List and install dependencies if they are not already present
for pkg in ["rasterio", "geopandas", "shapely", "tqdm", "scikit-learn", "pandas", "albumentations"]:
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
from tqdm import tqdm
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- Helper Functions ---
def extract_image_id_full(filename):
    """Extracts the full image ID from a filename."""
    match = re.search(r'(AOI_2_Vegas_img\d+)', filename)
    return match.group(1) if match else Path(filename).stem

def create_mask_from_geojson(image_path, geojson_path, output_mask_path):
    """Creates a binary mask from a GeoJSON file."""
    try:
        with rasterio.open(image_path) as src:
            transform, width, height, image_crs = src.transform, src.width, src.height, src.crs
    except rasterio.errors.RasterioIOError as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        return False

    try:
        gdf = gpd.read_file(geojson_path)
    except Exception as e:
        logger.warning(f"Skipping {geojson_path}: Error reading GeoJSON - {e}")
        return False

    if gdf.crs and gdf.crs != image_crs:
        gdf = gdf.to_crs(image_crs)

    geometries = [geom for geom in gdf.geometry if geom and geom.is_valid and isinstance(geom, (Polygon, MultiPolygon))]
    
    mask = np.zeros((height, width), dtype=np.uint8)
    if geometries:
        shapes_to_rasterize = [(geom, 1) for geom in geometries]
        try:
            temp_mask = rasterize(
                shapes=shapes_to_rasterize,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype=np.uint8
            )
            mask = np.logical_or(mask, temp_mask).astype(np.uint8)
        except Exception as e:
            logger.error(f"Error during rasterization for {geojson_path}: {e}")
            return False

    mask_meta = src.meta.copy()
    mask_meta.update(dtype=rasterio.uint8, count=1, driver='GTiff')
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with rasterio.open(output_mask_path, 'w', **mask_meta) as dst:
            dst.write(mask, 1)
        return True
    except Exception as e:
        logger.error(f"Failed to write mask to {output_mask_path}: {e}")
        return False

def process_spacenet_dataset_cloud(raw_input_dir, processed_output_dir, split_ratio=0.8, use_rgb=False):
    """Manages mask generation and data splitting for SpaceNet data."""
    raw_input_path = Path(raw_input_dir)
    processed_output_path = Path(processed_output_dir)

    extracted_train_data_base = raw_input_path / 'train' / 'AOI_2_Vegas_Train'
    geojson_train_dir = extracted_train_data_base / 'geojson' / 'buildings'

    if use_rgb:
        input_images_path_train = extracted_train_data_base / 'RGB-PanSharpen'
        logger.info("Processing RGB-PanSharpen (3-band) images.")
    else:
        # This will now process the PS-MS images
        input_images_path_train = extracted_train_data_base / 'MUL-PanSharpen'
        logger.info("Processing MUL-PanSharpen (8-band) PS-MS images.")

    all_image_files = sorted(list(input_images_path_train.glob('*.tif')))
    if not all_image_files:
        logger.error(f"No image files found in {input_images_path_train}. Aborting.")
        sys.exit(1)

    image_id_to_paths = {extract_image_id_full(p.name): p for p in all_image_files}
    all_geojson_files = sorted(list(geojson_train_dir.glob('*.geojson')))
    geojson_id_to_paths = {extract_image_id_full(p.name): p for p in all_geojson_files}

    common_image_ids = sorted(list(image_id_to_paths.keys() & geojson_id_to_paths.keys()))
    logger.info(f"Found {len(common_image_ids)} image/GeoJSON pairs.")

    # Create output directories
    output_dirs = {
        'train_masks': processed_output_path / 'train' / 'masks',
        'val_masks': processed_output_path / 'val' / 'masks',
        'train_images': processed_output_path / 'train' / 'images',
        'val_images': processed_output_path / 'val' / 'images'
    }
    for dir_path in output_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    train_ids, val_ids = train_test_split(common_image_ids, test_size=1-split_ratio, random_state=42)
    logger.info(f"Splitting data: {len(train_ids)} for training, {len(val_ids)} for validation.")

    for split_type, ids in [('train', train_ids), ('val', val_ids)]:
        logger.info(f"Processing {split_type} set...")
        for img_id in tqdm(ids, desc=f"Processing {split_type} data"):
            image_path = image_id_to_paths[img_id]
            geojson_path = geojson_id_to_paths[img_id]
            
            output_mask_path = output_dirs[f'{split_type}_masks'] / f"{img_id}_mask.tif"
            output_image_path = output_dirs[f'{split_type}_images'] / f"{img_id}.tif"
            
            # Generate the mask from the source image and geojson
            create_mask_from_geojson(image_path, geojson_path, output_mask_path)
            
            # Copy the source image to the corresponding output directory
            shutil.copy2(image_path, output_image_path)
            
    logger.info("Mask generation and data splitting complete.")


if __name__ == '__main__':
    input_data_path = Path(os.environ.get('SM_CHANNEL_INPUT_DATA', '/opt/ml/processing/input/data/'))
    output_masks_path = Path(os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output/'))

    logger.info(f"Input data directory: {input_data_path}")
    logger.info(f"Output directory for processed data: {output_masks_path}")

    # --- 1. Extract Tarballs ---
    for subdir_name in os.listdir(input_data_path):
        subdir_path = input_data_path / subdir_name
        if subdir_path.is_dir():
            for item in os.listdir(subdir_path):
                if item.endswith('.tar.gz'):
                    tarball_path = subdir_path / item
                    logger.info(f"Extracting {tarball_path} to {subdir_path}...")
                    try:
                        with tarfile.open(tarball_path, 'r:gz') as tar:
                            tar.extractall(path=subdir_path)
                        logger.info(f"Extraction of {item} complete.")
                    except Exception as e:
                        logger.error(f"Error extracting {tarball_path}: {e}")
                        raise

    # --- 2. Upload Extracted Raw Data to S3 ---
    s3_raw_data_dest = "s3://sagemaker-us-east-1-040571275415/spacenet-building-detection/raw_data/AOI_2_Vegas/"
    extracted_train_dir = input_data_path / 'train' / 'AOI_2_Vegas_Train'
    if extracted_train_dir.exists():
        logger.info(f"Uploading raw data from {extracted_train_dir} to {s3_raw_data_dest}")
        try:
            # Using aws s3 sync for efficient transfer
            subprocess.run(
                ['aws', 's3', 'sync', str(extracted_train_dir), s3_raw_data_dest, '--quiet'],
                check=True
            )
            logger.info("✅ Successfully uploaded raw training data to S3.")
        except Exception as e:
            logger.error(f"❌ Failed to upload raw data to S3: {e}")
    else:
        logger.warning(f"Could not find {extracted_train_dir}. Skipping raw data upload.")

    # --- 3. Process Data to Generate Masks ---
    # Set use_rgb to False to process the PS-MS (8-band) images.
    process_spacenet_dataset_cloud(
        raw_input_dir=str(input_data_path),
        processed_output_dir=str(output_masks_path),
        use_rgb=False  # This is the key change for processing PS-MS data
    )

###########################################################################
#Gemini Takeover
###########################################################################


#     # preprocessing/process_data_job.py - FINAL: UPLOADS BOTH MASKS AND EXTRACTED MUL-SPECTRAL IMAGES

# import subprocess
# import sys
# import os
# import tarfile
# from pathlib import Path
# import time
# import re
# import json
# import shutil
# from tqdm import tqdm 

# # Geospatial and scientific libraries
# import rasterio
# from rasterio.features import rasterize
# import geopandas as gpd
# from shapely.geometry import Polygon, MultiPolygon
# import numpy as np
# from sklearn.model_selection import train_test_split

# import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# # --- Dynamic Package Installation (same as before, ensure it's here) ---
# def install(package):
#     """Installs a pip package via subprocess."""
#     try:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#         logger.info(f"Successfully installed {package}")
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Error installing {package}: {e}")
#         sys.exit(1)

# required_packages = ["rasterio", "geopandas", "fiona", "shapely", "tqdm", "scikit-learn", "pandas"]

# for pkg in required_packages:
#     try:
#         __import__(pkg)
#     except ImportError:
#         install(pkg)
# # --- END Dynamic Package Installation ---


# def extract_image_id_full(filename):
#     """Extract full image ID from filename (e.g., AOI_2_Vegas_img1234)."""
#     # This regex is robust to both RGB-PanSharpen_ and MUL-PanSharpen_ prefixes
#     match = re.search(r'(AOI_2_Vegas_img\d+)', filename)
#     return match.group(1) if match else Path(filename).stem

# def create_mask_from_geojson(image_path, geojson_path, output_mask_path):
#     """
#     Creates a binary mask from a GeoJSON file, reprojecting to image pixel coordinates,
#     and saves the mask as a TIFF.
#     """
#     try:
#         with rasterio.open(image_path) as src:
#             transform = src.transform
#             width = src.width
#             height = src.height
#             image_crs = src.crs
#     except rasterio.errors.RasterioIOError as e:
#         logger.error(f"Failed to open image {image_path}: {e}")
#         return False

#     try:
#         gdf = gpd.read_file(geojson_path)
#     except Exception as e:
#         logger.warning(f"Skipping {geojson_path}: Error reading GeoJSON file - {e}")
#         return False

#     if gdf.crs and gdf.crs != image_crs:
#         gdf = gdf.to_crs(image_crs)
#         logger.info(f"Reprojected GeoJSON {geojson_path.name} from {gdf.crs} to {image_crs}")

#     geometries = []
#     for geom in gdf.geometry:
#         if geom and geom.is_valid:
#             if isinstance(geom, (Polygon, MultiPolygon)):
#                 geometries.append(geom)

#     mask = np.zeros((height, width), dtype=np.uint8)
#     if geometries:
#         shapes_to_rasterize = [(geom, 1) for geom in geometries]
#         if shapes_to_rasterize:
#             try:
#                 temp_mask = rasterize(
#                     shapes=shapes_to_rasterize,
#                     out_shape=(height, width),
#                     transform=transform,
#                     fill=0,
#                     all_touched=True,
#                     dtype=np.uint8
#                 )
#                 mask = np.logical_or(mask, temp_mask).astype(np.uint8)
#             except Exception as e:
#                 logger.error(f"Error rasterizing geometries from {geojson_path}: {e}")
#                 return False

#     mask_meta = src.meta.copy()
#     mask_meta.update(dtype=rasterio.uint8, count=1, driver='GTiff')
#     output_mask_path.parent.mkdir(parents=True, exist_ok=True)
#     try:
#         with rasterio.open(output_mask_path, 'w', **mask_meta) as dst:
#             dst.write(mask, 1)
#         return True
#     except Exception as e:
#         logger.error(f"Failed to write mask to {output_mask_path}: {e}")
#         return False

# def extract_tar(tar_path, dest_dir):
#     if not tar_path.exists():
#         logger.error(f"Tarball not found for extraction: {tar_path}")
#         return False
    
#     if 'train' in tar_path.name:
#         expected_folder_name = 'AOI_2_Vegas_Train'
#     elif 'test' in tar_path.name:
#         expected_folder_name = 'AOI_2_Vegas_Test_Public'
#     else:
#         expected_folder_name = tar_path.stem.replace('.tar', '') 

#     extracted_dir_check = dest_dir / expected_folder_name

#     if extracted_dir_check.exists() and list(extracted_dir_check.iterdir()):
#         logger.info(f"Directory '{extracted_dir_check.name}' already exists and is not empty. Skipping extraction of {tar_path.name}.")
#         return True 
#     else:
#         if extracted_dir_check.exists():
#             logger.warning(f"Removing empty/incomplete '{extracted_dir_check.name}' for fresh extraction.")
#             shutil.rmtree(extracted_dir_check)

#     logger.info(f"Extracting {tar_path.name} to {dest_dir}...")
#     try:
#         with tarfile.open(tar_path, 'r:gz') as tar:
#             tar.extractall(path=dest_dir) 
#         logger.info(f"Successfully extracted {tar_path.name}.")
#         return True
#     except Exception as e:
#         logger.error(f"Error extracting {tar_path.name}: {e}")
#         return False


# def process_spacenet_dataset_cloud(raw_input_dir, processed_masks_output_dir, raw_images_output_dir, split_ratio=0.8, rgb_only=False): # Changed default to False
#     """
#     Manages mask generation and raw image copying within a cloud processing environment.
#     """
#     raw_input_path = Path(raw_input_dir)
#     processed_masks_output_path = Path(processed_masks_output_dir)
#     raw_images_output_path = Path(raw_images_output_dir) # New output directory for raw images

#     # Step 1: Locate extracted tarball contents within the processing container
#     # Assuming tarballs extracted to /opt/ml/processing/input/data/train_tarball/
#     extracted_train_data_base = raw_input_path / 'train_tarball' / 'AOI_2_Vegas_Train' 

#     geojson_train_dir = extracted_train_data_base / 'geojson' / 'buildings'

#     if rgb_only:
#         input_images_source_path = extracted_train_data_base / 'RGB-PanSharpen'
#         image_suffix = '.tif'
#         logger.info("Using RGB-PanSharpen images (3 channels) for mask generation and copying.")
#     else:
#         input_images_source_path = extracted_train_data_base / 'MUL-PanSharpen' # Use MUL-PanSharpen for 8-band
#         image_suffix = '.tif'
#         logger.info("Using MUL-PanSharpen images (8 channels) for mask generation and copying.")

#     # --- Step 2: Copy Extracted Raw Images to a dedicated output channel ---
#     logger.info(f"Copying raw images from {input_images_source_path} to {raw_images_output_path}...")
    
#     raw_images_output_path.mkdir(parents=True, exist_ok=True) # Create output dir for raw images

#     all_raw_images = list(input_images_source_path.glob(f'*{image_suffix}'))
#     if not all_raw_images:
#         logger.error(f"No raw images found in {input_images_source_path}. Cannot copy raw images.")
#     else:
#         # Copy to the raw_images_output_path, which will then be uploaded to S3
#         for img_file in tqdm(all_raw_images, desc="Copying raw images"):
#             shutil.copy(img_file, raw_images_output_path / img_file.name)
#         logger.info(f"Successfully copied {len(all_raw_images)} raw images to {raw_images_output_path}.")


#     # --- Step 3: Generate Masks ---
#     logger.info("Starting mask generation for training/validation split...")
    
#     all_image_files_for_masks = sorted(list(input_images_source_path.glob(f'*{image_suffix}')))
#     if not all_image_files_for_masks:
#         logger.error(f"No image files found for mask generation in {input_images_source_path}. Exiting mask generation.")
#         sys.exit(1) # Exit if no images to generate masks from

#     image_id_to_paths = {extract_image_id_full(p.name): p for p in all_image_files_for_masks}
#     all_geojson_files = sorted(list(geojson_train_dir.glob('*.geojson')))
#     geojson_id_to_paths = {extract_image_id_full(p.name): p for p in all_geojson_files}

#     common_image_ids = sorted(list(image_id_to_paths.keys() & geojson_id_to_paths.keys()))
#     logger.info(f"Found {len(common_image_ids)} image/GeoJSON pairs for mask generation.")

#     output_train_masks_dir = processed_masks_output_path / 'train' / 'masks'
#     output_val_masks_dir = processed_masks_output_path / 'val' / 'masks'
#     output_train_masks_dir.mkdir(parents=True, exist_ok=True)
#     output_val_masks_dir.mkdir(parents=True, exist_ok=True)

#     train_ids, val_ids = train_test_split(common_image_ids, test_size=1-split_ratio, random_state=42)
#     logger.info(f"Splitting data: {len(train_ids)} for training, {len(val_ids)} for validation.")

#     for split_type, ids in [('train', train_ids), ('val', val_ids)]:
#         logger.info(f"Generating {split_type} masks...")
#         current_output_mask_dir = output_train_masks_dir if split_type == 'train' else output_val_masks_dir
#         for img_id in tqdm(ids, desc=f"Generating {split_type} masks"):
#             image_path = image_id_to_paths[img_id]
#             geojson_path = geojson_id_to_paths[img_id]
#             create_mask_from_geojson(image_path, geojson_path, current_output_mask_dir / f"{img_id}.tif")
            
#     logger.info("Mask generation complete.")
#     train_mask_count = len(list(output_train_masks_dir.iterdir()))
#     val_mask_count = len(list(output_val_masks_dir.iterdir()))
#     logger.info(f"Generated {train_mask_count} train masks and {val_mask_count} val masks.")


# if __name__ == '__main__':
#     # SageMaker Processing Job environment variables
#     input_data_path = os.environ.get('SM_CHANNEL_INPUT_DATA', '/opt/ml/processing/input/data/')
#     processed_masks_output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output/processed_masks/') 
#     raw_images_output_path_for_container = '/opt/ml/processing/output/raw_images/' # Dedicated output for raw images inside container

#     logger.info(f"Processing Job Input Data Path: {input_data_path}")
#     logger.info(f"Processing Job Output Masks Path: {processed_masks_output_path}")
#     logger.info(f"Processing Job Raw Images Output Path: {raw_images_output_path_for_container}")

#     # --- Initial Tarball Extraction within the main block (same as before) ---
#     train_tarball_path_in_container = Path(input_data_path) / 'train_tarball' / 'SN2_buildings_train_AOI_2_Vegas.tar.gz'
#     test_tarball_path_in_container = Path(input_data_path) / 'test_tarball' / 'AOI_2_Vegas_Test_public.tar.gz'

#     train_extract_dest_dir = Path(input_data_path) / 'train_tarball'
#     test_extract_dest_dir = Path(input_data_path) / 'test_tarball'

#     logger.info(f"DEBUG: Checking input_data_path ({input_data_path}) contents...")
#     logger.info(f"DEBUG: train_tarball_path_in_container ({train_tarball_path_in_container.exists()})")
#     logger.info(f"DEBUG: test_tarball_path_in_container ({test_tarball_path_in_container.exists()})")

#     if not extract_tar(train_tarball_path_in_container, train_extract_dest_dir):
#         sys.exit(1)
#     if not extract_tar(test_tarball_path_in_container, test_extract_dest_dir):
#         sys.exit(1)
#     logger.info("All tarball extractions complete by main block.")


#     # --- Call the main processing function ---
#     process_spacenet_dataset_cloud(
#         raw_input_dir=input_data_path, 
#         processed_masks_output_dir=processed_masks_output_path,
#         raw_images_output_dir=raw_images_output_path_for_container, # Pass the new output path
#         rgb_only=False # Set to False for MUL-PanSharpen (8-band) images
#     )