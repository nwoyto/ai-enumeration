# notebooks/launch_processing_job.py
import sagemaker
from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput, ScriptProcessor
import boto3
import os
from pathlib import Path
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sagemaker_session = sagemaker.Session()

# Your SageMaker Execution Role ARN (this is the one you provided)
role = 'arn:aws:iam::040571275415:role/service-role/AmazonSageMaker-ExecutionRole-20250320T164162'

logger.info(f"Using SageMaker Execution Role: {role}")

default_bucket = sagemaker_session.default_bucket()
s3_project_prefix = 'spacenet-building-detection' # Overall S3 prefix for your project data

# Define S3 paths for raw tarballs (public SpaceNet dataset)
public_spacenet_s3_base = 's3://spacenet-dataset/spacenet/SN2_buildings/tarballs/'
s3_raw_train_tarball = f'{public_spacenet_s3_base}SN2_buildings_train_AOI_2_Vegas.tar.gz'
s3_raw_test_tarball = f'{public_spacenet_s3_base}AOI_2_Vegas_test_public.tar.gz'

# Define S3 output path for processed masks (in YOUR S3 bucket)
s3_processed_masks_output_path = f's3://{default_bucket}/{s3_project_prefix}/processed_masks'
s3_raw_images_target_path = f's3://{default_bucket}/{s3_project_prefix}/raw_images/RGB-PanSharpen/'

logger.info(f"\nProcessing Job Configuration:")
logger.info(f"  Raw Training Tarball S3: {s3_raw_train_tarball}")
logger.info(f"  Raw Test Tarball S3: {s3_raw_test_tarball}")
logger.info(f"  Processed Masks Output S3: {s3_processed_masks_output_path}")
logger.info(f"  Raw Images Copy Target S3: {s3_raw_images_target_path}")


# Define the local path to your processing script and its requirements.
processing_script_path = str(Path('./spacenet-building-detection/preprocessing/process_data_job.py').resolve())
processing_requirements_path = str(Path('./preprocessing/requirements.txt').resolve())

# Retrieve the exact image URI for the scikit-learn processing container.
account_id = "683313688378" # AWS Account ID for US-WEST-2 public ECR images (change if your region is different)
region_name = sagemaker_session.boto_region_name # Get region from your session

# Use the specific URI for scikit-learn 1.2-1 Python 3 in your region's ECR
processing_image_uri = f"{account_id}.dkr.ecr.{region_name}.amazonaws.com/sagemaker-scikit-learn:1.2-1-py3"
logger.info(f"Using processing image URI: {processing_image_uri}")


# --- CORRECTED `ScriptProcessor` INSTANTIATION & `run()` CALL STARTS HERE ---
processor = ScriptProcessor(
    image_uri=processing_image_uri,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    volume_size_in_gb=150,
    max_runtime_in_seconds=3600 * 2,
    sagemaker_session=sagemaker_session,
    base_job_name="spacenet-data-preprocessor",
    command=["python3"]
)

logger.info("Launching SageMaker Processing Job...")

# Pass the main script to the 'code' argument in run().
processor.run(
    code=processing_script_path,
    inputs=[
        ProcessingInput(
            source=s3_raw_train_tarball,
            destination='/opt/ml/processing/input/data/SN2_buildings_train_AOI_2_Vegas.tar.gz'
        ),
        ProcessingInput(
            source=s3_raw_test_tarball,
            destination='/opt/ml/processing/input/data/AOI_2_Vegas_test_public.tar.gz'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output/',
            destination=s3_processed_masks_output_path,
            output_name="processed_masks"
        )
    ],
    wait=True
)
# --- CORRECTED `ScriptProcessor` INSTANTIATION & `run()` CALL ENDS HERE ---


logger.info(f"Processing job finished. Now copying raw images from public SpaceNet S3 to your S3 bucket: {s3_raw_images_target_path}")

public_s3_images_source = f's3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_2_Vegas_Train/RGB-PanSharpen/'

try:
    subprocess.run(['aws', 's3', 'cp', public_s3_images_source, s3_raw_images_target_path, '--recursive', '--quiet'], check=True)
    logger.info("✓ Raw images copied to your S3 bucket.")
except subprocess.CalledProcessError as e:
    logger.error(f"❌ Error copying raw images to S3: {e}")
    logger.error("Please ensure your IAM role has S3 read permissions for spacenet-dataset and write for your S3 bucket.")
    raise

logger.info("Data preprocessing and S3 upload steps completed via SageMaker Processing Job and S3 copy.")