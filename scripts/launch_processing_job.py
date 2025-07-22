import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
import boto3
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
region_name = "us-east-1"
boto_session = boto3.Session(region_name=region_name)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

role = 'arn:aws:iam::040571275415:role/service-role/AmazonSageMaker-ExecutionRole-20250320T164162'
image_uri = '040571275415.dkr.ecr.us-east-1.amazonaws.com/spacenet-geoprocessing:latest'

logger.info(f"Using SageMaker Execution Role: {role}")
logger.info(f"Using custom container image: {image_uri}")

default_bucket = sagemaker_session.default_bucket()
s3_project_prefix = 'spacenet-building-detection'

# S3 paths for raw data (public SpaceNet dataset)
public_spacenet_s3_base = 's3://spacenet-dataset/spacenet/SN2_buildings/tarballs/'
s3_raw_train_tarball = f'{public_spacenet_s3_base}SN2_buildings_train_AOI_2_Vegas.tar.gz'
s3_raw_test_tarball = f'{public_spacenet_s3_base}AOI_2_Vegas_Test_public.tar.gz'

# S3 output path for processed data
s3_processed_masks_output_path = f's3://{default_bucket}/{s3_project_prefix}/processed_masks'
s3_raw_images_target_path = f's3://{default_bucket}/{s3_project_prefix}/raw_images/RGB-PanSharpen/'

logger.info(f"\nProcessing Job Configuration:")
logger.info(f"  Raw Training Tarball S3: {s3_raw_train_tarball}")
logger.info(f"  Raw Test Tarball S3: {s3_raw_test_tarball}")
logger.info(f"  Processed Masks Output S3: {s3_processed_masks_output_path}")
logger.info(f"  Raw Images Copy Target S3: {s3_raw_images_target_path}")

# Local path to processing script
processing_script_path = str(Path('./preprocessing/process_data_job.py').resolve())

# --- ScriptProcessor using custom container ---
processor = ScriptProcessor(
    image_uri=image_uri,
    command=["python3"],
    role=role,
    instance_type="ml.m5.xlarge",
    instance_count=1,
    volume_size_in_gb=200,
    sagemaker_session=sagemaker_session,
    base_job_name="spacenet-data-preprocessor"
)

logger.info("Launching SageMaker Processing Job with custom container...")

processor.run(
    code=processing_script_path,
    inputs=[
        ProcessingInput(
            source=s3_raw_train_tarball,
            destination='/opt/ml/processing/input/data/train/'  # Directory, not file
        ),
        ProcessingInput(
            source=s3_raw_test_tarball,
            destination='/opt/ml/processing/input/data/test/'  # Directory, not file
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

logger.info("Processing job finished. Now copying raw images from public SpaceNet S3 to your S3 bucket: %s", s3_raw_images_target_path)

public_s3_images_source = f's3://spacenet-dataset/spacenet/SN2_buildings/tarballs/AOI_2_Vegas_Train/RGB-PanSharpen/'

try:
    import subprocess
    subprocess.run(['aws', 's3', 'cp', public_s3_images_source, s3_raw_images_target_path, '--recursive', '--quiet'], check=True)
    logger.info("✓ Raw images copied to your S3 bucket.")
except Exception as e:
    logger.error(f"❌ Error copying raw images to S3: {e}")
    logger.error("Please ensure your IAM role has S3 read permissions for spacenet-dataset and write for your S3 bucket.")
    raise

logger.info("Data preprocessing and S3 upload steps completed via SageMaker Processing Job and S3 copy.")
