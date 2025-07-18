# In launch_training_job.py

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris
import os
import logging
import sys

# Setup logging (similar to train.py for consistency)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Use INFO level for launch script
logger.addHandler(logging.StreamHandler(sys.stdout))

# --- SageMaker Session Setup ---
import boto3
aws_region = 'us-east-1'
boto3.setup_default_session(region_name=aws_region)
sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=aws_region))
logger.info(f"SageMaker session region: {aws_region}")

# Get SageMaker Execution Role
# CORRECTED: Use sagemaker.get_execution_role() directly
# Set your SageMaker execution role ARN directly
sagemaker_role = 'arn:aws:iam::040571275415:role/service-role/AmazonSageMaker-ExecutionRole-20250320T164162'
logger.info(f"Using SageMaker Execution Role: {sagemaker_role}")


# --- S3 Data Paths ---
# You can define these directly or use default_bucket()
bucket_name = f'sagemaker-{aws_region}-040571275415'
base_s3_uri = f"s3://{bucket_name}/spacenet-building-detection"

# Updated for unified S3 structure (all RGB images under raw_data/AOI_2_Vegas/PS-RGB)
s3_train_images = f"{base_s3_uri}/raw_data/AOI_2_Vegas/PS-RGB"
s3_train_masks = f"{base_s3_uri}/processed_masks/train/masks"
s3_val_images = f"{base_s3_uri}/raw_data/AOI_2_Vegas/PS-RGB"  # Update if you have a separate val split
s3_val_masks = f"{base_s3_uri}/processed_masks/val/masks"

# Important: Make sure these S3 paths exist and contain your data!
logger.info("\nTraining Job Configuration:")
logger.info(f"  Train Images: {s3_train_images}")
logger.info(f"  Train Masks: {s3_train_masks}")
logger.info(f"  Validation Images: {s3_val_images}")
logger.info(f"  Validation Masks: {s3_val_masks}")


# --- Hyperparameters for the training script ---
hyperparameters = {
    'batch-size': 16,
    'epochs': 50,
    'learning-rate': 0.0001,
    'log-interval': 10,
    'num-workers': 4,
    'seed': 42,
    'gradient-clip-val': 1.0,
    'lr-scheduler-patience': 5,
    'plot-interval': 5,
    'num-vis-samples': 3,
    'cnn-model': 'resnet50',
    'num-input-channels': 3, # RGB only
    'pretrained-cnn': True,
    'vit-embed-dim': 768,
    'vit-depth': 4, # As discussed, if you only have one block, this is a placeholder for future stacking
    'vit-heads': 12,
    'gat-heads': 4,
}

# --- Get the correct Docker image URI ---
framework_version = '2.0.0' # Or the PyTorch version you are using
py_version = 'py310' # Python version in the SageMaker container
instance_type = 'ml.g4dn.xlarge' # G-series GPU instance for debugging

try:
    training_image_uri = image_uris.retrieve(
        framework='pytorch',
        region=aws_region,
        version=framework_version,
        py_version=py_version,
        instance_type=instance_type,
        image_scope='training'
    )
    logger.info(f"Using SageMaker training image: {training_image_uri}")
except Exception as e:
    logger.error(f"Error retrieving SageMaker image URI: {e}")
    logger.error("Common issues: wrong framework version, py_version, or instance_type for your region.")
    sys.exit(1)


# --- SageMaker Estimator ---
estimator = PyTorch(
    entry_point='train.py',
    source_dir='code', # Points to the directory containing train.py, model.py, dataset.py
    role=sagemaker_role,
    instance_count=1,
    instance_type=instance_type,
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    image_uri=training_image_uri,
    output_path=f"s3://{bucket_name}/output/spacenet-building-detection", # Model artifacts location
)

logger.info("\nLaunching SageMaker training job...")
# Use TrainingInput for data channels
estimator.fit(
    inputs={
        'train_images': TrainingInput(s3_train_images, distribution='FullyReplicated'),
        'train_masks': TrainingInput(s3_train_masks, distribution='FullyReplicated'),
        'val_images': TrainingInput(s3_val_images, distribution='FullyReplicated'),
        'val_masks': TrainingInput(s3_val_masks, distribution='FullyReplicated')
    },
    wait=True, # Set to True to wait for completion, False to detach
    logs='All' # 'All' for full logs, 'None' for no logs in console
)

logger.info("Training job launched successfully!")

# You can add code here to deploy the model after training, if wait=True
# predictor = estimator.deploy(instance_type='ml.m5.xlarge', initial_instance_count=1)
# logger.info(f"Model deployed to endpoint: {predictor.endpoint_name}")