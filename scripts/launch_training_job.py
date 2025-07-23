# # In launch_training_job.py

# import sagemaker
# from sagemaker.pytorch import PyTorch
# from sagemaker.inputs import TrainingInput
# from sagemaker import image_uris
# import os
# import logging
# import sys

# # Setup logging (similar to train.py for consistency)
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Use INFO level for launch script
# logger.addHandler(logging.StreamHandler(sys.stdout))

# # --- SageMaker Session Setup ---
# import boto3
# aws_region = 'us-east-1'
# boto3.setup_default_session(region_name=aws_region)
# sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=aws_region))
# logger.info(f"SageMaker session region: {aws_region}")

# # Get SageMaker Execution Role
# # CORRECTED: Use sagemaker.get_execution_role() directly
# # Set your SageMaker execution role ARN directly
# sagemaker_role = 'arn:aws:iam::040571275415:role/service-role/AmazonSageMaker-ExecutionRole-20250320T164162'
# logger.info(f"Using SageMaker Execution Role: {sagemaker_role}")


# # --- S3 Data Paths ---
# # You can define these directly or use default_bucket()
# bucket_name = f'sagemaker-{aws_region}-040571275415'
# base_s3_uri = f"s3://{bucket_name}/spacenet-building-detection"

# # Use preprocessed multispectral images and masks for training and validation
# s3_train_images = f"{base_s3_uri}/processed_masks/train/images"
# s3_train_masks = f"{base_s3_uri}/processed_masks/train/masks"
# s3_val_images = f"{base_s3_uri}/processed_masks/val/images"
# s3_val_masks = f"{base_s3_uri}/processed_masks/val/masks"

# # Important: Make sure these S3 paths exist and contain your data!
# logger.info("\nTraining Job Configuration:")
# logger.info(f"  Train Images: {s3_train_images}")
# logger.info(f"  Train Masks: {s3_train_masks}")
# logger.info(f"  Validation Images: {s3_val_images}")
# logger.info(f"  Validation Masks: {s3_val_masks}")


# # --- Hyperparameters for the training script ---
# hyperparameters = {
#     'batch-size': 16,
#     'epochs': 100,
#     'learning-rate': 0.0001,
#     'log-interval': 10,
#     'num-workers': 4,
#     'seed': 42,
#     'gradient-clip-val': 1.0,
#     'lr-scheduler-patience': 5,
#     'plot-interval': 5,
#     'num-vis-samples': 3,
#     'cnn-model': 'resnet50',
#     'num-input-channels': 8, # RGB only = 3; MS = 8
#     'pretrained-cnn': True,
#     'vit-embed-dim': 768,
#     'vit-depth': 4, # As discussed, if you only have one block, this is a placeholder for future stacking
#     'vit-heads': 12,
#     'gat-heads': 4,
# }

# # --- Get the correct Docker image URI ---
# framework_version = '2.0.0' # Or the PyTorch version you are using
# py_version = 'py310' # Python version in the SageMaker container
# instance_type = 'ml.g4dn.xlarge' # G4 GPU instance for cost-effective debugging

# training_image_uri = "040571275415.dkr.ecr.us-east-1.amazonaws.com/spacenet-geoprocessing:latest"
# logger.info(f"Using custom SageMaker training image: {training_image_uri}")


# # --- SageMaker Estimator ---
# from datetime import datetime

# # Define a unique output path for this training job
# output_path = f"{base_s3_uri}/training-outputs/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

# estimator = PyTorch(
#     entry_point='train.py',
#     source_dir='code', # Points to the directory containing train.py, model.py, dataset.py
#     role=sagemaker_role,
#     instance_count=1,
#     instance_type=instance_type,
#     sagemaker_session=sagemaker_session,
#     hyperparameters=hyperparameters,
#     image_uri=training_image_uri,
#     output_path=output_path
# )

# logger.info("\nLaunching SageMaker training job...")
# # Use TrainingInput for data channels
# # --- S3 Data Inputs for SageMaker ---
# inputs = {
#     'train_images': TrainingInput('s3://sagemaker-us-east-1-040571275415/spacenet-building-detection/processed_masks/train/images/', distribution='FullyReplicated'),
#     'train_masks':  TrainingInput('s3://sagemaker-us-east-1-040571275415/spacenet-building-detection/processed_masks/train/masks/', distribution='FullyReplicated'),
#     'val_images':   TrainingInput('s3://sagemaker-us-east-1-040571275415/spacenet-building-detection/processed_masks/val/images/', distribution='FullyReplicated'),
#     'val_masks':    TrainingInput('s3://sagemaker-us-east-1-040571275415/spacenet-building-detection/processed_masks/val/masks/', distribution='FullyReplicated')
# }

# estimator.fit(
#     inputs=inputs,
#     wait=True, # Set to True to wait for completion, False to detach
#     logs='All' # 'All' for full logs, 'None' for no logs in console
# )

# logger.info("Training job launched successfully!")

# # You can add code here to deploy the model after training, if wait=True
# # predictor = estimator.deploy(instance_type='ml.m5.xlarge', initial_instance_count=1)
# # logger.info(f"Model deployed to endpoint: {predictor.endpoint_name}")





###########################################################################
#Gemini Takeover
###########################################################################

# scripts/launch_training_job.py - FIX: Align Training Input to New Processing Output Structure

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris
import os
import logging
import sys
from datetime import datetime # Import datetime for unique job naming

# Setup logging (similar to train.py for consistency)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Use INFO level for launch script
logger.addHandler(logging.StreamHandler(sys.stdout))

# --- SageMaker Session Setup ---
aws_region = 'us-east-1'
boto3.setup_default_session(region_name=aws_region)
sagemaker_session = sagemaker.Session(boto_session=boto3.Session(region_name=aws_region))
logger.info(f"SageMaker session region: {aws_region}")

# Get SageMaker Execution Role
sagemaker_role = 'arn:aws:iam::040571275415:role/service-role/AmazonSageMaker-ExecutionRole-20250320T164162'
logger.info(f"Using SageMaker Execution Role: {sagemaker_role}")


# --- S3 Data Paths (Now consistently matching Processing Job outputs) ---
bucket_name = f'sagemaker-{aws_region}-040571275415'
base_s3_uri = f"s3://{bucket_name}/spacenet-building-detection"

# FIX: Images are now from the 'raw_images' output channel of the preprocessing job (MUL-PanSharpen)
s3_train_images = f"{base_s3_uri}/raw_images/MUL-PanSharpen" # Corrected S3 path for images
s3_train_masks = f"{base_s3_uri}/processed_masks/train/masks"
s3_val_images = f"{base_s3_uri}/raw_images/MUL-PanSharpen"  # Corrected S3 path for images
s3_val_masks = f"{base_s3_uri}/processed_masks/val/masks"

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
    'num-input-channels': 8, # FIX: Set to 8 for MUL-PanSharpen as intended
    'pretrained-cnn': True,
    'vit-embed-dim': 768,
    'vit-depth': 4,
    'vit-heads': 12,
    'gat-heads': 4,
}

# --- Get the correct Docker image URI ---
# Using the custom image that should contain all dependencies for training
training_image_uri = '040571275415.dkr.ecr.us-east-1.amazonaws.com/spacenet-geoprocessing:latest'
logger.info(f"Using custom SageMaker training image: {training_image_uri}")


# --- SageMaker Estimator ---
# Generate a unique job name to ensure fresh code upload and avoid caching issues
current_time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
base_job_name = f"spacenet-geoprocessing-training-{current_time_str}" # New unique name for training job

estimator = PyTorch(
    entry_point='train.py',
    source_dir='code', # Points to the directory containing train.py, model.py, dataset.py
    role=sagemaker_role,
    instance_count=1,
    instance_type='ml.g4dn.xlarge', # GPU instance for training
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    image_uri=training_image_uri, # Use your custom image
    output_path=f"s3://{bucket_name}/output/spacenet-building-detection", # Model artifacts location
    base_job_name=base_job_name, # Use the new unique job name
)

logger.info("\nLaunching SageMaker training job...")
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

logger.info(f"Training job {estimator.latest_training_job.name} launched successfully!")