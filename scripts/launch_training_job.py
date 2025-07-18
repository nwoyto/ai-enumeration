# notebooks/launch_sagemaker_job.py
import sagemaker
from sagemaker.pytorch import PyTorch
import boto3
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sagemaker_session = sagemaker.Session()

# You MUST replace this with the actual ARN of your SageMaker execution role.
# This should be the same ARN you used in launch_processing_job.py
role = 'arn:aws:iam::040571275415:role/service-role/AmazonSageMaker-ExecutionRole-20250320T164162' # <<< PASTE YOUR ACTUAL ROLE ARN HERE!

logger.info(f"Using SageMaker Execution Role: {role}")

default_bucket = sagemaker_session.default_bucket()
s3_project_prefix = 'spacenet-building-detection'

# --- IMPORTANT: These S3 paths must now point to YOUR S3 bucket ---
# These are the paths where the preprocessing job would have uploaded the data.
s3_train_images = f's3://{default_bucket}/{s3_project_prefix}/raw_images/RGB-PanSharpen'
s3_val_images = f's3://{default_bucket}/{s3_project_prefix}/raw_images/RGB-PanSharpen' # Validation images come from the same raw source
s3_train_masks = f's3://{default_bucket}/{s3_project_prefix}/processed_masks/train/masks'
s3_val_masks = f's3://{default_bucket}/{s3_project_prefix}/processed_masks/val/masks'

logger.info(f"\nTraining Job Configuration:")
logger.info(f"  Train Images: {s3_train_images}")
logger.info(f"  Train Masks: {s3_train_masks}")
logger.info(f"  Validation Images: {s3_val_images}")
logger.info(f"  Validation Masks: {s3_val_masks}")

# Define the local path to your training scripts.
# This path is relative to where you run this 'launch_sagemaker_job.py' script.
# Assuming this script is in 'notebooks/' and your training code is in 'code/'
code_dir = './code'

# Hyperparameters for your model and training
hyperparameters = {
    'epochs': 50,
    'batch-size': 16,
    'learning-rate': 0.0001,
    'log-interval': 50,
    'num-workers': 8, # Number of data loading workers. Adjust based on instance vCPUs.
    'cnn-model': 'resnet50',
    'num-input-channels': 3, # Ensure this matches the images you preprocessed (RGB=3, MUL=8)
    'vit-embed-dim': 768,
    'vit-depth': 6,
    'vit-heads': 12,
    'gat-hidden-channels': 256,
    'gat-heads': 4
}

estimator = PyTorch(
    entry_point='train.py',
    source_dir=code_dir,
    role=role,
    image_uri='040571275415.dkr.ecr.us-east-1.amazonaws.com/spacenet-geoprocessing:latest',  # <-- Your custom ECR image
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    hyperparameters=hyperparameters,
    output_path=f's3://{default_bucket}/{s3_project_prefix}/output',
    sagemaker_session=sagemaker_session,
    metric_definitions=[
        {'Name': 'Train_Loss', 'Regex': 'Train_Loss=([0-9\.]+)'},
        {'Name': 'Validation_Loss', 'Regex': 'Validation_Loss=([0-9\.]+)'}
    ],
    debugger_hook_config=False,
    max_run=3600 * 4,
)

logger.info("\nLaunching SageMaker training job...")
estimator.fit(
    inputs={
        'train_images': sagemaker.inputs.TrainingInput(s3_train_images, distribution='FullyReplicated'),
        'train_masks': sagemaker.inputs.TrainingInput(s3_train_masks, distribution='FullyReplicated'),
        'val_images': sagemaker.inputs.TrainingInput(s3_val_images, distribution='FullyReplicated'),
        'val_masks': sagemaker.inputs.TrainingInput(s3_val_masks, distribution='FullyReplicated')
    },
    wait=False # Set to True to wait for completion in your local terminal, or False to submit and return control
)

training_job_name = estimator.latest_training_job.name
logger.info(f"Training job launched: {training_job_name}")
logger.info(f"Monitor its progress in the AWS SageMaker console under 'Training Jobs'.")