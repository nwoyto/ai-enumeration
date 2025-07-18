# In launch_training_job.py

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker import image_uris
import os
import logging
import sys # <-- Add this line
# Setup logging (similar to train.py for consistency)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Use INFO level for launch script
logger.addHandler(logging.StreamHandler(sys.stdout))

# --- SageMaker Session Setup ---
try:
    sagemaker_session = sagemaker.Session()
    # Ensure the region matches your S3 bucket region and IAM role setup
    aws_region = sagemaker_session.boto_region_name
    logger.info(f"SageMaker session region: {aws_region}")
except Exception as e:
    logger.error(f"Error setting up SageMaker session: {e}")
    sys.exit(1)

# Get SageMaker Execution Role
# The role from the traceback is 'arn:aws:iam::040571275415:role/service-role/AmazonSageMaker-ExecutionRole-20250320T164162'
# You can hardcode it if it's always the same, or get it dynamically if it's the default
try:
    sagemaker_role = sagemaker_session.get_execution_role()
    logger.info(f"Using SageMaker Execution Role: {sagemaker_role}")
except Exception as e:
    logger.error(f"Error getting SageMaker execution role. Ensure your AWS CLI is configured or the role is defined: {e}")
    sys.exit(1)


# --- S3 Data Paths ---
# You can define these directly or use default_bucket()
bucket_name = sagemaker_session.default_bucket() # Gets default bucket for the session's region
base_s3_uri = f"s3://{bucket_name}/spacenet-building-detection"

s3_train_images = f"{base_s3_uri}/raw_images/RGB-PanSharpen/train" # Assuming 'train' subdir
s3_train_masks = f"{base_s3_uri}/processed_masks/train/masks"
s3_val_images = f"{base_s3_uri}/raw_images/RGB-PanSharpen/val" # Assuming 'val' subdir
s3_val_masks = f"{base_s3_uri}/processed_masks/val/masks"

# Important: Make sure these S3 paths exist and contain your data!
# The paths from your traceback logs:
# Train Images: s3://sagemaker-us-west-2-040571275415/spacenet-building-detection/raw_images/RGB-PanSharpen
# Train Masks: s3://sagemaker-us-west-2-040571275415/spacenet-building-detection/processed_masks/train/masks
# Validation Images: s3://sagemaker-us-west-2-040571275415/spacenet-building-detection/raw_images/RGB-PanSharpen
# Validation Masks: s3://sagemaker-us-west-2-040571275415/spacenet-building-detection/processed_masks/val/masks
# NOTE: Your S3 paths in the traceback for validation images and masks don't explicitly show '/val/'
# Make sure your s3_val_images and s3_val_masks variables correctly reflect your S3 structure.
# If val images are also in RGB-PanSharpen, you might need subfolders like:
# s3_val_images = f"{base_s3_uri}/raw_images/RGB-PanSharpen/val"
# s3_val_masks = f"{base_s3_uri}/processed_masks/val/masks"
# Or if they are truly in the same folder as train:
# s3_val_images = s3_train_images
# s3_val_masks = f"{base_s3_uri}/processed_masks/val/masks" # Mask path seems distinct

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
    'num-input-channels': 8, # Important: SpaceNet often has 8-band imagery
    'pretrained-cnn': True,
    'vit-embed-dim': 768,
    'vit-depth': 4, # As discussed, if you only have one block, this is a placeholder for future stacking
    'vit-heads': 12,
    'gat-heads': 4,
}

# --- Get the correct Docker image URI ---
# This is the most likely fix for your error
framework_version = '2.0.0' # Or the PyTorch version you are using
py_version = 'py310' # Python version in the SageMaker container
instance_type = 'ml.g4dn.xlarge' # Or your chosen instance type

try:
    # This function retrieves the appropriate image URI for your region and setup
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
    source_dir='.', # Points to the directory containing train.py, model.py, dataset.py
    role=sagemaker_role,
    instance_count=1,
    instance_type=instance_type,
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    image_uri=training_image_uri, # <--- **Crucial Fix**
    # Other parameters: output_path, checkpoint_s3_uri etc.
    # output_path=f"s3://{bucket_name}/output/spacenet-geoprocessing", # Optional: custom output S3 path
    # checkpoint_s3_uri=f"s3://{bucket_name}/checkpoints/spacenet-geoprocessing", # Optional: for checkpointing
    # Keep wait=False if you want to launch and detach
    # wait=False,
    # logs=True, # Show logs in console
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